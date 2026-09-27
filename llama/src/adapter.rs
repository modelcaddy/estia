//! Protocol v2 requests, served by llama-server HTTP calls.
//!
//! | Request | llama-server |
//! |---|---|
//! | `hello`, `ping` | answered here; `ping` also checks `/health` once a model is up |
//! | `load`, `unload` | start or stop the process |
//! | `chat`, `chat_stream` | `POST /v1/chat/completions` |
//! | `generate`, `generate_stream` | the same, with `prompt` as one user turn, as the MLX runner does |
//! | `count_tokens` | `POST /tokenize` |
//! | `embed_batch`, `embed` | `POST /v1/embeddings` |
//!
//! A request for a model that is not running starts it first. One server at a
//! time: a request for a different model, or the same model as the other
//! kind, stops the current one.

use crate::http::{self, SseEvent, SseParser};
use crate::server::{self, Kind, Liveness, Server, Shared, WaitError, EMBED_MAX_TOKENS, STOP_GRACE};
use crate::AdapterOptions;
use serde_json::{json, Map, Value};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A stream that has been silent this long gets a keepalive line, so the
/// engine's silence deadline cannot trip during a long prefill or load.
const KEEPALIVE: Duration = Duration::from_secs(5);
/// `max_tokens` when the request leaves it null, as in the MLX runner.
const DEFAULT_MAX_TOKENS: u64 = 256;
/// Largest response body accepted (an embedding batch is the big one).
const MAX_BODY: usize = 1 << 30;

/// Protocol lines out, one JSON object per line, flushed each time.
pub(crate) struct Out {
    w: RefCell<Box<dyn Write>>,
    last: Cell<Instant>,
}

impl Out {
    pub(crate) fn new(w: Box<dyn Write>) -> Out {
        Out { w: RefCell::new(w), last: Cell::new(Instant::now()) }
    }

    pub(crate) fn emit(&self, v: &Value) -> io::Result<()> {
        let mut line = serde_json::to_vec(v).map_err(io::Error::other)?;
        line.push(b'\n');
        let mut w = self.w.borrow_mut();
        w.write_all(&line)?;
        w.flush()?;
        self.last.set(Instant::now());
        Ok(())
    }

    fn quiet_for(&self) -> Duration {
        self.last.get().elapsed()
    }
}

/// How a request failed.
#[derive(Debug)]
pub(crate) enum Fail {
    /// Answer `{"error": …}` and carry on.
    Msg(String),
    /// A cancel line stopped it.
    Cancelled,
    /// llama-server died while serving a loaded model: answer, then exit so
    /// the engine's `Session` respawns the adapter.
    Died(String),
    /// Stdout is gone; nobody is listening.
    Stdout(io::Error),
}

fn msg(s: impl Into<String>) -> Fail {
    Fail::Msg(s.into())
}

/// Marks an idle-hook error as a cancel.
#[derive(Debug)]
struct CancelledMark;
impl fmt::Display for CancelledMark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("cancelled")
    }
}
impl std::error::Error for CancelledMark {}

/// Marks an idle-hook error as a failed write to stdout.
#[derive(Debug)]
struct StdoutMark(io::Error);
impl fmt::Display for StdoutMark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stdout: {}", self.0)
    }
}
impl std::error::Error for StdoutMark {}

/// What the request loop does next.
pub(crate) enum Flow {
    Continue,
    /// llama-server died; the adapter exits.
    Exit(String),
}

enum Want {
    Kind(Kind),
    /// `count_tokens`: whatever is running for the model will do.
    Any,
}

/// The runner name in `hello`.
pub const RUNNER: &str = "estia-llama";

/// What the adapter declares in `hello`.
pub fn capabilities() -> estia_proto::Capabilities {
    estia_proto::Capabilities {
        generate: true,
        stream: true,
        embed: true,
        cancel: true,
        load: true,
        chat: true,
        tools: true,
        prompt_cache: true,
        count_tokens: true,
        structured: vec!["json".into(), "json_schema".into()],
        parses_tool_calls: true,
        // Read only by a model started with a projector (`mmproj.gguf`);
        // images for one without get a clear error, not silence.
        images: true,
        backend: Some("llama-cpp".into()),
    }
}

pub(crate) fn hello() -> Value {
    json!({
        "ok": true,
        "runner": RUNNER,
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": estia_proto::PROTOCOL_VERSION,
        "capabilities": capabilities(),
    })
}

pub(crate) struct Adapter {
    opts: AdapterOptions,
    shared: Arc<Shared>,
    server: Option<Server>,
    /// The `cache_key` of the conversation slot 0 holds. `None` after a
    /// request without a key, or before any.
    slot_key: Option<String>,
}

impl Drop for Adapter {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Adapter {
    pub(crate) fn new(opts: AdapterOptions, shared: Arc<Shared>) -> Adapter {
        Adapter { opts, shared, server: None, slot_key: None }
    }

    pub(crate) fn stop(&mut self) {
        self.shared.stop_server(STOP_GRACE);
        self.server = None;
        self.slot_key = None;
    }

    /// Serve one request, writing its response line(s). `Err` only when
    /// stdout fails.
    pub(crate) fn handle(&mut self, seq: u64, req: &Value, out: &Out) -> io::Result<Flow> {
        let ty = req.get("type").and_then(Value::as_str).unwrap_or("");
        let streaming = matches!(ty, "chat_stream" | "generate_stream");
        let r = match ty {
            "hello" => out.emit(&hello()).map_err(Fail::Stdout),
            "ping" => self.ping(out),
            "load" => self.load(seq, req, out),
            "unload" => self.unload(req, out),
            "chat" => self.chat(seq, req, Source::Messages, false, out),
            "chat_stream" => self.chat(seq, req, Source::Messages, true, out),
            "generate" => self.chat(seq, req, Source::Prompt, false, out),
            "generate_stream" => self.chat(seq, req, Source::Prompt, true, out),
            "count_tokens" => self.count_tokens(seq, req, out),
            "embed_batch" => self.embed(seq, req, true, out),
            "embed" => self.embed(seq, req, false, out),
            "" => Err(msg("request has no type")),
            other => Err(msg(format!("unknown type {other:?}"))),
        };
        match r {
            Ok(()) => Ok(Flow::Continue),
            Err(Fail::Msg(m)) => out.emit(&json!({ "error": m })).map(|_| Flow::Continue),
            Err(Fail::Cancelled) if streaming => out.emit(&json!({"done": true, "cancelled": true})).map(|_| Flow::Continue),
            Err(Fail::Cancelled) => out.emit(&json!({"error": "cancelled"})).map(|_| Flow::Continue),
            Err(Fail::Died(m)) => {
                self.server = None;
                out.emit(&json!({ "error": m }))?;
                Ok(Flow::Exit(m))
            }
            Err(Fail::Stdout(e)) => Err(e),
        }
    }

    fn ping(&mut self, out: &Out) -> Result<(), Fail> {
        if let Some(s) = &self.server {
            match self.shared.liveness() {
                Liveness::Running if s.ready => match server::health(&s.endpoint) {
                    Some(200) => {}
                    Some(code) => return Err(msg(format!("llama-server is not healthy (HTTP {code})"))),
                    None => return Err(msg("llama-server does not answer")),
                },
                Liveness::Running => {}
                Liveness::Exited(how) if s.ready => return Err(self.died(&how)),
                Liveness::Exited(_) | Liveness::Gone => self.server = None,
            }
        }
        out.emit(&json!({"ok": true})).map_err(Fail::Stdout)
    }

    fn load(&mut self, seq: u64, req: &Value, out: &Out) -> Result<(), Fail> {
        let model = model_of(req)?;
        let kind = match req.get("kind").and_then(Value::as_str) {
            None | Some("generation") => Kind::Generation,
            Some("embedding") => Kind::Embedding,
            Some(other) => return Err(msg(format!("unknown model kind {other:?} (generation|embedding)"))),
        };
        let ms = self.ensure(seq, &model, Want::Kind(kind), None)?.unwrap_or(0);
        out.emit(&json!({"ok": true, "loaded": true, "ms": ms})).map_err(Fail::Stdout)
    }

    fn unload(&mut self, req: &Value, out: &Out) -> Result<(), Fail> {
        let hit = match (&self.server, req.get("model_path").and_then(Value::as_str)) {
            (Some(s), Some(p)) => server::resolve_model(p).map(|m| m == s.model).unwrap_or(false) || Path::new(p) == s.model,
            // No model_path: whatever is loaded.
            (Some(_), None) => true,
            (None, _) => false,
        };
        if hit {
            self.stop();
        }
        out.emit(&json!({"ok": true, "unloaded": hit})).map_err(Fail::Stdout)
    }

    /// The failure for a server that died after it was ready.
    fn died(&mut self, how: &str) -> Fail {
        let (pid, tail) = self.server.as_ref().map(|s| (s.pid, s.log.summary())).unwrap_or_default();
        self.server = None;
        self.slot_key = None;
        let head = format!("llama-server {pid} exited ({how})");
        Fail::Died(if tail.is_empty() { head } else { format!("{head}: {tail}") })
    }

    /// Make sure a ready server runs `model` as `want`, starting it if need
    /// be. Returns the load time in ms when this call started it (or waited
    /// for it). `stream` gets keepalives while waiting.
    fn ensure(&mut self, seq: u64, model: &Path, want: Want, stream: Option<&Out>) -> Result<Option<u64>, Fail> {
        if let Some(s) = &self.server {
            match self.shared.liveness() {
                Liveness::Running => {
                    let fits = s.model == model
                        && match want {
                            Want::Kind(k) => s.kind == k,
                            Want::Any => true,
                        };
                    if fits && s.ready {
                        return Ok(None);
                    }
                    if !fits {
                        self.stop();
                    }
                }
                Liveness::Exited(how) if s.ready => return Err(self.died(&how)),
                Liveness::Exited(_) | Liveness::Gone => self.server = None,
            }
        }
        if self.server.is_none() {
            let kind = match want {
                Want::Kind(k) => k,
                Want::Any => Kind::Generation,
            };
            let s = Server::start(&self.opts, &self.shared, model, kind).map_err(Fail::Msg)?;
            self.server = Some(s);
            self.slot_key = None;
        }
        let shared = Arc::clone(&self.shared);
        let s = self.server.as_mut().expect("set above");
        let mut tick = idle_hook(&shared, seq, stream);
        match s.wait_ready(&shared, &mut tick) {
            Ok(ms) => Ok(Some(ms)),
            Err(WaitError::Exited(why)) => {
                self.server = None;
                Err(msg(format!("llama-server could not load {}: {why}", model.display())))
            }
            Err(WaitError::TimedOut) => {
                self.stop();
                Err(msg(format!("llama-server did not load {} in time", model.display())))
            }
            // The server keeps loading; the next request waits for it.
            Err(WaitError::Aborted(e)) => Err(classify_abort(e, &shared, seq)),
        }
    }

    fn chat(&mut self, seq: u64, req: &Value, source: Source, stream: bool, out: &Out) -> Result<(), Fail> {
        let model = model_of(req)?;
        let mut body = match source {
            Source::Messages => chat_body(req)?,
            Source::Prompt => prompt_body(req)?,
        };
        self.ensure(seq, &model, Want::Kind(Kind::Generation), stream.then_some(out))?;
        let images = matches!(source, Source::Messages) && has_images(req);
        if images && !self.server.as_ref().is_some_and(|s| s.projector) {
            return Err(msg(format!(
                "this model cannot read images: there is no image projector ({}) beside {}",
                server::MMPROJ_FILE,
                model.display()
            )));
        }

        // One slot. It reuses its cached prefix only for the conversation
        // that filled it: a new key, or no key, starts cold, so
        // `cached_tokens` never reveals another conversation's prompt. A turn
        // with images never reuses, and leaves the slot to nobody: image
        // embeddings sit in the cache as placeholders that do not compare
        // like text.
        let key = match source {
            Source::Messages if !images => req.get("cache_key").and_then(Value::as_str).filter(|k| !k.is_empty()),
            _ => None,
        };
        let reuse = key.is_some() && key == self.slot_key.as_deref();
        // Until this request succeeds nobody owns the slot: a request that
        // failed or was cancelled may have left the previous owner's tokens
        // in it, or half of its own prompt.
        self.slot_key = None;
        body["cache_prompt"] = json!(reuse);
        body["id_slot"] = json!(0);
        body["stream"] = json!(stream);
        let result = if stream {
            body["stream_options"] = json!({"include_usage": true});
            self.chat_streamed(seq, &body, out)
        } else {
            self.chat_once(seq, &body, source, out)
        };
        if result.is_ok() {
            self.slot_key = key.map(str::to_string);
        }
        result
    }

    fn chat_once(&mut self, seq: u64, body: &Value, source: Source, out: &Out) -> Result<(), Fail> {
        let v = self.post_json(seq, "/v1/chat/completions", body)?;
        let choice = v.pointer("/choices/0").ok_or_else(|| msg("llama-server returned no choices"))?;
        let text = choice.pointer("/message/content").and_then(Value::as_str).unwrap_or("");
        let calls = choice
            .pointer("/message/tool_calls")
            .and_then(Value::as_array)
            .filter(|c| !c.is_empty())
            .map(|c| c.iter().enumerate().map(|(i, c)| normalize_tool_call(c, i)).collect());
        let resp = match source {
            Source::Messages => {
                let finish = choice.get("finish_reason").and_then(Value::as_str);
                json!({"text": text, "meta": meta(v.get("usage"), v.get("timings"), calls, finish)})
            }
            Source::Prompt => json!({ "text": text }),
        };
        out.emit(&resp).map_err(Fail::Stdout)
    }

    fn chat_streamed(&mut self, seq: u64, body: &Value, out: &Out) -> Result<(), Fail> {
        let shared = Arc::clone(&self.shared);
        let mut acc = ChatAcc::default();
        let result = {
            let s = self.server.as_ref().expect("ensured");
            let bytes = serde_json::to_vec(body).map_err(|e| msg(e.to_string()))?;
            (|| -> io::Result<()> {
                let stream = s.endpoint.connect()?;
                // Registered so the stdin reader can shut it down on a cancel.
                shared.set_active(seq, stream.try_clone().ok());
                if shared.is_cancelled(seq) {
                    return Err(io::Error::other(CancelledMark));
                }
                let req = http::Request { method: "POST", path: "/v1/chat/completions", key: Some(&s.key), body: Some(&bytes) };
                let conn = http::send(stream, &req)?;
                let mut idle = idle_hook(&shared, seq, Some(out));
                let mut resp = conn.read_head(&mut idle)?;
                if resp.head.status != 200 {
                    let raw = resp.read_all(&mut idle, MAX_BODY).unwrap_or_default();
                    return Err(io::Error::other(ServerSaid(server_error(resp.head.status, &raw))));
                }
                let mut sse = SseParser::default();
                let mut events = Vec::new();
                let mut done = false;
                let mut ended = false;
                while !done && !ended {
                    match resp.next_chunk(&mut idle)? {
                        Some(piece) => sse.push(&piece, &mut events)?,
                        None => {
                            sse.finish(&mut events);
                            ended = true;
                        }
                    }
                    for ev in events.drain(..) {
                        if absorb_event(ev, &mut acc, out)? {
                            done = true;
                            break;
                        }
                    }
                }
                if !done && acc.finish_reason.is_none() {
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "llama-server ended the stream early"));
                }
                Ok(())
            })()
        };
        shared.set_active(seq, None);
        result.map_err(|e| self.io_fail(e, seq))?;
        let mut m = meta(acc.usage.as_ref(), acc.timings.as_ref(), acc.tool_calls(), acc.finish_reason.as_deref());
        m["type"] = json!("meta");
        out.emit(&m).map_err(Fail::Stdout)?;
        out.emit(&json!({"done": true})).map_err(Fail::Stdout)
    }

    /// Counted by whichever server runs `model_path`, chat or embedding. With
    /// none running, a chat server is started: an encoder-only embedding
    /// model cannot run as one, so load it with `kind: embedding` first.
    fn count_tokens(&mut self, seq: u64, req: &Value, out: &Out) -> Result<(), Fail> {
        let model = model_of(req)?;
        let text = req.get("text").and_then(Value::as_str).unwrap_or("");
        self.ensure(seq, &model, Want::Any, None)?;
        let tokens = self.tokenize(seq, text)?;
        out.emit(&json!({"tokens": tokens.len()})).map_err(Fail::Stdout)
    }

    fn tokenize(&mut self, seq: u64, text: &str) -> Result<Vec<i64>, Fail> {
        let v = self.post_json(seq, "/tokenize", &json!({"content": text, "add_special": true}))?;
        let ids = v.get("tokens").and_then(Value::as_array).ok_or_else(|| msg("llama-server /tokenize returned no tokens"))?;
        // With pieces the entries are objects; we never ask for them.
        Ok(ids.iter().filter_map(|t| t.as_i64().or_else(|| t.get("id").and_then(Value::as_i64))).collect())
    }

    fn embed(&mut self, seq: u64, req: &Value, batch: bool, out: &Out) -> Result<(), Fail> {
        let model = model_of(req)?;
        let inputs: Vec<String> = if batch {
            let list = req.get("inputs").and_then(Value::as_array).ok_or_else(|| msg("inputs must be a list of strings"))?;
            list.iter()
                .map(|v| v.as_str().map(str::to_string).ok_or_else(|| msg("inputs must be a list of strings")))
                .collect::<Result<_, _>>()?
        } else {
            vec![req.get("input").and_then(Value::as_str).ok_or_else(|| msg("input must be a string"))?.to_string()]
        };
        if inputs.is_empty() {
            return out.emit(&json!({"embeddings": []})).map_err(Fail::Stdout);
        }
        self.ensure(seq, &model, Want::Kind(Kind::Embedding), None)?;
        let prepared = self.cut_long_inputs(seq, inputs)?;
        let n = prepared.len();
        // embd_normalize 2 is L2, llama-server's default for this route,
        // sent so a changed default cannot change the vectors. The MLX runner
        // returns L2-normalised vectors too (mlx-embeddings' text_embeds).
        let v = self.post_json(seq, "/v1/embeddings", &json!({"input": prepared, "encoding_format": "float", "embd_normalize": 2}))?;
        let vectors = parse_embeddings(&v, n)?;
        let resp =
            if batch { json!({ "embeddings": vectors }) } else { json!({ "embedding": vectors.into_iter().next().unwrap_or_default() }) };
        out.emit(&resp).map_err(Fail::Stdout)
    }

    /// Inputs longer than [`EMBED_MAX_TOKENS`] tokens are cut to that length,
    /// keeping the special tokens the tokenizer adds at either end, and sent
    /// as token ids. The MLX runner truncates at 512 tokens too. Short inputs
    /// go as text: no text of that many bytes can reach the limit.
    fn cut_long_inputs(&mut self, seq: u64, inputs: Vec<String>) -> Result<Vec<Value>, Fail> {
        let mut prepared = Vec::with_capacity(inputs.len());
        for text in inputs {
            if text.len() + 16 <= EMBED_MAX_TOKENS {
                prepared.push(Value::String(text));
                continue;
            }
            let tokens = self.tokenize(seq, &text)?;
            if tokens.len() <= EMBED_MAX_TOKENS {
                prepared.push(Value::String(text));
                continue;
            }
            let specials = match self.server.as_ref().and_then(|s| s.specials.clone()) {
                Some(sp) => sp,
                None => {
                    let sp = self.tokenize(seq, "")?;
                    if let Some(s) = self.server.as_mut() {
                        s.specials = Some(sp.clone());
                    }
                    sp
                }
            };
            prepared.push(json!(truncate_tokens(&tokens, &specials, EMBED_MAX_TOKENS)));
        }
        Ok(prepared)
    }

    /// POST a JSON body and parse the JSON answer; non-200 is an error.
    fn post_json(&mut self, seq: u64, path: &str, body: &Value) -> Result<Value, Fail> {
        let shared = Arc::clone(&self.shared);
        let s = self.server.as_ref().ok_or_else(|| msg("no model is loaded"))?;
        let bytes = serde_json::to_vec(body).map_err(|e| msg(e.to_string()))?;
        let result = (|| -> io::Result<Value> {
            let stream = s.endpoint.connect()?;
            let conn = http::send(stream, &http::Request { method: "POST", path, key: Some(&s.key), body: Some(&bytes) })?;
            let mut idle = idle_hook(&shared, seq, None);
            let mut resp = conn.read_head(&mut idle)?;
            let raw = resp.read_all(&mut idle, MAX_BODY)?;
            if resp.head.status != 200 {
                return Err(io::Error::other(ServerSaid(server_error(resp.head.status, &raw))));
            }
            serde_json::from_slice(&raw).map_err(|e| io::Error::other(ServerSaid(format!("llama-server sent invalid JSON: {e}"))))
        })();
        result.map_err(|e| self.io_fail(e, seq))
    }

    /// Turn a failed HTTP exchange into the right answer: a cancel, a dead
    /// server, stdout gone, or an error message.
    fn io_fail(&mut self, e: io::Error, seq: u64) -> Fail {
        if let Some(inner) = e.get_ref() {
            if let Some(said) = inner.downcast_ref::<ServerSaid>() {
                return msg(said.0.clone());
            }
        }
        match classify_abort(e, &self.shared, seq) {
            Fail::Msg(m) => {
                // A reset connection may be the server dying: let it finish
                // exiting before looking.
                for _ in 0..10 {
                    match self.shared.liveness() {
                        Liveness::Running => std::thread::sleep(Duration::from_millis(50)),
                        Liveness::Exited(how) => return self.died(&how),
                        Liveness::Gone => break,
                    }
                }
                msg(format!("llama-server request failed: {m}"))
            }
            other => other,
        }
    }
}

/// An error llama-server answered with, already worded.
#[derive(Debug)]
struct ServerSaid(String);
impl fmt::Display for ServerSaid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ServerSaid {}

fn classify_abort(e: io::Error, shared: &Shared, seq: u64) -> Fail {
    if let Some(inner) = e.get_ref() {
        if inner.is::<CancelledMark>() {
            return Fail::Cancelled;
        }
        if let Some(StdoutMark(err)) = inner.downcast_ref::<StdoutMark>() {
            return Fail::Stdout(io::Error::new(err.kind(), err.to_string()));
        }
    }
    if shared.is_cancelled(seq) {
        return Fail::Cancelled;
    }
    Fail::Msg(e.to_string())
}

/// The hook run while a request waits on llama-server: stop on a cancel, and
/// in a stream keep the engine's silence deadline at bay.
fn idle_hook<'a>(shared: &'a Shared, seq: u64, stream: Option<&'a Out>) -> impl FnMut() -> io::Result<()> + 'a {
    move || {
        if shared.is_cancelled(seq) {
            return Err(io::Error::other(CancelledMark));
        }
        if let Some(out) = stream {
            if out.quiet_for() >= KEEPALIVE {
                out.emit(&json!({"type": "keepalive"})).map_err(|e| io::Error::other(StdoutMark(e)))?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Source {
    /// `chat`: `messages`.
    Messages,
    /// `generate`: `prompt`, sent as one user turn.
    Prompt,
}

fn model_of(req: &Value) -> Result<PathBuf, Fail> {
    let p = req.get("model_path").and_then(Value::as_str).ok_or_else(|| msg("model_path is required"))?;
    server::resolve_model(p).map_err(Fail::Msg)
}

/// `{"error": {"message": …}}` or whatever else came back, as one line.
fn server_error(status: u16, raw: &[u8]) -> String {
    let text = match serde_json::from_slice::<Value>(raw) {
        Ok(v) => match v.get("error") {
            Some(Value::Object(e)) => {
                e.get("message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| Value::Object(e.clone()).to_string())
            }
            Some(Value::String(s)) => s.clone(),
            _ => v.to_string(),
        },
        Err(_) => String::from_utf8_lossy(raw).trim().chars().take(500).collect(),
    };
    format!("llama-server HTTP {status}: {text}")
}

/// The OpenAI chat body for a `chat` / `chat_stream` request, without the
/// cache and stream fields.
pub(crate) fn chat_body(req: &Value) -> Result<Value, Fail> {
    let msgs = req.get("messages").and_then(Value::as_array).ok_or_else(|| msg("messages must be a list"))?;
    if msgs.is_empty() {
        return Err(msg("messages is empty"));
    }
    let messages: Vec<Value> = msgs.iter().map(map_message).collect::<Result<_, _>>()?;
    let mut body = json!({ "messages": messages });
    if let Some(tools) = req.get("tools").and_then(Value::as_array).filter(|t| !t.is_empty()) {
        body["tools"] = Value::Array(tools.clone());
    }
    if let Some(rf) = response_format(req.get("format"))? {
        body["response_format"] = rf;
    }
    sampling(&mut body, req);
    Ok(body)
}

/// `generate` / `generate_stream`: the prompt as one user turn.
pub(crate) fn prompt_body(req: &Value) -> Result<Value, Fail> {
    let prompt = req.get("prompt").and_then(Value::as_str).ok_or_else(|| msg("prompt must be a string"))?;
    let mut body = json!({ "messages": [{"role": "user", "content": prompt}] });
    if req.get("json").and_then(Value::as_bool) == Some(true) {
        body["response_format"] = json!({"type": "json_object"});
    }
    sampling(&mut body, req);
    Ok(body)
}

/// Every sampling value explicitly: llama-server's defaults (temperature
/// 0.8, top_k 40, top_p 0.95, min_p 0.05, unlimited tokens) are not the MLX
/// runner's. Null `max_tokens` / `temperature` mean 256 / 0.0, as there.
fn sampling(body: &mut Value, req: &Value) {
    let max_tokens = req.get("max_tokens").and_then(Value::as_u64).filter(|n| *n > 0).unwrap_or(DEFAULT_MAX_TOKENS);
    let temperature = req.get("temperature").and_then(Value::as_f64).filter(|t| t.is_finite() && *t > 0.0).unwrap_or(0.0);
    body["max_tokens"] = json!(max_tokens);
    body["temperature"] = json!(temperature);
    body["top_k"] = json!(0);
    body["top_p"] = json!(1.0);
    body["min_p"] = json!(0.0);
    body["repeat_penalty"] = json!(1.0);
    // Thinking off, so output matches the MLX runner until Estia exposes
    // reasoning.
    body["chat_template_kwargs"] = json!({"enable_thinking": false});
}

/// Whether any message in a `chat` request carries images.
fn has_images(req: &Value) -> bool {
    req.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|ms| ms.iter().any(|m| m.get("images").and_then(Value::as_array).is_some_and(|i| !i.is_empty())))
}

fn map_message(m: &Value) -> Result<Value, Fail> {
    let o = m.as_object().ok_or_else(|| msg("each message must be an object"))?;
    let role = o.get("role").and_then(Value::as_str).ok_or_else(|| msg("each message needs a role"))?;
    let mut content = match o.get("content") {
        None | Some(Value::Null) => Value::String(String::new()),
        Some(Value::String(s)) => Value::String(s.clone()),
        // OpenAI content parts; llama-server takes text parts.
        Some(Value::Array(parts)) => Value::Array(parts.clone()),
        Some(_) => return Err(msg("message content must be a string")),
    };
    // Images become OpenAI `image_url` parts with a data URL, before the
    // text, which is where Gemma's template puts them.
    if let Some(images) = o.get("images").and_then(Value::as_array).filter(|i| !i.is_empty()) {
        let mut parts = Vec::with_capacity(images.len() + 1);
        for img in images {
            let mime = img.get("mime").and_then(Value::as_str).ok_or_else(|| msg("each image needs a mime"))?;
            let data = img.get("data").and_then(Value::as_str).ok_or_else(|| msg("each image needs data"))?;
            parts.push(json!({"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{data}")}}));
        }
        match content {
            Value::String(t) if !t.is_empty() => parts.push(json!({"type": "text", "text": t})),
            Value::Array(existing) => parts.extend(existing),
            _ => {}
        }
        content = Value::Array(parts);
    }
    let mut out = Map::new();
    out.insert("role".into(), json!(role));
    out.insert("content".into(), content);
    for f in ["name", "tool_call_id"] {
        if let Some(v) = o.get(f).and_then(Value::as_str) {
            out.insert(f.into(), json!(v));
        }
    }
    if let Some(calls) = o.get("tool_calls").and_then(Value::as_array).filter(|c| !c.is_empty()) {
        out.insert("tool_calls".into(), Value::Array(calls.iter().enumerate().map(|(i, c)| normalize_tool_call(c, i)).collect()));
    }
    Ok(Value::Object(out))
}

/// OpenAI shape with `arguments` as a JSON string, whichever way it came.
fn normalize_tool_call(c: &Value, i: usize) -> Value {
    let f = c.get("function").cloned().unwrap_or(Value::Null);
    let name = f.get("name").and_then(Value::as_str).unwrap_or("");
    let arguments = match f.get("arguments") {
        Some(Value::String(s)) => s.clone(),
        None | Some(Value::Null) => "{}".to_string(),
        Some(other) => other.to_string(),
    };
    let id = c.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string).unwrap_or_else(|| format!("call_{i}"));
    json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}})
}

/// The engine's `OutputFormat` (`{"type":"json"}`, `{"type":"json_schema",
/// "schema":…}`) as an OpenAI `response_format`. The OpenAI spellings
/// (`json_object`, `json_schema.schema`) are accepted too.
pub(crate) fn response_format(f: Option<&Value>) -> Result<Option<Value>, Fail> {
    let ty = match f {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(s)) => s.as_str(),
        Some(Value::Object(o)) => o.get("type").and_then(Value::as_str).unwrap_or(""),
        Some(_) => return Err(msg("format must be an object")),
    };
    match ty {
        "" | "text" => Ok(None),
        "json" | "json_object" => Ok(Some(json!({"type": "json_object"}))),
        "json_schema" => {
            let f = f.expect("matched an object");
            let schema = f.get("schema").or_else(|| f.pointer("/json_schema/schema"));
            match schema {
                Some(s) if s.is_object() || s.is_boolean() => {
                    Ok(Some(json!({"type": "json_schema", "json_schema": {"name": "output", "schema": s}})))
                }
                _ => Err(msg("format json_schema needs a schema object")),
            }
        }
        other => Err(msg(format!("unsupported format type {other:?} (json | json_schema)"))),
    }
}

/// One parsed stream event: emit its text, collect the rest. `Ok(true)` at
/// the end of the stream.
fn absorb_event(ev: SseEvent, acc: &mut ChatAcc, out: &Out) -> io::Result<bool> {
    let data = ev.data.trim();
    if ev.event.as_deref() == Some("error") {
        return Err(io::Error::other(ServerSaid(format!("llama-server stream error: {}", error_text(data)))));
    }
    if data == "[DONE]" {
        return Ok(true);
    }
    if data.is_empty() {
        return Ok(false);
    }
    let v: Value = serde_json::from_str(data).map_err(|e| io::Error::other(ServerSaid(format!("llama-server sent a bad event: {e}"))))?;
    if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
        return Err(io::Error::other(ServerSaid(format!("llama-server stream error: {}", error_text(&err.to_string())))));
    }
    for text in acc.absorb(&v) {
        out.emit(&json!({"type": "token", "text": text})).map_err(|e| io::Error::other(StdoutMark(e)))?;
    }
    Ok(false)
}

fn error_text(data: &str) -> String {
    match serde_json::from_str::<Value>(data) {
        Ok(v) => v
            .get("message")
            .or_else(|| v.pointer("/error/message"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| v.to_string()),
        Err(_) => data.to_string(),
    }
}

#[derive(Default, Debug)]
struct ToolCallAcc {
    id: String,
    name: String,
    arguments: String,
}

/// What a chat stream carried besides its text.
#[derive(Default, Debug)]
pub(crate) struct ChatAcc {
    tool_calls: BTreeMap<u64, ToolCallAcc>,
    finish_reason: Option<String>,
    usage: Option<Value>,
    timings: Option<Value>,
}

impl ChatAcc {
    /// Take one `chat.completion.chunk`; returns its content text pieces.
    fn absorb(&mut self, chunk: &Value) -> Vec<String> {
        if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(u.clone());
        }
        if let Some(t) = chunk.get("timings").filter(|t| t.is_object()) {
            self.timings = Some(t.clone());
        }
        let mut texts = Vec::new();
        for choice in chunk.get("choices").and_then(Value::as_array).into_iter().flatten() {
            if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
                self.finish_reason = Some(fr.to_string());
            }
            let Some(delta) = choice.get("delta") else { continue };
            if let Some(t) = delta.get("content").and_then(Value::as_str) {
                if !t.is_empty() {
                    texts.push(t.to_string());
                }
            }
            // Tool calls arrive in pieces keyed by index: the id and name
            // once, the arguments as string fragments.
            for (i, call) in delta.get("tool_calls").and_then(Value::as_array).into_iter().flatten().enumerate() {
                let idx = call.get("index").and_then(Value::as_u64).unwrap_or(i as u64);
                let e = self.tool_calls.entry(idx).or_default();
                if let Some(id) = call.get("id").and_then(Value::as_str) {
                    if e.id.is_empty() {
                        e.id = id.to_string();
                    }
                }
                if let Some(f) = call.get("function") {
                    if let Some(n) = f.get("name").and_then(Value::as_str) {
                        e.name.push_str(n);
                    }
                    match f.get("arguments") {
                        Some(Value::String(a)) => e.arguments.push_str(a),
                        Some(Value::Null) | None => {}
                        Some(other) => e.arguments.push_str(&other.to_string()),
                    }
                }
            }
        }
        texts
    }

    fn tool_calls(&self) -> Option<Vec<Value>> {
        if self.tool_calls.is_empty() {
            return None;
        }
        Some(
            self.tool_calls
                .values()
                .enumerate()
                .map(|(i, c)| {
                    let id = if c.id.is_empty() { format!("call_{i}") } else { c.id.clone() };
                    let args = if c.arguments.is_empty() { "{}".to_string() } else { c.arguments.clone() };
                    json!({"id": id, "type": "function", "function": {"name": c.name, "arguments": args}})
                })
                .collect(),
        )
    }
}

/// The protocol's meta object from llama-server's `usage` and `timings`.
///
/// `usage.prompt_tokens` counts the whole prompt, cached part included
/// (`timings.prompt_n` is only the part prefilled now, `cache_n` the rest).
pub(crate) fn meta(usage: Option<&Value>, timings: Option<&Value>, tool_calls: Option<Vec<Value>>, finish_reason: Option<&str>) -> Value {
    let at = |v: Option<&Value>, ptr: &str| v.and_then(|v| v.pointer(ptr)).and_then(Value::as_u64);
    let cached = at(usage, "/prompt_tokens_details/cached_tokens").or_else(|| at(timings, "/cache_n"));
    let prompt = at(usage, "/prompt_tokens").or_else(|| match (at(timings, "/prompt_n"), at(timings, "/cache_n")) {
        (Some(p), c) => Some(p + c.unwrap_or(0)),
        _ => None,
    });
    let generated = at(usage, "/completion_tokens").or_else(|| at(timings, "/predicted_n"));
    let mut m = json!({
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "generation_tokens": generated,
        "template": "native",
    });
    if let Some(tps) = timings.and_then(|t| t.get("predicted_per_second")).and_then(Value::as_f64).filter(|t| t.is_finite()) {
        m["generation_tps"] = json!(tps);
    }
    if let Some(calls) = tool_calls {
        m["tool_calls"] = Value::Array(calls);
    }
    if let Some(fr) = finish_reason {
        m["finish_reason"] = json!(fr);
    }
    m
}

/// Cut `tokens` (a tokenization with special tokens added) to `max`, keeping
/// the trailing specials (`specials` is what the tokenizer adds to empty
/// text: BOS/CLS first, EOS/SEP last).
pub(crate) fn truncate_tokens(tokens: &[i64], specials: &[i64], max: usize) -> Vec<i64> {
    if tokens.len() <= max {
        return tokens.to_vec();
    }
    let prefix = specials.iter().zip(tokens).take_while(|(a, b)| a == b).count();
    let suffix = &specials[prefix..];
    let suffix: &[i64] = if !suffix.is_empty() && tokens.ends_with(suffix) { suffix } else { &[] };
    let keep = max.saturating_sub(suffix.len()).max(prefix);
    let mut t = tokens[..keep.min(tokens.len())].to_vec();
    t.extend_from_slice(suffix);
    t
}

/// `data[i].embedding` in input order. Non-finite values (JSON null) become
/// 0.0, as in the MLX runner: the line protocol cannot carry NaN.
pub(crate) fn parse_embeddings(v: &Value, n: usize) -> Result<Vec<Vec<f32>>, Fail> {
    let data = v.get("data").and_then(Value::as_array).ok_or_else(|| msg("llama-server returned no embeddings"))?;
    let mut slots: Vec<Option<Vec<f32>>> = vec![None; n];
    for (i, item) in data.iter().enumerate() {
        let idx = item.get("index").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(i);
        let emb =
            item.get("embedding").and_then(Value::as_array).ok_or_else(|| msg("llama-server returned an embedding that is not a list"))?;
        if emb.first().is_some_and(Value::is_array) {
            return Err(msg("the model returns one vector per token (no pooling); an embedding model needs pooling"));
        }
        let vec: Vec<f32> = emb.iter().map(|x| x.as_f64().filter(|f| f.is_finite()).unwrap_or(0.0) as f32).collect();
        let slot = slots.get_mut(idx).ok_or_else(|| msg(format!("llama-server returned embedding index {idx} for {n} inputs")))?;
        *slot = Some(vec);
    }
    let got = slots.iter().filter(|s| s.is_some()).count();
    if got != n {
        return Err(msg(format!("llama-server returned {got} embeddings for {n} inputs")));
    }
    Ok(slots.into_iter().map(|s| s.unwrap_or_default()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl Sink {
        fn lines(&self) -> Vec<Value> {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
        }
    }

    fn adapter() -> Adapter {
        let opts = AdapterOptions {
            server_bin: "/nonexistent/llama-server".into(),
            run_dir: std::env::temp_dir().join("estia-llama-unit"),
            context_length: None,
            extra_args: vec![],
        };
        Adapter::new(opts, Arc::new(Shared::default()))
    }

    fn ask(a: &mut Adapter, req: Value) -> Vec<Value> {
        let sink = Sink::default();
        let out = Out::new(Box::new(sink.clone()));
        assert!(matches!(a.handle(1, &req, &out).unwrap(), Flow::Continue));
        sink.lines()
    }

    #[test]
    fn hello_declares_the_contract() {
        let mut a = adapter();
        let got = ask(&mut a, json!({"type": "hello"}));
        assert_eq!(got.len(), 1);
        let h: estia_proto::HelloResp = serde_json::from_value(got[0].clone()).unwrap();
        assert_eq!(h.runner, "estia-llama");
        assert_eq!(h.protocol, 2);
        let c = h.capabilities;
        assert!(c.generate && c.stream && c.embed && c.cancel && c.load && c.chat && c.tools && c.prompt_cache && c.count_tokens);
        assert!(c.parses_tool_calls);
        assert_eq!(c.structured, ["json", "json_schema"]);
        assert_eq!(c.backend.as_deref(), Some("llama-cpp"));
        assert_eq!(got[0]["ok"], true);
    }

    #[test]
    fn bad_requests_get_error_lines_not_panics() {
        let mut a = adapter();
        assert_eq!(ask(&mut a, json!({"type": "ping"})), [json!({"ok": true})]);
        for req in [
            json!({}),
            json!([1, 2]),
            json!("chat"),
            json!({"type": "nope"}),
            json!({"type": 5}),
            json!({"type": "chat"}),
            json!({"type": "chat", "model_path": "/nonexistent/m.gguf", "messages": [{"role": "user", "content": "hi"}]}),
            json!({"type": "chat_stream", "model_path": 3}),
            json!({"type": "load", "model_path": "/nonexistent/m.gguf", "kind": "generation"}),
            json!({"type": "count_tokens"}),
            json!({"type": "embed_batch", "model_path": "/x", "inputs": "a"}),
            json!({"type": "embed", "model_path": "/x", "input": 7}),
            json!({"type": "generate_stream", "model_path": "/x"}),
        ] {
            let got = ask(&mut a, req.clone());
            assert_eq!(got.len(), 1, "{req}: {got:?}");
            assert!(got[0]["error"].is_string(), "{req}: {got:?}");
        }
        // Nothing loaded: unload says so.
        assert_eq!(ask(&mut a, json!({"type": "unload", "model_path": "/x"})), [json!({"ok": true, "unloaded": false})]);
    }

    #[test]
    fn a_missing_server_binary_is_an_error_line() {
        let dir = std::env::temp_dir().join(format!("estia-llama-unit-{}", server::random_hex(4).unwrap()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        let mut a = adapter();
        a.opts.run_dir = dir.join("run");
        let got = ask(&mut a, json!({"type": "load", "model_path": model, "kind": "embedding"}));
        assert!(got[0]["error"].as_str().unwrap().contains("cannot start"), "{got:?}");
        let got = ask(&mut a, json!({"type": "load", "model_path": model, "kind": "vision"}));
        assert!(got[0]["error"].as_str().unwrap().contains("unknown model kind"), "{got:?}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn images_become_data_url_parts_before_the_text() {
        let req = json!({"messages": [
            {"role": "user", "content": "What is in this picture?", "images": [
                {"mime": "image/png", "data": "iVBORw0KGgo="},
                {"mime": "image/jpeg", "data": "/9j/4AAQ"}
            ]},
            {"role": "assistant", "content": "A cat."},
            {"role": "user", "content": "", "images": [{"mime": "image/webp", "data": "UklGRg=="}]}
        ]});
        assert!(has_images(&req));
        let body = chat_body(&req).unwrap();
        let m = &body["messages"];
        assert_eq!(m[0]["content"][0], json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}));
        assert_eq!(m[0]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,/9j/4AAQ");
        assert_eq!(m[0]["content"][2], json!({"type": "text", "text": "What is in this picture?"}));
        assert_eq!(m[1]["content"], "A cat.", "a turn without images stays a string");
        assert_eq!(m[2]["content"].as_array().unwrap().len(), 1, "no empty text part");
        assert!(m[0].get("images").is_none(), "images are not passed on as a field");
        assert!(!has_images(&json!({"messages": [{"role": "user", "content": "hi"}]})));
        let bad = json!({"messages": [{"role": "user", "content": "x", "images": [{"data": "AAAA"}]}]});
        assert!(chat_body(&bad).is_err(), "an image without a mime is refused");
    }

    #[test]
    fn chat_body_maps_messages_tools_format_and_sampling() {
        let req = json!({
            "type": "chat",
            "messages": [
                {"role": "system", "content": "terse"},
                {"role": "user", "content": "weather in Athens?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Athens"}}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "{\"temp_c\":24}"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
            "format": {"type": "json_schema", "schema": {"type": "object"}},
            "max_tokens": null,
            "temperature": null,
        });
        let b = chat_body(&req).unwrap();
        assert_eq!(b["messages"][0], json!({"role": "system", "content": "terse"}));
        assert_eq!(b["messages"][2]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Athens\"}");
        assert_eq!(
            b["messages"][3],
            json!({"role": "tool", "content": "{\"temp_c\":24}", "name": "get_weather", "tool_call_id": "call_1"})
        );
        assert_eq!(b["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(b["response_format"], json!({"type": "json_schema", "json_schema": {"name": "output", "schema": {"type": "object"}}}));
        assert_eq!(b["max_tokens"], 256);
        assert_eq!(b["temperature"], 0.0);
        assert_eq!((b["top_k"].clone(), b["top_p"].clone(), b["min_p"].clone()), (json!(0), json!(1.0), json!(0.0)));
        assert_eq!(b["chat_template_kwargs"], json!({"enable_thinking": false}));

        let b =
            chat_body(&json!({"messages": [{"role": "user", "content": null}], "max_tokens": 9, "temperature": 0.7, "tools": []})).unwrap();
        assert_eq!(b["messages"][0]["content"], "");
        assert_eq!((b["max_tokens"].clone(), b["temperature"].clone()), (json!(9), json!(0.7)));
        assert!(b.get("tools").is_none() && b.get("response_format").is_none());

        assert!(chat_body(&json!({"messages": []})).is_err());
        assert!(chat_body(&json!({"messages": [{"content": "no role"}]})).is_err());
        assert!(chat_body(&json!({"messages": [{"role": "user", "content": 3}]})).is_err());
    }

    #[test]
    fn prompt_is_one_user_turn() {
        let b = prompt_body(&json!({"prompt": "Name a sea.", "max_tokens": 16, "temperature": 0.0, "json": true})).unwrap();
        assert_eq!(b["messages"], json!([{"role": "user", "content": "Name a sea."}]));
        assert_eq!(b["response_format"], json!({"type": "json_object"}));
        assert_eq!(b["max_tokens"], 16);
        assert!(prompt_body(&json!({"prompt": null})).is_err());
    }

    #[test]
    fn formats_map_to_response_format() {
        let rf = |v: Value| response_format(Some(&v));
        assert!(rf(json!(null)).unwrap().is_none());
        assert!(rf(json!({"type": "text"})).unwrap().is_none());
        assert_eq!(rf(json!({"type": "json"})).unwrap().unwrap(), json!({"type": "json_object"}));
        assert_eq!(rf(json!({"type": "json_object"})).unwrap().unwrap(), json!({"type": "json_object"}));
        let s = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        let want = json!({"type": "json_schema", "json_schema": {"name": "output", "schema": s}});
        assert_eq!(rf(json!({"type": "json_schema", "schema": s})).unwrap().unwrap(), want);
        assert_eq!(rf(json!({"type": "json_schema", "json_schema": {"name": "x", "schema": s}})).unwrap().unwrap(), want);
        assert!(rf(json!({"type": "json_schema"})).is_err());
        assert!(rf(json!({"type": "xml"})).is_err());
        assert!(rf(json!(5)).is_err());
    }

    #[test]
    fn stream_chunks_collect_text_tool_calls_and_usage() {
        let mut acc = ChatAcc::default();
        let chunks = [
            json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": null}, "finish_reason": null}]}),
            json!({"choices": [{"index": 0, "delta": {"content": "Let me"}}]}),
            json!({"choices": [{"index": 0, "delta": {"content": ""}}]}),
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c9", "type": "function", "function": {"name": "get_weather", "arguments": "{\"ci"}}]}}]}),
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ty\": \"Athens\"}"}}]}}]}),
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"name": "now", "arguments": ""}}]}}]}),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"completion_tokens": 12, "prompt_tokens": 40, "prompt_tokens_details": {"cached_tokens": 30}},
                   "timings": {"cache_n": 30, "prompt_n": 10, "predicted_n": 12, "predicted_per_second": 51.5}}),
        ];
        let texts: Vec<String> = chunks.iter().flat_map(|c| acc.absorb(c)).collect();
        assert_eq!(texts, ["Let me"]);
        let m = meta(acc.usage.as_ref(), acc.timings.as_ref(), acc.tool_calls(), acc.finish_reason.as_deref());
        assert_eq!(
            m,
            json!({
                "prompt_tokens": 40, "cached_tokens": 30, "generation_tokens": 12, "template": "native",
                "generation_tps": 51.5, "finish_reason": "tool_calls",
                "tool_calls": [
                    {"id": "c9", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Athens\"}"}},
                    {"id": "call_1", "type": "function", "function": {"name": "now", "arguments": "{}"}},
                ],
            })
        );
        // The meta line parses as the proto type.
        let g: estia_proto::GenerationMeta = serde_json::from_value(m).unwrap();
        assert_eq!(g.tool_calls.unwrap().len(), 2);
        assert_eq!(g.generation_tps, Some(51.5));
    }

    #[test]
    fn meta_falls_back_to_timings() {
        let t = json!({"cache_n": 4, "prompt_n": 6, "predicted_n": 3});
        let m = meta(None, Some(&t), None, None);
        assert_eq!(
            (m["prompt_tokens"].clone(), m["cached_tokens"].clone(), m["generation_tokens"].clone()),
            (json!(10), json!(4), json!(3))
        );
        let m = meta(None, None, None, None);
        assert!(m["prompt_tokens"].is_null() && m.get("generation_tps").is_none());
    }

    #[test]
    fn stream_events_become_token_lines() {
        let sink = Sink::default();
        let out = Out::new(Box::new(sink.clone()));
        let mut acc = ChatAcc::default();
        let ev = |d: &str| SseEvent { event: None, data: d.into() };
        assert!(!absorb_event(ev(r#"{"choices":[{"delta":{"content":"Hi"}}]}"#), &mut acc, &out).unwrap());
        assert!(absorb_event(ev("[DONE]"), &mut acc, &out).unwrap());
        assert_eq!(sink.lines(), [json!({"type": "token", "text": "Hi"})]);
        let err = absorb_event(SseEvent { event: Some("error".into()), data: r#"{"code":500,"message":"boom"}"#.into() }, &mut acc, &out)
            .unwrap_err();
        assert!(err.to_string().contains("boom"));
        let err = absorb_event(ev(r#"{"error":{"message":"context full"}}"#), &mut acc, &out).unwrap_err();
        assert!(err.to_string().contains("context full"));
        assert!(absorb_event(ev("{not json"), &mut acc, &out).is_err());
    }

    #[test]
    fn long_inputs_keep_their_end_tokens() {
        // BERT-like: [CLS] … [SEP].
        let toks: Vec<i64> = std::iter::once(101).chain(1000..1700).chain(std::iter::once(102)).collect();
        let t = truncate_tokens(&toks, &[101, 102], 512);
        assert_eq!(t.len(), 512);
        assert_eq!((t[0], t[1], t[510], t[511]), (101, 1000, 1509, 102));
        // Gemma-like: BOS only.
        let toks: Vec<i64> = std::iter::once(2).chain(10..900).collect();
        let t = truncate_tokens(&toks, &[2], 512);
        assert_eq!((t.len(), t[0], t[511]), (512, 2, 520));
        // BOS and EOS.
        let toks: Vec<i64> = std::iter::once(2).chain(10..900).chain(std::iter::once(1)).collect();
        let t = truncate_tokens(&toks, &[2, 1], 512);
        assert_eq!((t.len(), t[0], t[511]), (512, 2, 1));
        // Short enough: unchanged.
        assert_eq!(truncate_tokens(&[1, 2, 3], &[1], 512), [1, 2, 3]);
    }

    #[test]
    fn embeddings_come_back_in_input_order() {
        let v = json!({"data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, null]},
        ]});
        assert_eq!(parse_embeddings(&v, 2).unwrap(), vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert!(parse_embeddings(&v, 3).is_err());
        assert!(parse_embeddings(&json!({"data": [{"index": 0, "embedding": [[1.0]]}]}), 1).is_err());
        assert!(parse_embeddings(&json!({"data": [{"index": 5, "embedding": [1.0]}]}), 1).is_err());
        assert!(parse_embeddings(&json!({"error": "x"}), 1).is_err());
    }

    #[test]
    fn server_errors_are_worded() {
        assert_eq!(
            server_error(
                400,
                br#"{"error":{"code":400,"message":"the request exceeds the available context size","type":"exceed_context_size_error"}}"#
            ),
            "llama-server HTTP 400: the request exceeds the available context size"
        );
        assert_eq!(server_error(401, br#"{"error":"Invalid API Key"}"#), "llama-server HTTP 401: Invalid API Key");
        assert_eq!(server_error(502, b"bad gateway"), "llama-server HTTP 502: bad gateway");
    }

    #[test]
    fn idle_hook_sends_keepalives_in_streams_and_stops_on_cancel() {
        let shared = Shared::default();
        let seq = shared.next_seq();
        let sink = Sink::default();
        let out = Out::new(Box::new(sink.clone()));
        {
            let mut hook = idle_hook(&shared, seq, Some(&out));
            hook().unwrap();
            assert!(sink.lines().is_empty(), "not quiet long enough yet");
            out.last.set(Instant::now() - KEEPALIVE);
            hook().unwrap();
            hook().unwrap();
            assert_eq!(sink.lines(), [json!({"type": "keepalive"})], "one keepalive, then quiet again");
        }
        // Outside a stream nothing may be written: the next line is the answer.
        out.last.set(Instant::now() - KEEPALIVE * 2);
        idle_hook(&shared, seq, None)().unwrap();
        assert_eq!(sink.lines().len(), 1);
        shared.cancel();
        let err = idle_hook(&shared, seq, Some(&out))().unwrap_err();
        assert!(matches!(classify_abort(err, &shared, seq), Fail::Cancelled));
    }

    #[test]
    fn tool_calls_in_messages_are_normalized() {
        let c = normalize_tool_call(&json!({"function": {"name": "f", "arguments": {"a": 1}}}), 3);
        assert_eq!(c, json!({"id": "call_3", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}));
        let c = normalize_tool_call(&json!({"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}), 0);
        assert_eq!(c["id"], "x");
    }
}
