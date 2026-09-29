//! A daemon somewhere else, used through the same shapes as a local session.
//!
//! Blocking HTTP on purpose: hosts call sessions from blocking contexts
//! (`spawn_blocking`, worker threads), and a `Session` is blocking too. The
//! native `/engine/*` routes are used, not `/v1/*`, so the caller gets the
//! engine's own facts back: fingerprint, family, cached tokens, repairs.
//!
//! Safe from any thread, a Tokio worker included. reqwest's blocking client
//! panics when it is built, used or dropped inside an async runtime ("Cannot
//! drop a runtime in a context where blocking is not allowed"), and a host
//! cannot always tell where a handle will be created or dropped: ModelCaddy hit
//! it creating a handle in an async command and describing a picture inside an
//! async digest pass. So every step that touches the client runs on a plain
//! thread when the caller is inside a runtime ([`off_reactor`]); the caller
//! still blocks, as it would on a local session.

use crate::error::SessionError;
use crate::proto::{GenerationMeta, Message};
use crate::session::{CancelToken, Priority};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::time::Duration;

type Result<T> = std::result::Result<T, SessionError>;

fn err(msg: impl Into<String>) -> SessionError {
    SessionError::Runner(msg.into())
}

/// Connection to one daemon.
#[derive(Debug)]
pub struct RemoteEngine {
    base_url: String,
    token: Option<String>,
    /// Always `Some` until [`Drop`], which may have to move it to another thread.
    client: Option<reqwest::blocking::Client>,
}

/// Run `f` where reqwest's blocking client may run: here, unless this thread
/// is inside a Tokio runtime, in which case on a scoped plain thread that this
/// one waits for. A panic in `f` is re-raised on the caller.
fn off_reactor<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    if tokio::runtime::Handle::try_current().is_err() {
        return f();
    }
    std::thread::scope(|s| match s.spawn(f).join() {
        Ok(v) => v,
        Err(panic) => std::panic::resume_unwind(panic),
    })
}

impl Drop for RemoteEngine {
    fn drop(&mut self) {
        if let Some(client) = self.client.take() {
            if tokio::runtime::Handle::try_current().is_ok() {
                // Dropping it here would shut its runtime down inside ours.
                std::thread::spawn(move || drop(client));
            }
        }
    }
}

fn prio_str(p: Priority) -> &'static str {
    match p {
        Priority::Interactive => "interactive",
        Priority::Background => "background",
    }
}

impl RemoteEngine {
    /// `base_url` like `http://192.168.1.20:27200`.
    pub fn new(base_url: impl Into<String>, token: Option<String>) -> Result<Self> {
        let client = off_reactor(|| {
            reqwest::blocking::Client::builder()
                .connect_timeout(Duration::from_secs(5))
                // Generations run for minutes; this is the whole-request ceiling.
                .timeout(Duration::from_secs(600))
                .build()
        })
        .map_err(|e| err(format!("http client: {e}")))?;
        Ok(Self { base_url: base_url.into().trim_end_matches('/').to_string(), token, client: Some(client) })
    }

    fn client(&self) -> &reqwest::blocking::Client {
        self.client.as_ref().expect("client lives until drop")
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn post(&self, path: &str, body: &Value) -> reqwest::blocking::RequestBuilder {
        let mut r = self.client().post(format!("{}{path}", self.base_url)).json(body);
        if let Some(t) = &self.token {
            r = r.bearer_auth(t);
        }
        r
    }

    fn check(resp: reqwest::blocking::Response) -> Result<reqwest::blocking::Response> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let text = resp.text().unwrap_or_default();
        let message =
            serde_json::from_str::<Value>(&text).ok().and_then(|v| v["error"]["message"].as_str().map(str::to_string)).unwrap_or(text);
        Err(err(format!("remote engine {status}: {message}")))
    }

    /// `GET /engine/health` — reachable, and which API version.
    pub fn health(&self) -> Result<Value> {
        off_reactor(|| {
            let resp = self
                .client()
                .get(format!("{}/engine/health", self.base_url))
                .timeout(Duration::from_secs(5))
                .send()
                .map_err(|e| err(format!("remote engine unreachable: {e}")))?;
            Self::check(resp)?.json().map_err(|e| err(format!("health: {e}")))
        })
    }

    /// Non-streaming generation of a raw prompt.
    pub fn generate(&self, model: &str, prompt: &str, max_tokens: Option<u32>, temperature: Option<f32>, prio: Priority) -> Result<String> {
        off_reactor(|| {
            let body =
                json!({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "priority": prio_str(prio)});
            let resp = Self::check(self.post("/engine/generate", &body).send().map_err(|e| err(format!("remote generate: {e}")))?)?;
            let v: Value = resp.json().map_err(|e| err(format!("remote generate: {e}")))?;
            v["text"].as_str().map(str::to_string).ok_or_else(|| err("remote generate: no text in response"))
        })
    }

    /// Streaming generation; `on_token` per token line from the daemon's SSE.
    /// Dropping the connection (a flipped cancel token) cancels on the host.
    // Public API; mirrors `chat_stream` below. A params struct would break callers.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        prio: Priority,
        cancel: Option<&CancelToken>,
        on_token: impl FnMut(&str),
    ) -> Result<(String, Option<GenerationMeta>)> {
        let body = json!({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "priority": prio_str(prio), "stream": true});
        self.stream(body, cancel, on_token)
    }

    /// Streaming chat through the model's template on the host.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
        cache_key: Option<&str>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        prio: Priority,
        cancel: Option<&CancelToken>,
        on_token: impl FnMut(&str),
    ) -> Result<(String, Option<GenerationMeta>)> {
        let body = json!({"model": model, "messages": messages, "tools": tools, "cache_key": cache_key, "max_tokens": max_tokens, "temperature": temperature, "priority": prio_str(prio), "stream": true});
        self.stream(body, cancel, on_token)
    }

    fn stream(
        &self,
        body: Value,
        cancel: Option<&CancelToken>,
        mut on_token: impl FnMut(&str),
    ) -> Result<(String, Option<GenerationMeta>)> {
        // The request and the reading run on a plain thread (see
        // [`off_reactor`]); tokens come back over a channel, so `on_token`
        // runs here, on the caller's thread, and needs no `Send`.
        enum Line {
            Token(String),
            Done(Option<String>, Option<GenerationMeta>),
            Failed(SessionError),
        }
        std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::channel::<Line>();
            let body = &body;
            let reader = scope.spawn(move || {
                let resp = match self.post("/engine/generate", body).send() {
                    Ok(r) => r,
                    Err(e) => return drop(tx.send(Line::Failed(err(format!("remote stream: {e}"))))),
                };
                let resp = match Self::check(resp) {
                    Ok(r) => r,
                    Err(e) => return drop(tx.send(Line::Failed(e))),
                };
                for line in BufReader::new(resp).lines() {
                    let line = match line {
                        Ok(l) => l,
                        Err(e) => return drop(tx.send(Line::Failed(err(format!("remote stream read: {e}"))))),
                    };
                    let Some(data) = line.strip_prefix("data: ") else { continue };
                    let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
                    let msg = if let Some(e) = v.get("error") {
                        Line::Failed(err(format!("remote engine: {}", e.as_str().unwrap_or(&e.to_string()))))
                    } else if let Some(t) = v.get("token").and_then(Value::as_str) {
                        Line::Token(t.to_string())
                    } else if v.get("done").and_then(Value::as_bool) == Some(true) {
                        let meta = v.get("meta").cloned().and_then(|m| serde_json::from_value(m).ok());
                        Line::Done(v.get("text").and_then(Value::as_str).map(str::to_string), meta)
                    } else {
                        continue;
                    };
                    let last = !matches!(msg, Line::Token(_));
                    // A closed channel means the caller cancelled: returning
                    // drops the response, which closes the connection, and
                    // the daemon cancels the runner.
                    if tx.send(msg).is_err() || last {
                        return;
                    }
                }
                let _ = tx.send(Line::Failed(err("remote stream ended without a done line")));
            });
            let mut full = String::new();
            let outcome = loop {
                if cancel.map(CancelToken::is_cancelled).unwrap_or(false) {
                    break Err(SessionError::Cancelled { partial: std::mem::take(&mut full) });
                }
                match rx.recv_timeout(Duration::from_millis(200)) {
                    Ok(Line::Token(t)) => {
                        full.push_str(&t);
                        on_token(&t);
                    }
                    // Prefer the host's assembled text (identical bytes, no drift).
                    Ok(Line::Done(text, meta)) => break Ok((text.unwrap_or(std::mem::take(&mut full)), meta)),
                    Ok(Line::Failed(e)) => break Err(e),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break Err(err("remote stream ended without a done line")),
                }
            };
            drop(rx);
            if reader.join().is_err() {
                return Err(err("remote stream reader panicked"));
            }
            outcome
        })
    }

    /// Trade the same-user secret (`local-access.secret`) for a token named
    /// `local-<app>`. The daemon accepts this from loopback only; see
    /// [`crate::local::LocalEngine::claim_token`].
    pub fn local_token(&self, secret: &str, app: &str, scopes: Option<&[&str]>) -> Result<String> {
        off_reactor(|| {
            let body = json!({"secret": secret, "name": app, "scopes": scopes});
            let resp = self.client().post(format!("{}/engine/local-token", self.base_url)).json(&body).send();
            let resp = resp.map_err(|e| err(format!("local token: {e}")))?;
            let v: Value = Self::check(resp)?.json().map_err(|e| err(format!("local token: {e}")))?;
            v["token"].as_str().map(str::to_string).ok_or_else(|| err("local token: no token in the answer"))
        })
    }

    /// Ask a daemon for a token: the first half of pairing. Needs no token.
    /// Returns the pairing id to poll with [`RemoteEngine::pair_poll`].
    pub fn pair_request(&self, name: &str, scopes: &[&str]) -> Result<String> {
        off_reactor(|| {
            let resp = self
                .client()
                .post(format!("{}/engine/pair", self.base_url))
                .json(&json!({"name": name, "scopes": scopes}))
                .send()
                .map_err(|e| err(format!("pair request: {e}")))?;
            let v: Value = Self::check(resp)?.json().map_err(|e| err(format!("pair request: {e}")))?;
            v["id"].as_str().map(str::to_string).ok_or_else(|| err("pair request: no id"))
        })
    }

    /// Poll a pairing: `pending`, `approved` (with the token, once) or `denied`.
    pub fn pair_poll(&self, id: &str) -> Result<(String, Option<String>)> {
        off_reactor(|| {
            let resp =
                self.client().get(format!("{}/engine/pair/{id}", self.base_url)).send().map_err(|e| err(format!("pair poll: {e}")))?;
            let v: Value = Self::check(resp)?.json().map_err(|e| err(format!("pair poll: {e}")))?;
            Ok((v["status"].as_str().unwrap_or("pending").to_string(), v["token"].as_str().map(str::to_string)))
        })
    }

    /// Admin: pending and recent pairings on the daemon.
    pub fn pairings(&self) -> Result<Value> {
        off_reactor(|| {
            let mut r = self.client().get(format!("{}/engine/pairings", self.base_url));
            if let Some(t) = &self.token {
                r = r.bearer_auth(t);
            }
            Self::check(r.send().map_err(|e| err(format!("pairings: {e}")))?)?.json().map_err(|e| err(format!("pairings: {e}")))
        })
    }

    /// Admin: approve or deny a pairing from wherever this client is.
    pub fn decide_pairing(&self, id: &str, approve: bool) -> Result<Value> {
        off_reactor(|| {
            let verb = if approve { "approve" } else { "deny" };
            let resp =
                self.post(&format!("/engine/pairings/{id}/{verb}"), &json!({})).send().map_err(|e| err(format!("pairing {verb}: {e}")))?;
            Self::check(resp)?.json().map_err(|e| err(format!("pairing {verb}: {e}")))
        })
    }

    /// Embed already-prefixed inputs (`task: none`) and check the host serves
    /// the fingerprint the caller's index was built with.
    pub fn embed_batch(
        &self,
        model: &str,
        inputs: &[String],
        expect_fingerprint: Option<&str>,
        prio: Priority,
    ) -> Result<(Vec<Vec<f32>>, String)> {
        off_reactor(|| {
            let body = json!({"model": model, "inputs": inputs, "task": "none", "expect_fingerprint": expect_fingerprint, "priority": prio_str(prio)});
            let resp = Self::check(self.post("/engine/embed", &body).send().map_err(|e| err(format!("remote embed: {e}")))?)?;
            #[derive(Deserialize)]
            struct EmbedResp {
                vectors: Vec<Vec<f32>>,
                fingerprint: String,
            }
            let r: EmbedResp = resp.json().map_err(|e| err(format!("remote embed: {e}")))?;
            Ok((r.vectors, r.fingerprint))
        })
    }
}

/// A remote generation model under the local session's shape.
pub struct RemoteGen {
    engine: Arc<RemoteEngine>,
    model_id: String,
}

impl RemoteGen {
    pub fn new(engine: Arc<RemoteEngine>, model_id: impl Into<String>) -> Self {
        Self { engine, model_id: model_id.into() }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn engine(&self) -> &RemoteEngine {
        &self.engine
    }

    pub fn generate_with(&self, prompt: &str, max_tokens: Option<u32>, temperature: Option<f32>, prio: Priority) -> Result<String> {
        self.engine.generate(&self.model_id, prompt, max_tokens, temperature, prio)
    }

    pub fn generate_stream_with(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        prio: Priority,
        cancel: Option<&CancelToken>,
        on_token: impl FnMut(&str),
    ) -> Result<String> {
        self.engine.generate_stream(&self.model_id, prompt, max_tokens, temperature, prio, cancel, on_token).map(|(t, _)| t)
    }
}

/// A remote embedding model under the local session's shape. `fingerprint` is
/// what the caller expects; every call asks the host to refuse a mismatch.
pub struct RemoteEmbed {
    engine: Arc<RemoteEngine>,
    model_id: String,
    fingerprint: String,
}

impl RemoteEmbed {
    pub fn new(engine: Arc<RemoteEngine>, model_id: impl Into<String>, fingerprint: impl Into<String>) -> Self {
        Self { engine, model_id: model_id.into(), fingerprint: fingerprint.into() }
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn engine(&self) -> &RemoteEngine {
        &self.engine
    }

    pub fn embed_batch_with(&self, inputs: &[String], prio: Priority) -> Result<Vec<Vec<f32>>> {
        let (vectors, served) = self.engine.embed_batch(&self.model_id, inputs, Some(&self.fingerprint), prio)?;
        if served != self.fingerprint {
            return Err(err(format!("remote engine served fingerprint `{served}`, expected `{}`", self.fingerprint)));
        }
        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    /// A one-file HTTP daemon: `/engine/health` answers JSON, anything else
    /// streams two tokens and a done line, as `/engine/generate` does.
    fn fake_daemon() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { continue };
                std::thread::spawn(move || {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    // Headers, then the body its Content-Length announces.
                    let (head_end, len) = loop {
                        let n = conn.read(&mut chunk).unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            let head = String::from_utf8_lossy(&buf[..i]).to_lowercase();
                            let len = head
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(0);
                            break (i + 4, len);
                        }
                    };
                    while buf.len() < head_end + len {
                        let n = conn.read(&mut chunk).unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let first = String::from_utf8_lossy(&buf).lines().next().unwrap_or_default().to_string();
                    let (ctype, body) = if first.contains("/engine/health") {
                        ("application/json", r#"{"version":"test","backend":"fake"}"#.to_string())
                    } else {
                        (
                            "text/event-stream",
                            "data: {\"token\":\"a\"}\n\ndata: {\"token\":\"b\"}\n\ndata: {\"done\":true,\"text\":\"ab\"}\n\n".to_string(),
                        )
                    };
                    let _ = write!(
                        conn,
                        "HTTP/1.1 200 OK\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                });
            }
        });
        format!("http://{addr}")
    }

    fn use_it(base: &str) {
        let engine = RemoteEngine::new(base, Some("t".into())).unwrap();
        assert_eq!(engine.health().unwrap()["backend"], "fake");
        let mut seen = String::new();
        let (text, _) = engine
            .chat_stream("m", &[Message::new("user", "hi")], None, None, Some(8), None, Priority::Background, None, |t| seen.push_str(t))
            .unwrap();
        assert_eq!((text.as_str(), seen.as_str()), ("ab", "ab"));
        assert!(RemoteEngine::new("http://127.0.0.1:9", None).unwrap().health().is_err(), "unreachable is an error");
        // `engine` drops here, inside whatever runtime the caller is in.
    }

    /// The bug this guards: reqwest's blocking client panicked when it was
    /// built, used or dropped on a Tokio worker ("Cannot drop a runtime in a
    /// context where blocking is not allowed"), which is where a host's async
    /// commands call it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn works_on_a_multi_thread_runtime_worker() {
        use_it(&fake_daemon());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn works_on_a_current_thread_runtime() {
        use_it(&fake_daemon());
    }

    #[test]
    fn works_outside_any_runtime() {
        use_it(&fake_daemon());
    }
}
