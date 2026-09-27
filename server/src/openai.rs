//! `/v1/*` — OpenAI's shapes exactly, so anything that speaks to OpenAI speaks
//! to the engine. Extra engine facts ride in an `x_estia` field the client can
//! ignore.

use crate::access::{spawn_blocking_in_span, Access};
use crate::engine_api::{capped_max_tokens, check_embed_inputs};
use crate::{derive_cache_key, toolcalls, ApiError, AppState, Caller};
use axum::{
    extract::{Extension, State},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use estia_engine::models::embed::EmbedTask;
use estia_engine::models::{Artifact, Capability};
use estia_engine::proto::{Capabilities, ImageData, Message};
use estia_engine::structured::{self, OutputFormat, Structured};
use estia_engine::{CancelToken, Priority};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Debug, Deserialize)]
pub struct OaiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OaiMessage>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// OpenAI's end-user id; used as the prompt-cache key when present, scoped
    /// to the calling token like every cache key.
    #[serde(default)]
    pub user: Option<String>,
    /// Engine extension: `interactive` (default) or `background`.
    #[serde(default)]
    pub priority: Option<String>,
}

/// At most this many images in one request, across all its messages.
pub const MAX_IMAGES_PER_REQUEST: usize = 8;

/// The largest image accepted, decoded: 20 MiB.
pub const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// The text and the images of OpenAI message content: a string, or an array
/// of parts. `text` parts are joined with newlines; `image_url` parts must be
/// `data:` URLs (the engine never fetches a URL on a client's behalf). Any
/// other part type is refused rather than silently dropped.
fn content_parts(v: &Option<Value>) -> Result<(String, Vec<ImageData>), ApiError> {
    match v {
        None | Some(Value::Null) => Ok((String::new(), Vec::new())),
        Some(Value::String(s)) => Ok((s.clone(), Vec::new())),
        Some(Value::Array(parts)) => {
            let mut texts = Vec::new();
            let mut images = Vec::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => texts.push(p.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
                    Some("image_url") => {
                        let url = match p.get("image_url") {
                            Some(Value::String(u)) => u.as_str(),
                            Some(o) => o.get("url").and_then(Value::as_str).unwrap_or(""),
                            None => "",
                        };
                        images.push(image_from_data_url(url)?);
                    }
                    Some(other) => {
                        return Err(ApiError::bad_request(format!(
                            "content part type `{other}` is not supported; send `text` and `image_url` parts"
                        )))
                    }
                    None => texts.push(p.as_str().unwrap_or("").to_string()),
                }
            }
            Ok((texts.join("\n"), images))
        }
        Some(other) => Ok((other.to_string(), Vec::new())),
    }
}

/// A `data:image/...;base64,...` URL as an image the runner can take. The
/// format is read from the bytes, not trusted from the URL.
pub(crate) fn image_from_data_url(url: &str) -> Result<ImageData, ApiError> {
    use base64::Engine as _;
    let url = url.trim();
    let Some(rest) = url.strip_prefix("data:") else {
        let shown: String = url.chars().take(40).collect();
        return Err(ApiError::bad_request(format!(
            "image URLs are not fetched: send the image itself as a data URL (data:image/png;base64,...), not `{shown}`"
        )));
    };
    let (header, payload) = rest.split_once(',').ok_or_else(|| ApiError::bad_request("an image data URL needs a comma before its data"))?;
    let mut fields = header.split(';');
    let declared = fields.next().unwrap_or("").trim().to_ascii_lowercase();
    if !fields.any(|f| f.trim().eq_ignore_ascii_case("base64")) {
        return Err(ApiError::bad_request("image data URLs must be base64 (data:image/png;base64,...)"));
    }
    if !["image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"].contains(&declared.as_str()) {
        return Err(ApiError::bad_request(format!("image type `{declared}` is not supported; send PNG, JPEG, WebP or GIF")));
    }
    let compact: String = payload.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    // Rough size check before decoding: base64 is 4 characters per 3 bytes.
    if compact.len() / 4 * 3 > MAX_IMAGE_BYTES + 3 {
        return Err(ApiError::payload_too_large(format!("an image is over the {} MB limit", MAX_IMAGE_BYTES / (1024 * 1024))));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&compact)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(&compact))
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(&compact))
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(&compact))
        .map_err(|_| ApiError::bad_request("image data is not valid base64"))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(ApiError::payload_too_large(format!("an image is over the {} MB limit", MAX_IMAGE_BYTES / (1024 * 1024))));
    }
    image_from_bytes(&bytes).map_err(ApiError::bad_request)
}

/// An image file's bytes as an image a runner can take: the format read from
/// the bytes (PNG, JPEG, WebP or GIF), the data base64. Hosts that embed the
/// engine use this to attach a picture from disk.
pub fn image_from_bytes(bytes: &[u8]) -> Result<ImageData, String> {
    use base64::Engine as _;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(format!("the image is over the {} MB limit", MAX_IMAGE_BYTES / (1024 * 1024)));
    }
    let mime = sniff_image(bytes).ok_or("image data is not a PNG, JPEG, WebP or GIF file")?;
    Ok(ImageData { mime: mime.to_string(), data: base64::engine::general_purpose::STANDARD.encode(bytes) })
}

/// The image format from its first bytes.
fn sniff_image(b: &[u8]) -> Option<&'static str> {
    if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if b.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// OpenAI messages as runner messages, with their images. Refuses images
/// outside user turns and more than [`MAX_IMAGES_PER_REQUEST`] in total.
pub(crate) fn to_messages(msgs: &[OaiMessage]) -> Result<Vec<Message>, ApiError> {
    let mut total = 0usize;
    let mut out = Vec::with_capacity(msgs.len());
    for m in msgs {
        let (content, images) = content_parts(&m.content)?;
        if !images.is_empty() && m.role != "user" {
            return Err(ApiError::bad_request(format!("images are accepted in user messages only, not in a `{}` message", m.role)));
        }
        total += images.len();
        if total > MAX_IMAGES_PER_REQUEST {
            return Err(ApiError::bad_request(format!("at most {MAX_IMAGES_PER_REQUEST} images per request")));
        }
        out.push(
            Message {
                role: m.role.clone(),
                content,
                name: m.name.clone(),
                tool_call_id: m.tool_call_id.clone(),
                // Passed through structured: chat templates render a `tool` result
                // only after the assistant turn whose `tool_calls` it answers.
                tool_calls: m.tool_calls.clone().filter(|c| !c.is_empty()),
                images: None,
            }
            .with_images(images),
        );
    }
    Ok(out)
}

/// Refuse images for a model or runner that cannot read them, rather than
/// answering as if the images were not there.
pub(crate) fn check_images(messages: &[Message], artifact: &Artifact, caps: Option<&Capabilities>) -> Result<(), ApiError> {
    if !Message::any_images(messages) {
        return Ok(());
    }
    if !artifact.has(Capability::Vision) {
        return Err(ApiError::bad_request(format!(
            "model `{}` does not read images; use the `vision` role or another model that does",
            artifact.id
        )));
    }
    if caps.is_some_and(|c| !c.images) {
        return Err(ApiError::bad_request(format!(
            "the runner serving `{}` cannot take images; it is older than this engine",
            artifact.id
        )));
    }
    Ok(())
}

pub(crate) fn parse_response_format(v: &Option<Value>) -> Result<OutputFormat, ApiError> {
    let Some(v) = v else {
        return Ok(OutputFormat::Text);
    };
    match v.get("type").and_then(Value::as_str) {
        None | Some("text") => Ok(OutputFormat::Text),
        Some("json_object") => Ok(OutputFormat::Json),
        Some("json_schema") => {
            let schema = v
                .get("json_schema")
                .and_then(|j| j.get("schema"))
                .cloned()
                .ok_or_else(|| ApiError::bad_request("response_format.json_schema.schema is required"))?;
            Ok(OutputFormat::JsonSchema { schema })
        }
        Some(other) => Err(ApiError::bad_request(format!("unsupported response_format type `{other}`"))),
    }
}

/// The `format` to hand the runner: only when its `hello` lists that kind of
/// output under `structured` (the llama.cpp adapter constrains decoding to
/// it), and never together with tools, which llama-server refuses alongside
/// a grammar. The engine validates the output either way.
pub fn runner_format(caps: &Capabilities, format: &OutputFormat, has_tools: bool) -> Option<Value> {
    let kind = match format {
        OutputFormat::Text => return None,
        OutputFormat::Json => "json",
        OutputFormat::JsonSchema { .. } => "json_schema",
    };
    if has_tools || !caps.structured.iter().any(|s| s == kind) {
        return None;
    }
    serde_json::to_value(format).ok()
}

fn priority_of(p: &Option<String>) -> Priority {
    match p.as_deref() {
        Some("background") => Priority::Background,
        _ => Priority::Interactive,
    }
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// `chatcmpl-` and 24 random hex digits: unique across responses, engines
/// and restarts (clients and proxies key logs and caches on it).
fn completion_id() -> String {
    crate::unique_id("chatcmpl-")
}

/// Everything a finished completion needs to be rendered, streamed or not.
struct Finished {
    text: String,
    tool_calls: Vec<Value>,
    /// The runner parsed the calls itself, so `text` is plain content.
    runner_parsed: bool,
    structured: Option<Structured>,
    meta: estia_engine::proto::GenerationMeta,
    /// Generation time: the runner calls, retries included, not the load.
    ms: u128,
}

/// Why a generation ended, in OpenAI's words. Parsed tool calls win; then
/// the runner's own `meta.finish_reason`; a runner that does not say ran out
/// of budget when it produced `max_tokens` tokens. `None` without meta and
/// calls (a raw-prompt generation, whose runner call reports no accounting).
pub(crate) fn finish_reason(has_calls: bool, meta: Option<&estia_engine::proto::GenerationMeta>, max_tokens: u32) -> Option<&'static str> {
    if has_calls {
        return Some("tool_calls");
    }
    let meta = meta?;
    Some(match meta.finish_reason.as_deref() {
        Some("length") => "length",
        // `stop`, or `tool_calls` with no call that parsed.
        Some(_) => "stop",
        None if meta.generation_tokens.is_some_and(|n| n >= u64::from(max_tokens)) => "length",
        None => "stop",
    })
}

fn usage(meta: &estia_engine::proto::GenerationMeta) -> Value {
    let p = meta.prompt_tokens.unwrap_or(0);
    let c = meta.generation_tokens.unwrap_or(0);
    json!({"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
           "prompt_tokens_details": {"cached_tokens": meta.cached_tokens.unwrap_or(0)}})
}

/// `ms` is generation time; `load_ms` is how long this request waited for the
/// model to load (null when it was resident already), so the two add up to
/// the time the engine spent on the request.
fn x_estia(artifact: &estia_engine::models::Artifact, f: &Finished, backend: &str, load_ms: Option<u64>) -> Value {
    json!({
        "family": artifact.family,
        "backend": backend,
        "cached_tokens": f.meta.cached_tokens,
        "template": f.meta.template,
        "generation_tps": f.meta.generation_tps,
        "repaired": f.structured.as_ref().map(|s| s.repaired),
        "repairs": f.structured.as_ref().map(|s| s.repairs.clone()),
        "ms": f.ms,
        "load_ms": load_ms,
    })
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Extension(caller): Extension<Caller>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let artifact = state.resolve_generation(&req.model)?;
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let messages = to_messages(&req.messages)?;
    check_images(&messages, artifact, None)?;
    let format = parse_response_format(&req.response_format)?;
    let tools = req.tools.clone().filter(|t| !t.is_empty());
    // Per token: another token sending the same `user` gets its own entry.
    let cache_key = caller.scoped_key(&req.user.clone().unwrap_or_else(|| derive_cache_key(artifact.id, &messages)));
    // Same work caps as /engine/generate.
    let max_tokens = capped_max_tokens(req.max_completion_tokens.or(req.max_tokens), 1024);
    let temperature = req.temperature.unwrap_or(0.2);
    let prio = priority_of(&req.priority);
    let backend = state.embed_backend();
    let stream = req.stream.unwrap_or(false);
    let access = Access::current();
    if let Some(a) = &access {
        a.generation(&req.model, artifact.id, stream, max_tokens);
    }

    let st = Arc::clone(&state);
    let (session, load_ms) = spawn_blocking_in_span(move || st.gen_session_timed(artifact)).await??;
    if let (Some(a), Some(ms)) = (&access, load_ms) {
        a.loaded(ms);
    }
    let caps = session.capabilities();
    check_images(&messages, artifact, Some(&caps))?;
    let parses_calls = caps.parses_tool_calls;
    let constrain = runner_format(&caps, &format, tools.is_some());
    // A runner that cannot constrain decoding (MLX) is shown the schema in
    // the system prompt instead. The cache key above was derived without it.
    let messages = if constrain.is_none() && tools.is_none() { structured::with_prompt_hint(&messages, &format) } else { messages };

    if !stream {
        let s = Arc::clone(&session);
        let (msgs, tls, ck, fmt) = (messages.clone(), tools.clone(), cache_key.clone(), format.clone());
        let acc = access.clone();
        let finished: Finished = spawn_blocking_in_span(move || -> Result<Finished, ApiError> {
            let t0 = Instant::now();
            let call = |m: &[Message]| {
                if let Some(a) = &acc {
                    a.call_start();
                }
                let out = s.chat_with(m, tls.as_deref(), Some(&ck), constrain.as_ref(), Some(max_tokens), Some(temperature), prio)?;
                if let Some(a) = &acc {
                    a.call_end(Some(&out.meta));
                }
                Ok::<_, ApiError>(out)
            };
            let mut out = call(&msgs)?;
            let mut structured = None;
            if fmt.wants_json() {
                structured = Some(match structured::enforce(&out.text, &fmt) {
                    Ok(v) => v,
                    Err(first_err) => {
                        // One retry with the validator's complaint appended.
                        tracing::debug!("structured output did not validate; retrying once");
                        let mut retry = msgs.clone();
                        retry.push(Message::new("assistant", out.text.clone()));
                        retry.push(Message::new("user", Structured::retry_hint(&first_err).trim().to_string()));
                        out = call(&retry)?;
                        structured::enforce(&out.text, &fmt)
                            .map_err(|e| ApiError::unprocessable(format!("structured output failed after retry: {e}")))?
                    }
                });
            }
            let tool_calls =
                if tls.is_some() { toolcalls::from_output(parses_calls, &out.text, out.meta.tool_calls.as_deref()) } else { Vec::new() };
            Ok(Finished {
                text: out.text,
                tool_calls,
                runner_parsed: parses_calls,
                structured,
                meta: out.meta,
                ms: t0.elapsed().as_millis(),
            })
        })
        .await??;
        let finish = finish_reason(!finished.tool_calls.is_empty(), Some(&finished.meta), max_tokens).unwrap_or("stop");
        if let Some(a) = &access {
            a.finish(finish);
        }

        // Text next to parsed calls is the call syntax itself, unless the
        // runner parsed the calls out and left only prose.
        let content: Value = match &finished.structured {
            Some(s) => Value::String(s.value.to_string()),
            None if !finished.tool_calls.is_empty() && (!finished.runner_parsed || finished.text.trim().is_empty()) => Value::Null,
            None => Value::String(finished.text.clone()),
        };
        let mut message = json!({"role": "assistant", "content": content});
        if !finished.tool_calls.is_empty() {
            message["tool_calls"] = Value::Array(finished.tool_calls.clone());
        }
        let body = json!({
            "id": completion_id(),
            "object": "chat.completion",
            "created": now_unix(),
            "model": artifact.id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage(&finished.meta),
            "x_estia": x_estia(artifact, &finished, backend, load_ms),
        });
        return Ok(Json(body).into_response());
    }

    // Streaming. Tokens go over a channel from the blocking generation to the
    // SSE stream. If the client disconnects, the receiver drops and the
    // generation is cancelled in the runner, even mid-prefill, before any
    // token was sent (see `cancel_on_disconnect`).
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let cancel = CancelToken::new();
    let watch = crate::cancel_on_disconnect(&tx, cancel.clone());
    let id = completion_id();
    let created = now_unix();
    let model_id = artifact.id.to_string();
    let wants_json = format.wants_json();
    let has_tools = tools.is_some();
    let s = Arc::clone(&session);
    let acc = access.clone();
    let request_id = access.as_ref().map(|a| a.id().to_string());
    spawn_blocking_in_span(move || {
        let chunk = |delta: Value, finish: Option<&str>, extra: Option<Value>| -> Value {
            let mut v = json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
            });
            if let Some(extra) = extra {
                if let Some(obj) = v.as_object_mut() {
                    if let Some(map) = extra.as_object() {
                        for (k, val) in map {
                            obj.insert(k.clone(), val.clone());
                        }
                    }
                }
            }
            v
        };
        let send = |v: Value| -> bool { tx.send(Ok(Event::default().data(v.to_string()))).is_ok() };
        let _watch = watch;
        let _ = send(chunk(json!({"role": "assistant"}), None, None));

        let flip = cancel.clone();
        // Hold the first characters back to decide whether this is a tool
        // call (buffer everything, emit tool_calls at the end) or prose (stream).
        // A runner that parses calls itself streams prose only and sends the
        // calls in its meta line, so there is nothing to hold back.
        let mut buffered = String::new();
        let mut decided: Option<bool> = if wants_json || !has_tools { Some(!wants_json && !has_tools) } else { None };
        // decided == Some(true): stream live; Some(false): buffer to the end.
        if (!has_tools || parses_calls) && !wants_json {
            decided = Some(true);
        }
        let t0 = Instant::now();
        if let Some(a) = &acc {
            a.call_start();
        }
        let mut first = true;
        let result = s.chat_stream_with(
            &messages,
            tools.as_deref(),
            Some(&cache_key),
            constrain.as_ref(),
            Some(max_tokens),
            Some(temperature),
            prio,
            Some(&cancel),
            |tok| {
                if std::mem::take(&mut first) {
                    if let Some(a) = &acc {
                        a.token();
                    }
                }
                match decided {
                    Some(true) => {
                        if !send(chunk(json!({"content": tok}), None, None)) {
                            flip.cancel();
                        }
                    }
                    Some(false) => buffered.push_str(tok),
                    None => {
                        buffered.push_str(tok);
                        let head = buffered.trim_start();
                        if head.len() >= 12 || head.starts_with("<|tool") || head.starts_with('{') {
                            let looks_like_call = head.starts_with("<|tool") || head.starts_with("{\"tool") || head.starts_with("{\"name");
                            if looks_like_call {
                                decided = Some(false);
                            } else {
                                decided = Some(true);
                                if !send(chunk(json!({"content": buffered.clone()}), None, None)) {
                                    flip.cancel();
                                }
                            }
                        }
                    }
                }
            },
        );
        let ms = t0.elapsed().as_millis();
        // Our hold on the access record goes before the stream ends (the
        // sender drops with this closure), so the response body is normally
        // the last holder and the line is written as the response finishes.
        let settle = |acc: Option<Access>, f: &dyn Fn(&Access)| {
            if let Some(a) = acc {
                f(&a);
            }
        };
        match result {
            Ok(outcome) => {
                if let Some(a) = &acc {
                    a.call_end(Some(&outcome.meta));
                }
                let full = outcome.text;
                let mut finish = finish_reason(false, Some(&outcome.meta), max_tokens).unwrap_or("stop");
                let mut final_delta = json!({});
                // A parsing runner's calls come from its meta line, whether or
                // not its prose was streamed.
                let calls: Vec<Value> = if has_tools && (parses_calls || decided != Some(true)) {
                    toolcalls::from_output(parses_calls, &full, outcome.meta.tool_calls.as_deref())
                } else {
                    Vec::new()
                };
                if !calls.is_empty() {
                    finish = "tool_calls";
                    final_delta = json!({"tool_calls": toolcalls::indexed(&calls)});
                    // Prose a parsing runner sent next to its calls and that
                    // was still held back.
                    if decided != Some(true) && parses_calls && !full.trim().is_empty() {
                        final_delta["content"] = json!(full);
                    }
                } else if decided != Some(true) {
                    // Everything is still buffered: JSON, or short prose.
                    if wants_json {
                        match structured::enforce(&full, &format) {
                            Ok(sv) => final_delta = json!({"content": sv.value.to_string()}),
                            Err(e) => {
                                // The message quotes the output, so the log gets a fixed one.
                                settle(acc, &|a| a.failed("structured output failed validation"));
                                let _ = send(
                                    json!({"error": {"message": format!("structured output failed: {e}"), "type": "invalid_request_error", "request_id": request_id}}),
                                );
                                let _ = tx.send(Ok(Event::default().data("[DONE]")));
                                return;
                            }
                        }
                    } else {
                        final_delta = json!({"content": full});
                    }
                }
                settle(acc, &|a| a.finish(finish));
                let extra = json!({"usage": usage(&outcome.meta), "x_estia": {
                    "backend": backend, "cached_tokens": outcome.meta.cached_tokens, "template": outcome.meta.template,
                    "generation_tps": outcome.meta.generation_tps, "ms": ms, "load_ms": load_ms,
                }});
                let _ = send(chunk(final_delta, Some(finish), Some(extra)));
                let _ = tx.send(Ok(Event::default().data("[DONE]")));
            }
            Err(estia_engine::SessionError::Cancelled { .. }) => {
                // Client went away; nothing to tell it.
                settle(acc, &|a| a.finish("cancelled"));
            }
            Err(e) => {
                settle(acc, &|a| a.failed(&e.to_string()));
                let _ = send(json!({"error": {"message": e.to_string(), "type": "server_error", "request_id": request_id}}));
                let _ = tx.send(Ok(Event::default().data("[DONE]")));
            }
        }
    });
    let stream = UnboundedReceiverStream::new(rx);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: Value,
    /// Engine extension: `document` (default), `query`, `clustering`.
    #[serde(default)]
    pub task: Option<String>,
    /// Engine extension: refuse unless the served fingerprint matches.
    #[serde(default)]
    pub expect_fingerprint: Option<String>,
    #[serde(default)]
    pub priority: Option<String>,
    /// OpenAI's `float` (default) or `base64`. OpenAI's SDKs ask for base64
    /// unless told otherwise and decode it themselves.
    #[serde(default)]
    pub encoding_format: Option<String>,
}

/// How `/v1/embeddings` writes each vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Encoding {
    /// A JSON array of numbers.
    Float,
    /// The vector's little-endian f32 bytes in standard base64, one string.
    Base64,
}

impl Encoding {
    pub(crate) fn parse(s: Option<&str>) -> Result<Self, ApiError> {
        match s {
            None | Some("float") => Ok(Encoding::Float),
            Some("base64") => Ok(Encoding::Base64),
            Some(other) => Err(ApiError::bad_request(format!("unsupported encoding_format `{other}` (float|base64)"))),
        }
    }

    pub(crate) fn encode(self, v: &[f32]) -> Value {
        match self {
            Encoding::Float => json!(v),
            Encoding::Base64 => Value::String(embedding_base64(v)),
        }
    }
}

/// A vector as `encoding_format: "base64"` carries it: its f32 values as
/// little-endian bytes, in standard base64 with padding. What OpenAI returns,
/// and what its SDKs decode (`Float32Array` in JavaScript, `numpy.frombuffer`
/// with `dtype=float32` in Python).
pub(crate) fn embedding_base64(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    crate::base64_std(&bytes)
}

/// `None` = the caller already prefixed its inputs (a remote client whose
/// index layer owns the prefix policy); nothing is prepended.
pub(crate) fn embed_task(s: &Option<String>) -> Result<Option<EmbedTask>, ApiError> {
    match s.as_deref() {
        None | Some("document") => Ok(Some(EmbedTask::Document)),
        Some("query") => Ok(Some(EmbedTask::Query)),
        Some("clustering") => Ok(Some(EmbedTask::Clustering)),
        Some("none") => Ok(None),
        Some(other) => Err(ApiError::bad_request(format!("unknown task `{other}` (document|query|clustering|none)"))),
    }
}

pub(crate) fn apply_prefix(model: &estia_engine::models::embed::EmbedModel, task: Option<EmbedTask>, inputs: &[String]) -> Vec<String> {
    match task {
        Some(t) => inputs.iter().map(|s| format!("{}{}", model.prefix(t), s)).collect(),
        None => inputs.to_vec(),
    }
}

pub(crate) fn task_name(task: Option<EmbedTask>) -> String {
    task.map(|t| format!("{t:?}").to_lowercase()).unwrap_or_else(|| "none".into())
}

pub(crate) fn inputs_of(v: &Value) -> Result<Vec<String>, ApiError> {
    match v {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(items) => items
            .iter()
            .map(|i| i.as_str().map(str::to_string).ok_or_else(|| ApiError::bad_request("input array must hold strings")))
            .collect(),
        _ => Err(ApiError::bad_request("input must be a string or an array of strings")),
    }
}

pub async fn embeddings(State(state): State<Arc<AppState>>, Json(req): Json<EmbeddingsRequest>) -> Result<Response, ApiError> {
    let model = state.resolve_embedding(Some(&req.model))?;
    let inputs = inputs_of(&req.input)?;
    let access = Access::current();
    if let Some(a) = &access {
        a.embedding(&req.model, model.id, inputs.len(), model.dims);
    }
    if inputs.is_empty() {
        return Err(ApiError::bad_request("input must not be empty"));
    }
    check_embed_inputs(inputs.len())?;
    let task = embed_task(&req.task)?;
    let encoding = Encoding::parse(req.encoding_format.as_deref())?;
    let fingerprint = state.embed_fingerprint(model)?;
    if let Some(expected) = &req.expect_fingerprint {
        if expected != &fingerprint {
            return Err(ApiError::unprocessable(format!(
                "fingerprint mismatch: this host serves `{fingerprint}`, you expected `{expected}` — re-embed before mixing"
            )));
        }
    }
    let prio = priority_of(&req.priority);
    let st = Arc::clone(&state);
    let prefixed = apply_prefix(model, task, &inputs);
    let (vectors, load_ms) = spawn_blocking_in_span(move || -> Result<(Vec<Vec<f32>>, Option<u64>), ApiError> {
        let (session, load_ms) = st.embed_session_timed(model)?;
        Ok((session.embed_batch_with(&prefixed, prio)?, load_ms))
    })
    .await??;
    if let (Some(a), Some(ms)) = (&access, load_ms) {
        a.loaded(ms);
    }
    let data: Vec<Value> =
        vectors.iter().enumerate().map(|(i, v)| json!({"object": "embedding", "index": i, "embedding": encoding.encode(v)})).collect();
    Ok(Json(json!({
        "object": "list",
        "data": data,
        "model": model.id,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
        "x_estia": {"fingerprint": fingerprint, "dims": model.dims, "task": task_name(task), "load_ms": load_ms}
    }))
    .into_response())
}

/// Roles first, then every generation artifact (built-in and imported), then
/// every embedding model. `x_estia.runnable` says whether this engine's
/// backend loads the artifact: on a llama.cpp engine the MLX artifacts are
/// listed but not runnable, and the other way round.
pub async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let roles = state.engine.roles();
    let store = state.engine.store();
    let backend = state.backend();
    let mut data = Vec::new();
    for (role, b) in roles.iter() {
        data.push(json!({"id": role, "object": "model", "created": 0, "owned_by": "estia", "x_estia": {"role": true, "family": b.family}}));
    }
    for a in estia_engine::models::generation_artifacts() {
        data.push(json!({"id": a.id, "object": "model", "created": 0, "owned_by": "estia",
            "x_estia": {"family": a.family, "format": a.format.id(), "backend": a.backend().id(), "runnable": a.format == backend.format(),
                        "imported": crate::catalog::is_imported(store, a.id), "installed": store.is_installed(a.id),
                        "context_length": a.context_length, "capabilities": a.capabilities}}));
    }
    for e in estia_engine::models::embed_models() {
        let artifact = e.artifact_for(backend);
        data.push(json!({"id": e.id, "object": "model", "created": 0, "owned_by": "estia",
            "x_estia": {"kind": "embedding", "dims": e.dims, "runnable": artifact.is_some(),
                        "artifact": artifact.map(|a| a.id), "format": artifact.map(|a| a.format.id()),
                        "fingerprint": state.engine.embed_fingerprint(e),
                        "imported": crate::catalog::is_imported(store, e.id),
                        "installed": artifact.is_some_and(|a| store.is_installed(a.id))}}));
    }
    Json(json!({"object": "list", "data": data}))
}

#[cfg(test)]
mod image_tests {
    use super::*;
    use base64::Engine as _;

    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR";

    fn data_url(mime: &str, bytes: &[u8]) -> String {
        format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    fn user(content: Value) -> OaiMessage {
        OaiMessage { role: "user".into(), content: Some(content), name: None, tool_call_id: None, tool_calls: None }
    }

    #[test]
    fn data_urls_become_images_with_the_sniffed_type() {
        let img = image_from_data_url(&data_url("image/png", PNG)).unwrap();
        assert_eq!(img.mime, "image/png");
        assert_eq!(base64::engine::general_purpose::STANDARD.decode(&img.data).unwrap(), PNG);
        // The type is read from the bytes: a PNG declared as JPEG is a PNG.
        assert_eq!(image_from_data_url(&data_url("image/jpg", PNG)).unwrap().mime, "image/png");
        let jpeg = image_from_data_url(&data_url("image/jpeg", &[0xFF, 0xD8, 0xFF, 0xE0, 0, 0])).unwrap();
        assert_eq!(jpeg.mime, "image/jpeg");
        let webp = image_from_data_url(&data_url("image/webp", b"RIFF\0\0\0\0WEBPVP8 ")).unwrap();
        assert_eq!(webp.mime, "image/webp");
        // Whitespace and URL-safe base64 are tolerated.
        let spaced = format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(PNG).replace('A', "A\n"));
        assert!(image_from_data_url(&spaced).is_ok());
    }

    #[test]
    fn what_is_not_an_inline_image_is_refused() {
        let msg = |url: &str| image_from_data_url(url).unwrap_err().message;
        assert!(msg("https://example.com/cat.png").contains("not fetched"));
        assert!(msg("file:///etc/passwd").contains("not fetched"));
        assert!(msg("data:image/png,rawbytes").contains("base64"));
        assert!(msg("data:image/svg+xml;base64,PHN2Zz4=").contains("not supported"));
        assert!(msg("data:image/png;base64,!!!notbase64!!!").contains("base64"));
        assert!(msg(&data_url("image/png", b"GIF-looking but not")).contains("not a PNG"));
        let big = vec![0u8; MAX_IMAGE_BYTES + 16];
        let err = image_from_data_url(&data_url("image/png", &big)).unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn messages_carry_images_from_user_turns_only() {
        let url = data_url("image/png", PNG);
        let msgs = vec![user(json!([
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Be brief."}
        ]))];
        let out = to_messages(&msgs).unwrap();
        assert_eq!(out[0].content, "What is this?\nBe brief.");
        assert_eq!(out[0].images.as_ref().unwrap().len(), 1);
        // The bare-string form of image_url works too.
        let out = to_messages(&[user(json!([{"type": "image_url", "image_url": url}]))]).unwrap();
        assert!(Message::any_images(&out));
        // Not in a system message.
        let sys = OaiMessage { role: "system".into(), ..user(json!([{"type": "image_url", "image_url": {"url": url}}])) };
        assert!(to_messages(&[sys]).unwrap_err().message.contains("user messages only"));
        // Unknown part types are refused, not dropped.
        let err = to_messages(&[user(json!([{"type": "input_audio", "input_audio": {}}]))]).unwrap_err();
        assert!(err.message.contains("input_audio"));
        // At most MAX_IMAGES_PER_REQUEST across the request.
        let many: Vec<Value> = (0..=MAX_IMAGES_PER_REQUEST).map(|_| json!({"type": "image_url", "image_url": {"url": url}})).collect();
        assert!(to_messages(&[user(Value::Array(many))]).unwrap_err().message.contains("at most"));
        // Plain strings stay plain.
        let out = to_messages(&[user(json!("hi"))]).unwrap();
        assert!(out[0].images.is_none());
    }

    #[test]
    fn images_need_a_vision_model_and_runner() {
        let text_only = Artifact {
            id: "t",
            family: "t",
            label: "t",
            kind: estia_engine::models::ModelKind::Generation,
            format: estia_engine::models::Format::Gguf,
            repo_id: "",
            revision: "",
            required_disk_bytes: 0,
            capabilities: &[Capability::Text],
            context_length: None,
            license: "",
            files: &[],
        };
        let with_image = vec![Message::new("user", "x").with_images(vec![ImageData { mime: "image/png".into(), data: "AA==".into() }])];
        let plain = vec![Message::new("user", "x")];
        assert!(check_images(&plain, &text_only, None).is_ok());
        assert!(check_images(&with_image, &text_only, None).unwrap_err().message.contains("does not read images"));
        let vision = estia_engine::models::find_artifact("gemma4-e2b-it-4bit-mlx").unwrap();
        assert!(check_images(&with_image, vision, None).is_ok());
        let old_runner = Capabilities { chat: true, ..Default::default() };
        assert!(check_images(&with_image, vision, Some(&old_runner)).unwrap_err().message.contains("cannot take images"));
        let runner = Capabilities { images: true, ..Default::default() };
        assert!(check_images(&with_image, vision, Some(&runner)).is_ok());
    }
}
