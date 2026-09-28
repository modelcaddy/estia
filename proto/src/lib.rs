//! Wire types for the estia runner protocol.
//!
//! **Protocol v1** is the dialect the existing runners already speak, frozen
//! here so the engine and every runner can be tested against one definition:
//!
//! - One JSON object per line on stdin, one per line on stdout.
//! - A runner-level failure is `{"error": "…"}` (resident runners) or
//!   `{"ok": false, "error": "…"}` (one-shot runners). [`check_error`] accepts
//!   both.
//! - Streaming generation emits `{"type":"token","text":"…"}` lines while
//!   decoding, then a terminal `{"done":true}` (or `{"ok":true,"done":true}`).
//!   A runner that cannot stream may instead answer with a single
//!   `{"text":"…"}` envelope; [`parse_stream_line`] surfaces that as
//!   [`StreamEvent::Final`] so callers never regress below one-shot behaviour.
//! - `{"type":"keepalive"}` lines may appear at any time and carry nothing.
//!
//! - `{"type":"cancel"}` asks the runner to stop the generation in flight.
//!   A runner that honours it ends the stream with `{"done":true,"cancelled":true}`
//!   and answers nothing else; an older runner reads it as an unknown request
//!   *after* the generation finishes and replies with an error line, which the
//!   client drains.
//!
//! **Protocol v2** adds, on top of v1 (every v1 line is still valid):
//! - `{"type":"hello"}` → the runner's name, version, protocol number and
//!   capabilities. A v1 runner answers it with an unknown-type error, which the
//!   client reads as "v1, no capabilities". Send it once after spawn.
//! - `{"type":"load","model_path":…,"kind":"generation"|"embedding"}` loads a
//!   model into the runner's cache ahead of the first request and reports the
//!   time it took; `{"type":"unload","model_path":…}` drops it, freeing memory.
//!
//! - `{"type":"chat"}` / `{"type":"chat_stream"}` take `messages` (OpenAI roles:
//!   system, user, assistant, tool), optional `tools` (OpenAI tool schemas,
//!   declared through the model's chat template where it supports them), an
//!   optional `cache_key` (the runner keeps the KV cache per conversation and
//!   prefills only the new suffix) and an optional `format` (honoured only by a
//!   runner that lists it under `capabilities.structured`). A stream ends with
//!   a `{"type":"meta",…}` line — prompt, cached and generated token counts —
//!   before `{"done":true}`.
//! - `{"type":"count_tokens","model_path":…,"text":…}` → `{"tokens":N}`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The dialect this crate describes. Bumped only on an incompatible change.
pub const PROTOCOL_VERSION: u32 = 2;

/// A request to a runner. Borrowed so a large embed batch is serialized once
/// without being copied first.
///
/// Field order and optional-field handling are deliberate: they reproduce the
/// exact bytes earlier clients of these runners sent, so swapping the client changes
/// nothing on the wire. `max_tokens` / `temperature` serialize as `null` when
/// absent (as before); `json` is omitted entirely when `None` because the
/// resident runner never received it and the one-shot runner always did.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request<'a> {
    /// Resident runners: `{"ok": true}` when alive.
    Ping,
    /// One-shot runners: `{"ok": true, "mlx_available": …, "version": …}`.
    Health,
    EmbedBatch {
        model_path: &'a str,
        inputs: &'a [String],
    },
    Embed {
        model_path: &'a str,
        input: &'a str,
    },
    Generate {
        model_path: &'a str,
        prompt: &'a str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        json: Option<bool>,
    },
    GenerateStream {
        model_path: &'a str,
        prompt: &'a str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    },
    /// Stop the generation in flight. Serialized while a stream is open, so
    /// it is the one request written outside the call/response rhythm.
    Cancel,
    /// Protocol v2: who are you, what can you do.
    Hello,
    /// Protocol v2: load a model now rather than on first use.
    Load {
        model_path: &'a str,
        /// `generation` or `embedding`.
        kind: &'a str,
    },
    /// Protocol v2: drop a loaded model.
    Unload {
        model_path: &'a str,
    },
    /// Protocol v2: a conversation rendered by the model's chat template.
    Chat {
        model_path: &'a str,
        messages: &'a [Message],
        #[serde(skip_serializing_if = "Option::is_none")]
        tools: Option<&'a [Value]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_key: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<&'a Value>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    },
    /// Protocol v2: [`Request::Chat`], streamed.
    ChatStream {
        model_path: &'a str,
        messages: &'a [Message],
        #[serde(skip_serializing_if = "Option::is_none")]
        tools: Option<&'a [Value]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_key: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<&'a Value>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    },
    /// Protocol v2: tokens in `text` under the model's tokenizer.
    CountTokens {
        model_path: &'a str,
        text: &'a str,
    },
    /// Apple Foundation Models health (Swift runners only).
    AppleHealth,
    /// Apple Foundation Models generation (Swift runners only).
    AppleGenerate {
        prompt: &'a str,
        temperature: Option<f32>,
    },
}

/// One turn of a conversation, OpenAI-shaped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// OpenAI-shaped tool calls an assistant turn made
    /// (`[{"id", "type": "function", "function": {"name", "arguments"}}]`).
    /// Chat templates only show a `tool` result after the assistant turn that
    /// called it, so dropping these dropped the results too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    /// Images attached to this turn, in order. The runner places them before
    /// the turn's text, as the model's chat template expects. Only runners
    /// that declare [`Capabilities::images`] receive them; the server refuses
    /// images for any other runner rather than dropping them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImageData>>,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into(), name: None, tool_call_id: None, tool_calls: None, images: None }
    }

    /// This turn with `images` attached.
    pub fn with_images(mut self, images: Vec<ImageData>) -> Self {
        self.images = if images.is_empty() { None } else { Some(images) };
        self
    }

    /// Whether any turn in `messages` carries an image.
    pub fn any_images(messages: &[Message]) -> bool {
        messages.iter().any(|m| m.images.as_ref().is_some_and(|i| !i.is_empty()))
    }
}

/// An image attached to a [`Message`]: the encoded file (PNG, JPEG, WebP or
/// GIF), base64 with the standard alphabet and padding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageData {
    /// `image/png`, `image/jpeg`, `image/webp` or `image/gif`.
    pub mime: String,
    /// The file's bytes, base64.
    pub data: String,
}

/// Token accounting a runner reports for one generation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationMeta {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    /// Prompt tokens served from the KV cache instead of prefilled.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    #[serde(default)]
    pub generation_tokens: Option<u64>,
    /// `native` when the model's own chat template rendered the messages,
    /// `manual` for the runner's fallback rendering.
    #[serde(default)]
    pub template: Option<String>,
    /// OpenAI-shaped tool calls the runner parsed itself. Only runners that
    /// declare [`Capabilities::parses_tool_calls`] send it; otherwise the
    /// client parses the model's text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    /// Decode rate the runner measured (tokens/s, generation only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_tps: Option<f64>,
    /// Why decoding stopped, in OpenAI's words: `stop` (end of turn or a stop
    /// sequence), `length` (the `max_tokens` budget ran out) or `tool_calls`.
    /// Optional: without it the server infers `length` from
    /// `generation_tokens` reaching `max_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatResp {
    pub text: String,
    #[serde(default)]
    pub meta: Option<GenerationMeta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CountTokensResp {
    pub tokens: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PingResp {
    pub ok: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HealthResp {
    pub version: Option<String>,
    pub mlx_available: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedBatchResp {
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedResp {
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerateResp {
    pub text: String,
}

/// What a runner can do, as declared in its `hello`. Every field defaults to
/// off so a runner that omits one is read conservatively.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    #[serde(default)]
    pub generate: bool,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub embed: bool,
    #[serde(default)]
    pub cancel: bool,
    #[serde(default)]
    pub load: bool,
    /// `chat` / `chat_stream` with template-rendered messages.
    #[serde(default)]
    pub chat: bool,
    /// Tool schemas declared in the model's native format.
    #[serde(default)]
    pub tools: bool,
    /// `cache_key` reuses the KV cache across turns.
    #[serde(default)]
    pub prompt_cache: bool,
    #[serde(default)]
    pub count_tokens: bool,
    /// Output formats the runner can constrain decoding to (`json`,
    /// `json_schema`). Empty means the client validates and repairs instead.
    #[serde(default)]
    pub structured: Vec<String>,
    /// The runner parses tool calls itself and returns them in
    /// `meta.tool_calls` (llama.cpp does); text is not re-parsed.
    #[serde(default)]
    pub parses_tool_calls: bool,
    /// The runner accepts [`Message::images`] and passes them to the model.
    #[serde(default)]
    pub images: bool,
    /// The backend id vectors and outputs from this runner carry
    /// (`mlx-python`, `llama-cpp`). Absent on older runners.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
}

/// Answer to `hello`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HelloResp {
    pub runner: String,
    pub version: String,
    pub protocol: u32,
    #[serde(default)]
    pub capabilities: Capabilities,
}

/// Answer to `load`.
#[derive(Debug, Clone, Deserialize)]
pub struct LoadResp {
    pub loaded: bool,
    #[serde(default)]
    pub ms: u64,
}

/// Answer to `unload`.
#[derive(Debug, Clone, Deserialize)]
pub struct UnloadResp {
    pub unloaded: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ProtoError {
    /// The runner answered with an error envelope.
    #[error("runner error: {0}")]
    Runner(String),
    /// The runner refused the request as invalid (llama-server's chat template
    /// rejecting the conversation, say). The client's request is at fault, not
    /// the runner.
    #[error("runner refused the request: {0}")]
    Refused(String),
    /// The line was not JSON.
    #[error("runner returned invalid JSON: {0}")]
    Invalid(#[from] serde_json::Error),
}

fn error_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "runner failed without an error message".to_string(),
        other => other.to_string(),
    }
}

/// Parse one response line, mapping either error envelope to
/// [`ProtoError::Runner`] and returning the payload otherwise. Typed
/// deserialization is the caller's next step (`serde_json::from_value`).
pub fn check_error(line: &str) -> Result<Value, ProtoError> {
    let v: Value = serde_json::from_str(line.trim())?;
    if let Some(err) = v.get("error") {
        if !err.is_null() {
            if v.get("refused").and_then(Value::as_bool) == Some(true) {
                return Err(ProtoError::Refused(error_text(err)));
            }
            return Err(ProtoError::Runner(error_text(err)));
        }
    }
    if v.get("ok").and_then(Value::as_bool) == Some(false) {
        return Err(ProtoError::Runner(error_text(v.get("error").unwrap_or(&Value::Null))));
    }
    Ok(v)
}

/// One line of a streaming generation, classified.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// A decoded piece of text. May be empty; callers usually skip empties.
    Token(String),
    /// The stream finished normally.
    Done,
    /// The runner stopped because it was asked to. Tokens already delivered
    /// stand; nothing else follows.
    Cancelled,
    /// The runner failed. Tokens already delivered stand; nothing is retried.
    Error(String),
    /// The runner refused the request as invalid. Tokens already delivered
    /// stand; nothing else follows.
    Refused(String),
    /// Heartbeat with no content.
    Keepalive,
    /// Token accounting for the generation, sent just before `Done`.
    Meta(Value),
    /// A runner that could not stream answered with the whole text at once.
    Final(String),
    /// Valid JSON the protocol does not define. Ignored.
    Other,
}

/// Classify one stdout line of a streaming call. `None` for blank lines and
/// non-JSON chatter, which callers skip to keep the protocol alive.
pub fn parse_stream_line(line: &str) -> Option<StreamEvent> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let v: Value = serde_json::from_str(trimmed).ok()?;
    if let Some(err) = v.get("error") {
        if !err.is_null() {
            if v.get("refused").and_then(Value::as_bool) == Some(true) {
                return Some(StreamEvent::Refused(error_text(err)));
            }
            return Some(StreamEvent::Error(error_text(err)));
        }
    }
    if v.get("ok").and_then(Value::as_bool) == Some(false) {
        return Some(StreamEvent::Error(error_text(v.get("error").unwrap_or(&Value::Null))));
    }
    match v.get("type").and_then(Value::as_str) {
        Some("token") => {
            let text = v.get("text").and_then(Value::as_str).unwrap_or("");
            return Some(StreamEvent::Token(text.to_string()));
        }
        Some("keepalive") => return Some(StreamEvent::Keepalive),
        Some("meta") => return Some(StreamEvent::Meta(v)),
        _ => {}
    }
    if v.get("done").and_then(Value::as_bool) == Some(true) {
        if v.get("cancelled").and_then(Value::as_bool) == Some(true) {
            return Some(StreamEvent::Cancelled);
        }
        return Some(StreamEvent::Done);
    }
    if let Some(text) = v.get("text").and_then(Value::as_str) {
        return Some(StreamEvent::Final(text.to_string()));
    }
    Some(StreamEvent::Other)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_serialize_to_the_legacy_bytes() {
        let inputs = vec!["a".to_string(), "b".to_string()];
        let r = Request::EmbedBatch { model_path: "/m", inputs: &inputs };
        assert_eq!(serde_json::to_string(&r).unwrap(), r#"{"type":"embed_batch","model_path":"/m","inputs":["a","b"]}"#);
        // Resident generate: no `json` field at all when None.
        let r = Request::Generate { model_path: "/m", prompt: "p", max_tokens: Some(64), temperature: Some(0.0), json: None };
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"type":"generate","model_path":"/m","prompt":"p","max_tokens":64,"temperature":0.0}"#
        );
        // One-shot generate: `json` present, nulls preserved for the optionals.
        let r = Request::Generate { model_path: "/m", prompt: "p", max_tokens: None, temperature: None, json: Some(true) };
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"type":"generate","model_path":"/m","prompt":"p","max_tokens":null,"temperature":null,"json":true}"#
        );
        assert_eq!(serde_json::to_string(&Request::Ping).unwrap(), r#"{"type":"ping"}"#);
        assert_eq!(serde_json::to_string(&Request::AppleHealth).unwrap(), r#"{"type":"apple_health"}"#);
    }

    #[test]
    fn check_error_accepts_both_envelopes() {
        assert!(matches!(check_error(r#"{"error":"boom"}"#), Err(ProtoError::Runner(m)) if m == "boom"));
        assert!(matches!(check_error(r#"{"ok":false,"error":"boom"}"#), Err(ProtoError::Runner(m)) if m == "boom"));
        assert!(matches!(check_error(r#"{"ok":false}"#), Err(ProtoError::Runner(_))));
        assert!(matches!(check_error("not json"), Err(ProtoError::Invalid(_))));
        let v = check_error(r#"{"embeddings":[[1.0]]}"#).unwrap();
        assert_eq!(v["embeddings"][0][0], 1.0);
        let v = check_error(r#"{"ok":true,"text":"hi","error":null}"#).unwrap();
        assert_eq!(v["text"], "hi");
    }

    #[test]
    fn stream_lines_classify() {
        assert_eq!(parse_stream_line(""), None);
        assert_eq!(parse_stream_line("loading weights..."), None);
        assert_eq!(parse_stream_line(r#"{"type":"token","text":"he"}"#), Some(StreamEvent::Token("he".into())));
        assert_eq!(parse_stream_line(r#"{"type":"keepalive"}"#), Some(StreamEvent::Keepalive));
        assert_eq!(parse_stream_line(r#"{"done":true}"#), Some(StreamEvent::Done));
        assert_eq!(parse_stream_line(r#"{"ok":true,"done":true}"#), Some(StreamEvent::Done));
        assert_eq!(parse_stream_line(r#"{"done":true,"cancelled":true}"#), Some(StreamEvent::Cancelled));
        assert_eq!(serde_json::to_string(&Request::Cancel).unwrap(), r#"{"type":"cancel"}"#);
        assert_eq!(serde_json::to_string(&Request::Hello).unwrap(), r#"{"type":"hello"}"#);
        assert_eq!(
            serde_json::to_string(&Request::Load { model_path: "/m", kind: "generation" }).unwrap(),
            r#"{"type":"load","model_path":"/m","kind":"generation"}"#
        );
        let msgs = vec![Message::new("system", "terse"), Message::new("user", "hi")];
        let r = Request::ChatStream {
            model_path: "/m",
            messages: &msgs,
            tools: None,
            cache_key: Some("c1"),
            format: None,
            max_tokens: Some(8),
            temperature: Some(0.0),
        };
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"type":"chat_stream","model_path":"/m","messages":[{"role":"system","content":"terse"},{"role":"user","content":"hi"}],"cache_key":"c1","max_tokens":8,"temperature":0.0}"#
        );
        let h: HelloResp = serde_json::from_str(
            r#"{"ok":true,"runner":"mlx-python","version":"2","protocol":2,"capabilities":{"cancel":true,"stream":true}}"#,
        )
        .unwrap();
        assert!(h.capabilities.cancel && h.capabilities.stream && !h.capabilities.load);
        assert_eq!(h.protocol, 2);
        assert_eq!(parse_stream_line(r#"{"text":"all at once"}"#), Some(StreamEvent::Final("all at once".into())));
        assert_eq!(parse_stream_line(r#"{"ok":true,"text":"all at once"}"#), Some(StreamEvent::Final("all at once".into())));
        assert_eq!(parse_stream_line(r#"{"error":"gpu fell over"}"#), Some(StreamEvent::Error("gpu fell over".into())));
        assert_eq!(parse_stream_line(r#"{"ok":false,"error":"nope"}"#), Some(StreamEvent::Error("nope".into())));
        assert_eq!(parse_stream_line(r#"{"progress":0.5}"#), Some(StreamEvent::Other));
        assert!(matches!(parse_stream_line(r#"{"type":"meta","prompt_tokens":10,"cached_tokens":4}"#), Some(StreamEvent::Meta(_))));
    }

    #[test]
    fn refused_error_envelopes_carry_their_kind() {
        assert!(matches!(
            check_error(r#"{"error":"template refuses","refused":true}"#),
            Err(ProtoError::Refused(m)) if m == "template refuses"
        ));
        assert!(matches!(
            parse_stream_line(r#"{"error":"template refuses","refused":true}"#),
            Some(StreamEvent::Refused(m)) if m == "template refuses"
        ));
        // Without the flag the envelope stays a plain runner error.
        assert!(matches!(check_error(r#"{"error":"nope"}"#), Err(ProtoError::Runner(m)) if m == "nope"));
        assert_eq!(parse_stream_line(r#"{"error":"gpu fell over"}"#), Some(StreamEvent::Error("gpu fell over".into())));
    }
}
