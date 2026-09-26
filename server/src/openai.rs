//! `/v1/*` — OpenAI's shapes exactly, so anything that speaks to OpenAI speaks
//! to the engine. Extra engine facts ride in an `x_estia` field the client can
//! ignore.

use crate::{derive_cache_key, toolcalls, ApiError, AppState};
use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use estia_engine::models::embed::EmbedTask;
use estia_engine::proto::Message;
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
    /// OpenAI's end-user id; used as the prompt-cache key when present.
    #[serde(default)]
    pub user: Option<String>,
    /// Engine extension: `interactive` (default) or `background`.
    #[serde(default)]
    pub priority: Option<String>,
}

/// Flatten OpenAI content (a string, or an array of `{type:"text",text}` parts)
/// to text. Non-text parts are dropped with a marker.
fn content_text(v: &Option<Value>) -> String {
    match v {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") => p.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
                Some(other) => format!("[{other} omitted]"),
                None => p.as_str().unwrap_or("").to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => other.to_string(),
    }
}

pub(crate) fn to_messages(msgs: &[OaiMessage]) -> Vec<Message> {
    msgs.iter()
        .map(|m| {
            let mut content = content_text(&m.content);
            if let Some(calls) = &m.tool_calls {
                // The model sees its earlier call as text; the template has no
                // slot for structured prior calls we can rely on across models.
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&Value::Array(calls.clone()).to_string());
            }
            Message { role: m.role.clone(), content, name: m.name.clone(), tool_call_id: m.tool_call_id.clone() }
        })
        .collect()
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

fn priority_of(p: &Option<String>) -> Priority {
    match p.as_deref() {
        Some("background") => Priority::Background,
        _ => Priority::Interactive,
    }
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn completion_id() -> String {
    format!("chatcmpl-{}-{}", std::process::id(), now_unix())
}

/// Everything a finished completion needs to be rendered, streamed or not.
struct Finished {
    text: String,
    tool_calls: Vec<Value>,
    structured: Option<Structured>,
    meta: estia_engine::proto::GenerationMeta,
    ms: u128,
}

fn finish_reason(f: &Finished) -> &'static str {
    if f.tool_calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    }
}

fn usage(meta: &estia_engine::proto::GenerationMeta) -> Value {
    let p = meta.prompt_tokens.unwrap_or(0);
    let c = meta.generation_tokens.unwrap_or(0);
    json!({"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
           "prompt_tokens_details": {"cached_tokens": meta.cached_tokens.unwrap_or(0)}})
}

fn x_estia(artifact: &estia_engine::models::Artifact, f: &Finished, backend: &str) -> Value {
    json!({
        "family": artifact.family,
        "backend": backend,
        "cached_tokens": f.meta.cached_tokens,
        "template": f.meta.template,
        "repaired": f.structured.as_ref().map(|s| s.repaired),
        "repairs": f.structured.as_ref().map(|s| s.repairs.clone()),
        "ms": f.ms,
    })
}

pub async fn chat_completions(State(state): State<Arc<AppState>>, Json(req): Json<ChatCompletionRequest>) -> Result<Response, ApiError> {
    let artifact = state.resolve_generation(&req.model)?;
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let messages = to_messages(&req.messages);
    let format = parse_response_format(&req.response_format)?;
    let tools = req.tools.clone().filter(|t| !t.is_empty());
    let cache_key = req.user.clone().unwrap_or_else(|| derive_cache_key(artifact.id, &messages));
    let max_tokens = req.max_completion_tokens.or(req.max_tokens).unwrap_or(1024);
    let temperature = req.temperature.unwrap_or(0.2);
    let prio = priority_of(&req.priority);
    let backend = state.embed_backend();
    let stream = req.stream.unwrap_or(false);

    let st = Arc::clone(&state);
    let session = tokio::task::spawn_blocking(move || st.gen_session(artifact)).await??;

    if !stream {
        let s = Arc::clone(&session);
        let (msgs, tls, ck, fmt) = (messages.clone(), tools.clone(), cache_key.clone(), format.clone());
        let finished: Finished = tokio::task::spawn_blocking(move || -> Result<Finished, ApiError> {
            let t0 = Instant::now();
            let mut out = s.chat_with(&msgs, tls.as_deref(), Some(&ck), None, Some(max_tokens), Some(temperature), prio)?;
            let mut structured = None;
            if fmt.wants_json() {
                structured = Some(match structured::enforce(&out.text, &fmt) {
                    Ok(v) => v,
                    Err(first_err) => {
                        // One retry with the validator's complaint appended.
                        let mut retry = msgs.clone();
                        retry.push(Message::new("assistant", out.text.clone()));
                        retry.push(Message::new("user", Structured::retry_hint(&first_err).trim().to_string()));
                        out = s.chat_with(&retry, tls.as_deref(), Some(&ck), None, Some(max_tokens), Some(temperature), prio)?;
                        structured::enforce(&out.text, &fmt)
                            .map_err(|e| ApiError::unprocessable(format!("structured output failed after retry: {e}")))?
                    }
                });
            }
            let tool_calls = if tls.is_some() {
                toolcalls::parse(&out.text).iter().enumerate().map(|(i, c)| c.to_openai(i)).collect()
            } else {
                Vec::new()
            };
            Ok(Finished { text: out.text, tool_calls, structured, meta: out.meta, ms: t0.elapsed().as_millis() })
        })
        .await??;

        let content: Value = match &finished.structured {
            Some(s) if !finished.tool_calls.is_empty() => Value::String(s.value.to_string()),
            Some(s) => Value::String(s.value.to_string()),
            None if !finished.tool_calls.is_empty() => Value::Null,
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
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(&finished)}],
            "usage": usage(&finished.meta),
            "x_estia": x_estia(artifact, &finished, backend),
        });
        return Ok(Json(body).into_response());
    }

    // Streaming. Tokens go over a channel from the blocking generation to the
    // SSE stream. If the client disconnects, the receiver drops, the next send
    // fails, and the generation is cancelled in the runner.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let id = completion_id();
    let created = now_unix();
    let model_id = artifact.id.to_string();
    let wants_json = format.wants_json();
    let has_tools = tools.is_some();
    let s = Arc::clone(&session);
    tokio::task::spawn_blocking(move || {
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
        let _ = send(chunk(json!({"role": "assistant"}), None, None));

        let cancel = CancelToken::new();
        let flip = cancel.clone();
        // Hold the first characters back to decide whether this is a tool
        // call (buffer everything, emit tool_calls at the end) or prose (stream).
        let mut buffered = String::new();
        let mut decided: Option<bool> = if wants_json || !has_tools { Some(!wants_json && !has_tools) } else { None };
        // decided == Some(true): stream live; Some(false): buffer to the end.
        if !has_tools && !wants_json {
            decided = Some(true);
        }
        let t0 = Instant::now();
        let result = s.chat_stream_with(
            &messages,
            tools.as_deref(),
            Some(&cache_key),
            None,
            Some(max_tokens),
            Some(temperature),
            prio,
            Some(&cancel),
            |tok| match decided {
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
            },
        );
        let ms = t0.elapsed().as_millis();
        match result {
            Ok(outcome) => {
                let full = outcome.text;
                let mut finish = "stop";
                let mut final_delta = json!({});
                if decided != Some(true) {
                    // Everything is still buffered: tool call, JSON, or short prose.
                    let calls: Vec<Value> = if has_tools {
                        toolcalls::parse(&full).iter().enumerate().map(|(i, c)| c.to_openai(i)).collect()
                    } else {
                        Vec::new()
                    };
                    if !calls.is_empty() {
                        finish = "tool_calls";
                        final_delta = json!({"tool_calls": calls});
                    } else if wants_json {
                        match structured::enforce(&full, &format) {
                            Ok(sv) => final_delta = json!({"content": sv.value.to_string()}),
                            Err(e) => {
                                let _ = send(
                                    json!({"error": {"message": format!("structured output failed: {e}"), "type": "invalid_request_error"}}),
                                );
                                let _ = tx.send(Ok(Event::default().data("[DONE]")));
                                return;
                            }
                        }
                    } else {
                        final_delta = json!({"content": full});
                    }
                }
                let extra = json!({"usage": usage(&outcome.meta), "x_estia": {"cached_tokens": outcome.meta.cached_tokens, "template": outcome.meta.template, "ms": ms}});
                let _ = send(chunk(final_delta, Some(finish), Some(extra)));
                let _ = tx.send(Ok(Event::default().data("[DONE]")));
            }
            Err(estia_engine::SessionError::Cancelled { .. }) => {
                // Client went away; nothing to tell it.
            }
            Err(e) => {
                let _ = send(json!({"error": {"message": e.to_string(), "type": "server_error"}}));
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
    if inputs.is_empty() {
        return Err(ApiError::bad_request("input must not be empty"));
    }
    let task = embed_task(&req.task)?;
    let fingerprint = model.fingerprint_for(state.embed_backend());
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
    let vectors = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>, ApiError> {
        let session = st.embed_session(model)?;
        Ok(session.embed_batch_with(&prefixed, prio)?)
    })
    .await??;
    let data: Vec<Value> = vectors.iter().enumerate().map(|(i, v)| json!({"object": "embedding", "index": i, "embedding": v})).collect();
    Ok(Json(json!({
        "object": "list",
        "data": data,
        "model": model.id,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
        "x_estia": {"fingerprint": fingerprint, "dims": model.dims, "task": task_name(task)}
    }))
    .into_response())
}

pub async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let roles = state.engine.roles();
    let mut data = Vec::new();
    for (role, b) in roles.iter() {
        data.push(json!({"id": role, "object": "model", "created": 0, "owned_by": "estia", "x_estia": {"role": true, "family": b.family}}));
    }
    for a in estia_engine::models::GENERATION_MODELS {
        data.push(json!({"id": a.id, "object": "model", "created": 0, "owned_by": "estia",
            "x_estia": {"family": a.family, "format": format!("{:?}", a.format).to_lowercase(), "installed": state.engine.store().is_installed(a.id), "context_length": a.context_length}}));
    }
    for e in estia_engine::models::EMBEDDING_MODELS {
        data.push(json!({"id": e.id, "object": "model", "created": 0, "owned_by": "estia",
            "x_estia": {"kind": "embedding", "dims": e.dims, "installed": state.engine.store().is_installed(e.id)}}));
    }
    Json(json!({"object": "list", "data": data}))
}
