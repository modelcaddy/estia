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
use estia_engine::proto::{Capabilities, Message};
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
        .map(|m| Message {
            role: m.role.clone(),
            content: content_text(&m.content),
            name: m.name.clone(),
            tool_call_id: m.tool_call_id.clone(),
            // Passed through structured: chat templates render a `tool` result
            // only after the assistant turn whose `tool_calls` it answers.
            tool_calls: m.tool_calls.clone().filter(|c| !c.is_empty()),
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

fn completion_id() -> String {
    format!("chatcmpl-{}-{}", std::process::id(), now_unix())
}

/// Everything a finished completion needs to be rendered, streamed or not.
struct Finished {
    text: String,
    tool_calls: Vec<Value>,
    /// The runner parsed the calls itself, so `text` is plain content.
    runner_parsed: bool,
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
        "generation_tps": f.meta.generation_tps,
        "repaired": f.structured.as_ref().map(|s| s.repaired),
        "repairs": f.structured.as_ref().map(|s| s.repairs.clone()),
        "ms": f.ms,
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
    let messages = to_messages(&req.messages);
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
        if let Some(a) = &access {
            a.finish(finish_reason(&finished));
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
        let _ = send(chunk(json!({"role": "assistant"}), None, None));

        let cancel = CancelToken::new();
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
                let mut finish = "stop";
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
                    "generation_tps": outcome.meta.generation_tps, "ms": ms,
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
                        "context_length": a.context_length}}));
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
