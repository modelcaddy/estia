//! `/engine/*` — what OpenAI's shape cannot say.

use crate::access::{spawn_blocking_in_span, Access};
use crate::jobs::JobStatus;
use crate::openai::{apply_prefix, embed_task, inputs_of, parse_response_format, task_name, to_messages, OaiMessage};
use crate::pairing::PairingError;
use crate::{derive_cache_key, toolcalls, ApiError, AppState, API_VERSION};
use axum::{
    extract::{Path, State},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use estia_engine::models::{find_artifact, find_embed_model, DownloadSpec, EMBEDDING_MODELS, GENERATION_MODELS};
use estia_engine::proto::Message;
use estia_engine::structured::{self, Structured};
use estia_engine::{CancelToken, Priority, Roles};
use serde::Deserialize;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::Instrument as _;

// Work caps: one request must not be able to hold a model indefinitely.
// Public so the `/v1/*` handlers apply the same limits.

/// Ceiling on `max_tokens`; larger requests are clamped to it.
pub const MAX_TOKENS_CEILING: u32 = 8192;
/// Most structured-output attempts one `/engine/generate` may make; larger
/// `max_attempts` values are clamped to it (and 0 is raised to 1).
pub const MAX_ATTEMPTS: u32 = 3;
/// Most inputs one embed request may carry; more is a 400.
pub const MAX_EMBED_INPUTS: usize = 256;

/// `max_tokens` as requested (or `default`), clamped to [`MAX_TOKENS_CEILING`].
pub fn capped_max_tokens(requested: Option<u32>, default: u32) -> u32 {
    requested.unwrap_or(default).min(MAX_TOKENS_CEILING)
}

/// 400 when an embed request carries more than [`MAX_EMBED_INPUTS`] inputs.
pub fn check_embed_inputs(count: usize) -> Result<(), ApiError> {
    if count > MAX_EMBED_INPUTS {
        return Err(ApiError::bad_request(format!(
            "{count} inputs in one request; at most {MAX_EMBED_INPUTS} (split them across requests)"
        )));
    }
    Ok(())
}

/// A pairing failure as HTTP: bad input 400, a full queue 429, storage
/// trouble 500 (details to the log, not to the unauthenticated caller), and
/// an unknown id `missing` (404 when polling; 400 on approve/deny, as
/// documented).
fn pairing_error(e: PairingError, missing: fn(String) -> ApiError) -> ApiError {
    match e {
        PairingError::NotFound { .. } => missing(e.to_string()),
        PairingError::Invalid(_) => ApiError::bad_request(e.to_string()),
        PairingError::TooMany(_) => {
            ApiError { status: axum::http::StatusCode::TOO_MANY_REQUESTS, kind: "rate_limit_error", message: e.to_string() }
        }
        PairingError::Storage(_) => {
            tracing::error!(error = %e, "pairing store failure");
            ApiError::internal("the pairing store could not be read or written; see the engine's log")
        }
    }
}

pub async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let rt = state.engine.runtime().status();
    Json(json!({
        "ok": true,
        "engine": "estia",
        "version": env!("CARGO_PKG_VERSION"),
        "api_version": API_VERSION,
        "protocol_version": estia_engine::proto::PROTOCOL_VERSION,
        "bind": state.bind.to_string(),
        "auth_required": state.require_auth,
        "uptime_s": state.started.elapsed().as_secs(),
        "backends": [{"id": "mlx-python", "runtime_installed": rt.state == estia_engine::runtime::RuntimeState::Installed}],
        "loaded": state.loaded(),
    }))
}

pub async fn get_defaults(State(state): State<Arc<AppState>>) -> Json<Roles> {
    Json(state.engine.roles())
}

pub async fn put_defaults(State(state): State<Arc<AppState>>, Json(roles): Json<Roles>) -> Result<Json<Roles>, ApiError> {
    // Re-bind through the checked path so capabilities are enforced.
    let mut checked = Roles::default();
    for (role, b) in roles.iter() {
        checked.bind(role, b.clone()).map_err(|e| ApiError::bad_request(e.to_string()))?;
    }
    // Persist the same file `estia role set` writes, keeping any other keys, so
    // roles set from a client survive a restart.
    let path = state.data_dir.join("config.json");
    let mut cfg: Value = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_else(|| json!({}));
    if !cfg.is_object() {
        cfg = json!({});
    }
    cfg["roles"] = serde_json::to_value(&checked).map_err(|e| ApiError::internal(e.to_string()))?;
    std::fs::write(&path, serde_json::to_string_pretty(&cfg).unwrap_or_default())
        .map_err(|e| ApiError::internal(format!("write config.json: {e}")))?;
    state.engine.set_roles(checked.clone());
    let table: Vec<String> = checked.iter().map(|(role, b)| format!("{role}={}", b.family)).collect();
    tracing::info!(roles = %table.join(","), "role table replaced");
    Ok(Json(checked))
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let store = state.engine.store();
    let gen: Vec<Value> = GENERATION_MODELS
        .iter()
        .map(|a| {
            json!({
                "id": a.id, "family": a.family, "kind": "generation", "format": format!("{:?}", a.format).to_lowercase(),
                "label": a.label, "repo_id": a.repo_id, "revision": a.revision, "license": a.license,
                "context_length": a.context_length, "capabilities": a.capabilities,
                "installed": store.is_installed(a.id), "bytes_on_disk": store.bytes_on_disk(a.id),
                "partial_bytes": store.partial_bytes_on_disk(a.id), "required_disk_bytes": a.required_disk_bytes,
                "pulling": state.jobs.running_pull(a.id).map(|j| j.id),
            })
        })
        .collect();
    let emb: Vec<Value> = EMBEDDING_MODELS
        .iter()
        .map(|e| {
            json!({
                "id": e.id, "kind": "embedding", "format": "mlx", "label": e.label, "repo_id": e.repo_id, "revision": e.revision,
                "license": e.license, "dims": e.dims, "arch": e.arch.model_type(), "multilingual": e.multilingual,
                "fingerprint": e.fingerprint_for(state.embed_backend()),
                "installed": store.is_installed(e.id), "bytes_on_disk": store.bytes_on_disk(e.id), "required_disk_bytes": e.required_disk_bytes,
            })
        })
        .collect();
    Json(json!({"generation": gen, "embedding": emb, "models_dir": store.models_dir()}))
}

#[derive(Debug, Deserialize)]
pub struct PullRequest {
    pub id: String,
}

pub async fn pull_model(State(state): State<Arc<AppState>>, Json(req): Json<PullRequest>) -> Result<Response, ApiError> {
    let spec = if let Some(a) = find_artifact(&req.id) {
        DownloadSpec::from(a)
    } else if let Some(e) = find_embed_model(&req.id) {
        DownloadSpec { id: e.id.into(), repo_id: e.repo_id.into(), revision: e.revision.into(), required_disk_bytes: e.required_disk_bytes }
    } else {
        return Err(ApiError::not_found(format!("unknown model `{}`", req.id)));
    };
    if let Some(job) = state.jobs.running_pull(&spec.id) {
        if let Some(a) = Access::current() {
            a.job(&job.id);
        }
        return Ok((axum::http::StatusCode::ACCEPTED, Json(json!({"job_id": job.id, "already_running": true}))).into_response());
    }
    let (job_id, view) = state.jobs.create("pull", &spec.id);
    if let Some(a) = Access::current() {
        a.job(&job_id);
    }
    tracing::info!(job_id = %job_id, model = %spec.id, repo = %spec.repo_id, revision = %spec.revision, "model pull started");
    let store = state.engine.store().clone();
    let view2 = view.clone();
    // The job outlives the request: its events carry the job id, not the request's.
    let span = tracing::info_span!(parent: None, "job", job_id = %job_id);
    let (jid, model) = (job_id.clone(), spec.id.clone());
    let (end_jid, end_model) = (job_id.clone(), spec.id.clone());
    tokio::spawn(
        async move {
            let t0 = Instant::now();
            let mut milestones = PullMilestones::default();
            let result = store
                .download(&spec, move |p| {
                    if let Some(pct) = milestones.crossed(&p) {
                        tracing::info!(job_id = %jid, model = %model, percent = pct, bytes = p.bytes_downloaded, total_bytes = p.total_bytes, "model pull progress");
                    }
                    view2.send_modify(|v| v.progress = Some(p));
                })
                .await;
            match &result {
                Ok(summary) => tracing::info!(
                    job_id = %end_jid,
                    model = %summary.model_id,
                    files = summary.files_downloaded,
                    bytes = summary.bytes_downloaded,
                    secs = t0.elapsed().as_secs(),
                    "model pull finished"
                ),
                Err(e) => tracing::warn!(job_id = %end_jid, model = %end_model, error = %format!("{e:#}"), "model pull failed"),
            }
            view.send_modify(|v| match &result {
            Ok(summary) => {
                v.status = JobStatus::Done;
                v.result = serde_json::to_value(summary).ok();
            }
                Err(e) => {
                    v.status = JobStatus::Failed;
                    v.error = Some(e.to_string());
                }
            });
        }
        .instrument(span),
    );
    Ok((axum::http::StatusCode::ACCEPTED, Json(json!({"job_id": job_id}))).into_response())
}

/// Which tenth of a download was last logged, so a pull logs at 10 %, 20 %, …
#[derive(Default)]
struct PullMilestones {
    logged: Option<u64>,
}

impl PullMilestones {
    /// The percentage to log for this progress report, when it reaches a new
    /// tenth (10 to 90; the end is logged as "finished").
    fn crossed(&mut self, p: &estia_engine::models::DownloadProgress) -> Option<u64> {
        let total = p.total_bytes.filter(|t| *t > 0)?;
        if p.phase != "downloading" {
            return None;
        }
        let tenth = (p.bytes_downloaded.min(total).saturating_mul(10) / total).min(9);
        if tenth == 0 || self.logged.is_some_and(|l| l >= tenth) {
            return None;
        }
        self.logged = Some(tenth);
        Some(tenth * 10)
    }
}

pub async fn delete_model(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    if find_artifact(&id).is_none() && find_embed_model(&id).is_none() {
        return Err(ApiError::not_found(format!("unknown model `{id}`")));
    }
    let removed = state.engine.store().remove(&id).await?;
    tracing::info!(model = %id, removed, "model removed");
    Ok(Json(json!({"removed": removed})))
}

/// SSE of the running pull for a model (one event per progress change).
pub async fn model_progress(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Response, ApiError> {
    let job = state.jobs.running_pull(&id).ok_or_else(|| ApiError::not_found(format!("no running pull for `{id}`")))?;
    job_stream(&state, &job.id)
}

/// SSE of any job by id (one event per change, ends when the job does).
pub async fn job_events(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Response, ApiError> {
    job_stream(&state, &id)
}

fn job_stream(state: &AppState, job_id: &str) -> Result<Response, ApiError> {
    let mut rx = state.jobs.subscribe(job_id).ok_or_else(|| ApiError::not_found(format!("no job `{job_id}`")))?;
    let (tx, out) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
    tokio::spawn(async move {
        loop {
            let view = rx.borrow_and_update().clone();
            let done = !matches!(view.status, JobStatus::Running);
            if tx.send(Ok(Event::default().data(serde_json::to_string(&view).unwrap_or_default()))).is_err() {
                break;
            }
            if done || rx.changed().await.is_err() {
                break;
            }
        }
    });
    Ok(Sse::new(UnboundedReceiverStream::new(out)).keep_alive(KeepAlive::default()).into_response())
}

/// Install the Python runtime the MLX backends run in, as a job (admin). The
/// same work as `estia setup`'s first step, so a browser or a host application
/// can set a fresh machine up without a terminal.
pub async fn install_runtime(State(state): State<Arc<AppState>>) -> Result<Response, ApiError> {
    const ID: &str = "python-runtime";
    if let Some(job) = state.jobs.running("runtime", ID) {
        return Ok((axum::http::StatusCode::ACCEPTED, Json(json!({"job_id": job.id, "already_running": true}))).into_response());
    }
    let (job_id, view) = state.jobs.create("runtime", ID);
    if let Some(a) = Access::current() {
        a.job(&job_id);
    }
    tracing::info!(job_id = %job_id, "runtime install started");
    let runtime = state.engine.runtime().clone();
    let view2 = view.clone();
    let span = tracing::info_span!(parent: None, "job", job_id = %job_id);
    let (jid, end_jid) = (job_id.clone(), job_id.clone());
    tokio::spawn(
        async move {
            let t0 = Instant::now();
            let mut phase: &'static str = "";
            let mut phase_started = Instant::now();
            let result = runtime
                .install(move |p| {
                    if p.phase != phase {
                        if !phase.is_empty() {
                            tracing::debug!(job_id = %jid, phase = %phase, ms = phase_started.elapsed().as_millis() as u64, "runtime install phase done");
                        }
                        tracing::info!(job_id = %jid, phase = %p.phase, step = %p.message, "runtime install phase");
                        phase = p.phase;
                        phase_started = Instant::now();
                    }
                    view2.send_modify(|v| v.setup = Some(p))
                })
                .await;
            match &result {
                Ok(summary) => tracing::info!(
                    job_id = %end_jid,
                    python = %summary.python_version,
                    mlx_lm = %summary.mlx_lm_version,
                    secs = t0.elapsed().as_secs(),
                    "runtime install finished"
                ),
                Err(e) => tracing::warn!(job_id = %end_jid, error = %format!("{e:#}"), "runtime install failed"),
            }
            view.send_modify(|v| match &result {
                Ok(summary) => {
                    v.status = JobStatus::Done;
                    v.result = serde_json::to_value(summary).ok();
                }
                Err(e) => {
                    v.status = JobStatus::Failed;
                    v.error = Some(e.to_string());
                }
            });
        }
        .instrument(span),
    );
    Ok((axum::http::StatusCode::ACCEPTED, Json(json!({"job_id": job_id}))).into_response())
}

pub async fn list_jobs(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"jobs": state.jobs.list()}))
}

pub async fn get_job(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    state
        .jobs
        .get(&id)
        .map(|j| Json(serde_json::to_value(j).unwrap_or_default()))
        .ok_or_else(|| ApiError::not_found(format!("no job `{id}`")))
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    /// Role, family or artifact id. Default: the `text` role.
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<OaiMessage>>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub cache_key: Option<String>,
    /// `{"type":"text"|"json_object"|"json_schema", …}` (OpenAI's response_format shape).
    #[serde(default)]
    pub format: Option<Value>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub priority: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Attempts for structured output (default 2 = one retry).
    #[serde(default)]
    pub max_attempts: Option<u32>,
}

pub async fn generate(State(state): State<Arc<AppState>>, Json(req): Json<GenerateRequest>) -> Result<Response, ApiError> {
    let name = req.model.clone().unwrap_or_else(|| "text".to_string());
    let artifact = state.resolve_generation(&name)?;
    let format = parse_response_format(&req.format)?;
    let prio = if req.priority.as_deref() == Some("background") { Priority::Background } else { Priority::Interactive };
    let max_tokens = capped_max_tokens(req.max_tokens, 1024);
    let temperature = req.temperature.unwrap_or(0.2);
    let attempts = req.max_attempts.unwrap_or(2).clamp(1, MAX_ATTEMPTS);
    let streaming = req.stream.unwrap_or(false) && !format.wants_json();
    let access = Access::current();
    if let Some(a) = &access {
        a.generation(&name, artifact.id, streaming, max_tokens);
    }
    let st = Arc::clone(&state);
    let (session, load_ms) = spawn_blocking_in_span(move || st.gen_session_timed(artifact)).await??;
    if let (Some(a), Some(ms)) = (&access, load_ms) {
        a.loaded(ms);
    }

    // Either a raw prompt (the resident runner's `generate_stream`) or messages
    // (protocol v2 `chat_stream` through the template).
    let messages: Option<Vec<Message>> = req.messages.as_ref().map(|m| to_messages(m));
    let prompt = req.prompt.clone();
    if messages.is_none() && prompt.is_none() {
        return Err(ApiError::bad_request("prompt or messages is required"));
    }
    let tools = req.tools.clone().filter(|t| !t.is_empty());
    let cache_key = req
        .cache_key
        .clone()
        .or_else(|| messages.as_ref().map(|m| derive_cache_key(artifact.id, m)))
        .map(|k| crate::Caller::current().scoped_key(&k));

    if streaming {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let s = Arc::clone(&session);
        let acc = access.clone();
        let request_id = access.as_ref().map(|a| a.id().to_string());
        spawn_blocking_in_span(move || {
            let cancel = CancelToken::new();
            let flip = cancel.clone();
            let send = |v: Value| tx.send(Ok(Event::default().data(v.to_string()))).is_ok();
            let t0 = Instant::now();
            if let Some(a) = &acc {
                a.call_start();
            }
            let mut first = true;
            let on_token = |tok: &str| {
                if std::mem::take(&mut first) {
                    if let Some(a) = &acc {
                        a.token();
                    }
                }
                if !send(json!({"token": tok})) {
                    flip.cancel();
                }
            };
            let result = match (&messages, &prompt) {
                (Some(m), _) => s
                    .chat_stream_with(
                        m,
                        tools.as_deref(),
                        cache_key.as_deref(),
                        None,
                        Some(max_tokens),
                        Some(temperature),
                        prio,
                        Some(&cancel),
                        on_token,
                    )
                    .map(|o| (o.text, Some(o.meta))),
                (None, Some(p)) => {
                    s.generate_stream_with(p, Some(max_tokens), Some(temperature), prio, Some(&cancel), on_token).map(|t| (t, None))
                }
                _ => unreachable!(),
            };
            // Let go of the access record before the stream ends (see chat_completions).
            match result {
                Ok((text, meta)) => {
                    if let Some(a) = acc {
                        a.call_end(meta.as_ref());
                        a.finish("stop");
                    }
                    let _ = send(
                        json!({"done": true, "text": text, "meta": meta, "model": artifact.id, "family": artifact.family, "backend": "mlx-python", "ms": t0.elapsed().as_millis()}),
                    );
                }
                Err(estia_engine::SessionError::Cancelled { .. }) => {
                    if let Some(a) = acc {
                        a.finish("cancelled");
                    }
                }
                Err(e) => {
                    if let Some(a) = acc {
                        a.failed(&e.to_string());
                    }
                    let _ = send(json!({"error": e.to_string(), "request_id": request_id}));
                }
            }
        });
        return Ok(Sse::new(UnboundedReceiverStream::new(rx)).keep_alive(KeepAlive::default()).into_response());
    }

    let s = Arc::clone(&session);
    let fmt = format.clone();
    let acc = access.clone();
    let body = spawn_blocking_in_span(move || -> Result<Value, ApiError> {
        let t0 = Instant::now();
        let run = |extra_turns: &[Message]| -> Result<(String, Option<estia_engine::proto::GenerationMeta>), ApiError> {
            if let Some(a) = &acc {
                a.call_start();
            }
            let out = match (&messages, &prompt) {
                (Some(m), _) => {
                    let mut all = m.clone();
                    all.extend_from_slice(extra_turns);
                    let o = s.chat_with(&all, tools.as_deref(), cache_key.as_deref(), None, Some(max_tokens), Some(temperature), prio)?;
                    (o.text, Some(o.meta))
                }
                (None, Some(p)) => {
                    let mut full = p.clone();
                    for t in extra_turns {
                        full.push_str("\n\n");
                        full.push_str(&t.content);
                    }
                    (s.generate_with(&full, Some(max_tokens), Some(temperature), prio)?, None)
                }
                _ => unreachable!(),
            };
            if let Some(a) = &acc {
                a.call_end(out.1.as_ref());
            }
            Ok(out)
        };
        let (mut text, mut meta) = run(&[])?;
        let mut structured: Option<Structured> = None;
        let mut attempt = 1;
        if fmt.wants_json() {
            loop {
                match structured::enforce(&text, &fmt) {
                    Ok(v) => {
                        structured = Some(v);
                        break;
                    }
                    Err(e) if attempt < attempts => {
                        attempt += 1;
                        tracing::debug!(attempt, "structured output did not validate; retrying");
                        let hint = Structured::retry_hint(&e);
                        let (t, m) = run(&[Message::new("assistant", text.clone()), Message::new("user", hint.trim().to_string())])?;
                        text = t;
                        meta = m;
                    }
                    Err(e) => return Err(ApiError::unprocessable(format!("structured output failed after {attempt} attempt(s): {e}"))),
                }
            }
        }
        let tool_calls: Vec<Value> =
            if tools.is_some() { toolcalls::parse(&text).iter().enumerate().map(|(i, c)| c.to_openai(i)).collect() } else { Vec::new() };
        if let Some(a) = &acc {
            a.finish(if tool_calls.is_empty() { "stop" } else { "tool_calls" });
        }
        Ok(json!({
            "text": text,
            "json": structured.as_ref().map(|s| s.value.clone()),
            "repaired": structured.as_ref().map(|s| s.repaired).unwrap_or(false),
            "repairs": structured.as_ref().map(|s| s.repairs.clone()).unwrap_or_default(),
            "attempts": attempt,
            "tool_calls": tool_calls,
            "meta": meta,
            "model": artifact.id, "family": artifact.family, "backend": "mlx-python",
            "ms": t0.elapsed().as_millis(),
        }))
    })
    .await??;
    Ok(Json(body).into_response())
}

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub inputs: Value,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub expect_fingerprint: Option<String>,
    #[serde(default)]
    pub priority: Option<String>,
}

pub async fn embed(State(state): State<Arc<AppState>>, Json(req): Json<EmbedRequest>) -> Result<Json<Value>, ApiError> {
    let model = state.resolve_embedding(req.model.as_deref())?;
    let inputs = inputs_of(&req.inputs)?;
    let access = Access::current();
    if let Some(a) = &access {
        a.embedding(req.model.as_deref().unwrap_or("embed"), model.id, inputs.len(), model.dims);
    }
    check_embed_inputs(inputs.len())?;
    let task = embed_task(&req.task)?;
    let fingerprint = model.fingerprint_for(state.embed_backend());
    if let Some(expected) = &req.expect_fingerprint {
        if expected != &fingerprint {
            return Err(ApiError::unprocessable(format!("fingerprint mismatch: host serves `{fingerprint}`, expected `{expected}`")));
        }
    }
    let prio = if req.priority.as_deref() == Some("background") { Priority::Background } else { Priority::Interactive };
    let prefixed = apply_prefix(model, task, &inputs);
    let st = Arc::clone(&state);
    let t0 = Instant::now();
    let (vectors, load_ms) = spawn_blocking_in_span(move || -> Result<(Vec<Vec<f32>>, Option<u64>), ApiError> {
        let (session, load_ms) = st.embed_session_timed(model)?;
        Ok((session.embed_batch_with(&prefixed, prio)?, load_ms))
    })
    .await??;
    if let (Some(a), Some(ms)) = (&access, load_ms) {
        a.loaded(ms);
    }
    Ok(Json(json!({
        "vectors": vectors, "fingerprint": fingerprint, "model": model.id, "dims": model.dims,
        "task": task_name(task), "backend": state.embed_backend(), "ms": t0.elapsed().as_millis()
    })))
}

#[derive(Debug, Deserialize)]
pub struct PairRequestBody {
    pub name: String,
    #[serde(default)]
    pub scopes: Option<Vec<String>>,
}

/// A client on the LAN asks for a token. Open route; the operator decides.
pub async fn pair_request(
    State(state): State<Arc<AppState>>,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<std::net::SocketAddr>,
    Json(body): Json<PairRequestBody>,
) -> Result<Response, ApiError> {
    // The store checks the name (it ends up in the operator's terminal) and
    // the scopes, and caps pending requests overall and per address.
    let scopes = body.scopes.unwrap_or_else(|| vec!["generate".into(), "embed".into(), "models:read".into()]);
    let from = peer.ip().to_string();
    // The store takes a file lock (the CLI edits the same file): off the runtime.
    let p = spawn_blocking_in_span(move || state.pairings.request(&body.name, &scopes, Some(from)))
        .await?
        .map_err(|e| pairing_error(e, ApiError::not_found))?;
    if let Some(a) = Access::current() {
        a.pairing(&p.id);
    }
    Ok((
        axum::http::StatusCode::ACCEPTED,
        Json(json!({
            "id": p.id, "status": p.status, "expires_in": crate::pairing::PAIRING_TTL_SECS,
            "how": "the operator approves with `estia pair approve <id>`; poll GET /engine/pair/<id> until approved"
        })),
    )
        .into_response())
}

/// The client polls; the token is returned exactly once when approved.
pub async fn pair_poll(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    if let Some(a) = Access::current() {
        a.pairing(&id);
    }
    let key = id.clone();
    let (status, token) =
        spawn_blocking_in_span(move || state.pairings.poll(&key)).await?.map_err(|e| pairing_error(e, ApiError::not_found))?;
    Ok(Json(json!({"id": id, "status": status, "token": token})))
}

/// Operator view of pairing requests (admin scope) — a paired admin client,
/// such as a host application, can approve other devices from wherever it is.
pub async fn list_pairings(State(state): State<Arc<AppState>>) -> Json<Value> {
    let list: Vec<Value> = state
        .pairings
        .list()
        .into_iter()
        .map(|p| {
            json!({
                "id": p.id, "name": p.name, "scopes": p.scopes, "status": p.status, "from": p.from,
                "created_unix": p.created_unix, "claimed": p.claimed, "revoked": p.revoked,
            })
        })
        .collect();
    Json(json!({"pairings": list}))
}

pub async fn approve_pairing(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    if let Some(a) = Access::current() {
        a.pairing(&id);
    }
    let p = spawn_blocking_in_span(move || state.pairings.approve(&id, &state.tokens))
        .await?
        .map_err(|e| pairing_error(e, ApiError::bad_request))?;
    Ok(Json(json!({"id": p.id, "name": p.name, "status": p.status, "scopes": p.scopes})))
}

/// Deny a pending request, or take back an approved one: its token is revoked
/// (`"revoked": true`) whether or not the device has collected it yet.
pub async fn deny_pairing(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    if let Some(a) = Access::current() {
        a.pairing(&id);
    }
    let p = spawn_blocking_in_span(move || state.pairings.deny_with(&id, &state.tokens))
        .await?
        .map_err(|e| pairing_error(e, ApiError::bad_request))?;
    Ok(Json(json!({"id": p.id, "name": p.name, "status": p.status, "revoked": p.revoked, "token_name": p.token_name})))
}

pub async fn stats(State(state): State<Arc<AppState>>) -> Json<Value> {
    let (interactive, background) = state.queue_depths();
    Json(json!({
        "uptime_s": state.started.elapsed().as_secs(),
        "loaded": state.loaded(),
        "queue": {"interactive": interactive, "background": background},
        "jobs": state.jobs.list().len(),
    }))
}

#[cfg(test)]
mod tests {
    use super::PullMilestones;
    use estia_engine::models::DownloadProgress;

    fn at(phase: &'static str, done: u64, total: Option<u64>) -> DownloadProgress {
        DownloadProgress { phase, file_name: None, file_index: 1, file_count: 2, bytes_downloaded: done, total_bytes: total }
    }

    /// A pull logs each tenth once, skips tenths it jumps over, and never
    /// logs 100 % (that is the "finished" line) or an unknown total.
    #[test]
    fn pull_progress_is_logged_once_per_tenth() {
        let mut m = PullMilestones::default();
        let seen: Vec<u64> = [0, 5, 10, 11, 19, 20, 55, 56, 90, 99, 100, 100]
            .iter()
            .filter_map(|pct| m.crossed(&at("downloading", *pct, Some(100))))
            .collect();
        assert_eq!(seen, [10, 20, 50, 90]);
        let mut resumed = PullMilestones::default();
        assert_eq!(resumed.crossed(&at("downloading", 42, Some(100))), Some(40), "a resumed pull starts where it is");
        assert_eq!(PullMilestones::default().crossed(&at("downloading", 50, None)), None);
        assert_eq!(PullMilestones::default().crossed(&at("finalizing", 50, Some(100))), None);
    }
}
