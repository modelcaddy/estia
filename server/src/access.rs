//! The access log and request ids.
//!
//! Every request gets an id: the client's own `X-Request-Id` when it is
//! short and plain (letters, digits, `.`, `_`, `:`, `-`; at most 64
//! characters), a fresh random one otherwise. The id goes back in the
//! `X-Request-Id` response header, inside every JSON error body as
//! `error.request_id`, and on every log event emitted while the request is
//! handled (as the `request_id` field of the `request` span).
//!
//! When the response is finished, one event with target
//! `estia_server::access` records the request: method, path (never the query
//! string), status, duration, peer, the calling token's name, and for model
//! work the model, token counts and timings. Handlers add to it through
//! [`Access`]. It never holds prompts, completions, embeddings, tokens or
//! request bodies.
//!
//! The line is written when the last holder lets go of the record: the
//! response body (so a streamed response is logged when the stream ends, not
//! when its headers go out) and, for streamed generations, the worker thread
//! (so a stream cut short by the client still gets its `cancelled`).
//!
//! Levels: `warn` for 5xx responses and generations that failed mid-stream;
//! `debug` for successful polls (health, stats, pairing and job polls), which
//! clients repeat every few seconds; `info` for everything else.

use axum::{
    body::{Body, Bytes},
    extract::ConnectInfo,
    http::{HeaderMap, HeaderValue, Request},
    middleware::Next,
    response::Response,
};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;
use tracing::Instrument as _;

/// Request and response header carrying the request id.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Longest client-supplied request id we keep.
pub const MAX_REQUEST_ID_LEN: usize = 64;

/// Longest path written to the log; longer ones are cut with `…`.
const MAX_LOGGED_PATH: usize = 200;

tokio::task_local! {
    static CURRENT: Access;
}

/// The access record of one request. Cheap to clone; the log line is written
/// when the last clone is dropped.
#[derive(Clone)]
pub struct Access(Arc<Record>);

struct Record {
    id: String,
    method: String,
    path: String,
    peer: Option<SocketAddr>,
    started: Instant,
    fields: Mutex<Fields>,
}

/// What handlers learn while serving a request. Every field is optional:
/// a line carries only what applies to its route.
#[derive(Debug, Default, Clone)]
struct Fields {
    status: Option<u16>,
    caller: Option<String>,
    /// Why the request was refused or failed. For 401/403 a short reason set
    /// by the guard; otherwise the error message, except for 422, whose
    /// message can quote model output.
    error: Option<String>,
    error_type: Option<&'static str>,
    /// The `model` the client asked for (a role, family or id).
    asked: Option<String>,
    /// The artifact or embedding model that served it.
    model: Option<String>,
    stream: Option<bool>,
    /// Starting the runner and loading the model, when this request did it.
    load_ms: Option<u64>,
    max_tokens: Option<u32>,
    prompt_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    attempts: Option<u32>,
    first_token_at: Option<Instant>,
    call_started: Option<Instant>,
    call_ended: Option<Instant>,
    finish: Option<&'static str>,
    inputs: Option<usize>,
    dims: Option<usize>,
    job_id: Option<String>,
    pairing_id: Option<String>,
}

impl Access {
    /// The record of the request this task is handling, if any.
    pub fn current() -> Option<Access> {
        CURRENT.try_with(Access::clone).ok()
    }

    pub fn id(&self) -> &str {
        &self.0.id
    }

    fn with<R>(&self, f: impl FnOnce(&mut Fields) -> R) -> R {
        f(&mut self.0.fields.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// The token that made the request (its name, never the token).
    pub fn caller(&self, name: &str) {
        self.with(|f| f.caller = Some(name.to_string()));
    }

    /// Why a guard refused the request (401, 403). Wins over the error message.
    pub fn refused(&self, reason: impl Into<String>) {
        self.with(|f| f.error = Some(reason.into()));
    }

    /// An error response is being sent. See [`Fields::error`] for what is kept.
    pub(crate) fn error(&self, status: u16, kind: &'static str, message: &str) {
        self.with(|f| {
            f.error_type = Some(kind);
            if f.error.is_none() && status != 422 {
                f.error = Some(message.to_string());
            }
        });
    }

    /// A generation request: what was asked, what serves it.
    pub fn generation(&self, asked: &str, model: &str, stream: bool, max_tokens: u32) {
        self.with(|f| {
            f.asked = Some(asked.to_string());
            f.model = Some(model.to_string());
            f.stream = Some(stream);
            f.max_tokens = Some(max_tokens);
        });
    }

    /// An embedding request.
    pub fn embedding(&self, asked: &str, model: &str, inputs: usize, dims: usize) {
        self.with(|f| {
            f.asked = Some(asked.to_string());
            f.model = Some(model.to_string());
            f.inputs = Some(inputs);
            f.dims = Some(dims);
        });
    }

    /// This request waited this long for the model to load: it started the
    /// runner and loaded the model, or joined a load already under way.
    pub fn loaded(&self, ms: u64) {
        self.with(|f| f.load_ms = Some(ms));
    }

    /// A runner call starts (one per attempt).
    pub fn call_start(&self) {
        self.with(|f| {
            f.call_started = Some(Instant::now());
            f.first_token_at = None;
            f.attempts = Some(f.attempts.unwrap_or(0) + 1);
        });
    }

    /// A token arrived from the runner. Only the first one counts.
    pub fn token(&self) {
        self.with(|f| {
            if f.first_token_at.is_none() {
                f.first_token_at = Some(Instant::now());
            }
        });
    }

    /// A runner call finished with this accounting.
    pub fn call_end(&self, meta: Option<&estia_engine::proto::GenerationMeta>) {
        self.with(|f| {
            f.call_ended = Some(Instant::now());
            if let Some(m) = meta {
                f.prompt_tokens = m.prompt_tokens;
                f.cached_tokens = m.cached_tokens;
                f.completion_tokens = m.generation_tokens;
            }
        });
    }

    /// How the generation ended: `stop`, `length`, `tool_calls`, `cancelled` or `error`.
    pub fn finish(&self, finish: &'static str) {
        self.with(|f| {
            if f.call_started.is_some() && f.call_ended.is_none() {
                f.call_ended = Some(Instant::now());
            }
            f.finish = Some(finish);
        });
    }

    /// A generation failed after the response had started (a stream error).
    pub fn failed(&self, message: &str) {
        self.with(|f| {
            f.finish = Some("error");
            f.error = Some(message.to_string());
        });
    }

    pub fn job(&self, id: &str) {
        self.with(|f| f.job_id = Some(id.to_string()));
    }

    pub fn pairing(&self, id: &str) {
        self.with(|f| f.pairing_id = Some(id.to_string()));
    }

    fn set_status(&self, status: u16) {
        self.with(|f| f.status = Some(status));
    }

    /// The line as it would be written now.
    #[cfg(test)]
    fn line(&self, now: Instant) -> Line {
        let r = &self.0;
        let f = self.with(|f| f.clone());
        Line::from_record(r, &f, now)
    }
}

impl Drop for Record {
    fn drop(&mut self) {
        let f = self.fields.get_mut().map(|f| f.clone()).unwrap_or_else(|e| e.into_inner().clone());
        Line::from_record(self, &f, Instant::now()).emit();
    }
}

/// One access-log line, with the derived numbers worked out.
#[derive(Debug, Clone, PartialEq)]
struct Line {
    request_id: String,
    method: String,
    path: String,
    status: Option<u16>,
    duration_ms: u64,
    peer: Option<SocketAddr>,
    caller: Option<String>,
    error_type: Option<&'static str>,
    error: Option<String>,
    asked: Option<String>,
    model: Option<String>,
    stream: Option<bool>,
    load_ms: Option<u64>,
    max_tokens: Option<u32>,
    prompt_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    time_to_first_token_ms: Option<u64>,
    tokens_per_s: Option<f64>,
    finish: Option<&'static str>,
    attempts: Option<u32>,
    inputs: Option<usize>,
    dims: Option<usize>,
    job_id: Option<String>,
    pairing_id: Option<String>,
    level: tracing::Level,
}

impl Line {
    fn from_record(r: &Record, f: &Fields, now: Instant) -> Line {
        let ms = |a: Instant, b: Instant| b.saturating_duration_since(a).as_millis() as u64;
        // Completion tokens over the whole runner call, prompt processing
        // included, streamed or not. (Not from the first streamed token: the
        // runner holds back the last few characters to filter control
        // markers, so text reaches us in bursts and a first-token-to-end rate
        // would be meaningless for short answers.)
        let tokens_per_s = match (f.completion_tokens, f.call_started, f.call_ended) {
            (Some(n), Some(a), Some(b)) if n > 0 => {
                let secs = b.saturating_duration_since(a).as_secs_f64();
                (secs >= 0.001).then(|| (n as f64 / secs * 10.0).round() / 10.0)
            }
            _ => None,
        };
        let failed = f.finish == Some("error") || f.status.is_some_and(|s| s >= 500);
        let level = if failed {
            tracing::Level::WARN
        } else if f.status.is_some_and(|s| s < 400) && is_poll(&r.method, &r.path) {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        };
        Line {
            request_id: r.id.clone(),
            method: r.method.clone(),
            path: r.path.clone(),
            status: f.status,
            duration_ms: ms(r.started, now),
            peer: r.peer,
            caller: f.caller.clone(),
            error_type: f.error_type,
            error: f.error.clone(),
            asked: f.asked.clone(),
            model: f.model.clone(),
            stream: f.stream,
            load_ms: f.load_ms,
            max_tokens: f.max_tokens,
            prompt_tokens: f.prompt_tokens,
            cached_tokens: f.cached_tokens,
            completion_tokens: f.completion_tokens,
            time_to_first_token_ms: if f.stream == Some(true) { f.first_token_at.map(|t| ms(r.started, t)) } else { None },
            tokens_per_s,
            finish: f.finish,
            attempts: f.attempts.filter(|n| *n > 1),
            inputs: f.inputs,
            dims: f.dims,
            job_id: f.job_id.clone(),
            pairing_id: f.pairing_id.clone(),
            level,
        }
    }

    fn emit(&self) {
        // Strings a client chose (`caller`, `asked`, `error`) are recorded as
        // `&str`, which the text format prints quoted and escaped; our own
        // (ids, model ids, `finish`) are printed bare.
        macro_rules! access_event {
            ($level:expr) => {
                tracing::event!(
                    target: "estia_server::access",
                    parent: None,
                    $level,
                    request_id = %self.request_id,
                    method = %self.method,
                    path = %self.path,
                    status = self.status,
                    duration_ms = self.duration_ms,
                    peer = self.peer.map(tracing::field::display),
                    caller = self.caller.as_deref(),
                    error_type = self.error_type.map(tracing::field::display),
                    error = self.error.as_deref(),
                    asked = self.asked.as_deref(),
                    model = self.model.as_deref().map(tracing::field::display),
                    stream = self.stream,
                    load_ms = self.load_ms,
                    max_tokens = self.max_tokens,
                    prompt_tokens = self.prompt_tokens,
                    cached_tokens = self.cached_tokens,
                    completion_tokens = self.completion_tokens,
                    time_to_first_token_ms = self.time_to_first_token_ms,
                    tokens_per_s = self.tokens_per_s,
                    finish = self.finish.map(tracing::field::display),
                    attempts = self.attempts,
                    inputs = self.inputs.map(|n| n as u64),
                    dims = self.dims.map(|n| n as u64),
                    job_id = self.job_id.as_deref().map(tracing::field::display),
                    pairing_id = self.pairing_id.as_deref().map(tracing::field::display),
                )
            };
        }
        match self.level {
            tracing::Level::WARN => access_event!(tracing::Level::WARN),
            tracing::Level::DEBUG => access_event!(tracing::Level::DEBUG),
            _ => access_event!(tracing::Level::INFO),
        }
    }
}

/// Routes clients poll every few seconds; logged at `debug` when they succeed.
fn is_poll(method: &str, path: &str) -> bool {
    method == "GET"
        && (matches!(path, "/engine/health" | "/engine/stats" | "/engine/pairings" | "/engine/jobs")
            || path.starts_with("/engine/pair/")
            || (path.starts_with("/engine/jobs/") && !path.ends_with("/events")))
}

/// The client's `X-Request-Id`, when it is safe to reuse: 1 to
/// [`MAX_REQUEST_ID_LEN`] characters of ASCII letters, digits, `.`, `_`, `:`
/// and `-`. Anything else is replaced, not cleaned, so an id in the log is
/// always one a client could have sent.
pub fn incoming_request_id(headers: &HeaderMap) -> Option<String> {
    let v = headers.get(REQUEST_ID_HEADER)?.to_str().ok()?.trim();
    let ok = !v.is_empty()
        && v.len() <= MAX_REQUEST_ID_LEN
        && v.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '-'));
    ok.then(|| v.to_string())
}

/// A fresh request id: 16 random hex digits.
pub fn new_request_id() -> String {
    let mut bytes = [0u8; 8];
    if getrandom::getrandom(&mut bytes).is_err() {
        // No OS randomness: fall back to the clock and a counter. Ids only
        // need to be unique enough to find a line.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        bytes = (t ^ n.rotate_left(48)).to_be_bytes();
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The path as logged: at most [`MAX_LOGGED_PATH`] bytes, and anything but
/// printable ASCII escaped, since it is printed bare in text logs.
fn logged_path(path: &str) -> String {
    let mut end = path.len().min(MAX_LOGGED_PATH);
    while !path.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = String::with_capacity(end + 1);
    for c in path[..end].chars() {
        if c.is_ascii_graphic() {
            out.push(c);
        } else {
            out.extend(c.escape_default());
        }
    }
    if end < path.len() {
        out.push('…');
    }
    out
}

/// Outermost middleware: assigns the request id, runs the request inside a
/// `request` span and the [`Access`] scope, sets `X-Request-Id` on the
/// response, and ties the access line to the end of the response body.
pub async fn middleware(mut req: Request<Body>, next: Next) -> Response {
    let id = incoming_request_id(req.headers()).unwrap_or_else(new_request_id);
    let peer = req.extensions().get::<ConnectInfo<SocketAddr>>().map(|c| c.0);
    let access = Access(Arc::new(Record {
        id,
        method: req.method().as_str().to_string(),
        path: logged_path(req.uri().path()),
        peer,
        started: Instant::now(),
        fields: Mutex::new(Fields::default()),
    }));
    req.extensions_mut().insert(access.clone());
    let span = tracing::info_span!("request", request_id = %access.id());
    let mut resp = CURRENT.scope(access.clone(), next.run(req).instrument(span)).await;
    access.set_status(resp.status().as_u16());
    if let Ok(v) = HeaderValue::from_str(access.id()) {
        resp.headers_mut().insert(REQUEST_ID_HEADER, v);
    }
    let (parts, body) = resp.into_parts();
    Response::from_parts(parts, Body::new(Tracked { inner: body, _access: access }))
}

/// A response body that holds the access record until it is finished or
/// dropped.
struct Tracked {
    inner: Body,
    _access: Access,
}

impl http_body::Body for Tracked {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<http_body::Frame<Bytes>, axum::Error>>> {
        Pin::new(&mut self.get_mut().inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// `spawn_blocking` that keeps the current span (and so the request id) on
/// the events the blocking work emits, engine events included.
pub fn spawn_blocking_in_span<F, R>(f: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let span = tracing::Span::current();
    tokio::task::spawn_blocking(move || span.in_scope(f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn record(method: &str, path: &str) -> Access {
        Access(Arc::new(Record {
            id: "abc".into(),
            method: method.into(),
            path: path.into(),
            peer: Some("127.0.0.1:5000".parse().unwrap()),
            started: Instant::now(),
            fields: Mutex::new(Fields::default()),
        }))
    }

    #[test]
    fn request_ids_from_clients_are_checked() {
        let mut h = HeaderMap::new();
        assert_eq!(incoming_request_id(&h), None);
        h.insert(REQUEST_ID_HEADER, HeaderValue::from_static("app-42:turn.3_x"));
        assert_eq!(incoming_request_id(&h).as_deref(), Some("app-42:turn.3_x"));
        for bad in ["", " ", "has space", "semi;colon", "quote\"", "slash/", "é"] {
            if let Ok(v) = HeaderValue::from_str(bad) {
                h.insert(REQUEST_ID_HEADER, v);
                assert_eq!(incoming_request_id(&h), None, "{bad:?}");
            }
        }
        h.insert(REQUEST_ID_HEADER, HeaderValue::from_str(&"a".repeat(MAX_REQUEST_ID_LEN)).unwrap());
        assert!(incoming_request_id(&h).is_some());
        h.insert(REQUEST_ID_HEADER, HeaderValue::from_str(&"a".repeat(MAX_REQUEST_ID_LEN + 1)).unwrap());
        assert_eq!(incoming_request_id(&h), None);
        let (a, b) = (new_request_id(), new_request_id());
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }

    #[test]
    fn stream_line_has_ttft_rate_and_counts() {
        let a = record("POST", "/v1/chat/completions");
        a.caller("editor");
        a.generation("fast", "gemma4-e2b-it-4bit-mlx", true, 256);
        a.call_start();
        let t0 = a.0.started;
        a.with(|f| {
            f.first_token_at = Some(t0 + Duration::from_millis(300));
        });
        a.call_end(Some(&estia_engine::proto::GenerationMeta {
            prompt_tokens: Some(20),
            cached_tokens: Some(8),
            generation_tokens: Some(50),
            template: None,
            ..Default::default()
        }));
        a.with(|f| {
            f.call_started = Some(t0 + Duration::from_millis(100));
            f.call_ended = Some(t0 + Duration::from_millis(1350));
        });
        a.finish("stop");
        a.set_status(200);
        let l = a.line(t0 + Duration::from_millis(1360));
        assert_eq!(l.time_to_first_token_ms, Some(300), "from the request's start");
        assert_eq!(l.tokens_per_s, Some(40.0), "50 tokens over the 1.25 s runner call");
        assert_eq!((l.prompt_tokens, l.cached_tokens, l.completion_tokens), (Some(20), Some(8), Some(50)));
        assert_eq!(l.duration_ms, 1360);
        assert_eq!(l.caller.as_deref(), Some("editor"));
        assert_eq!(l.finish, Some("stop"));
        assert_eq!(l.attempts, None, "one attempt is not worth a field");
        assert_eq!(l.level, tracing::Level::INFO);
    }

    #[test]
    fn non_stream_rate_spans_the_call_and_has_no_ttft() {
        let a = record("POST", "/engine/generate");
        a.generation("text", "m", false, 64);
        a.call_start();
        a.call_start(); // a structured-output retry
        let start = a.with(|f| f.call_started.unwrap());
        a.token();
        a.call_end(Some(&estia_engine::proto::GenerationMeta { generation_tokens: Some(10), ..Default::default() }));
        a.with(|f| f.call_ended = Some(start + Duration::from_millis(500)));
        let l = a.line(start + Duration::from_millis(600));
        assert_eq!(l.time_to_first_token_ms, None);
        assert_eq!(l.tokens_per_s, Some(20.0));
        assert_eq!(l.attempts, Some(2));
    }

    #[test]
    fn errors_levels_and_what_is_kept() {
        let a = record("POST", "/v1/chat/completions");
        a.error(422, "invalid_request_error", "structured output failed: raw head: {\"secret");
        a.set_status(422);
        let l = a.line(Instant::now());
        assert_eq!(l.error, None, "a 422 message can quote model output");
        assert_eq!(l.error_type, Some("invalid_request_error"));

        let a = record("POST", "/v1/embeddings");
        a.refused("missing bearer token");
        a.error(401, "authentication_error", "missing bearer token (Authorization: Bearer …)");
        a.set_status(401);
        let l = a.line(Instant::now());
        assert_eq!(l.error.as_deref(), Some("missing bearer token"), "the guard's reason wins");
        assert_eq!(l.level, tracing::Level::INFO);

        let a = record("POST", "/v1/chat/completions");
        a.error(500, "server_error", "runner: runner closed stdout (EOF)");
        a.set_status(500);
        assert_eq!(a.line(Instant::now()).level, tracing::Level::WARN);

        let a = record("POST", "/v1/chat/completions");
        a.set_status(200);
        a.failed("runner exited mid-stream");
        assert_eq!(a.line(Instant::now()).level, tracing::Level::WARN);

        for (m, p, poll) in [
            ("GET", "/engine/health", true),
            ("GET", "/engine/pair/0011", true),
            ("GET", "/engine/jobs/job_1_2", true),
            ("GET", "/engine/jobs/job_1_2/events", false),
            ("POST", "/engine/pair", false),
            ("GET", "/v1/models", false),
        ] {
            let a = record(m, p);
            a.set_status(200);
            let want = if poll { tracing::Level::DEBUG } else { tracing::Level::INFO };
            assert_eq!(a.line(Instant::now()).level, want, "{m} {p}");
            a.set_status(404);
            assert_eq!(a.line(Instant::now()).level, tracing::Level::INFO, "a failed poll is info: {m} {p}");
        }
    }

    #[test]
    fn long_paths_are_cut_and_odd_characters_escaped() {
        assert_eq!(logged_path("/v1/models"), "/v1/models");
        let long = format!("/engine/models/{}", "a".repeat(300));
        let cut = logged_path(&long);
        assert!(cut.ends_with('…') && cut.len() == MAX_LOGGED_PATH + '…'.len_utf8(), "{cut}");
        assert_eq!(logged_path("/engine/models/\u{9b}31m\u{1b}é"), "/engine/models/\\u{9b}31m\\u{1b}\\u{e9}");
        assert!(logged_path(&format!("/{}", "é".repeat(300))).ends_with('…'), "cut on a character boundary");
    }
}

/// Request ids and the access line over real HTTP, with the log captured.
#[cfg(test)]
mod http_tests {
    use crate::tokens::{TokenStore, SCOPE_EMBED};
    use crate::{router, serve_router, AppState, ConnLimits};
    use estia_engine::models::ModelStore;
    use estia_engine::runtime::PythonRuntime;
    use estia_engine::{Engine, EngineConfig};
    use serde_json::Value;
    use std::sync::{Arc, Mutex, OnceLock};

    static LOG: Mutex<Vec<u8>> = Mutex::new(Vec::new());

    struct Capture;
    impl std::io::Write for Capture {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            LOG.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Every event from every test in this binary, as JSON lines.
    fn capture() {
        static INIT: OnceLock<()> = OnceLock::new();
        INIT.get_or_init(|| {
            use tracing_subscriber::layer::SubscriberExt as _;
            use tracing_subscriber::Layer as _;
            let layer = tracing_subscriber::fmt::layer()
                .json()
                .flatten_event(true)
                .with_writer(|| Capture)
                .with_filter(tracing_subscriber::filter::LevelFilter::DEBUG);
            tracing::subscriber::set_global_default(tracing_subscriber::registry().with(layer)).expect("one subscriber");
        });
    }

    fn logged() -> String {
        String::from_utf8_lossy(&LOG.lock().unwrap()).into_owned()
    }

    /// The access line for `id`, waiting for it: it is written when the
    /// response body is dropped, which can be after the client has read it.
    async fn access_line(id: &str) -> Value {
        for _ in 0..200 {
            let found = logged()
                .lines()
                .filter_map(|l| serde_json::from_str::<Value>(l).ok())
                .find(|v| v["target"] == "estia_server::access" && v["request_id"] == id);
            if let Some(v) = found {
                return v;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("no access line for {id}:\n{}", logged());
    }

    struct Server {
        base: String,
        dir: std::path::PathBuf,
        embed_only: String,
    }

    async fn start() -> Server {
        capture();
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "estia-access-test-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), dir.join("no-runner.py"));
        let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
        let embed_only = tokens.mint("embedder", &[SCOPE_EMBED]).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let state = Arc::new(AppState::new(Arc::new(Engine::new(cfg)), tokens, true, addr));
        let app = router(state);
        tokio::spawn(async move { serve_router(listener, app, ConnLimits::default(), std::future::pending()).await.unwrap() });
        Server { base: format!("http://{addr}"), dir, embed_only }
    }

    async fn call(s: &Server, method: reqwest::Method, path: &str, token: Option<&str>, request_id: Option<&str>) -> (u16, String, Value) {
        let mut req = reqwest::Client::new().request(method.clone(), format!("{}{path}", s.base));
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        if let Some(id) = request_id {
            req = req.header("X-Request-Id", id);
        }
        if method == reqwest::Method::POST {
            req = req.json(&serde_json::json!({"model": "embed", "input": ["x"]}));
        }
        let r = req.send().await.unwrap();
        let status = r.status().as_u16();
        let id = r.headers().get("x-request-id").expect("every response carries X-Request-Id").to_str().unwrap().to_string();
        (status, id, r.json().await.unwrap_or(Value::Null))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn request_ids_reach_headers_bodies_and_the_log() {
        let s = start().await;

        // A fresh id, in the header and in the error body.
        let (status, id, body) = call(&s, reqwest::Method::GET, "/nope?secret=1", None, None).await;
        assert_eq!(status, 404);
        assert_eq!(id.len(), 16, "{id}");
        assert_eq!(body["error"]["request_id"], id.as_str(), "{body}");
        let line = access_line(&id).await;
        assert_eq!(line["status"], 404);
        assert_eq!(line["path"], "/nope", "no query string: {line}");
        assert_eq!(line["method"], "GET");
        assert!(line["duration_ms"].is_u64() && line["peer"].as_str().unwrap().starts_with("127.0.0.1:"), "{line}");
        assert_eq!(line["level"], "INFO");

        // The client's own id is kept when it is plain, replaced when not.
        let (_, id, body) = call(&s, reqwest::Method::GET, "/nope", None, Some("app-7:turn.2")).await;
        assert_eq!(id, "app-7:turn.2");
        assert_eq!(body["error"]["request_id"], "app-7:turn.2");
        let (_, id, _) = call(&s, reqwest::Method::GET, "/nope", None, Some("two words")).await;
        assert_ne!(id, "two words");
        assert_eq!(id.len(), 16);

        // Successful polls: header, no body id, logged at debug.
        let (status, id, body) = call(&s, reqwest::Method::GET, "/engine/health", None, None).await;
        assert_eq!(status, 200);
        assert!(body.get("error").is_none());
        assert_eq!(access_line(&id).await["level"], "DEBUG");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn refusals_are_logged_with_a_reason_and_never_the_token() {
        let s = start().await;

        let (status, id, body) = call(&s, reqwest::Method::POST, "/v1/embeddings", None, None).await;
        assert_eq!(status, 401);
        assert_eq!(body["error"]["request_id"], id.as_str());
        let line = access_line(&id).await;
        assert_eq!(line["error"], "missing bearer token", "{line}");
        assert_eq!(line["error_type"], "authentication_error");

        let unknown = "estia_00000000000000000000000000000000000000000000cafe";
        let (status, id, _) = call(&s, reqwest::Method::POST, "/v1/embeddings", Some(unknown), None).await;
        assert_eq!(status, 401);
        assert_eq!(access_line(&id).await["error"], "unknown token");

        // Scope: the caller is named, the token is not.
        let (status, id, body) = call(&s, reqwest::Method::GET, "/v1/models", Some(&s.embed_only), None).await;
        assert_eq!(status, 403, "{body}");
        let line = access_line(&id).await;
        assert_eq!(line["caller"], "embedder");
        assert_eq!(line["error"], "token lacks the `models:read` scope");

        // Minted and used, then revoked from another process while the server runs.
        let cli = TokenStore::open(s.dir.join("tokens.json")).unwrap();
        let doomed = cli.mint("old-phone", &[SCOPE_EMBED]).unwrap();
        let (status, id, _) = call(&s, reqwest::Method::GET, "/v1/models", Some(&doomed), None).await;
        assert_eq!(status, 403, "known to the server now");
        assert_eq!(access_line(&id).await["caller"], "old-phone");
        assert!(cli.revoke("old-phone").unwrap());
        let (status, id, body) = call(&s, reqwest::Method::POST, "/v1/embeddings", Some(&doomed), None).await;
        assert_eq!(status, 401);
        assert_eq!(body["error"]["message"], "unknown token", "the client is not told more");
        let line = access_line(&id).await;
        assert_eq!(line["error"], "revoked token", "{line}");
        assert_eq!(line["caller"], "old-phone");

        let all = logged();
        for secret in [unknown, s.embed_only.as_str(), doomed.as_str()] {
            assert!(!all.contains(secret), "a token reached the log");
        }
        let _ = std::fs::remove_dir_all(&s.dir);
    }
}
