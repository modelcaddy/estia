//! estia-server — the HTTP front for the Estia engine.
//!
//! Two surfaces on one listener:
//! - `/v1/*` in OpenAI's exact shapes (`chat/completions`, `embeddings`,
//!   `models`) so SDKs, coding agents and scripts work unchanged. `model` may
//!   be a role, a family or an artifact id.
//! - `/engine/*` for what OpenAI's shape cannot say: health, role defaults,
//!   model pulls with progress, generate with a JSON Schema, embed with an
//!   expected fingerprint, stats, jobs.
//!
//! Auth is a bearer token from `tokens.json` with scopes; `/engine/health` is
//! open. A non-loopback bind needs [`ServeOptions::lan`] (the CLI's `--lan`)
//! and auth on; LAN clients pair for a token. One engine per data directory:
//! `engine.json` in the data directory records where it bound, and a second
//! `serve` refuses when that port answers.
//!
//! In front of auth sits a DNS-rebinding guard ([`HostPolicy`]): a request is
//! served only when its `Host` is an IP literal, `localhost`, a `.local` name,
//! this machine's hostname or a name the operator allowed, and a state-changing
//! request carrying an `Origin` must come from the same origin it is sent to.
//!
//! Outermost of all is the access log ([`access`]): every request gets an id
//! (`X-Request-Id`) and one log line when its response ends. The crate only
//! emits `tracing` events; the binary installs the subscriber.

pub mod access;
pub mod catalog;
pub mod engine_api;
pub mod jobs;
pub mod openai;
pub mod pairing;
pub mod tokens;
pub mod toolcalls;

use axum::{
    body::Body,
    extract::State,
    http::{header, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use estia_engine::models::embed::EmbedModel;
use estia_engine::models::{Artifact, ResolveError};
use estia_engine::{Backend, EmbedSession, Engine, EngineError, GenSession};
use jobs::JobTable;
use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use tokens::TokenStore;

/// Version of the `/engine/*` contract. Clients check it in `/engine/health`.
pub const API_VERSION: u32 = 1;

/// The commit this crate was built from: 9 hex digits, with `+dirty` when
/// compiled-in files differed from it, or `unknown` (a build without git or
/// package metadata). Set by `build.rs`; `/engine/health` reports it as
/// `build.commit`. See docs/versioning.md.
pub const BUILD_COMMIT: &str = env!("ESTIA_BUILD_COMMIT");

/// The UTC day this crate was built, `YYYY-MM-DD` (the day of
/// `SOURCE_DATE_EPOCH` when that was set). `/engine/health` `build.date`.
pub const BUILD_DATE: &str = env!("ESTIA_BUILD_DATE");

/// Environment variable with extra `Host` names to answer to, comma-separated.
/// Same syntax as [`ServeOptions::allowed_hosts`].
pub const ALLOWED_HOSTS_ENV: &str = "ESTIA_ALLOWED_HOSTS";

pub struct AppState {
    pub engine: Arc<Engine>,
    pub tokens: TokenStore,
    pub pairings: pairing::PairingStore,
    pub jobs: JobTable,
    pub started: Instant,
    pub require_auth: bool,
    pub bind: SocketAddr,
    /// Where `tokens.json`, `pairings.json` and `config.json` live.
    pub data_dir: std::path::PathBuf,
    gen: Mutex<HashMap<String, Arc<GenSession>>>,
    embed: Mutex<HashMap<String, Arc<EmbedSession>>>,
    hosts: RwLock<HostPolicy>,
}

impl AppState {
    /// Answers to IP literals, `localhost`, `.local` names, this machine's
    /// hostname and whatever `ESTIA_ALLOWED_HOSTS` lists; add more with
    /// [`AppState::allow_hosts`].
    pub fn new(engine: Arc<Engine>, tokens: TokenStore, require_auth: bool, bind: SocketAddr) -> Self {
        let data_dir = tokens.path().parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let mut hosts = HostPolicy::default();
        hosts.allow(machine_hostnames());
        if let Ok(v) = std::env::var(ALLOWED_HOSTS_ENV) {
            hosts.allow([v]);
        }
        Self {
            engine,
            tokens,
            pairings: pairing::PairingStore::new(&data_dir),
            data_dir,
            jobs: JobTable::default(),
            started: Instant::now(),
            require_auth,
            bind,
            gen: Mutex::new(HashMap::new()),
            embed: Mutex::new(HashMap::new()),
            hosts: RwLock::new(hosts),
        }
    }

    /// Also answer to these `Host` names (see [`ServeOptions::allowed_hosts`]).
    pub fn allow_hosts<I, S>(&self, names: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.hosts.write().unwrap_or_else(|e| e.into_inner()).allow(names);
    }

    /// The `Host` names this server currently answers to, beyond the built-in rules.
    pub fn host_policy(&self) -> HostPolicy {
        self.hosts.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The backend this server's engine runs.
    pub fn backend(&self) -> Backend {
        self.engine.backend()
    }

    /// The backend id outputs and vectors from this server carry
    /// (`x_estia.backend`, the suffix of every embedding fingerprint).
    pub fn embed_backend(&self) -> &'static str {
        self.engine.backend().id()
    }

    /// Role name, family or artifact id → the artifact this host's backend
    /// loads (`fast` is the MLX artifact on an MLX engine and the GGUF one on
    /// a llama.cpp engine). Imported models resolve by id and family too.
    pub fn resolve_generation(&self, name: &str) -> Result<&'static Artifact, ApiError> {
        self.engine.resolve_generation(name).map_err(resolve_error)
    }

    /// Model id, artifact id, fingerprint or the `embed` role (`None` too) →
    /// the embedding model. Refused when this backend has no artifact for it.
    pub fn resolve_embedding(&self, name: Option<&str>) -> Result<&'static EmbedModel, ApiError> {
        self.engine.resolve_embedding(name).map(|(m, _)| m).map_err(resolve_error)
    }

    /// The fingerprint vectors of `model` carry on this host:
    /// `<artifact id>@<backend>`.
    pub fn embed_fingerprint(&self, model: &EmbedModel) -> Result<String, ApiError> {
        let backend = self.backend();
        self.engine
            .embed_fingerprint(model)
            .ok_or_else(|| resolve_error(ResolveError::NoArtifactForBackend { family: model.id.to_string(), backend }))
    }

    /// Get or spawn the resident generation session for an artifact. Blocking
    /// (spawns a process on first use) — call from `spawn_blocking`.
    pub fn gen_session(&self, artifact: &Artifact) -> Result<Arc<GenSession>, ApiError> {
        self.gen_session_timed(artifact).map(|(s, _)| s)
    }

    /// [`AppState::gen_session`], plus how long starting it took (runner,
    /// handshake, model load) when this call started it.
    pub(crate) fn gen_session_timed(&self, artifact: &Artifact) -> Result<(Arc<GenSession>, Option<u64>), ApiError> {
        if let Some(s) = self.gen.lock().unwrap().get(artifact.id) {
            return Ok((Arc::clone(s), None));
        }
        let t0 = Instant::now();
        let s = self.engine.spawn_gen_session(artifact.id)?;
        // Load now rather than inside the first generation, so the log can say
        // how long the load took. The first request waits the same either way.
        s.load()?;
        let ms = t0.elapsed().as_millis() as u64;
        tracing::info!(model = %artifact.id, family = %artifact.family, ready_ms = ms, "generation model ready");
        let s = Arc::new(s);
        self.gen.lock().unwrap().insert(artifact.id.to_string(), Arc::clone(&s));
        Ok((s, Some(ms)))
    }

    pub fn embed_session(&self, model: &EmbedModel) -> Result<Arc<EmbedSession>, ApiError> {
        self.embed_session_timed(model).map(|(s, _)| s)
    }

    /// [`AppState::embed_session`], plus how long starting it took. Sessions
    /// are keyed by the artifact this backend loads, so `loaded` names the
    /// weights actually resident.
    pub(crate) fn embed_session_timed(&self, model: &EmbedModel) -> Result<(Arc<EmbedSession>, Option<u64>), ApiError> {
        let backend = self.backend();
        let artifact = model
            .artifact_for(backend)
            .ok_or_else(|| resolve_error(ResolveError::NoArtifactForBackend { family: model.id.to_string(), backend }))?;
        if let Some(s) = self.embed.lock().unwrap().get(artifact.id) {
            return Ok((Arc::clone(s), None));
        }
        let t0 = Instant::now();
        let s = self.engine.spawn_embed_model(model)?;
        s.load()?;
        let ms = t0.elapsed().as_millis() as u64;
        tracing::info!(model = %model.id, artifact = %artifact.id, dims = model.dims, ready_ms = ms, "embedding model ready");
        let s = Arc::new(s);
        self.embed.lock().unwrap().insert(artifact.id.to_string(), Arc::clone(&s));
        Ok((s, Some(ms)))
    }

    /// Artifact ids with a live session.
    pub fn loaded(&self) -> Vec<String> {
        let mut v: Vec<String> = self.gen.lock().unwrap().keys().cloned().collect();
        v.extend(self.embed.lock().unwrap().keys().cloned());
        v.sort();
        v
    }

    /// Drop resident sessions idle for `idle_after`, freeing their model
    /// memory in the runner; the next request respawns. Returns what was dropped.
    pub fn reap_idle(&self, idle_after: std::time::Duration) -> Vec<String> {
        let mut dropped = Vec::new();
        {
            let mut g = self.gen.lock().unwrap();
            let idle: Vec<String> = g.iter().filter(|(_, s)| s.is_idle(idle_after)).map(|(k, _)| k.clone()).collect();
            for k in idle {
                g.remove(&k);
                dropped.push(k);
            }
        }
        {
            let mut e = self.embed.lock().unwrap();
            let idle: Vec<String> = e.iter().filter(|(_, s)| s.is_idle(idle_after)).map(|(k, _)| k.clone()).collect();
            for k in idle {
                e.remove(&k);
                dropped.push(k);
            }
        }
        dropped
    }

    /// Queue depth per priority across live generation sessions.
    pub fn queue_depths(&self) -> (usize, usize) {
        let mut interactive = 0;
        let mut background = 0;
        for s in self.gen.lock().unwrap().values() {
            interactive += s.waiting(estia_engine::Priority::Interactive);
            background += s.waiting(estia_engine::Priority::Background);
        }
        (interactive, background)
    }
}

/// OpenAI-shaped error body with a status.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(m: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, kind: "invalid_request_error", message: m.into() }
    }
    pub fn unauthorized(m: impl Into<String>) -> Self {
        Self { status: StatusCode::UNAUTHORIZED, kind: "authentication_error", message: m.into() }
    }
    pub fn forbidden(m: impl Into<String>) -> Self {
        Self { status: StatusCode::FORBIDDEN, kind: "permission_error", message: m.into() }
    }
    pub fn not_found(m: impl Into<String>) -> Self {
        Self { status: StatusCode::NOT_FOUND, kind: "not_found_error", message: m.into() }
    }
    pub fn unprocessable(m: impl Into<String>) -> Self {
        Self { status: StatusCode::UNPROCESSABLE_ENTITY, kind: "invalid_request_error", message: m.into() }
    }
    pub fn internal(m: impl Into<String>) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, kind: "server_error", message: m.into() }
    }
}

impl IntoResponse for ApiError {
    /// Inside a request (always, for the router's handlers) the body also
    /// carries `request_id`, and the access log notes the error.
    fn into_response(self) -> Response {
        let mut err = serde_json::json!({"message": self.message, "type": self.kind, "code": self.status.as_u16()});
        if let Some(a) = access::Access::current() {
            a.error(self.status.as_u16(), self.kind, &self.message);
            err["request_id"] = serde_json::Value::String(a.id().to_string());
        }
        (self.status, Json(serde_json::json!({ "error": err }))).into_response()
    }
}

impl From<estia_engine::SessionError> for ApiError {
    fn from(e: estia_engine::SessionError) -> Self {
        ApiError::internal(format!("runner: {e}"))
    }
}
impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        match e {
            EngineError::ModelMissing { .. } => ApiError::not_found(e.to_string()),
            EngineError::Resolve(r) => resolve_error(r),
            EngineError::FingerprintMismatch { .. } => ApiError::unprocessable(e.to_string()),
            other => ApiError::internal(other.to_string()),
        }
    }
}

/// A name that does not resolve on this backend: unknown is 404, an artifact
/// of the other format is 400 (ask by family or role), a family without an
/// artifact for this backend is 404.
pub fn resolve_error(e: ResolveError) -> ApiError {
    match e {
        ResolveError::Unknown(name) => ApiError::not_found(format!("unknown model `{name}`")),
        e @ ResolveError::WrongFormat { .. } => ApiError::bad_request(e.to_string()),
        e @ ResolveError::NoArtifactForBackend { .. } => ApiError::not_found(e.to_string()),
    }
}
impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError::internal(e.to_string())
    }
}
impl From<tokio::task::JoinError> for ApiError {
    fn from(e: tokio::task::JoinError) -> Self {
        ApiError::internal(format!("worker task failed: {e}"))
    }
}

/// Which scope a request needs, by method and path. `None` = open.
pub fn required_scope(method: &Method, path: &str) -> Option<&'static str> {
    match (method, path) {
        (_, "/engine/health") => None,
        // The bundled test client is a static page; everything it does goes
        // through the routes below with the token the user gives it.
        (_, "/") | (_, "/client") | (_, "/client/") => None,
        (_, "/engine/runtime/install") => Some(tokens::SCOPE_ADMIN),
        // Pairing is how a client *gets* a token; it cannot require one.
        (_, p) if p == "/engine/pair" || p.starts_with("/engine/pair/") => None,
        // Deciding pairings is the operator's job: a paired admin client may do it remotely.
        (_, p) if p.starts_with("/engine/pairings") => Some(tokens::SCOPE_ADMIN),
        (_, "/v1/chat/completions") | (_, "/engine/generate") => Some(tokens::SCOPE_GENERATE),
        (_, "/v1/embeddings") | (_, "/engine/embed") => Some(tokens::SCOPE_EMBED),
        (m, "/engine/defaults") if *m == Method::PUT => Some(tokens::SCOPE_ADMIN),
        (m, p) if p.starts_with("/engine/models") && (*m == Method::POST || *m == Method::DELETE) => Some(tokens::SCOPE_MODELS_WRITE),
        _ => Some(tokens::SCOPE_MODELS_READ),
    }
}

/// Runs on matched routes only (a `route_layer`), so an unknown path is a 404
/// from the fallback rather than a 401. Attaches the [`Caller`] every handler
/// sees.
async fn auth(State(state): State<Arc<AppState>>, mut req: Request<Body>, next: Next) -> Response {
    let access = access::Access::current();
    let refuse = |reason: String, e: ApiError| {
        if let Some(a) = &access {
            a.refused(reason);
        }
        e.into_response()
    };
    let caller = match required_scope(req.method(), req.uri().path()) {
        None => Caller::anonymous(),
        Some(_) if !state.require_auth => Caller::anonymous(),
        Some(scope) => {
            let bearer = req.headers().get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).unwrap_or("");
            let token = bearer.strip_prefix("Bearer ").unwrap_or("").trim();
            if token.is_empty() {
                let why = if bearer.is_empty() { "missing bearer token" } else { "Authorization is not `Bearer <token>`" };
                return refuse(why.into(), ApiError::unauthorized("missing bearer token (Authorization: Bearer …)"));
            }
            match state.tokens.verify(token) {
                Some(record) if record.allows(scope) => Caller::token(&record),
                Some(record) => {
                    if let Some(a) = &access {
                        a.caller(&record.name);
                    }
                    return refuse(
                        format!("token lacks the `{scope}` scope"),
                        ApiError::forbidden(format!("token lacks the `{scope}` scope")),
                    );
                }
                None => {
                    // The client is told only "unknown token"; the log says
                    // whether it is one that was revoked while this server ran.
                    let why = match state.tokens.revoked_name(token) {
                        Some(name) => {
                            if let Some(a) = &access {
                                a.caller(&name);
                            }
                            "revoked token".to_string()
                        }
                        None => "unknown token".to_string(),
                    };
                    return refuse(why, ApiError::unauthorized("unknown token"));
                }
            }
        }
    };
    if let (Some(a), Some(name)) = (&access, &caller.name) {
        a.caller(name);
    }
    req.extensions_mut().insert(caller.clone());
    CURRENT_CALLER.scope(caller, next.run(req)).await
}

/// Who a request is from, as far as state shared between clients is concerned.
///
/// The auth middleware attaches one to every routed request, both as a request
/// extension and as [`Caller::current`] for the handler's duration, so the
/// prompt cache can be partitioned per token. That matters because the cache
/// is an oracle: responses report how many prompt tokens came from it (and
/// prefill time says the same), so a token that could reach another token's
/// cache entry could test guesses against that conversation's prefix. With
/// [`Caller::scoped_key`] two tokens never share an entry, even when they send
/// the same `user` / `cache_key` or the same opening messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Caller {
    /// The token's name, for display. `None` when unauthenticated.
    pub name: Option<String>,
    /// Cache namespace: the token's hash rather than its name, so a revoked
    /// name minted again for another device does not inherit the old caches.
    namespace: String,
}

tokio::task_local! {
    static CURRENT_CALLER: Caller;
}

impl Caller {
    /// A request authenticated with this token.
    pub fn token(record: &tokens::TokenRecord) -> Self {
        Caller { name: Some(record.name.clone()), namespace: format!("token:{}", record.sha256) }
    }

    /// No identity: auth is off (`--no-auth`, loopback only) or the route is
    /// open. All such requests share one namespace.
    pub fn anonymous() -> Self {
        Caller { name: None, namespace: "anonymous".to_string() }
    }

    /// The caller of the request this task is handling — the same value as the
    /// request's `Caller` extension. [`Caller::anonymous`] outside a request.
    pub fn current() -> Self {
        CURRENT_CALLER.try_with(Caller::clone).unwrap_or_else(|_| Caller::anonymous())
    }

    /// The prompt-cache key to hand the runner for this caller, from the
    /// client's key (`user`, `cache_key`) or [`derive_cache_key`]. An empty
    /// key stays empty: the runner reads that as "no cache", as before.
    pub fn scoped_key(&self, raw: &str) -> String {
        use sha2::{Digest, Sha256};
        if raw.is_empty() {
            return String::new();
        }
        let mut h = Sha256::new();
        h.update(self.namespace.as_bytes());
        h.update([0]);
        h.update(raw.as_bytes());
        h.finalize().iter().take(16).map(|b| format!("{b:02x}")).collect()
    }
}

/// Which `Host` names the engine answers to: the DNS-rebinding guard.
///
/// A web page can point a DNS name it controls at 127.0.0.1 (or a LAN
/// address) and then talk to the engine as a same-origin page, reading every
/// response. What it cannot do is make the browser send a `Host` other than
/// its own name, so the engine refuses names it does not know. Always allowed:
/// IP literals (v4 and v6, any port), `localhost` and `*.localhost`, `*.local`
/// (mDNS, which a remote site cannot answer), and this machine's hostname,
/// short and fully qualified. The port is never checked: the attacker picks
/// the name, not the port, and port-forwards would break.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostPolicy {
    /// `*` was given: the check is off (a reverse proxy that validates Host).
    any: bool,
    exact: Vec<String>,
    /// From `*.example.com`: matches `a.example.com`, not `example.com`.
    suffixes: Vec<String>,
}

impl HostPolicy {
    /// Add names. Each item may itself be a comma-separated list; a port is
    /// ignored; `*.example.com` allows every subdomain; `*` allows any name.
    pub fn allow<I, S>(&mut self, names: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for item in names {
            for raw in item.as_ref().split(',') {
                let raw = raw.trim();
                if raw == "*" {
                    self.any = true;
                    continue;
                }
                let (wild, rest) = match raw.strip_prefix("*.").or_else(|| raw.strip_prefix('.')) {
                    Some(r) => (true, r),
                    None => (false, raw),
                };
                let Some(HostName::Name(name)) = parse_host(rest) else { continue };
                let list = if wild { &mut self.suffixes } else { &mut self.exact };
                let entry = if wild { format!(".{name}") } else { name };
                if !list.contains(&entry) {
                    list.push(entry);
                }
            }
        }
    }

    /// Is a request with this `Host` header value for us?
    pub fn allows(&self, host: &str) -> bool {
        match parse_host(host) {
            None => false,
            Some(HostName::Ip) => true,
            Some(HostName::Name(n)) => {
                self.any
                    || n == "localhost"
                    || n.ends_with(".localhost")
                    || n.ends_with(".local")
                    || self.exact.contains(&n)
                    || self.suffixes.iter().any(|s| n.ends_with(s.as_str()))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum HostName {
    Ip,
    /// Lowercased, without port or trailing dot.
    Name(String),
}

/// Split `host[:port]` / `[v6][:port]` into the host (lowercased, trailing dot
/// dropped, brackets kept for v6) and the port. `None` when malformed.
fn split_host_port(s: &str) -> Option<(String, Option<u16>)> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (host, port) = if s.starts_with('[') {
        let end = s.find(']')?;
        let (h, rest) = s.split_at(end + 1);
        match rest {
            "" => (h, None),
            r => (h, Some(r.strip_prefix(':')?)),
        }
    } else if s.parse::<std::net::IpAddr>().is_ok() {
        // A bare IPv6 literal (not valid in a Host header, but unambiguous).
        (s, None)
    } else {
        match s.rsplit_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (s, None),
        }
    };
    let port = match port {
        None => None,
        Some(p) => Some(p.parse::<u16>().ok()?),
    };
    let host = host.strip_suffix('.').unwrap_or(host).to_ascii_lowercase();
    if host.is_empty() || (host.contains(':') && !host.starts_with('[') && host.parse::<std::net::IpAddr>().is_err()) {
        return None;
    }
    Some((host, port))
}

fn parse_host(s: &str) -> Option<HostName> {
    let (host, _) = split_host_port(s)?;
    let bare = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')).unwrap_or(&host);
    if bare.parse::<std::net::IpAddr>().is_ok() {
        return Some(HostName::Ip);
    }
    if host.starts_with('[') {
        return None;
    }
    Some(HostName::Name(host))
}

/// Does `origin` (an `Origin` header) name the same scheme-host-port the
/// request was sent to (`host`, its `Host` header)? `null` never does.
fn same_origin(origin: &str, host: &str) -> bool {
    let Some((scheme, rest)) = origin.trim().split_once("://") else { return false };
    let default_port = match scheme.to_ascii_lowercase().as_str() {
        "http" => 80,
        "https" => 443,
        _ => return false,
    };
    let authority = rest.split('/').next().unwrap_or("");
    match (split_host_port(authority), split_host_port(host)) {
        (Some((oh, op)), Some((hh, hp))) => oh == hh && op.unwrap_or(default_port) == hp.unwrap_or(default_port),
        _ => false,
    }
}

/// This machine's hostname, short and fully qualified (`hostname -s`,
/// `hostname`; the kernel's name on a Linux without the command), computed
/// once per process.
fn machine_hostnames() -> Vec<String> {
    static NAMES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    NAMES
        .get_or_init(|| {
            let run = |args: &[&str]| {
                std::process::Command::new("hostname")
                    .args(args)
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            };
            let mut v: Vec<String> = [run(&["-s"]), run(&[])].into_iter().flatten().collect();
            if v.is_empty() {
                if let Ok(s) = std::fs::read_to_string("/proc/sys/kernel/hostname") {
                    v.extend(Some(s.trim().to_string()).filter(|s| !s.is_empty()));
                }
            }
            v.dedup();
            v
        })
        .clone()
}

/// The DNS-rebinding and cross-origin guard; outermost, so it covers open
/// routes (health, pairing, the client page) and unknown paths too.
async fn host_guard(State(state): State<Arc<AppState>>, req: Request<Body>, next: Next) -> Response {
    let refused = |reason: String| {
        if let Some(a) = access::Access::current() {
            a.refused(reason);
        }
    };
    // HTTP/2 carries the authority in the URI rather than a Host header.
    let host = match req.headers().get(header::HOST) {
        Some(v) => match v.to_str() {
            Ok(s) => Some(s.to_string()),
            Err(_) => {
                refused("unreadable Host header".into());
                return ApiError::forbidden("unreadable Host header").into_response();
            }
        },
        None => req.uri().authority().map(|a| a.to_string()),
    };
    // No Host at all is not a browser (browsers always send one), so it
    // cannot be a rebinding page; let it through.
    if let Some(h) = &host {
        if !state.hosts.read().unwrap_or_else(|e| e.into_inner()).allows(h) {
            let Some(HostName::Name(name)) = parse_host(h) else {
                refused(format!("malformed Host header `{h}`"));
                return ApiError::forbidden(format!("malformed Host header `{h}`")).into_response();
            };
            refused(format!("host `{name}` not allowed (DNS-rebinding guard)"));
            return ApiError::forbidden(format!(
                "this engine does not answer to the host name `{name}` (DNS-rebinding guard). Use an IP address, \
                 localhost or <machine>.local, or allow the name with `estia serve --allow-host {name}` \
                 or {ALLOWED_HOSTS_ENV}={name}"
            ))
            .into_response();
        }
    }
    // A state-changing request from a browser page must come from the origin
    // it is sent to. Same-origin pages (the /client page) send a matching
    // Origin; non-browser clients send none.
    if !matches!(*req.method(), Method::GET | Method::HEAD | Method::OPTIONS) {
        if let Some(origin) = req.headers().get(header::ORIGIN) {
            let origin = origin.to_str().unwrap_or("");
            let ok = host.as_deref().is_some_and(|h| same_origin(origin, h));
            if !ok {
                refused(format!("cross-origin {} from Origin `{origin}`", req.method()));
                return ApiError::forbidden(format!(
                    "cross-origin request refused: Origin `{origin}` is not the origin this request was sent to (`{}`)",
                    host.as_deref().unwrap_or("")
                ))
                .into_response();
            }
        }
    }
    next.run(req).await
}

/// JSON 404 for paths no route matches (instead of a bare 404, or the 401 an
/// unknown path got while auth ran on everything).
async fn no_route(method: Method, uri: axum::http::Uri) -> ApiError {
    ApiError::not_found(format!("no route for {method} {}", uri.path()))
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        // OpenAI-compatible
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/v1/models", get(openai::models))
        // native
        .route("/engine/health", get(engine_api::health))
        .route("/engine/defaults", get(engine_api::get_defaults).put(engine_api::put_defaults))
        .route("/engine/models", get(engine_api::list_models))
        .route("/engine/models/pull", post(engine_api::pull_model))
        .route("/engine/models/:id", delete(engine_api::delete_model))
        .route("/engine/models/:id/progress", get(engine_api::model_progress))
        .route("/engine/generate", post(engine_api::generate))
        .route("/engine/embed", post(engine_api::embed))
        .route("/engine/stats", get(engine_api::stats))
        .route("/engine/pair", post(engine_api::pair_request))
        .route("/engine/pair/:id", get(engine_api::pair_poll))
        .route("/engine/pairings", get(engine_api::list_pairings))
        .route("/engine/pairings/:id/approve", post(engine_api::approve_pairing))
        .route("/engine/pairings/:id/deny", post(engine_api::deny_pairing))
        .route("/engine/jobs", get(engine_api::list_jobs))
        .route("/engine/jobs/:id", get(engine_api::get_job))
        .route("/engine/jobs/:id/events", get(engine_api::job_events))
        .route("/engine/runtime/install", post(engine_api::install_runtime))
        .route("/client", get(client_page))
        .route("/client/", get(|| async { axum::response::Redirect::permanent("/client") }))
        .route("/", get(|| async { axum::response::Redirect::temporary("/client") }))
        // Auth on matched routes only; `required_scope` stays fail-closed for
        // every one of them.
        .route_layer(middleware::from_fn_with_state(Arc::clone(&state), auth))
        .fallback(no_route)
        .layer(middleware::from_fn_with_state(Arc::clone(&state), host_guard))
        // Outermost: request ids and the access log see every request,
        // including the ones the guards refuse and unknown paths.
        .layer(middleware::from_fn(access::middleware))
        .with_state(state)
}

/// What `engine.json` records while a daemon runs.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct EngineRecord {
    pub pid: u32,
    pub port: u16,
    pub bind: String,
    pub started_unix: u64,
    pub api_version: u32,
}

pub fn engine_record_path(data_dir: &std::path::Path) -> PathBuf {
    data_dir.join("engine.json")
}

/// Is something already answering on the recorded port?
pub fn another_engine_running(data_dir: &std::path::Path) -> Option<EngineRecord> {
    let text = std::fs::read_to_string(engine_record_path(data_dir)).ok()?;
    let rec: EngineRecord = serde_json::from_str(&text).ok()?;
    let addr = rec.dial_addr()?;
    std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(300)).ok().map(|_| rec)
}

impl EngineRecord {
    /// Where a local client reaches this engine: the bound address, with a
    /// wildcard bind (`0.0.0.0`, `::`) mapped to loopback of the same family.
    /// Parsed as an IP rather than as `"{bind}:{port}"`, which an IPv6 bind
    /// such as `::1` does not survive.
    pub fn dial_addr(&self) -> Option<SocketAddr> {
        let ip: std::net::IpAddr = self.bind.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
        let ip = match ip {
            std::net::IpAddr::V4(v4) if v4.is_unspecified() => std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            std::net::IpAddr::V6(v6) if v6.is_unspecified() => std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
            other => other,
        };
        Some(SocketAddr::new(ip, self.port))
    }
}

/// How the daemon presents itself on the network.
#[derive(Debug, Clone, Default)]
pub struct ServeOptions {
    /// Allow a non-loopback bind. Requires auth: every LAN client pairs first.
    pub lan: bool,
    /// Advertise `_estia._tcp` over Bonjour/mDNS (LAN only).
    pub advertise: bool,
    /// Instance name for the advertisement; defaults to the hostname.
    pub name: Option<String>,
    /// Drop a resident model after this much idle time (None = never).
    pub idle_unload: Option<std::time::Duration>,
    /// Extra `Host` names to answer to (the CLI's `--allow-host`, repeatable),
    /// on top of IP literals, `localhost`, `*.local`, this machine's hostname
    /// and `ESTIA_ALLOWED_HOSTS`. For a reverse proxy or a name like
    /// `studio.lan`. An entry may be a comma-separated list; `*.example.com`
    /// allows the subdomains; `*` switches the DNS-rebinding guard off.
    pub allowed_hosts: Vec<String>,
}

pub const MDNS_SERVICE_TYPE: &str = "_estia._tcp.local.";

fn hostname() -> String {
    std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "estia".to_string())
}

/// This machine's non-loopback, non-link-local IPv4 addresses (via `ifconfig`).
pub fn lan_ipv4_addresses() -> Vec<std::net::Ipv4Addr> {
    let mut out = Vec::new();
    if let Ok(o) = std::process::Command::new("ifconfig").output() {
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            if let Some(rest) = line.trim().strip_prefix("inet ") {
                if let Some(ip) = rest.split_whitespace().next().and_then(|s| s.parse::<std::net::Ipv4Addr>().ok()) {
                    if !ip.is_loopback() && !ip.is_link_local() {
                        out.push(ip);
                    }
                }
            }
        }
    }
    out
}

/// The TXT record every registration path publishes.
fn advert_props() -> Vec<(&'static str, String)> {
    vec![
        ("api_version", API_VERSION.to_string()),
        ("engine_version", env!("CARGO_PKG_VERSION").to_string()),
        ("protocol_version", estia_engine::proto::PROTOCOL_VERSION.to_string()),
    ]
}

/// Who is doing the multicast for us.
enum Registration {
    /// macOS: `dns-sd -R`, so mDNSResponder announces on our behalf. See
    /// [`register_via_dns_sd`] for why this is the default there.
    #[cfg(target_os = "macos")]
    System(std::process::Child),
    /// Our own multicast sockets, via mdns-sd.
    Rust(Box<mdns_sd::ServiceDaemon>),
}

impl Registration {
    fn stop(self) {
        match self {
            #[cfg(target_os = "macos")]
            Registration::System(c) => stop_tethered(c),
            Registration::Rust(d) => {
                let _ = d.shutdown();
            }
        }
    }
}

/// Register through mDNSResponder rather than our own sockets.
///
/// macOS gates local-network traffic behind a privacy permission. A process
/// that does not hold it gets `No route to host` (`EHOSTUNREACH`) on its first
/// multicast send — and only there, so the socket, the bind and the
/// registration all look successful. mdns-sd tests each interface with exactly
/// such a send and drops the ones that fail, which on a blocked process is
/// every IPv4 interface: the daemon then advertises over link-local IPv6 only
/// and resolves for nobody, while reporting itself as advertising.
///
/// A binary launched from a terminal inherits the terminal's grant, so this is
/// invisible in development. Under launchd there is no grant to inherit and no
/// prompt to answer — the permission is per *app bundle*, and a launchd agent
/// running a plain executable has no bundle identity — so a service installed
/// with `estia service install` could never be discovered, which is exactly the
/// bug this replaces.
///
/// `dns-sd` sidesteps all of it: the multicast is emitted by mDNSResponder, a
/// system daemon that is not gated, and we only talk to it over a local socket.
/// It also buys correct conflict resolution and re-announcement on wake, which
/// our own sockets did not survive.
#[cfg(target_os = "macos")]
fn register_via_dns_sd(port: u16, instance: &str) -> anyhow::Result<std::process::Child> {
    let ty = MDNS_SERVICE_TYPE.trim_end_matches(".local.");
    let mut args: Vec<String> = vec!["-R".into(), instance.into(), ty.into(), "local.".into(), port.to_string()];
    args.extend(advert_props().into_iter().map(|(k, v)| format!("{k}={v}")));
    // The registration lives exactly as long as this process; nothing is read back.
    let mut child = spawn_tethered("dns-sd", &args)?;
    // `dns-sd` exits immediately on a bad type or a missing responder; give it
    // a moment and fail loudly rather than supervising a corpse.
    std::thread::sleep(std::time::Duration::from_millis(400));
    if let Ok(Some(status)) = child.try_wait() {
        anyhow::bail!("dns-sd -R exited immediately ({status})");
    }
    Ok(child)
}

/// Runs `program` under a small `sh` that ends it when our end of its stdin
/// pipe closes, which happens however this process ends: a clean stop, a
/// panic, SIGKILL, an OOM kill. A `dns-sd -R` child that merely had its stdin
/// set to null outlived a killed server, was adopted by launchd and kept
/// advertising an engine that no longer existed (one more after every crash
/// under the service's restart policy).
///
/// The returned child is the `sh`; it exits when `program` does, so
/// `try_wait` on it still reports a registrar that died. The watcher is
/// stopped once `program` has exited, so it never signals a recycled pid.
/// Stop it with [`stop_tethered`], not `kill`: SIGKILL on the `sh` alone
/// would leave `program` behind until the pipe closes.
#[cfg(target_os = "macos")]
fn spawn_tethered(program: &str, args: &[String]) -> std::io::Result<std::process::Child> {
    const TETHER: &str = "exec 3<&0 </dev/null
\"$@\" >/dev/null 2>&1 3<&- &
p=$!
{ read _ <&3; kill \"$p\" 2>/dev/null; } &
r=$!
exec 3<&-
wait \"$p\"
s=$?
kill \"$r\" 2>/dev/null
exit \"$s\"";
    std::process::Command::new("/bin/sh")
        .arg("-c")
        .arg(TETHER)
        .arg("estia-tether")
        .arg(program)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
}

/// Stop a [`spawn_tethered`] child: close its pipe so the watcher ends the
/// program, then reap the `sh`, killing it only if it has not gone within
/// two seconds (the closed pipe still takes the program down).
#[cfg(target_os = "macos")]
fn stop_tethered(mut child: std::process::Child) {
    drop(child.stdin.take());
    let deadline = Instant::now() + std::time::Duration::from_secs(2);
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return,
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Register the mDNS advertisement with our own multicast sockets.
fn register_via_mdns_sd(port: u16, instance: &str) -> anyhow::Result<mdns_sd::ServiceDaemon> {
    let mdns = mdns_sd::ServiceDaemon::new()?;
    let host = hostname();
    let props: std::collections::HashMap<String, String> = advert_props().into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    // addr_auto only. Registering explicit IPv4 addresses (alone or alongside
    // addr_auto) made the instance invisible to both the mdns-sd browser and
    // the system resolver on macOS; addr_auto is announced fine, it just
    // lists link-local IPv6 first — `discover` sorts IPv4 first and resolves
    // `<host>.local` when no IPv4 is announced.
    let info = mdns_sd::ServiceInfo::new(MDNS_SERVICE_TYPE, instance, &format!("{host}.local."), "", port, props)?.enable_addr_auto();
    mdns.register(info)?;
    Ok(mdns)
}

fn register(port: u16, instance: &str) -> anyhow::Result<Registration> {
    #[cfg(target_os = "macos")]
    {
        match register_via_dns_sd(port, instance) {
            Ok(c) => {
                tracing::debug!(instance = %instance, port, via = "dns-sd", "Bonjour registration");
                return Ok(Registration::System(c));
            }
            Err(e) => tracing::warn!(error = %e, "registering via dns-sd failed; falling back to our own multicast sockets"),
        }
    }
    let d = register_via_mdns_sd(port, instance)?;
    tracing::debug!(instance = %instance, port, via = "mdns-sd", "Bonjour registration");
    Ok(Registration::Rust(Box::new(d)))
}

/// How often the advertiser looks for a reason to re-register.
const ADVERT_POLL: std::time::Duration = std::time::Duration::from_secs(15);
/// Re-register unconditionally at least this often.
const ADVERT_REFRESH: std::time::Duration = std::time::Duration::from_secs(300);
/// Wall clock running ahead of the monotonic clock by more than this means the
/// machine was asleep in between.
const SLEEP_SLOP: std::time::Duration = std::time::Duration::from_secs(10);

/// A Bonjour advertisement that is supervised rather than registered once.
///
/// Registering once is not enough on either path. `dns-sd -R` is a child
/// process that can die (mDNSResponder restarting takes it with it), so it is
/// watched and respawned. Our own sockets are worse: mdns-sd binds one per
/// interface **at registration time**, and those do not survive a sleep/wake
/// cycle or a Wi-Fi change — the registration still looks healthy from inside
/// the process while the instance has silently stopped answering. Measured on
/// macOS: a freshly started daemon holds an IPv4 socket on port 5353 and
/// resolves; the same binary after 12 hours holds only IPv6 sockets and
/// resolves for nobody. So that path is rebuilt outright when the machine's
/// IPv4 set changes, when it wakes from sleep, and on a slow heartbeat that
/// covers whatever we failed to notice.
struct Advertiser {
    port: u16,
    instance: String,
    reg: Option<Registration>,
    ips: Vec<std::net::Ipv4Addr>,
    at_mono: Instant,
    at_wall: std::time::SystemTime,
}

impl Advertiser {
    fn start(port: u16, instance: String) -> anyhow::Result<Self> {
        let mut a =
            Advertiser { port, instance, reg: None, ips: Vec::new(), at_mono: Instant::now(), at_wall: std::time::SystemTime::now() };
        a.rebuild()?;
        Ok(a)
    }

    /// True when mDNSResponder is doing the announcing, which needs only a
    /// liveness check rather than periodic re-registration.
    fn delegated(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            return matches!(self.reg, Some(Registration::System(_)));
        }
        #[allow(unreachable_code)]
        false
    }

    fn rebuild(&mut self) -> anyhow::Result<()> {
        if let Some(r) = self.reg.take() {
            r.stop();
        }
        self.reg = Some(register(self.port, &self.instance)?);
        self.ips = lan_ipv4_addresses();
        self.at_mono = Instant::now();
        self.at_wall = std::time::SystemTime::now();
        Ok(())
    }

    /// Re-register if anything could have invalidated the advertisement.
    /// Returns the reason when it did.
    fn tick(&mut self) -> Option<&'static str> {
        let mono = self.at_mono.elapsed();
        // `Instant` does not advance while macOS sleeps; `SystemTime` does.
        let wall = self.at_wall.elapsed().unwrap_or(mono);
        let reason = if self.delegated() {
            // mDNSResponder handles address changes and wake itself; the only
            // failure we can see from here is the registrar dying.
            #[cfg(target_os = "macos")]
            {
                let alive = match &mut self.reg {
                    Some(Registration::System(c)) => !matches!(c.try_wait(), Ok(Some(_))),
                    _ => true,
                };
                if alive {
                    return None;
                }
                "registrar exited"
            }
            #[cfg(not(target_os = "macos"))]
            return None;
        } else if lan_ipv4_addresses() != self.ips {
            "address change"
        } else if wall > mono + SLEEP_SLOP {
            "wake from sleep"
        } else if mono >= refresh_interval() {
            "heartbeat"
        } else {
            return None;
        };
        match self.rebuild() {
            Ok(()) => Some(reason),
            Err(e) => {
                tracing::warn!(reason = %reason, error = %e, "Bonjour re-registration failed");
                None
            }
        }
    }

    fn stop(mut self) {
        if let Some(r) = self.reg.take() {
            r.stop();
        }
    }
}

/// Heartbeat period. `ESTIA_MDNS_REFRESH_SECS` shortens it so the rebuild path
/// can be exercised without waiting five minutes.
fn refresh_interval() -> std::time::Duration {
    std::env::var("ESTIA_MDNS_REFRESH_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|s| *s > 0)
        .map(std::time::Duration::from_secs)
        .unwrap_or(ADVERT_REFRESH)
}

/// The test client: one static page, no build step, same origin as the API so
/// a phone on the LAN opens `http://<host>:<port>/client` and nothing else.
async fn client_page() -> impl axum::response::IntoResponse {
    use axum::http::header;
    (
        [
            (header::CONTENT_SECURITY_POLICY, client_csp()),
            (header::X_FRAME_OPTIONS, "DENY"),
            (header::REFERRER_POLICY, "no-referrer"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        ],
        axum::response::Html(CLIENT_PAGE),
    )
}

const CLIENT_PAGE: &str = include_str!("../client/index.html");

/// Content-Security-Policy for the test page. Its one inline script is allowed
/// by hash, so markup injected through a server-supplied string cannot run
/// script (`'unsafe-inline'` would let an injected `onerror=` run). The page
/// may talk to any engine URL the user types, hence `connect-src *`.
fn client_csp() -> &'static str {
    static CSP: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CSP.get_or_init(|| {
        use sha2::{Digest, Sha256};
        let start = CLIENT_PAGE.find("<script>").map(|i| i + "<script>".len()).unwrap_or(0);
        let end = CLIENT_PAGE[start..].find("</script>").map(|i| start + i).unwrap_or(start);
        let digest = Sha256::digest(&CLIENT_PAGE.as_bytes()[start..end]);
        format!(
            "default-src 'none'; script-src 'sha256-{}'; style-src 'unsafe-inline'; connect-src *; img-src 'self' data:; \
             frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
            base64_std(&digest)
        )
    })
}

fn base64_std(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Bind and serve until Ctrl-C. A non-loopback bind needs `opts.lan` and
/// auth on; refuses when another engine already answers for this data dir.
pub async fn serve(state: Arc<AppState>, data_dir: PathBuf, opts: ServeOptions) -> anyhow::Result<()> {
    let addr = state.bind;
    if !addr.ip().is_loopback() {
        if !opts.lan {
            anyhow::bail!("binding {} exposes the engine to the network; pass --lan to confirm (clients then pair for a token)", addr.ip());
        }
        if !state.require_auth {
            anyhow::bail!("--no-auth is loopback-only: a LAN engine must require pairing");
        }
    }
    if let Some(rec) = another_engine_running(&data_dir) {
        let host = if rec.bind.contains(':') { format!("[{}]", rec.bind) } else { rec.bind.clone() };
        anyhow::bail!(
            "an engine is already running for this data directory (pid {} on {host}:{}); one engine per data directory",
            rec.pid,
            rec.port
        );
    }
    state.allow_hosts(&opts.allowed_hosts);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let record = EngineRecord {
        pid: std::process::id(),
        port: bound.port(),
        bind: bound.ip().to_string(),
        started_unix: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        api_version: API_VERSION,
    };
    std::fs::create_dir_all(&data_dir)?;
    std::fs::write(engine_record_path(&data_dir), serde_json::to_string_pretty(&record)?)?;
    tracing::info!(
        version = %env!("CARGO_PKG_VERSION"),
        commit = %BUILD_COMMIT,
        api_version = API_VERSION,
        protocol_version = estia_engine::proto::PROTOCOL_VERSION,
        url = %format!("http://{bound}"),
        bind = %bound,
        auth = state.require_auth,
        lan = opts.lan,
        advertise = opts.advertise && !bound.ip().is_loopback(),
        idle_unload_s = opts.idle_unload.map(|d| d.as_secs()),
        allowed_hosts = %if opts.allowed_hosts.is_empty() { "-".to_string() } else { opts.allowed_hosts.join(",") },
        data_dir = %data_dir.display(),
        backend = %state.engine.backend(),
        runner = %runner_description(&state.engine),
        pid = std::process::id(),
        "estia serving"
    );
    let advertiser = if opts.advertise && !bound.ip().is_loopback() {
        let instance = opts.name.clone().unwrap_or_else(hostname);
        match Advertiser::start(bound.port(), instance.clone()) {
            Ok(a) => {
                tracing::info!(service = MDNS_SERVICE_TYPE, instance = %instance, port = bound.port(), "advertising over Bonjour");
                Some(Arc::new(std::sync::Mutex::new(Some(a))))
            }
            Err(e) => {
                tracing::warn!(error = %e, "Bonjour advertisement failed; pairing by address still works");
                None
            }
        }
    } else {
        None
    };
    // Keep the advertisement alive: see `Advertiser`.
    let advert_task = advertiser.as_ref().map(|a| {
        let a = Arc::clone(a);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(ADVERT_POLL).await;
                let Ok(mut guard) = a.lock() else { return };
                let Some(ad) = guard.as_mut() else { return };
                if let Some(why) = ad.tick() {
                    tracing::info!(reason = %why, "re-registered the Bonjour advertisement");
                }
            }
        })
    });
    if let Some(idle) = opts.idle_unload {
        let st = Arc::clone(&state);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                let dropped = st.reap_idle(idle);
                for d in dropped {
                    tracing::info!(model = %d, idle_s = idle.as_secs(), "released idle model");
                }
            }
        });
    }
    let app = router(Arc::clone(&state));
    let result = serve_router(listener, app, ConnLimits::default(), shutdown_signal()).await;
    let loaded = state.loaded();
    let _ = std::fs::remove_file(engine_record_path(&data_dir));
    if let Some(t) = advert_task {
        t.abort();
    }
    if let Some(a) = advertiser {
        if let Ok(mut guard) = a.lock() {
            if let Some(ad) = guard.take() {
                ad.stop();
                tracing::info!("stopped the Bonjour advertisement");
            }
        }
    }
    match &result {
        Ok(()) => tracing::info!(uptime_s = state.started.elapsed().as_secs(), loaded = %loaded.join(","), "estia stopped"),
        Err(e) => tracing::error!(error = %e, "estia stopped with an error"),
    }
    result?;
    Ok(())
}

/// What starts this engine's runners, for the startup log line: the MLX
/// script, or the llama adapter and the `llama-server` it drives.
fn runner_description(engine: &Engine) -> String {
    match engine.backend() {
        Backend::MlxPython => engine.resident_runner().display().to_string(),
        Backend::LlamaCpp => {
            let adapter = engine.config().llama.as_ref().map(|l| l.program.display().to_string()).unwrap_or_else(|| "(none)".into());
            match engine.llama_server_path() {
                Ok(server) => format!("{adapter} → {}", server.display()),
                Err(e) => format!("{adapter} ({e})"),
            }
        }
    }
}

/// Connection limits for [`serve_router`].
///
/// Without them a few hundred connections that never finish their request
/// head exhaust the process's file descriptors (launchd starts agents with a
/// soft limit of 256): `accept` fails, and so do runner spawns, which need
/// pipes. Nothing needs a token to do that.
#[derive(Debug, Clone)]
pub struct ConnLimits {
    /// Longest a client may take to send a request head. The timer restarts
    /// at every request, so it also closes idle keep-alive connections.
    pub header_read_timeout: std::time::Duration,
    /// Most connections one peer may hold (IPv6 counted per /64).
    pub per_peer: usize,
    /// Loopback peers skip `per_peer` (a local process can do worse anyway).
    pub exempt_loopback: bool,
    /// Most connections in total; `None` sizes it from the open-files limit,
    /// which [`serve_router`] first raises.
    pub total: Option<usize>,
}

impl Default for ConnLimits {
    fn default() -> Self {
        ConnLimits { header_read_timeout: std::time::Duration::from_secs(10), per_peer: 32, exempt_loopback: true, total: None }
    }
}

/// File descriptors kept free for runner pipes, `tokens.json` / `config.json`
/// writes and logs, below the connection cap.
const FD_HEADROOM: u64 = 64;

/// Raise the soft open-files limit toward the hard one; returns the soft limit
/// now in force. macOS refuses more than `OPEN_MAX` (10240) here.
#[allow(clippy::unnecessary_cast)] // rlim_t is not u64 on every target
fn raise_fd_limit() -> u64 {
    #[cfg(unix)]
    // SAFETY: getrlimit/setrlimit only read and write the struct we pass.
    unsafe {
        let mut rl = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl) != 0 {
            return 256;
        }
        for want in [rl.rlim_max.min(65_536), 10_240, 4_096] {
            if want <= rl.rlim_cur {
                break;
            }
            let new = libc::rlimit { rlim_cur: want, rlim_max: rl.rlim_max };
            if libc::setrlimit(libc::RLIMIT_NOFILE, &new) == 0 {
                return want as u64;
            }
        }
        rl.rlim_cur as u64
    }
    #[cfg(not(unix))]
    1_024
}

/// A peer's share of the connection budget, returned when the connection ends.
struct PeerSlot {
    map: Arc<Mutex<HashMap<std::net::IpAddr, usize>>>,
    key: Option<std::net::IpAddr>,
}

impl PeerSlot {
    fn take(map: &Arc<Mutex<HashMap<std::net::IpAddr, usize>>>, ip: std::net::IpAddr, limits: &ConnLimits) -> Option<Self> {
        let ip = ip.to_canonical();
        if limits.exempt_loopback && ip.is_loopback() {
            return Some(PeerSlot { map: Arc::clone(map), key: None });
        }
        // One IPv6 host can use a whole /64; count it as one peer.
        let key = match ip {
            std::net::IpAddr::V6(v6) => std::net::IpAddr::V6((u128::from(v6) & !0u128 << 64).into()),
            v4 => v4,
        };
        let mut m = map.lock().unwrap_or_else(|e| e.into_inner());
        let n = m.entry(key).or_insert(0);
        if *n >= limits.per_peer {
            return None;
        }
        *n += 1;
        Some(PeerSlot { map: Arc::clone(map), key: Some(key) })
    }
}

impl Drop for PeerSlot {
    fn drop(&mut self) {
        let Some(key) = self.key else { return };
        let mut m = self.map.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = m.get_mut(&key) {
            *n -= 1;
            if *n == 0 {
                m.remove(&key);
            }
        }
    }
}

/// Serve `app` on `listener` over HTTP/1.1 until `shutdown` resolves, with
/// the limits `axum::serve` lacks: a request-head timeout, a per-peer
/// connection cap and a total cap below the open-files limit. Accept errors
/// and refused connections are logged (`warn`). On shutdown, in-flight
/// requests get ten seconds; a long SSE stream does not hold the process
/// hostage.
pub async fn serve_router(
    listener: tokio::net::TcpListener,
    app: Router,
    limits: ConnLimits,
    shutdown: impl std::future::Future<Output = ()>,
) -> std::io::Result<()> {
    use tower_service::Service as _;
    let soft = raise_fd_limit();
    let total = limits.total.unwrap_or_else(|| soft.saturating_sub(FD_HEADROOM).clamp(16, 4_096) as usize);
    tracing::debug!(open_files = soft, max_connections = total, per_client = limits.per_peer, "connection limits");
    // A client over its connection cap is logged at most every 10 s, with a count.
    let mut refused_since_log: u64 = 0;
    let mut last_refusal_log: Option<Instant> = None;
    let budget = Arc::new(tokio::sync::Semaphore::new(total));
    let peers: Arc<Mutex<HashMap<std::net::IpAddr, usize>>> = Arc::default();
    let mut http = hyper::server::conn::http1::Builder::new();
    http.timer(hyper_util::rt::TokioTimer::new()).header_read_timeout(limits.header_read_timeout);
    let graceful = hyper_util::server::graceful::GracefulShutdown::new();
    let mut make = app.into_make_service_with_connect_info::<SocketAddr>();
    tokio::pin!(shutdown);
    loop {
        // Take a slot before accepting: at the cap, new connections wait in
        // the kernel's backlog instead of costing us a descriptor.
        let permit = tokio::select! {
            p = Arc::clone(&budget).acquire_owned() => p.expect("the semaphore is never closed"),
            () = &mut shutdown => break,
        };
        let (stream, peer) = tokio::select! {
            r = listener.accept() => match r {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, "accept failed");
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                    continue;
                }
            },
            () = &mut shutdown => break,
        };
        let Some(slot) = PeerSlot::take(&peers, peer.ip(), &limits) else {
            // This peer holds its share already; close at once.
            refused_since_log += 1;
            if last_refusal_log.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(10)) {
                tracing::warn!(
                    peer = %peer.ip(),
                    per_client = limits.per_peer,
                    refused = refused_since_log,
                    "connection refused: this client already holds its share of connections"
                );
                refused_since_log = 0;
                last_refusal_log = Some(Instant::now());
            }
            continue;
        };
        let _ = stream.set_nodelay(true);
        let svc = make.call(peer).await.unwrap_or_else(|e| match e {});
        let conn = http.serve_connection(hyper_util::rt::TokioIo::new(stream), hyper_util::service::TowerToHyperService::new(svc));
        let conn = graceful.watch(conn);
        tokio::spawn(async move {
            let _ = conn.await;
            drop(slot);
            drop(permit);
        });
    }
    drop(listener);
    tracing::info!("shutting down: no new connections; waiting up to 10 s for requests in flight");
    if tokio::time::timeout(std::time::Duration::from_secs(10), graceful.shutdown()).await.is_err() {
        tracing::warn!("requests still in flight after 10 s; closing them");
    }
    Ok(())
}

/// Ctrl-C, or SIGTERM from launchd/systemd.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
        let signal = tokio::select! {
            _ = tokio::signal::ctrl_c() => "SIGINT",
            _ = term.recv() => "SIGTERM",
        };
        tracing::info!(signal = %signal, "shutdown requested");
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!(signal = "ctrl-c", "shutdown requested");
    }
}

/// One engine found on the LAN.
#[derive(Debug, Clone, Serialize)]
pub struct Discovered {
    pub name: String,
    pub host: String,
    pub addresses: Vec<String>,
    pub port: u16,
    pub api_version: Option<String>,
    pub engine_version: Option<String>,
}

/// Order the addresses by what a client can actually dial: routable IPv4
/// first, then routable IPv6, then link-local IPv6, loopback last.
///
/// An engine on *this* machine is a special case worth handling rather than
/// printing: macOS resolves the machine's own `.local` name to 127.0.0.1, so a
/// local engine is discovered at an address that is correct here and useless in
/// the "point your other device at this" line the CLI prints from it. When
/// nothing routable came back, this machine's own LAN addresses are added.
fn usable_addresses(addrs: Vec<String>) -> Vec<String> {
    fn rank(a: &str) -> u8 {
        match a.parse::<std::net::IpAddr>() {
            Ok(ip) if ip.is_loopback() => 3,
            Ok(std::net::IpAddr::V4(_)) => 0,
            Ok(std::net::IpAddr::V6(v6)) if (v6.segments()[0] & 0xffc0) == 0xfe80 => 2,
            Ok(std::net::IpAddr::V6(_)) => 1,
            Err(_) => 3,
        }
    }
    let mut out = addrs;
    let routable_v4 = out.iter().any(|a| rank(a) == 0);
    if !routable_v4 && out.iter().any(|a| rank(a) == 3) {
        for ip in lan_ipv4_addresses() {
            out.push(ip.to_string());
        }
    }
    out.sort_by_key(|a| rank(a));
    let mut seen = std::collections::HashSet::new();
    out.retain(|a| seen.insert(a.clone()));
    out
}

/// Browse for `_estia._tcp` for `window`. Blocking.
pub fn discover(window: std::time::Duration) -> anyhow::Result<Vec<Discovered>> {
    let mdns = mdns_sd::ServiceDaemon::new()?;
    let rx = mdns.browse(MDNS_SERVICE_TYPE)?;
    let deadline = Instant::now() + window;
    let mut found: Vec<Discovered> = Vec::new();
    #[allow(unused_mut)]
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(remaining) {
            Ok(mdns_sd::ServiceEvent::ServiceResolved(info)) => {
                let name = info.get_fullname().split('.').next().unwrap_or("").to_string();
                let mut addresses: Vec<String> = info.get_addresses().iter().map(|a| a.to_string()).collect();
                let host = info.get_hostname().trim_end_matches('.').to_string();
                // No IPv4 in the announcement: ask the resolver for `host.local`,
                // which on macOS and Linux goes through mDNS and returns IPv4.
                if !addresses.iter().any(|a| a.parse::<std::net::Ipv4Addr>().is_ok()) {
                    use std::net::ToSocketAddrs;
                    addresses.extend(
                        (host.as_str(), info.get_port())
                            .to_socket_addrs()
                            .map(|it| it.filter(|a| a.is_ipv4()).map(|a| a.ip().to_string()).collect::<Vec<_>>())
                            .unwrap_or_default(),
                    );
                }
                let addresses = usable_addresses(addresses);
                let d = Discovered {
                    name,
                    host,
                    addresses,
                    port: info.get_port(),
                    api_version: info.get_property_val_str("api_version").map(str::to_string),
                    engine_version: info.get_property_val_str("engine_version").map(str::to_string),
                };
                if !found.iter().any(|f| f.name == d.name && f.port == d.port) {
                    found.push(d);
                }
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    let _ = mdns.shutdown();
    if found.is_empty() {
        // The mdns-sd browser misses instances the system resolver sees (a
        // daemon under launchd, in practice). Ask mDNSResponder itself.
        #[cfg(target_os = "macos")]
        {
            found = discover_via_dns_sd(window.min(std::time::Duration::from_secs(4)));
        }
    }
    Ok(found)
}

/// macOS fallback: `dns-sd -B` to list instances, `dns-sd -L` per instance for
/// host, port and TXT, then the OS resolver for the host's IPv4.
#[cfg(target_os = "macos")]
fn discover_via_dns_sd(window: std::time::Duration) -> Vec<Discovered> {
    use std::io::Read;
    use std::net::ToSocketAddrs;
    fn run_for(args: &[&str], window: std::time::Duration) -> String {
        // `dns-sd` browses forever and block-buffers whenever its stdout is not
        // a terminal: killed at the end of the window it flushes nothing, so
        // reading it straight into a pipe or a file returns an empty string no
        // matter which signal it is sent. Run it under `script`, which gives it
        // a pty and forwards what it prints, line by line, to `script`'s own
        // stdout — which we point at a file and read once the window closes.
        //
        // Two details that both look like the same empty-output bug:
        // the typescript `script` writes (its first argument) is itself
        // buffered and only flushed on a clean exit, hence `/dev/null` there
        // and the redirect for the real capture; and `script` ends the session
        // when its stdin reaches EOF, so stdin is a pipe we hold open rather
        // than `/dev/null`, which would kill `dns-sd` immediately.
        let out_path =
            std::env::temp_dir().join(format!("estia-dns-sd-{}-{}.txt", std::process::id(), args.join("_").replace(['/', '.', ' '], "-")));
        let Ok(out_file) = std::fs::File::create(&out_path) else {
            return String::new();
        };
        let mut cmd = std::process::Command::new("script");
        cmd.arg("-q").arg("/dev/null").arg("dns-sd").args(args);
        cmd.stdout(std::process::Stdio::from(out_file)).stderr(std::process::Stdio::null()).stdin(std::process::Stdio::piped());
        let Ok(mut child) = cmd.spawn() else {
            let _ = std::fs::remove_file(&out_path);
            return String::new();
        };
        let stdin = child.stdin.take();
        std::thread::sleep(window);
        let _ = child.kill();
        let _ = child.wait();
        drop(stdin);
        let mut out = String::new();
        if let Ok(mut f) = std::fs::File::open(&out_path) {
            let mut buf = Vec::new();
            let _ = f.read_to_end(&mut buf);
            out = String::from_utf8_lossy(&buf).replace('\r', "");
        }
        let _ = std::fs::remove_file(&out_path);
        out
    }
    let ty = MDNS_SERVICE_TYPE.trim_end_matches(".local.");
    let browse = run_for(&["-B", ty, "local."], window);
    let mut instances: Vec<String> = Vec::new();
    for line in browse.lines() {
        // "Timestamp  A/R Flags if Domain  Service Type  Instance Name"
        if line.contains(" Add ") && line.contains(ty) {
            if let Some(idx) = line.find(ty) {
                let name = line[idx + ty.len()..].trim().trim_start_matches('.').trim().to_string();
                if !name.is_empty() && !instances.contains(&name) {
                    instances.push(name);
                }
            }
        }
    }
    let mut found = Vec::new();
    for inst in instances {
        let lookup = run_for(&["-L", &inst, ty, "local."], std::time::Duration::from_secs(2));
        let mut host = String::new();
        let mut port: u16 = 0;
        let mut api_version = None;
        let mut engine_version = None;
        for line in lookup.lines() {
            if let Some(idx) = line.find("can be reached at ") {
                let rest = line[idx + "can be reached at ".len()..].trim();
                let hp = rest.split_whitespace().next().unwrap_or("");
                if let Some((h, p)) = hp.rsplit_once(':') {
                    host = h.trim_end_matches('.').to_string();
                    port = p.parse().unwrap_or(0);
                }
            }
            for kv in line.split_whitespace() {
                if let Some(v) = kv.strip_prefix("api_version=") {
                    api_version = Some(v.to_string());
                }
                if let Some(v) = kv.strip_prefix("engine_version=") {
                    engine_version = Some(v.to_string());
                }
            }
        }
        if host.is_empty() || port == 0 {
            continue;
        }
        let addresses: Vec<String> = (host.as_str(), port)
            .to_socket_addrs()
            .map(|it| it.filter(|a| a.is_ipv4()).map(|a| a.ip().to_string()).collect())
            .unwrap_or_default();
        let addresses = usable_addresses(addresses);
        found.push(Discovered { name: inst, host, addresses, port, api_version, engine_version });
    }
    found
}

/// Stable conversation key for the prompt cache when the client gives none:
/// the model, the leading system message(s) and the first non-system turn.
/// Handlers pass it (or the client's key) through [`Caller::scoped_key`], so
/// two tokens with the same opening still get separate cache entries.
/// That prefix is identical on every turn of a conversation that grows by
/// appending, so turn two reuses turn one's KV cache. (Hashing a fixed count
/// of messages did not: turn one has one message, turn two starts
/// user + assistant, so the key changed exactly when reuse mattered most.)
pub fn derive_cache_key(model: &str, messages: &[estia_engine::proto::Message]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(model.as_bytes());
    let opening = messages.iter().position(|m| m.role != "system").map(|i| i + 1).unwrap_or(messages.len());
    for m in &messages[..opening] {
        h.update(m.role.as_bytes());
        h.update([0]);
        h.update(m.content.as_bytes());
        h.update([0]);
    }
    let d = h.finalize();
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod cache_key_tests {
    use super::{derive_cache_key, tokens::TokenRecord, Caller};
    use estia_engine::proto::Message;

    fn caller(name: &str, sha: &str) -> Caller {
        Caller::token(&TokenRecord { name: name.into(), sha256: sha.into(), scopes: vec![], created_unix: 0 })
    }

    /// What reaches the runner is always per caller: the same raw key (a
    /// client's `user`, or a derived key from the same opening) never names
    /// the same cache entry for two tokens.
    #[test]
    fn scoped_per_caller() {
        let (a, b) = (caller("alice", "aa"), caller("bob", "bb"));
        let raw = derive_cache_key("m", &[Message::new("user", "hi")]);
        assert_eq!(a.scoped_key(&raw), a.scoped_key(&raw), "stable for one caller");
        assert_ne!(a.scoped_key(&raw), b.scoped_key(&raw));
        assert_ne!(a.scoped_key("conv-1"), b.scoped_key("conv-1"));
        assert_ne!(a.scoped_key("conv-1"), "conv-1");
        assert_ne!(a.scoped_key("conv-1"), a.scoped_key("conv-2"));
        assert_ne!(a.scoped_key("conv-1"), Caller::anonymous().scoped_key("conv-1"));
        // Keyed by the token's hash, not its name: a name minted again for
        // another device does not inherit the old caches.
        assert_ne!(caller("alice", "aa").scoped_key("k"), caller("alice", "a2").scoped_key("k"));
        assert_eq!(a.scoped_key(""), "", "an empty key still means no cache");
        assert_eq!(Caller::current(), Caller::anonymous(), "outside a request");
    }

    #[test]
    fn stable_as_a_conversation_grows() {
        let t1 = vec![Message::new("user", "hi")];
        let t2 = vec![Message::new("user", "hi"), Message::new("assistant", "hello"), Message::new("user", "more")];
        assert_eq!(derive_cache_key("m", &t1), derive_cache_key("m", &t2));
        let s1 = vec![Message::new("system", "terse"), Message::new("user", "hi")];
        let s2 = vec![
            Message::new("system", "terse"),
            Message::new("user", "hi"),
            Message::new("assistant", "yo"),
            Message::new("user", "again"),
        ];
        assert_eq!(derive_cache_key("m", &s1), derive_cache_key("m", &s2));
        assert_ne!(derive_cache_key("m", &t1), derive_cache_key("m", &s1), "system prompt is part of the key");
        assert_ne!(derive_cache_key("m", &t1), derive_cache_key("other", &t1), "model is part of the key");
        assert_ne!(derive_cache_key("m", &t1), derive_cache_key("m", &[Message::new("user", "bye")]));
    }
}

#[cfg(test)]
mod host_guard_tests {
    use super::{same_origin, HostPolicy};

    #[test]
    fn built_in_names_and_ip_literals() {
        let p = HostPolicy::default();
        for ok in [
            "127.0.0.1",
            "127.0.0.1:27200",
            "10.0.0.5:1",
            "[::1]",
            "[::1]:27200",
            "[fe80::1]:80",
            "::1",
            "localhost",
            "LOCALHOST:27200",
            "localhost.",
            "a.localhost",
            "mac.local",
            "Mac.Local.:27200",
        ] {
            assert!(p.allows(ok), "{ok} should be allowed");
        }
        for bad in [
            "",
            "evil.example",
            "evil.example:27200",
            "localhost.evil.example",
            "127.0.0.1.evil.example",
            "local",
            "notlocalhost",
            "evil.localhost.example",
            "[::1",
            "[evil.example]:80",
            "a:b:c",
            "mac.local:notaport",
        ] {
            assert!(!p.allows(bad), "{bad:?} should be refused");
        }
    }

    #[test]
    fn operator_names() {
        let mut p = HostPolicy::default();
        p.allow(["studio.lan:27200,  *.tail.example", "", "10.0.0.1", ".corp.example"]);
        assert!(p.allows("studio.lan") && p.allows("STUDIO.lan.:1"));
        assert!(p.allows("mac.tail.example") && !p.allows("tail.example") && !p.allows("mac.tail.example.evil"));
        assert!(p.allows("x.corp.example"));
        assert!(!p.allows("evil.example"));
        p.allow(["*"]);
        assert!(p.allows("evil.example"), "`*` switches the guard off");
        assert!(!p.allows("[evil"), "but a malformed Host is still refused");
    }

    #[test]
    fn origin_matching() {
        assert!(same_origin("http://127.0.0.1:27200", "127.0.0.1:27200"));
        assert!(same_origin("http://Mac.local:27200", "mac.local:27200"));
        assert!(same_origin("http://[::1]:27200", "[::1]:27200"));
        assert!(same_origin("http://example.lan", "example.lan:80"), "default port");
        assert!(same_origin("https://proxy.example", "proxy.example"), "TLS-terminating proxy that keeps Host");
        assert!(!same_origin("http://127.0.0.1:27201", "127.0.0.1:27200"));
        assert!(!same_origin("http://localhost:27200", "127.0.0.1:27200"));
        assert!(!same_origin("http://evil.example", "127.0.0.1:27200"));
        assert!(!same_origin("null", "127.0.0.1:27200"));
        assert!(!same_origin("", "127.0.0.1:27200"));
        assert!(!same_origin("file://", "127.0.0.1:27200"));
        assert!(!same_origin("http://127.0.0.1:27200", ""));
    }
}

#[cfg(test)]
mod advert_tests {
    use super::*;

    /// The heartbeat is what keeps a long-lived daemon discoverable, so a
    /// misparsed override would quietly disable the whole repair.
    #[test]
    fn refresh_interval_override() {
        assert_eq!(refresh_interval(), ADVERT_REFRESH);
        std::env::set_var("ESTIA_MDNS_REFRESH_SECS", "20");
        assert_eq!(refresh_interval(), std::time::Duration::from_secs(20));
        std::env::set_var("ESTIA_MDNS_REFRESH_SECS", "0");
        assert_eq!(refresh_interval(), ADVERT_REFRESH);
        std::env::set_var("ESTIA_MDNS_REFRESH_SECS", "not-a-number");
        assert_eq!(refresh_interval(), ADVERT_REFRESH);
        std::env::remove_var("ESTIA_MDNS_REFRESH_SECS");
    }

    /// What actually matters: a registration this process makes is resolvable
    /// by a browser, with an IPv4 address to connect to. Touches the network,
    /// so it is opt-in: `cargo test -p estia-server -- --ignored
    /// advertised_engine_is_discoverable`.
    #[test]
    #[ignore]
    fn advertised_engine_is_discoverable() {
        let a = Advertiser::start(27299, "estia-test-probe".to_string()).expect("register");
        std::thread::sleep(std::time::Duration::from_secs(1));
        let found = discover(std::time::Duration::from_secs(5)).expect("browse");
        a.stop();
        let probe = found.iter().find(|d| d.name == "estia-test-probe");
        let probe = probe.unwrap_or_else(|| panic!("probe not discovered: {found:?}"));
        assert_eq!(probe.port, 27299);
        assert!(probe.addresses.iter().any(|a| a.parse::<std::net::Ipv4Addr>().is_ok()), "no IPv4 address to connect to: {probe:?}");
    }

    /// The system-resolver fallback: it must parse `dns-sd` output rather than
    /// come back empty, which is what happened while `dns-sd` was block
    /// buffering into a pipe. Needs an engine advertising on the LAN, so it is
    /// opt-in: `cargo test -p estia-server -- --ignored dns_sd_sees`.
    #[test]
    #[ignore]
    #[cfg(target_os = "macos")]
    fn dns_sd_sees_an_advertised_engine() {
        let a = Advertiser::start(27298, "estia-test-probe".to_string()).expect("register");
        std::thread::sleep(std::time::Duration::from_secs(2));
        let found = discover_via_dns_sd(std::time::Duration::from_secs(4));
        a.stop();
        assert!(found.iter().any(|d| d.name == "estia-test-probe"), "dns-sd fallback did not see the probe: {found:?}");
    }
}

#[cfg(test)]
mod engine_record_tests {
    use super::*;

    fn rec(bind: &str) -> EngineRecord {
        EngineRecord { pid: 1, port: 27182, bind: bind.to_string(), started_unix: 0, api_version: API_VERSION }
    }

    /// `"{bind}:{port}"` does not parse for an IPv6 bind, which made a running
    /// `serve --bind ::1` invisible to `status`, `dashboard` and the
    /// one-engine-per-data-dir guard.
    #[test]
    fn dial_addr_handles_ipv6_and_wildcards() {
        assert_eq!(rec("127.0.0.1").dial_addr(), Some("127.0.0.1:27182".parse().unwrap()));
        assert_eq!(rec("::1").dial_addr(), Some("[::1]:27182".parse().unwrap()));
        assert_eq!(rec("[::1]").dial_addr(), Some("[::1]:27182".parse().unwrap()));
        assert_eq!(rec("0.0.0.0").dial_addr(), Some("127.0.0.1:27182".parse().unwrap()));
        assert_eq!(rec("::").dial_addr(), Some("[::1]:27182".parse().unwrap()));
        assert_eq!(rec("192.168.1.7").dial_addr(), Some("192.168.1.7:27182".parse().unwrap()));
        assert_eq!(rec("not-an-ip").dial_addr(), None);
    }

    #[test]
    fn another_engine_running_sees_an_ipv6_listener() {
        let Ok(listener) = std::net::TcpListener::bind("[::1]:0") else {
            eprintln!("skip: no IPv6 loopback");
            return;
        };
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "estia-engine-record-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut r = rec("::1");
        r.port = listener.local_addr().unwrap().port();
        std::fs::write(engine_record_path(&dir), serde_json::to_string(&r).unwrap()).unwrap();
        assert!(another_engine_running(&dir).is_some(), "an engine bound to ::1 is running");
        drop(listener);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tether_tests {
    use super::*;

    fn alive(pid: i32) -> bool {
        // SAFETY: signal 0 only checks that the pid exists.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    fn within(ms: u64, mut cond: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + std::time::Duration::from_millis(ms);
        while Instant::now() < deadline {
            if cond() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        cond()
    }

    /// A tethered `sleep` that reports its own pid, standing in for `dns-sd -R`.
    fn tethered_sleeper() -> (std::process::Child, i32, PathBuf) {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "estia-tether-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let pidfile = dir.join("pid");
        let script = format!("echo $$ > '{}'; exec sleep 30", pidfile.display());
        let child = spawn_tethered("/bin/sh", &["-c".to_string(), script]).expect("spawn");
        let mut pid = 0;
        assert!(
            within(3_000, || {
                pid = std::fs::read_to_string(&pidfile).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                pid > 0
            }),
            "the tethered program started"
        );
        (child, pid, dir)
    }

    /// What a killed server looks like from the child's side: its end of the
    /// pipe closes. The program must go, and the `sh` with it.
    #[test]
    fn closing_the_pipe_ends_the_program() {
        let (mut child, pid, dir) = tethered_sleeper();
        assert!(alive(pid));
        drop(child.stdin.take());
        assert!(within(3_000, || !alive(pid)), "the program outlived its parent's pipe");
        assert!(within(3_000, || matches!(child.try_wait(), Ok(Some(_)))), "the sh exits too");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn stop_reaps_the_program() {
        let (child, pid, dir) = tethered_sleeper();
        stop_tethered(child);
        assert!(within(1_000, || !alive(pid)), "stop left the program running");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The advertiser's liveness check is `try_wait` on the `sh`, so a program
    /// that dies on its own must end the `sh` while our end is still open.
    #[test]
    fn a_program_that_exits_ends_the_sh() {
        let mut child = spawn_tethered("/bin/sh", &["-c".to_string(), "exit 3".to_string()]).expect("spawn");
        assert!(within(3_000, || matches!(child.try_wait(), Ok(Some(_)))), "the sh outlived its program");
        assert_eq!(child.wait().unwrap().code(), Some(3));
        drop(child.stdin.take());
    }
}

#[cfg(test)]
mod client_page_tests {
    #[test]
    fn base64_matches_rfc4648_vectors() {
        for (i, o) in
            [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("fooba", "Zm9vYmE="), ("foobar", "Zm9vYmFy")]
        {
            assert_eq!(super::base64_std(i.as_bytes()), o);
        }
    }

    #[test]
    fn csp_pins_the_one_inline_script() {
        let csp = super::client_csp();
        assert!(csp.contains("script-src 'sha256-"), "{csp}");
        assert!(!csp.split(';').any(|d| d.trim().starts_with("script-src") && d.contains("unsafe-inline")), "{csp}");
        assert_eq!(super::CLIENT_PAGE.matches("<script").count(), 1, "one inline script, allowed by hash");
    }
}
