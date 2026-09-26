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
//! and auth on; LAN clients pair for a token. One engine per machine:
//! `engine.json` in the data directory records where it bound, and a second
//! `serve` refuses when that port answers.

pub mod engine_api;
pub mod jobs;
pub mod openai;
pub mod pairing;
pub mod tokens;
pub mod toolcalls;

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use estia_engine::models::embed::{find_embed_model, EmbedModel, BACKEND_MLX_PYTHON};
use estia_engine::models::{find_artifact, find_family_default, Artifact, Format};
use estia_engine::{EmbedSession, Engine, GenSession};
use jobs::JobTable;
use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokens::TokenStore;

/// Version of the `/engine/*` contract. Clients check it in `/engine/health`.
pub const API_VERSION: u32 = 1;

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
}

impl AppState {
    pub fn new(engine: Arc<Engine>, tokens: TokenStore, require_auth: bool, bind: SocketAddr) -> Self {
        let data_dir = tokens.path().parent().map(|p| p.to_path_buf()).unwrap_or_default();
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
        }
    }

    /// The backend identity vectors from this server carry.
    pub fn embed_backend(&self) -> &'static str {
        BACKEND_MLX_PYTHON
    }

    /// Role name, family or artifact id → the artifact this host loads.
    pub fn resolve_generation(&self, name: &str) -> Result<&'static Artifact, ApiError> {
        if let Some(a) = find_artifact(name) {
            return Ok(a);
        }
        if let Some(a) = find_family_default(name, Format::Mlx) {
            return Ok(a);
        }
        let roles = self.engine.roles();
        match roles.resolve_artifact(name, Format::Mlx) {
            Ok((_, a)) => Ok(a),
            Err(e) => Err(ApiError::not_found(format!("unknown model `{name}`: {e}"))),
        }
    }

    pub fn resolve_embedding(&self, name: Option<&str>) -> Result<&'static EmbedModel, ApiError> {
        let roles = self.engine.roles();
        let id = match name {
            None | Some("embed") => roles.get("embed").map(|b| b.family.clone()).unwrap_or_else(|| "embeddinggemma-300m-4bit".to_string()),
            Some(other) => other.to_string(),
        };
        find_embed_model(&id).ok_or_else(|| ApiError::not_found(format!("unknown embedding model `{id}`")))
    }

    /// Get or spawn the resident generation session for an artifact. Blocking
    /// (spawns a process on first use) — call from `spawn_blocking`.
    pub fn gen_session(&self, artifact: &Artifact) -> Result<Arc<GenSession>, ApiError> {
        if let Some(s) = self.gen.lock().unwrap().get(artifact.id) {
            return Ok(Arc::clone(s));
        }
        let s = Arc::new(self.engine.spawn_gen_session(artifact.id)?);
        self.gen.lock().unwrap().insert(artifact.id.to_string(), Arc::clone(&s));
        Ok(s)
    }

    pub fn embed_session(&self, model: &EmbedModel) -> Result<Arc<EmbedSession>, ApiError> {
        if let Some(s) = self.embed.lock().unwrap().get(model.id) {
            return Ok(Arc::clone(s));
        }
        let fingerprint = model.fingerprint_for(self.embed_backend());
        let s = Arc::new(self.engine.spawn_embed_session(model.id, &fingerprint)?);
        self.embed.lock().unwrap().insert(model.id.to_string(), Arc::clone(&s));
        Ok(s)
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
    fn into_response(self) -> Response {
        let body = serde_json::json!({"error": {"message": self.message, "type": self.kind, "code": self.status.as_u16()}});
        (self.status, Json(body)).into_response()
    }
}

impl From<estia_engine::SessionError> for ApiError {
    fn from(e: estia_engine::SessionError) -> Self {
        ApiError::internal(format!("runner: {e}"))
    }
}
impl From<estia_engine::EngineError> for ApiError {
    fn from(e: estia_engine::EngineError) -> Self {
        match e {
            estia_engine::EngineError::ModelMissing { .. } => ApiError::not_found(e.to_string()),
            other => ApiError::internal(other.to_string()),
        }
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
pub fn required_scope(method: &axum::http::Method, path: &str) -> Option<&'static str> {
    use axum::http::Method;
    match (method, path) {
        (_, "/engine/health") => None,
        // The bundled test client is a static page; everything it does goes
        // through the routes below with the token the user gives it.
        (_, "/") | (_, "/client") => None,
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

async fn auth(State(state): State<Arc<AppState>>, req: Request<Body>, next: Next) -> Response {
    let needed = required_scope(req.method(), req.uri().path());
    let Some(scope) = needed else {
        return next.run(req).await;
    };
    if !state.require_auth {
        return next.run(req).await;
    }
    let header = req.headers().get(axum::http::header::AUTHORIZATION).and_then(|v| v.to_str().ok()).unwrap_or("");
    let token = header.strip_prefix("Bearer ").unwrap_or("").trim();
    if token.is_empty() {
        return ApiError::unauthorized("missing bearer token (Authorization: Bearer …)").into_response();
    }
    match state.tokens.verify(token) {
        Some(record) if record.allows(scope) => next.run(req).await,
        Some(_) => ApiError::forbidden(format!("token lacks the `{scope}` scope")).into_response(),
        None => ApiError::unauthorized("unknown token").into_response(),
    }
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
        .route("/", get(|| async { axum::response::Redirect::temporary("/client") }))
        .layer(middleware::from_fn_with_state(Arc::clone(&state), auth))
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
    let addr: SocketAddr = format!("{}:{}", rec.bind, rec.port).parse().ok()?;
    std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(300)).ok().map(|_| rec)
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
            Registration::System(mut c) => {
                let _ = c.kill();
                let _ = c.wait();
            }
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
    let mut cmd = std::process::Command::new("dns-sd");
    cmd.arg("-R").arg(instance).arg(ty).arg("local.").arg(port.to_string());
    for (k, v) in advert_props() {
        cmd.arg(format!("{k}={v}"));
    }
    // The registration lives as long as the process; nothing is read back.
    cmd.stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).stdin(std::process::Stdio::null());
    let mut child = cmd.spawn()?;
    // `dns-sd` exits immediately on a bad type or a missing responder; give it
    // a moment and fail loudly rather than supervising a corpse.
    std::thread::sleep(std::time::Duration::from_millis(400));
    if let Ok(Some(status)) = child.try_wait() {
        anyhow::bail!("dns-sd -R exited immediately ({status})");
    }
    Ok(child)
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
            Ok(c) => return Ok(Registration::System(c)),
            Err(e) => eprintln!("warning: registering via dns-sd failed ({e}); falling back to our own sockets"),
        }
    }
    Ok(Registration::Rust(Box::new(register_via_mdns_sd(port, instance)?)))
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
                eprintln!("warning: Bonjour re-registration failed ({reason}): {e}");
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
async fn client_page() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../../clients/web/index.html"))
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
        anyhow::bail!(
            "an engine is already running for this data directory (pid {} on {}:{}); one engine per machine",
            rec.pid,
            rec.bind,
            rec.port
        );
    }
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
    eprintln!("estia serving on http://{bound}  (api v{API_VERSION}; Ctrl-C to stop)");
    let advertiser = if opts.advertise && !bound.ip().is_loopback() {
        let instance = opts.name.clone().unwrap_or_else(hostname);
        match Advertiser::start(bound.port(), instance.clone()) {
            Ok(a) => {
                eprintln!("advertising {MDNS_SERVICE_TYPE} as `{instance}` on port {}", bound.port());
                Some(Arc::new(std::sync::Mutex::new(Some(a))))
            }
            Err(e) => {
                eprintln!("warning: Bonjour advertisement failed: {e}");
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
                    eprintln!("re-registered Bonjour advertisement ({why})");
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
                    eprintln!("idle {}s: released {d}", idle.as_secs());
                }
            }
        });
    }
    let app = router(state);
    let result =
        axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).with_graceful_shutdown(shutdown_signal()).await;
    let _ = std::fs::remove_file(engine_record_path(&data_dir));
    if let Some(t) = advert_task {
        t.abort();
    }
    if let Some(a) = advertiser {
        if let Ok(mut guard) = a.lock() {
            if let Some(ad) = guard.take() {
                ad.stop();
            }
        }
    }
    result?;
    Ok(())
}

/// Ctrl-C, or SIGTERM from launchd/systemd.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = term.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
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
    use super::derive_cache_key;
    use estia_engine::proto::Message;

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
