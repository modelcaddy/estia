//! The engine: one value that owns the model store, the backend runtime, the
//! resident sessions, and the role table.
//!
//! Hosts hold one `Engine` per data directory. The resident sessions are
//! shared `Arc<Mutex<Option<…>>>` handles on purpose: a host's long-running
//! job (a batch that embeds or classifies many items, say) borrows the
//! session under the guard for the whole job, and an idle reaper on
//! another thread can only take the lock between jobs. Lock order when both
//! are held: **embed, then gen** — never the reverse, or a co-holding job
//! deadlocks against the reaper or another job.
//!
//! One backend per engine ([`EngineConfig::backend`]): `mlx-python` starts
//! `<python> estia-runner.py` and hands it model directories; `llama-cpp`
//! starts the llama adapter (`estia-llama`, or `estia runner llama`) with the
//! `llama-server` to drive, and hands it `.gguf` files. Everything after the
//! launch — sessions, deadlines, respawn, cancel — is the same.

use crate::backend::Backend;
use crate::models::embed::{self, EmbedModel};
use crate::models::registry::{self, Artifact, ResolveError};
use crate::models::ModelStore;
use crate::resident::{EmbedSession, GenSession};
use crate::roles::Roles;
use crate::runtime::{LlamaRuntime, PythonRuntime};
use crate::session::{Launch, NoopObserver, SessionConfig, SessionObserver};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Which `llama-server` the llama adapter drives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlamaServer {
    /// The build the runtime installer put under the runtime root
    /// ([`LlamaRuntime::server_path`]).
    Installed,
    /// A binary the user provides: a distribution package, or a build for a
    /// GPU the prebuilt binaries miss.
    Path(PathBuf),
}

impl LlamaServer {
    /// `ESTIA_LLAMA_SERVER` when it is set, else the installed runtime.
    pub fn from_env() -> Self {
        match std::env::var_os(crate::runtime::llama::ENV_LLAMA_SERVER) {
            Some(p) if !p.is_empty() => LlamaServer::Path(PathBuf::from(p)),
            _ => LlamaServer::Installed,
        }
    }
}

/// How the engine starts the llama.cpp adapter. The command line is
/// `<program> [prefix args] --server <llama-server> --run-dir <run dir>
/// [--ctx <n>] [-- <extra llama-server args>]`, where `--ctx` is the
/// artifact's context length.
#[derive(Debug, Clone)]
pub struct LlamaLaunch {
    /// The adapter: the standalone `estia-llama` binary, or a host binary
    /// that runs the adapter in a mode (the `estia` CLI, with
    /// `prefix_args = ["runner", "llama"]`).
    pub program: PathBuf,
    pub prefix_args: Vec<OsString>,
    pub server: LlamaServer,
    /// A private directory for the adapter's socket, API-key file and pid
    /// records (`<data dir>/run`). The adapter creates it if missing.
    pub run_dir: PathBuf,
    /// Extra `llama-server` arguments from the operator (`-ngl 0`, say).
    pub extra_server_args: Vec<String>,
}

impl LlamaLaunch {
    /// Run `program` with no prefix arguments against the installed
    /// `llama-server`.
    pub fn new(program: impl Into<PathBuf>, run_dir: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            prefix_args: Vec::new(),
            server: LlamaServer::Installed,
            run_dir: run_dir.into(),
            extra_server_args: Vec::new(),
        }
    }

    /// Add an argument before the adapter's own (`runner`, `llama`).
    pub fn prefix_arg(mut self, arg: impl Into<OsString>) -> Self {
        self.prefix_args.push(arg.into());
        self
    }

    pub fn with_server(mut self, server: LlamaServer) -> Self {
        self.server = server;
        self
    }

    /// Add an argument passed through to `llama-server`.
    pub fn server_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_server_args.push(arg.into());
        self
    }
}

/// Where the engine keeps things and how it starts runners.
#[derive(Clone)]
pub struct EngineConfig {
    /// `<models_dir>/<artifact id>/`.
    pub store: ModelStore,
    /// The Python runtime root (`…/runtime`).
    pub runtime: PythonRuntime,
    /// The resident runner script (`estia-runner.py` today).
    pub resident_runner: PathBuf,
    /// Interpreter override for the resident runner. `None` → the installed
    /// runtime's `python3`; hosts that allow a system interpreter in debug
    /// builds resolve that themselves and pass it here.
    pub python: Option<PathBuf>,
    pub session: SessionConfig,
    pub observer: Arc<dyn SessionObserver>,
    pub roles: Roles,
    /// Which backend runs models. [`EngineConfig::new`] sets `mlx-python`;
    /// [`EngineConfig::with_llama`] switches to `llama-cpp`.
    pub backend: Backend,
    /// How to start the llama adapter; required when `backend` is `llama-cpp`.
    pub llama: Option<LlamaLaunch>,
    /// The llama.cpp runtime, under the same root as the Python runtime by
    /// default.
    pub llama_runtime: LlamaRuntime,
}

impl EngineConfig {
    /// An MLX engine, exactly as before the llama.cpp backend existed.
    pub fn new(store: ModelStore, runtime: PythonRuntime, resident_runner: impl Into<PathBuf>) -> Self {
        let llama_runtime = LlamaRuntime::new(runtime.root());
        Self {
            store,
            runtime,
            resident_runner: resident_runner.into(),
            python: None,
            session: SessionConfig::default(),
            observer: Arc::new(NoopObserver),
            roles: Roles::defaults(),
            backend: Backend::MlxPython,
            llama: None,
            llama_runtime,
        }
    }

    pub fn with_python(mut self, python: impl Into<PathBuf>) -> Self {
        self.python = Some(python.into());
        self
    }

    pub fn with_observer(mut self, observer: Arc<dyn SessionObserver>) -> Self {
        self.observer = observer;
        self
    }

    pub fn with_roles(mut self, roles: Roles) -> Self {
        self.roles = roles;
        self
    }

    /// Choose the backend. `llama-cpp` also needs [`EngineConfig::with_llama`].
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Run models through llama.cpp, started as `launch` says. Sets the
    /// backend to `llama-cpp`.
    pub fn with_llama(mut self, launch: LlamaLaunch) -> Self {
        self.llama = Some(launch);
        self.backend = Backend::LlamaCpp;
        self
    }

    pub fn with_llama_runtime(mut self, runtime: LlamaRuntime) -> Self {
        self.llama_runtime = runtime;
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("no Python interpreter for the resident runner: the runtime is not installed")]
    NoInterpreter,
    #[error("resident runner script not found at {0}")]
    NoRunner(PathBuf),
    #[error("model `{id}` is not downloaded at {path}")]
    ModelMissing { id: String, path: PathBuf },
    #[error("the llama-cpp backend needs the llama adapter's launch settings (EngineConfig::with_llama)")]
    NoLlamaAdapter,
    #[error("llama adapter not found at {0}")]
    NoLlamaAdapterBinary(PathBuf),
    #[error("no llama-server: {0}")]
    NoLlamaServer(String),
    #[error(transparent)]
    Resolve(#[from] ResolveError),
    #[error("fingerprint `{given}` does not match what this engine produces (`{expected}`)")]
    FingerprintMismatch { given: String, expected: String },
    #[error(transparent)]
    Session(#[from] crate::error::SessionError),
}

/// Which resident sessions an idle sweep released.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Reaped {
    pub embed: bool,
    pub gen: bool,
}

pub struct Engine {
    cfg: EngineConfig,
    /// Long-lived embed runner: model loaded once, stays warm. `None` until
    /// first use and after an idle sweep.
    pub resident_embed: Arc<Mutex<Option<EmbedSession>>>,
    /// Long-lived generation runner. Lock order: embed before gen.
    pub resident_gen: Arc<Mutex<Option<GenSession>>>,
    roles: Mutex<Roles>,
}

impl Engine {
    /// Build the engine. Registers the models imported into the store (see
    /// `models::custom`), so they resolve like built-ins from here on.
    /// On the llama backend it also stops `llama-server` processes whose
    /// adapter was killed outright (see [`reap_orphaned_llama_servers`]).
    pub fn new(cfg: EngineConfig) -> Self {
        let imported = cfg.store.load_imported();
        if !imported.is_empty() {
            tracing::debug!(count = imported.len(), "registered imported models");
        }
        if let (Backend::LlamaCpp, Some(l)) = (cfg.backend, &cfg.llama) {
            let stopped = reap_orphaned_llama_servers(&l.run_dir);
            if !stopped.is_empty() {
                tracing::info!(pids = ?stopped, run_dir = %l.run_dir.display(), "stopped llama-server processes left by a killed adapter");
            }
        }
        let roles = Mutex::new(cfg.roles.clone());
        Self { cfg, resident_embed: Arc::new(Mutex::new(None)), resident_gen: Arc::new(Mutex::new(None)), roles }
    }

    pub fn store(&self) -> &ModelStore {
        &self.cfg.store
    }

    /// The Python (MLX) runtime.
    pub fn runtime(&self) -> &PythonRuntime {
        &self.cfg.runtime
    }

    /// The llama.cpp runtime.
    pub fn llama_runtime(&self) -> &LlamaRuntime {
        &self.cfg.llama_runtime
    }

    pub fn backend(&self) -> Backend {
        self.cfg.backend
    }

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    /// A snapshot of the role table.
    pub fn roles(&self) -> Roles {
        self.roles.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn set_roles(&self, roles: Roles) {
        *self.roles.lock().unwrap_or_else(|e| e.into_inner()) = roles;
    }

    /// Artifact id, family or role → the generation artifact this engine's
    /// backend loads. An artifact id of another format is an error (see
    /// [`registry::resolve_generation`]).
    pub fn resolve_generation(&self, name: &str) -> Result<&'static Artifact, ResolveError> {
        match registry::resolve_generation(name, self.backend()) {
            Err(ResolveError::Unknown(_)) => {
                let res = self.roles().resolve(name).map_err(|_| ResolveError::Unknown(name.to_string()))?;
                registry::resolve_generation(&res.binding.family, self.backend())
            }
            other => other,
        }
    }

    /// Model id, artifact id or fingerprint (or `None`/`"embed"` for the
    /// `embed` role, defaulting to EmbeddingGemma) → the embedding model and
    /// the artifact this engine's backend loads for it.
    pub fn resolve_embedding(&self, name: Option<&str>) -> Result<(&'static EmbedModel, &'static Artifact), ResolveError> {
        let id = match name {
            None | Some(crate::roles::EMBED) => self
                .roles()
                .get(crate::roles::EMBED)
                .map(|b| b.family.clone())
                .unwrap_or_else(|| embed::EMBEDDING_GEMMA_300M_4BIT.id.to_string()),
            Some(other) => other.to_string(),
        };
        embed::resolve_embed(&id, self.backend())
    }

    /// The fingerprint vectors of `model` carry on this engine
    /// (`<artifact id>@<backend id>`); `None` when the backend has no
    /// artifact for it.
    pub fn embed_fingerprint(&self, model: &EmbedModel) -> Option<String> {
        model.fingerprint_on(self.backend())
    }

    fn interpreter(&self) -> Result<PathBuf, EngineError> {
        if let Some(p) = &self.cfg.python {
            return Ok(p.clone());
        }
        self.cfg.runtime.python_path().ok_or(EngineError::NoInterpreter)
    }

    /// The `llama-server` the adapter will be told to run.
    pub fn llama_server_path(&self) -> Result<PathBuf, EngineError> {
        let launch = self.cfg.llama.as_ref().ok_or(EngineError::NoLlamaAdapter)?;
        match &launch.server {
            LlamaServer::Path(p) if p.is_file() => Ok(p.clone()),
            LlamaServer::Path(p) => Err(EngineError::NoLlamaServer(format!("{} does not exist", p.display()))),
            LlamaServer::Installed => self.cfg.llama_runtime.server_path().ok_or_else(|| {
                EngineError::NoLlamaServer(format!(
                    "the llama.cpp runtime is not installed under {} and {} is not set",
                    self.cfg.llama_runtime.llama_dir().display(),
                    crate::runtime::llama::ENV_LLAMA_SERVER
                ))
            }),
        }
    }

    /// The runner command for `artifact_id` on this engine's backend.
    pub fn launch_for(&self, artifact_id: &str) -> Result<Launch, EngineError> {
        match self.cfg.backend {
            Backend::MlxPython => {
                if !self.cfg.resident_runner.exists() {
                    return Err(EngineError::NoRunner(self.cfg.resident_runner.clone()));
                }
                Ok(Launch::new(self.interpreter()?).arg(self.cfg.resident_runner.clone()))
            }
            Backend::LlamaCpp => {
                let l = self.cfg.llama.as_ref().ok_or(EngineError::NoLlamaAdapter)?;
                // A bare name is looked up on PATH when spawned; only a path
                // can be checked here.
                if l.program.components().count() > 1 && !l.program.exists() {
                    return Err(EngineError::NoLlamaAdapterBinary(l.program.clone()));
                }
                let server = self.llama_server_path()?;
                let mut launch = Launch::new(&l.program);
                for a in &l.prefix_args {
                    launch = launch.arg(a.clone());
                }
                launch = launch.arg("--server").arg(server).arg("--run-dir").arg(l.run_dir.clone());
                if let Some(ctx) = registry::find_any_artifact(artifact_id).and_then(|a| a.context_length) {
                    launch = launch.arg("--ctx").arg(ctx.to_string());
                }
                if !l.extra_server_args.is_empty() {
                    launch = launch.arg("--");
                    for a in &l.extra_server_args {
                        launch = launch.arg(a.clone());
                    }
                }
                Ok(launch)
            }
        }
    }

    /// What the runner is told to load for `id`, checked to be on disk: the
    /// artifact directory for MLX (as always), the `.gguf` file for GGUF.
    fn installed_path(&self, id: &str) -> Result<PathBuf, EngineError> {
        let (path, present) = match self.cfg.backend {
            Backend::MlxPython => {
                let p = self.cfg.store.path(id);
                let present = p.exists();
                (p, present)
            }
            // The runner contract names the `.gguf` by absolute path.
            Backend::LlamaCpp => {
                let p = self.cfg.store.load_path(id);
                (std::path::absolute(&p).unwrap_or(p), self.cfg.store.is_installed(id))
            }
        };
        if !present {
            return Err(EngineError::ModelMissing { id: id.to_string(), path });
        }
        Ok(path)
    }

    /// Refuse a registry artifact of another format: MLX weights cannot run
    /// on llama.cpp and the other way round. Unknown ids (a host's own
    /// directory, an override) pass.
    fn check_format(&self, id: &str) -> Result<(), EngineError> {
        if let Some(a) = registry::find_any_artifact(id) {
            if a.format != self.backend().format() {
                return Err(ResolveError::WrongFormat { id: id.to_string(), format: a.format, backend: self.backend() }.into());
            }
        }
        Ok(())
    }

    /// The artifact this engine loads for an embedding model named by `id`:
    /// a model id resolves to the backend's artifact (so the MLX-era id
    /// `embeddinggemma-300m-4bit` loads the GGUF file on llama.cpp); an
    /// artifact id must be this backend's own.
    fn embed_artifact_id(&self, id: &str) -> Result<String, EngineError> {
        let Some(model) = embed::find_embed_model(id) else {
            return Ok(id.to_string());
        };
        let backend = self.backend();
        let want =
            model.artifact_for(backend).ok_or_else(|| ResolveError::NoArtifactForBackend { family: model.id.to_string(), backend })?;
        if id == want.id || id == model.id {
            return Ok(want.id.to_string());
        }
        let format = model.artifacts.iter().find(|a| a.id == id).map(|a| a.format).unwrap_or(backend.format());
        Err(ResolveError::WrongFormat { id: id.to_string(), format, backend }.into())
    }

    /// Spawn a fresh, ping-verified embed session for `model_id`, stamped
    /// with `fingerprint` (model **and** backend — the identity every vector
    /// carries). Does not touch `resident_embed`; callers decide where it lives.
    ///
    /// `model_id` may be an embedding model's id or its artifact id; it
    /// resolves to the artifact this backend loads. A fingerprint naming a
    /// known backend must match this engine's (`<artifact id>@<backend>`):
    /// vectors stamped with another backend's fingerprint would silently mix
    /// two spaces in one index.
    pub fn spawn_embed_session(&self, model_id: &str, fingerprint: &str) -> Result<EmbedSession, EngineError> {
        let artifact_id = self.embed_artifact_id(model_id)?;
        self.check_format(&artifact_id)?;
        if let Some((stem, backend)) = fingerprint.rsplit_once('@') {
            if let Ok(b) = backend.parse::<Backend>() {
                let expected = format!("{artifact_id}@{}", self.backend().id());
                let known_model = embed::find_embed_model(&artifact_id).is_some();
                if b != self.backend() || (known_model && stem != artifact_id) {
                    return Err(EngineError::FingerprintMismatch { given: fingerprint.to_string(), expected });
                }
            }
        }
        let launch = self.launch_for(&artifact_id)?.with_label(artifact_id.clone());
        let model_path = self.installed_path(&artifact_id)?;
        let s = EmbedSession::spawn(
            launch,
            self.cfg.session.clone(),
            Arc::clone(&self.cfg.observer),
            model_path.to_string_lossy().into_owned(),
            fingerprint,
        )?;
        s.ping()?;
        Ok(s)
    }

    /// [`Engine::spawn_embed_session`] for a model, with the fingerprint this
    /// engine gives it.
    pub fn spawn_embed_model(&self, model: &EmbedModel) -> Result<EmbedSession, EngineError> {
        let backend = self.backend();
        let artifact =
            model.artifact_for(backend).ok_or_else(|| ResolveError::NoArtifactForBackend { family: model.id.to_string(), backend })?;
        let fingerprint = format!("{}@{}", artifact.id, backend.id());
        self.spawn_embed_session(artifact.id, &fingerprint)
    }

    /// Spawn a fresh, ping-verified generation session for `model_id` (an
    /// artifact id in this backend's format).
    pub fn spawn_gen_session(&self, model_id: &str) -> Result<GenSession, EngineError> {
        self.check_format(model_id)?;
        let launch = self.launch_for(model_id)?.with_label(model_id);
        let model_path = self.installed_path(model_id)?;
        let s = GenSession::spawn(
            launch,
            self.cfg.session.clone(),
            Arc::clone(&self.cfg.observer),
            model_path.to_string_lossy().into_owned(),
            model_id,
        )?;
        s.ping()?;
        Ok(s)
    }

    /// Release resident sessions that have been idle for `idle_after`,
    /// freeing their model memory; the next use respawns. Uses `try_lock`, so
    /// a session in the middle of a job is left alone rather than waited on.
    pub fn reap_idle(&self, idle_after: Duration) -> Reaped {
        let mut reaped = Reaped::default();
        if let Ok(mut guard) = self.resident_embed.try_lock() {
            if guard.as_ref().map(|s| s.is_idle(idle_after)).unwrap_or(false) {
                let model = guard.as_ref().map(|s| s.model_path().to_string()).unwrap_or_default();
                *guard = None;
                reaped.embed = true;
                tracing::info!(model_path = %model, idle_s = idle_after.as_secs(), "released idle embedding session");
            }
        }
        if let Ok(mut guard) = self.resident_gen.try_lock() {
            if guard.as_ref().map(|s| s.is_idle(idle_after)).unwrap_or(false) {
                let model = guard.as_ref().map(|s| s.model_id().to_string()).unwrap_or_default();
                *guard = None;
                reaped.gen = true;
                tracing::info!(model = %model, idle_s = idle_after.as_secs(), "released idle generation session");
            }
        }
        reaped
    }

    /// The MLX runner script.
    pub fn resident_runner(&self) -> &Path {
        &self.cfg.resident_runner
    }
}

/// Stop `llama-server` processes whose adapter died without cleaning up.
///
/// The adapter writes `llama-<adapter pid>.json` (`{adapter_pid, server_pid,
/// socket, …}`) into its run directory. Linux kills an orphaned server
/// through `PR_SET_PDEATHSIG` and Windows through a job object; macOS has
/// neither, so a `kill -9` of the adapter leaves the server running with a
/// model in memory. For each record whose adapter is gone: the server is
/// stopped if it is still running *and* is still a `llama-server` (its pid
/// may have been reused), then the record, key file and socket are removed.
/// A record whose adapter is alive belongs to a running engine and is left
/// alone. Returns the pids stopped.
#[cfg(unix)]
pub fn reap_orphaned_llama_servers(run_dir: &Path) -> Vec<u32> {
    let mut stopped = Vec::new();
    let Ok(entries) = std::fs::read_dir(run_dir) else {
        return stopped;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(pid_part) = name.strip_prefix("llama-").and_then(|n| n.strip_suffix(".json")) else {
            continue;
        };
        let Ok(record) =
            std::fs::read(e.path()).map_err(|_| ()).and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).map_err(|_| ()))
        else {
            continue;
        };
        let adapter = record["adapter_pid"].as_u64().or_else(|| pid_part.parse().ok());
        if adapter.is_some_and(|p| process_name(p).is_some()) {
            continue;
        }
        if let Some(server) = record["server_pid"].as_u64() {
            if process_name(server).is_some_and(|n| n.contains("llama-server")) {
                terminate(server);
                stopped.push(server as u32);
            }
        }
        let _ = std::fs::remove_file(e.path());
        let _ = std::fs::remove_file(run_dir.join(format!("llama-{pid_part}.key")));
        if let Some(sock) = record["socket"].as_str().map(PathBuf::from) {
            if sock.file_name().is_some_and(|n| n.to_string_lossy() == format!("llama-{pid_part}.sock")) {
                let _ = std::fs::remove_file(&sock);
                // A private directory made for a short socket path; only
                // removed when empty.
                if let Some(parent) = sock.parent().filter(|p| *p != run_dir) {
                    let _ = std::fs::remove_dir(parent);
                }
            }
        }
    }
    stopped
}

/// Windows ties the server to the adapter with a job object; nothing to reap.
#[cfg(not(unix))]
pub fn reap_orphaned_llama_servers(_run_dir: &Path) -> Vec<u32> {
    Vec::new()
}

/// The command of a running process (`ps -o comm=`), `None` when it is not
/// running or has exited.
#[cfg(unix)]
fn process_name(pid: u64) -> Option<String> {
    if pid == 0 || pid > i32::MAX as u64 {
        return None;
    }
    let out = std::process::Command::new("ps").args(["-p", &pid.to_string(), "-o", "stat=,comm="]).output().ok()?;
    let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let (stat, name) = line.split_once(char::is_whitespace)?;
    // A zombie has exited; it only waits for its parent to reap it.
    (out.status.success() && !stat.starts_with('Z')).then(|| name.trim().to_string())
}

/// SIGTERM, a short grace, then SIGKILL.
#[cfg(unix)]
fn terminate(pid: u64) {
    let signal = |sig: &str| std::process::Command::new("kill").args([sig, &pid.to_string()]).output();
    let _ = signal("-TERM");
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while std::time::Instant::now() < deadline {
        if process_name(pid).is_none() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = signal("-KILL");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("estia-engine-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn args(l: &Launch) -> Vec<String> {
        l.args().iter().map(|a| a.to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn mlx_launch_is_unchanged() {
        let dir = scratch("mlx");
        let runner = dir.join("estia-runner.py");
        std::fs::write(&runner, b"").unwrap();
        let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), &runner)
            .with_python("/usr/bin/python3");
        assert_eq!(cfg.backend, Backend::MlxPython);
        let engine = Engine::new(cfg);
        let l = engine.launch_for("gemma4-e2b-it-4bit-mlx").unwrap();
        assert_eq!(l.program(), Path::new("/usr/bin/python3"));
        assert_eq!(args(&l), vec![runner.display().to_string()]);
        // The MLX store path is the directory, as before.
        std::fs::create_dir_all(dir.join("models/gemma4-e2b-it-4bit-mlx")).unwrap();
        assert_eq!(engine.installed_path("gemma4-e2b-it-4bit-mlx").unwrap(), dir.join("models/gemma4-e2b-it-4bit-mlx"));
        // GGUF artifacts are refused before anything is spawned.
        assert!(matches!(
            engine.spawn_gen_session("gemma4-e2b-it-qat-q4_0-gguf"),
            Err(EngineError::Resolve(ResolveError::WrongFormat { .. }))
        ));
        assert_eq!(engine.resolve_generation("fast").unwrap().id, "gemma4-e2b-it-4bit-mlx");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn llama_launch_names_the_server_run_dir_and_context() {
        let dir = scratch("llama");
        let server = dir.join("llama-server");
        std::fs::write(&server, b"").unwrap();
        let run = dir.join("run");
        let launch = LlamaLaunch::new("/opt/estia/bin/estia", &run)
            .prefix_arg("runner")
            .prefix_arg("llama")
            .with_server(LlamaServer::Path(server.clone()))
            .server_arg("-ngl")
            .server_arg("0");
        let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), dir.join("none.py"))
            .with_llama(launch);
        assert_eq!(cfg.backend, Backend::LlamaCpp);
        let engine = Engine::new(cfg);
        // The adapter binary does not exist at that absolute path.
        assert!(matches!(engine.launch_for("gemma4-e2b-it-qat-q4_0-gguf"), Err(EngineError::NoLlamaAdapterBinary(_))));

        let adapter = dir.join("estia-llama");
        std::fs::write(&adapter, b"").unwrap();
        let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), dir.join("none.py"))
            .with_llama(
                LlamaLaunch::new(&adapter, &run)
                    .prefix_arg("runner")
                    .prefix_arg("llama")
                    .with_server(LlamaServer::Path(server.clone()))
                    .server_arg("-ngl")
                    .server_arg("0"),
            );
        let engine = Engine::new(cfg);
        let l = engine.launch_for("gemma4-e2b-it-qat-q4_0-gguf").unwrap();
        assert_eq!(l.program(), adapter.as_path());
        assert_eq!(
            args(&l),
            vec![
                "runner".to_string(),
                "llama".into(),
                "--server".into(),
                server.display().to_string(),
                "--run-dir".into(),
                run.display().to_string(),
                "--ctx".into(),
                "32768".into(),
                "--".into(),
                "-ngl".into(),
                "0".into()
            ]
        );
        // Embedding artifacts pass their own context; unknown ids pass none.
        assert!(args(&engine.launch_for("embeddinggemma-300m-q8_0-gguf").unwrap()).windows(2).any(|w| w == ["--ctx", "2048"]));
        assert!(!args(&engine.launch_for("someones-own-dir").unwrap()).contains(&"--ctx".to_string()));

        // model_path is the .gguf file, and a missing one is ModelMissing.
        let err = engine.installed_path("gemma4-e2b-it-qat-q4_0-gguf").unwrap_err();
        assert!(
            matches!(&err, EngineError::ModelMissing { path, .. } if path.ends_with("gemma4-e2b-it-qat-q4_0-gguf/model.gguf")),
            "{err}"
        );
        std::fs::create_dir_all(dir.join("models/gemma4-e2b-it-qat-q4_0-gguf")).unwrap();
        assert!(engine.installed_path("gemma4-e2b-it-qat-q4_0-gguf").is_err(), "a directory without model.gguf is not installed");
        std::fs::write(dir.join("models/gemma4-e2b-it-qat-q4_0-gguf/model.gguf"), b"GGUF").unwrap();
        assert_eq!(
            engine.installed_path("gemma4-e2b-it-qat-q4_0-gguf").unwrap(),
            dir.join("models/gemma4-e2b-it-qat-q4_0-gguf/model.gguf")
        );

        // Resolution by backend.
        assert_eq!(engine.resolve_generation("fast").unwrap().id, "gemma4-e2b-it-qat-q4_0-gguf");
        assert_eq!(engine.resolve_generation("text").unwrap().id, "gemma4-e4b-it-qat-q4_0-gguf");
        assert!(matches!(engine.resolve_generation("gemma4-e2b-it-4bit-mlx"), Err(ResolveError::WrongFormat { .. })));
        assert!(matches!(engine.spawn_gen_session("gemma4-e2b-it-4bit-mlx"), Err(EngineError::Resolve(ResolveError::WrongFormat { .. }))));
        let (m, a) = engine.resolve_embedding(None).unwrap();
        assert_eq!((m.id, a.id), ("embeddinggemma-300m-4bit", "embeddinggemma-300m-q8_0-gguf"));
        assert_eq!(engine.embed_fingerprint(m).as_deref(), Some("embeddinggemma-300m-q8_0-gguf@llama-cpp"));
        assert_eq!(engine.embed_artifact_id("embeddinggemma-300m-4bit").unwrap(), "embeddinggemma-300m-q8_0-gguf");
        assert_eq!(engine.embed_artifact_id("embeddinggemma-300m-q8_0-gguf").unwrap(), "embeddinggemma-300m-q8_0-gguf");
        assert!(engine.embed_artifact_id("multilingual-e5-small-mlx").is_err(), "no GGUF artifact");
        // Another backend's fingerprint is refused before anything spawns.
        let err = engine.spawn_embed_session("embeddinggemma-300m-4bit", "embeddinggemma-300m-4bit@mlx-python").err().unwrap();
        assert!(matches!(err, EngineError::FingerprintMismatch { .. }), "{err}");
        let err = engine.spawn_embed_session("embeddinggemma-300m-4bit", "embeddinggemma-300m-4bit@llama-cpp").err().unwrap();
        assert!(matches!(err, EngineError::FingerprintMismatch { .. }), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One role table works on both backends: roles, families and the embed
    /// role resolve to each backend's own artifact and fingerprint.
    #[test]
    fn roles_resolve_to_each_backends_artifact() {
        use crate::roles::{RoleBinding, EMBED};
        let dir = scratch("roles");
        let mut roles = Roles::defaults();
        roles.bind(EMBED, RoleBinding::family("nomic-embed-text-v1.5")).unwrap();
        let cases = [
            (
                Backend::MlxPython,
                "gemma4-e2b-it-4bit-mlx",
                "gemma4-e4b-it-4bit-mlx",
                "gemma4-12b-it-qat-4bit-mlx",
                "nomic-embed-text-v1.5",
                "nomic-embed-text-v1.5@mlx-python",
            ),
            (
                Backend::LlamaCpp,
                "gemma4-e2b-it-qat-q4_0-gguf",
                "gemma4-e4b-it-qat-q4_0-gguf",
                "gemma4-12b-it-qat-q4_0-gguf",
                "nomic-embed-text-v1.5-q8_0-gguf",
                "nomic-embed-text-v1.5-q8_0-gguf@llama-cpp",
            ),
        ];
        for (backend, fast, text, big, embed_artifact, fingerprint) in cases {
            let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), dir.join("none.py"))
                .with_roles(roles.clone())
                .with_backend(backend);
            let engine = Engine::new(cfg);
            assert_eq!(engine.resolve_generation("fast").unwrap().id, fast, "{backend}");
            assert_eq!(engine.resolve_generation("code").unwrap().id, text, "code falls back to text on {backend}");
            assert_eq!(engine.resolve_generation("gemma4-12b-qat").unwrap().id, big, "{backend}");
            assert_eq!(engine.resolve_generation("nobody").unwrap_err(), ResolveError::Unknown("nobody".into()));
            let (m, a) = engine.resolve_embedding(Some(EMBED)).unwrap();
            assert_eq!((m.id, a.id), ("nomic-embed-text-v1.5", embed_artifact), "{backend}");
            assert_eq!(engine.embed_fingerprint(m).as_deref(), Some(fingerprint));
            assert_eq!(m.fingerprint_for(backend.id()), fingerprint);
            // Asked by either fingerprint, the model is the same; the
            // artifact is always this backend's.
            let (_, a) = engine.resolve_embedding(Some("nomic-embed-text-v1.5-q8_0-gguf@llama-cpp")).unwrap();
            assert_eq!(a.id, embed_artifact);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn llama_without_a_server_says_so() {
        let dir = scratch("noserver");
        let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), dir.join("none.py"))
            .with_llama(LlamaLaunch::new("estia-llama", dir.join("run")));
        let engine = Engine::new(cfg);
        let err = engine.launch_for("gemma4-e2b-it-qat-q4_0-gguf").unwrap_err();
        assert!(matches!(err, EngineError::NoLlamaServer(_)), "{err}");
        assert!(err.to_string().contains("ESTIA_LLAMA_SERVER"), "{err}");
        let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), dir.join("none.py"))
            .with_backend(Backend::LlamaCpp);
        assert!(matches!(Engine::new(cfg).launch_for("x"), Err(EngineError::NoLlamaAdapter)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A record left by a killed adapter: its server is stopped and its files
    /// removed; a record whose adapter is alive, and a pid that is no longer
    /// a llama-server, are left alone.
    #[cfg(unix)]
    #[test]
    fn orphaned_llama_servers_are_stopped() {
        let dir = scratch("orphans");
        let run = dir.join("run");
        std::fs::create_dir_all(&run).unwrap();
        // A stand-in server: `sleep` under the name llama-server.
        let fake = dir.join("llama-server");
        std::fs::copy("/bin/sleep", &fake).unwrap();
        let mut server = std::process::Command::new(&fake).arg("30").spawn().unwrap();
        let mut other = std::process::Command::new("/bin/sleep").arg("30").spawn().unwrap();
        // A dead adapter pid: a process that has exited and been reaped.
        let mut gone = std::process::Command::new("/usr/bin/true").spawn().unwrap();
        let dead = gone.id();
        gone.wait().unwrap();
        let mut gone2 = std::process::Command::new("/usr/bin/true").spawn().unwrap();
        let dead2 = gone2.id();
        gone2.wait().unwrap();
        let me = std::process::id();
        let write = |pid: u32, adapter: u32, server_pid: u32| {
            let sock = run.join(format!("llama-{pid}.sock"));
            std::fs::write(&sock, b"").unwrap();
            std::fs::write(run.join(format!("llama-{pid}.key")), b"k").unwrap();
            let rec = serde_json::json!({"adapter_pid": adapter, "server_pid": server_pid, "socket": sock.to_string_lossy(), "addr": null, "model": "m", "kind": "generation"});
            std::fs::write(run.join(format!("llama-{pid}.json")), serde_json::to_vec(&rec).unwrap()).unwrap();
        };
        write(dead, dead, server.id()); // orphan: stop it
        write(me, me, server.id()); // adapter alive (this test): leave it
        write(dead2, dead2, other.id()); // pid now belongs to something else: do not kill

        let stopped = reap_orphaned_llama_servers(&run);
        assert_eq!(stopped, vec![server.id()]);
        let status = server.wait().unwrap();
        assert!(!status.success(), "the orphan was signalled");
        assert!(!run.join(format!("llama-{dead}.json")).exists());
        assert!(!run.join(format!("llama-{dead}.key")).exists());
        assert!(!run.join(format!("llama-{dead}.sock")).exists());
        assert!(run.join(format!("llama-{me}.json")).exists(), "a live adapter's record stays");
        assert!(other.try_wait().unwrap().is_none(), "an unrelated process is never signalled");
        assert!(!run.join(format!("llama-{}.json", dead2)).exists(), "its stale record is removed");
        let _ = other.kill();
        let _ = other.wait();
        assert!(reap_orphaned_llama_servers(&dir.join("missing")).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The whole path, live: imported GGUF files, the engine's launch, a
    /// real adapter and a real `llama-server`. Runs when `ESTIA_LLAMA_ADAPTER`
    /// (the `estia-llama` binary), `ESTIA_LLAMA_SERVER` and the two test
    /// models are set; skipped otherwise.
    #[test]
    fn live_llama_sessions_through_the_engine() {
        use crate::models::custom::{ImportMode, ImportOptions};
        use crate::session::Priority;
        let (Some(adapter), Ok(chat), Ok(emb)) = (
            std::env::var_os("ESTIA_LLAMA_ADAPTER"),
            std::env::var("ESTIA_LLAMA_TEST_MODEL"),
            std::env::var("ESTIA_LLAMA_TEST_EMBED_MODEL"),
        ) else {
            return;
        };
        let LlamaServer::Path(server) = LlamaServer::from_env() else {
            return;
        };
        let dir = scratch("live");
        let store = ModelStore::new(dir.join("models"));
        let gen_id = store
            .import_gguf(
                Path::new(&chat),
                ImportOptions {
                    id: Some("live-chat-gguf".into()),
                    mode: ImportMode::Symlink,
                    context_length: Some(2048),
                    ..Default::default()
                },
            )
            .unwrap()
            .id;
        let embed_id = store
            .import_gguf(
                Path::new(&emb),
                ImportOptions { id: Some("live-embed-gguf".into()), mode: ImportMode::Symlink, ..Default::default() },
            )
            .unwrap()
            .id;
        let cfg = EngineConfig::new(store, PythonRuntime::new(dir.join("runtime")), dir.join("none.py"))
            .with_llama(LlamaLaunch::new(PathBuf::from(adapter), dir.join("run")).with_server(LlamaServer::Path(server)));
        let engine = Engine::new(cfg);
        assert!(args(&engine.launch_for(&gen_id).unwrap()).windows(2).any(|w| w == ["--ctx", "2048"]));

        let gen = engine.spawn_gen_session(&gen_id).unwrap();
        let caps = gen.capabilities();
        assert_eq!(caps.backend.as_deref(), Some("llama-cpp"));
        assert!(caps.chat && caps.load && caps.count_tokens);
        let loaded = gen.load().unwrap().unwrap();
        let msgs = vec![crate::proto::Message::new("user", "Say hi.")];
        let out = gen.chat_with(&msgs, None, None, None, Some(8), Some(0.0), Priority::Interactive).unwrap();
        let tokens = gen.count_tokens("hello world").unwrap().unwrap();
        eprintln!(
            "gen: load {loaded:?}, {} generated, text {:?}, count_tokens {tokens}",
            out.meta.generation_tokens.unwrap_or(0),
            out.text
        );
        assert!(tokens > 0);
        drop(gen);

        let (model, artifact) = crate::models::embed::resolve_embed(&embed_id, Backend::LlamaCpp).unwrap();
        assert_eq!(artifact.id, "live-embed-gguf");
        let emb = engine.spawn_embed_model(model).unwrap();
        assert_eq!(emb.fingerprint(), "live-embed-gguf@llama-cpp");
        let v = emb.embed_batch(&["a cat".to_string(), "a dog".to_string(), "tax law".to_string()]).unwrap();
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|x| x.len() == model.dims));
        eprintln!("embed: {} vectors of {} dims", v.len(), v[0].len());
        drop(emb);
        crate::models::custom::unregister(&gen_id);
        crate::models::custom::unregister(&embed_id);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
