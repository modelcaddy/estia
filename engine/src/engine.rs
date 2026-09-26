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

use crate::models::ModelStore;
use crate::resident::{EmbedSession, GenSession};
use crate::roles::Roles;
use crate::runtime::PythonRuntime;
use crate::session::{Launch, NoopObserver, SessionConfig, SessionObserver};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
}

impl EngineConfig {
    pub fn new(store: ModelStore, runtime: PythonRuntime, resident_runner: impl Into<PathBuf>) -> Self {
        Self {
            store,
            runtime,
            resident_runner: resident_runner.into(),
            python: None,
            session: SessionConfig::default(),
            observer: Arc::new(NoopObserver),
            roles: Roles::defaults(),
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
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("no Python interpreter for the resident runner: the runtime is not installed")]
    NoInterpreter,
    #[error("resident runner script not found at {0}")]
    NoRunner(PathBuf),
    #[error("model `{id}` is not downloaded at {path}")]
    ModelMissing { id: String, path: PathBuf },
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
    pub fn new(cfg: EngineConfig) -> Self {
        let roles = Mutex::new(cfg.roles.clone());
        Self { cfg, resident_embed: Arc::new(Mutex::new(None)), resident_gen: Arc::new(Mutex::new(None)), roles }
    }

    pub fn store(&self) -> &ModelStore {
        &self.cfg.store
    }

    pub fn runtime(&self) -> &PythonRuntime {
        &self.cfg.runtime
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

    fn interpreter(&self) -> Result<PathBuf, EngineError> {
        if let Some(p) = &self.cfg.python {
            return Ok(p.clone());
        }
        self.cfg.runtime.python_path().ok_or(EngineError::NoInterpreter)
    }

    fn launch(&self) -> Result<Launch, EngineError> {
        if !self.cfg.resident_runner.exists() {
            return Err(EngineError::NoRunner(self.cfg.resident_runner.clone()));
        }
        Ok(Launch::new(self.interpreter()?).arg(self.cfg.resident_runner.clone()))
    }

    fn installed_path(&self, id: &str) -> Result<PathBuf, EngineError> {
        let path = self.cfg.store.path(id);
        if !path.exists() {
            return Err(EngineError::ModelMissing { id: id.to_string(), path });
        }
        Ok(path)
    }

    /// Spawn a fresh, ping-verified embed session for `model_id`, stamped
    /// with `fingerprint` (model **and** backend — the identity every vector
    /// carries). Does not touch `resident_embed`; callers decide where it lives.
    pub fn spawn_embed_session(&self, model_id: &str, fingerprint: &str) -> Result<EmbedSession, EngineError> {
        let launch = self.launch()?;
        let model_path = self.installed_path(model_id)?;
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

    /// Spawn a fresh, ping-verified generation session for `model_id`.
    pub fn spawn_gen_session(&self, model_id: &str) -> Result<GenSession, EngineError> {
        let launch = self.launch()?;
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
                *guard = None;
                reaped.embed = true;
            }
        }
        if let Ok(mut guard) = self.resident_gen.try_lock() {
            if guard.as_ref().map(|s| s.is_idle(idle_after)).unwrap_or(false) {
                *guard = None;
                reaped.gen = true;
            }
        }
        reaped
    }

    pub fn resident_runner(&self) -> &Path {
        &self.cfg.resident_runner
    }
}
