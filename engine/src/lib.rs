//! Estia: a local LLM inference engine.
//!
//! Models run inside runner processes that speak JSON lines over stdin and
//! stdout. This crate starts and supervises those runners, manages the model
//! files they load, and gives a host application one value to hold.
//!
//! - [`Session`]: the single implementation of "keep a JSON-over-pipe runner
//!   process alive safely": per-call deadline, one bounded respawn, a
//!   priority gate, cancel. [`EmbedSession`] / [`GenSession`] pair a session
//!   with the model it serves.
//! - [`OneShot`]: one process per request, for the older runners that read
//!   stdin to EOF.
//! - [`models`]: the artifact registry (families with per-backend artifacts:
//!   MLX and GGUF), the Hugging Face downloader with resume and SHA-256
//!   verification, the on-disk [`models::ModelStore`], and imported GGUF
//!   files ([`models::custom`]).
//! - [`runtime`]: backend runtime installers: the Python MLX runtime
//!   (`python-mlx` feature) and upstream llama.cpp's `llama-server`
//!   (`llama-runtime` feature).
//! - [`Backend`]: `mlx-python` or `llama-cpp`, one per engine
//!   ([`EngineConfig::backend`]).
//! - [`Roles`]: stable names a client asks for (`text`, `fast`, `embed`, …),
//!   bound to model families.
//! - [`structured`]: JSON output repair and JSON Schema validation.
//! - [`procmem`]: a process's physical memory by pid ([`phys_footprint`]),
//!   for per-runner memory figures.
//! - [`Engine`]: one value owning the store, the runtime, the resident
//!   sessions and the role table.
//! - [`RemoteEngine`], [`GenHandle`] / [`EmbedHandle`]: the same shapes over
//!   HTTP, for a model served by an `estia serve` daemon on another machine.
//!
//! The wire types live in [`proto`] (re-exported `estia-proto`).

pub mod backend;
pub mod engine;
pub mod error;
pub mod handles;
pub mod location;
pub mod models;
pub mod oneshot;
pub mod procmem;
pub mod remote;
pub mod resident;
pub mod roles;
pub mod runtime;
pub mod session;
pub mod structured;

pub use backend::{Backend, BACKEND_LLAMA_CPP};
pub use engine::{Engine, EngineConfig, EngineError, LlamaLaunch, LlamaServer, Reaped};
pub use error::SessionError;
pub use estia_proto as proto;
pub use handles::{EmbedHandle, GenHandle};
pub use location::EngineLocation;
pub use oneshot::{OneShot, OneShotConfig};
pub use procmem::phys_footprint;
pub use remote::{RemoteEmbed, RemoteEngine, RemoteGen};
pub use resident::{ChatOutcome, EmbedSession, GenSession};
pub use roles::{RoleBinding, RoleError, Roles};
pub use session::{CancelToken, Launch, NoopObserver, Priority, Session, SessionConfig, SessionObserver, StreamOutcome};
pub use structured::{OutputFormat, Structured, StructuredError};
