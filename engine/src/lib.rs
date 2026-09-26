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
//! - [`models`]: the artifact registry (families with per-backend artifacts),
//!   the Hugging Face downloader with resume and SHA-256 verification, and the
//!   on-disk [`models::ModelStore`].
//! - [`runtime`]: the Python MLX backend's runtime installer (`python-mlx`
//!   feature).
//! - [`Roles`]: stable names a client asks for (`text`, `fast`, `embed`, …),
//!   bound to model families.
//! - [`structured`]: JSON output repair and JSON Schema validation.
//! - [`Engine`]: one value owning the store, the runtime, the resident
//!   sessions and the role table.
//! - [`RemoteEngine`], [`GenHandle`] / [`EmbedHandle`]: the same shapes over
//!   HTTP, for a model served by an `estia serve` daemon on another machine.
//!
//! The wire types live in [`proto`] (re-exported `estia-proto`).

pub mod engine;
pub mod error;
pub mod handles;
pub mod location;
pub mod models;
pub mod oneshot;
pub mod remote;
pub mod resident;
pub mod roles;
pub mod runtime;
pub mod session;
pub mod structured;

pub use engine::{Engine, EngineConfig, EngineError, Reaped};
pub use error::SessionError;
pub use estia_proto as proto;
pub use handles::{EmbedHandle, GenHandle};
pub use location::EngineLocation;
pub use oneshot::{OneShot, OneShotConfig};
pub use remote::{RemoteEmbed, RemoteEngine, RemoteGen};
pub use resident::{ChatOutcome, EmbedSession, GenSession};
pub use roles::{RoleBinding, RoleError, Roles};
pub use session::{CancelToken, Launch, NoopObserver, Priority, Session, SessionConfig, SessionObserver, StreamOutcome};
pub use structured::{OutputFormat, Structured, StructuredError};
