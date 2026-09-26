//! Models: what the engine knows how to fetch, where they live on disk, and
//! how they get there.
//!
//! - [`registry`] — the built-in artifacts, grouped into **families**. A family
//!   (`gemma4-e4b`) is what a client or a tier asks for; an artifact
//!   (`gemma4-e4b-it-4bit-mlx`, `gemma4-e4b-it-qat-q4_0-gguf`) is what a
//!   backend loads, and each backend loads one format
//!   ([`registry::resolve_generation`]).
//! - [`hf`] — the Hugging Face downloader: tree listing with sizes and LFS
//!   hashes (or an artifact's explicit, pinned file list), resumable
//!   single-stream and parallel chunked transfers, SHA-256 verification,
//!   pause/cancel, and metadata top-up for installs that predate a wider file
//!   filter.
//! - [`embed`] — embedding models with the task prefixes and architecture facts
//!   that must travel with the weights, and one artifact per format.
//! - [`store`] — the on-disk layout (`<models_dir>/<artifact id>/`), per-host
//!   path overrides, status queries, download and remove.
//! - [`custom`] — imported GGUF files ("bring your own model"), registered at
//!   startup like built-ins; [`gguf`] reads their metadata.

pub mod custom;
pub mod embed;
pub mod gguf;
pub mod hf;
pub mod registry;
pub mod store;

pub use custom::{ImportManifest, ImportMode, ImportOptions};
pub use embed::{
    embed_models, find_embed_artifact, find_embed_model, resolve_embed, EmbedArch, EmbedModel, EmbedTask, Pooling, EMBEDDING_MODELS,
};
pub use hf::{request_download_cancel, DownloadProgress, DownloadSpec, DownloadSummary, DOWNLOAD_PAUSED_MARKER};
pub use registry::{
    find_any_artifact, find_artifact, find_family_default, generation_artifacts, resolve_generation, Artifact, ArtifactFile, Capability,
    Format, ModelKind, ResolveError, DEFAULT_GENERATION_MODEL_ID, GENERATION_MODELS, GGUF_MODEL_FILE,
};
pub use store::{dir_size, ModelStore};
