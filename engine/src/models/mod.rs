//! Models: what the engine knows how to fetch, where they live on disk, and
//! how they get there.
//!
//! - [`registry`] — the built-in artifacts, grouped into **families**. A family
//!   (`gemma4-e4b`) is what a client or a tier asks for; an artifact
//!   (`gemma4-e4b-it-4bit-mlx`) is what a backend loads. Today every built-in
//!   artifact is MLX; GGUF artifacts join when the llama backend does.
//! - [`hf`] — the Hugging Face downloader: tree listing with sizes and LFS
//!   hashes, resumable single-stream and parallel chunked transfers,
//!   SHA-256 verification, pause/cancel, and metadata top-up for installs
//!   that predate a wider file filter.
//! - [`embed`] — embedding models with the task prefixes and architecture facts
//!   that must travel with the weights.
//! - [`store`] — the on-disk layout (`<models_dir>/<artifact id>/`), per-host
//!   path overrides, status queries, download and remove.

pub mod embed;
pub mod hf;
pub mod registry;
pub mod store;

pub use embed::{find_embed_model, EmbedArch, EmbedModel, EmbedTask, EMBEDDING_MODELS};
pub use hf::{request_download_cancel, DownloadProgress, DownloadSpec, DownloadSummary, DOWNLOAD_PAUSED_MARKER};
pub use registry::{
    find_artifact, find_family_default, Artifact, Capability, Format, ModelKind, DEFAULT_GENERATION_MODEL_ID, GENERATION_MODELS,
};
pub use store::{dir_size, ModelStore};
