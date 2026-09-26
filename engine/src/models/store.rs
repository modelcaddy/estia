//! Where models live on disk.
//!
//! Each artifact is a directory, `<models_dir>/<artifact id>/`. An MLX
//! artifact holds the repository's files under their own names; a GGUF
//! artifact holds its listed files under fixed names (`model.gguf`), and an
//! imported one also holds its manifest (see [`super::custom`]).

use super::custom::{self, ImportManifest, ImportOptions};
use super::hf::{self, DownloadProgress, DownloadSpec, DownloadSummary};
use super::registry::{self, Format};
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The models directory plus any per-artifact path overrides a host wants
/// (a developer pointing one id at a checkout, say). Cheap to construct;
/// build one per call from the host's configuration.
#[derive(Debug, Clone)]
pub struct ModelStore {
    models_dir: PathBuf,
    overrides: HashMap<String, PathBuf>,
    user_agent: String,
}

impl ModelStore {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self { models_dir: models_dir.into(), overrides: HashMap::new(), user_agent: format!("estia/{}", env!("CARGO_PKG_VERSION")) }
    }

    /// Serve `id` from `path` instead of the models directory. An overridden
    /// artifact reports as external: never downloaded, never removed.
    pub fn with_override(mut self, id: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        self.overrides.insert(id.into(), path.into());
        self
    }

    /// The `User-Agent` sent to Hugging Face. Hosts should name themselves.
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// On-disk directory for an artifact id.
    pub fn path(&self, id: &str) -> PathBuf {
        self.overrides.get(id).cloned().unwrap_or_else(|| self.models_dir.join(id))
    }

    pub fn is_override(&self, id: &str) -> bool {
        self.overrides.contains_key(id)
    }

    /// Where a download stages before its atomic rename into place.
    pub fn partial_path(&self, id: &str) -> PathBuf {
        hf::staging_path(&self.path(id))
    }

    /// Whether the artifact is on disk. A GGUF artifact counts only when its
    /// `model.gguf` is there (a symlinked import whose source moved does not).
    pub fn is_installed(&self, id: &str) -> bool {
        let p = self.path(id);
        if !p.exists() {
            return false;
        }
        match self.gguf_file_name(id) {
            Some(file) if p.is_dir() => p.join(file).exists(),
            _ => true,
        }
    }

    /// What a runner is told to load (`model_path` in the runner protocol):
    /// the artifact's directory for MLX, the `.gguf` file itself for GGUF. An
    /// override that points at a file is used as it is.
    pub fn load_path(&self, id: &str) -> PathBuf {
        let p = self.path(id);
        match self.gguf_file_name(id) {
            Some(file) if !p.is_file() => p.join(file),
            _ => p,
        }
    }

    /// `model.gguf` for a known GGUF artifact (built-in or imported), or for
    /// an unknown id whose directory holds one.
    fn gguf_file_name(&self, id: &str) -> Option<&'static str> {
        match registry::find_any_artifact(id) {
            Some(a) => a.model_file(),
            None => self.path(id).join(registry::GGUF_MODEL_FILE).exists().then_some(registry::GGUF_MODEL_FILE),
        }
    }

    /// The format of what is stored under `id`, when the registry or the
    /// directory tells.
    pub fn format_of(&self, id: &str) -> Option<Format> {
        match registry::find_any_artifact(id) {
            Some(a) => Some(a.format),
            None => self.gguf_file_name(id).map(|_| Format::Gguf),
        }
    }

    /// Import a local GGUF file as a model (see [`custom::import_gguf`]).
    pub fn import_gguf(&self, source: &Path, opts: ImportOptions) -> Result<ImportManifest> {
        custom::import_gguf(self, source, opts)
    }

    /// Register every imported model in this store. `Engine::new` does this;
    /// a host that lists or resolves models without an engine calls it once
    /// at startup.
    pub fn load_imported(&self) -> Vec<ImportManifest> {
        custom::load_imported(self)
    }

    pub fn bytes_on_disk(&self, id: &str) -> Option<u64> {
        let p = self.path(id);
        if p.exists() {
            dir_size(&p)
        } else {
            None
        }
    }

    /// Bytes sitting in the resumable partial from a paused or interrupted
    /// download, so a UI can say "Resume · 2.1 GB done" instead of presenting
    /// a fresh download as if the earlier progress was lost. `None` when the
    /// artifact is installed or nothing is staged.
    pub fn partial_bytes_on_disk(&self, id: &str) -> Option<u64> {
        if self.is_installed(id) {
            return None;
        }
        let partial = self.partial_path(id);
        if partial.exists() {
            dir_size(&partial).filter(|b| *b > 0)
        } else {
            None
        }
    }

    /// Download (or top up) an artifact into its place in this store.
    pub async fn download<F>(&self, spec: &DownloadSpec, on_progress: F) -> Result<DownloadSummary>
    where
        F: FnMut(DownloadProgress),
    {
        hf::download_model(self.path(&spec.id), spec, &self.user_agent, on_progress).await
    }

    /// Remove an installed artifact and any partial beside it. Returns whether
    /// anything was removed. A paused attempt never creates the destination,
    /// so the partial has to count — otherwise Remove reports false and leaves
    /// gigabytes on disk.
    pub async fn remove(&self, id: &str) -> Result<bool> {
        if self.is_override(id) {
            // Not ours to delete: the host pointed this id at its own directory.
            return Ok(false);
        }
        let destination = self.path(id);
        let partial = self.partial_path(id);
        let imported = custom::is_imported(self, id);
        let mut removed = false;
        // `exists()` follows links; `symlink_metadata` also sees a dangling one.
        if destination.symlink_metadata().is_ok() {
            if destination.is_dir() {
                // Does not follow symlinks: a symlinked import loses its link,
                // never the file it points at.
                tokio::fs::remove_dir_all(&destination).await?;
            } else {
                tokio::fs::remove_file(&destination).await?;
            }
            removed = true;
        }
        if imported {
            custom::unregister(id);
        }
        if partial.exists() {
            tokio::fs::remove_dir_all(&partial).await?;
            removed = true;
        }
        Ok(removed)
    }
}

/// Recursive size of a file or directory. `None` when it does not exist or
/// cannot be read.
pub fn dir_size(path: &Path) -> Option<u64> {
    if path.is_file() {
        return std::fs::metadata(path).ok().map(|m| m.len());
    }
    if !path.is_dir() {
        return None;
    }

    let mut total = 0_u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let meta = entry.metadata().ok()?;
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                total = total.saturating_add(meta.len());
            }
        }
    }
    Some(total)
}
