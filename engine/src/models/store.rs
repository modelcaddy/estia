//! Where models live on disk.

use super::hf::{self, DownloadProgress, DownloadSpec, DownloadSummary};
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
        self.path(id).with_extension("download")
    }

    pub fn is_installed(&self, id: &str) -> bool {
        self.path(id).exists()
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
        let mut removed = false;
        if destination.exists() {
            tokio::fs::remove_dir_all(&destination).await?;
            removed = true;
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
