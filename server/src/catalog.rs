//! What an engine can list, fetch and remove, shared by the HTTP routes and
//! the `estia` CLI so both answer the same way on either backend.
//!
//! A pull names weights to download. An embedding model id (`embed`'s
//! default, `embeddinggemma-300m-4bit`) means the artifact the engine's
//! backend loads; any other artifact id means exactly that artifact, in
//! either format, so an MLX engine can fetch GGUF files ahead of a switch;
//! a family or a role (`embed` included) means this backend's artifact of
//! it. Imported models were never downloaded and cannot be pulled.

use estia_engine::models::{custom, embed, find_any_artifact, Artifact, DownloadSpec, ModelStore, ResolveError};
use estia_engine::Engine;

/// Why a pull could not be planned.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PullError {
    #[error("unknown model `{0}`")]
    Unknown(String),
    #[error("`{0}` is an imported model: it is already on disk and has nothing to download")]
    Imported(String),
    #[error(transparent)]
    Resolve(ResolveError),
}

impl From<PullError> for crate::ApiError {
    fn from(e: PullError) -> Self {
        match e {
            PullError::Unknown(_) => crate::ApiError::not_found(e.to_string()),
            PullError::Imported(_) => crate::ApiError::bad_request(e.to_string()),
            PullError::Resolve(r) => crate::resolve_error(r),
        }
    }
}

/// The artifact a pull of `id` downloads on `engine` (see the module docs).
pub fn pull_artifact(engine: &Engine, id: &str) -> Result<&'static Artifact, PullError> {
    let backend = engine.backend();
    let artifact = if id == estia_engine::roles::EMBED {
        engine.resolve_embedding(Some(id)).map_err(PullError::Resolve)?.1
    } else if let Some(model) = embed::find_embed_model(id).filter(|m| m.id == id) {
        model
            .artifact_for(backend)
            .ok_or_else(|| PullError::Resolve(ResolveError::NoArtifactForBackend { family: model.id.to_string(), backend }))?
    } else if let Some(a) = find_any_artifact(id) {
        a
    } else {
        match engine.resolve_generation(id) {
            Ok(a) => a,
            Err(ResolveError::Unknown(_)) => return Err(PullError::Unknown(id.to_string())),
            Err(e) => return Err(PullError::Resolve(e)),
        }
    };
    if artifact.repo_id.is_empty() {
        return Err(PullError::Imported(artifact.id.to_string()));
    }
    Ok(artifact)
}

/// [`pull_artifact`] as a download.
pub fn pull_spec(engine: &Engine, id: &str) -> Result<DownloadSpec, PullError> {
    pull_artifact(engine, id).map(DownloadSpec::from)
}

/// Whether `id` names something `DELETE /engine/models/<id>` and `estia rm`
/// may remove: a known artifact or embedding model, or an import in `store`.
pub fn removable(store: &ModelStore, id: &str) -> bool {
    find_any_artifact(id).is_some() || embed::find_embed_model(id).is_some_and(|m| m.id == id) || custom::is_imported(store, id)
}

/// Whether `id` is a model the user imported into `store`.
pub fn is_imported(store: &ModelStore, id: &str) -> bool {
    custom::is_imported(store, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use estia_engine::runtime::PythonRuntime;
    use estia_engine::{Backend, EngineConfig};

    fn engine(backend: Backend) -> Engine {
        let dir = std::env::temp_dir().join(format!("estia-catalog-{}-{backend}", std::process::id()));
        let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), dir.join("none.py"))
            .with_backend(backend);
        Engine::new(cfg)
    }

    #[test]
    fn pulls_resolve_per_backend() {
        let mlx = engine(Backend::MlxPython);
        let llama = engine(Backend::LlamaCpp);
        // Roles and families: this backend's artifact.
        assert_eq!(pull_artifact(&mlx, "fast").unwrap().id, "gemma4-e2b-it-4bit-mlx");
        assert_eq!(pull_artifact(&llama, "fast").unwrap().id, "gemma4-e2b-it-qat-q4_0-gguf");
        assert_eq!(pull_artifact(&llama, "gemma4-e4b").unwrap().id, "gemma4-e4b-it-qat-q4_0-gguf");
        // An exact artifact id is that artifact, whatever the backend.
        assert_eq!(pull_artifact(&mlx, "gemma4-e2b-it-qat-q4_0-gguf").unwrap().id, "gemma4-e2b-it-qat-q4_0-gguf");
        assert_eq!(pull_artifact(&llama, "gemma4-e2b-it-4bit-mlx").unwrap().id, "gemma4-e2b-it-4bit-mlx");
        // An embedding model id: this backend's artifact of it.
        assert_eq!(pull_artifact(&mlx, "embeddinggemma-300m-4bit").unwrap().id, "embeddinggemma-300m-4bit");
        assert_eq!(pull_artifact(&llama, "embeddinggemma-300m-4bit").unwrap().id, "embeddinggemma-300m-q8_0-gguf");
        assert_eq!(pull_artifact(&mlx, "embeddinggemma-300m-q8_0-gguf").unwrap().id, "embeddinggemma-300m-q8_0-gguf");
        assert!(matches!(
            pull_artifact(&llama, "multilingual-e5-small-mlx"),
            Err(PullError::Resolve(ResolveError::NoArtifactForBackend { .. }))
        ));
        assert_eq!(pull_artifact(&llama, "nope").unwrap_err(), PullError::Unknown("nope".into()));
        assert_eq!(pull_artifact(&llama, "embed").unwrap().id, "embeddinggemma-300m-q8_0-gguf", "the embed role");
        let spec = pull_spec(&llama, "fast").unwrap();
        assert_eq!(spec.repo_id, "google/gemma-4-E2B-it-qat-q4_0-gguf");
    }
}
