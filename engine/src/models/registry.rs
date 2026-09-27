//! Built-in artifacts and the families they belong to, plus the models a user
//! imported (see [`super::custom`]).
//!
//! A family (`gemma4-e4b`) is what a client or a role asks for. An artifact
//! (`gemma4-e4b-it-4bit-mlx`, `gemma4-e4b-it-qat-q4_0-gguf`) is one set of
//! weights in one format, and a backend loads only its own format
//! ([`Backend::format`]). So a family resolves to a different artifact on
//! each backend, and one role table works everywhere.

use crate::backend::Backend;
use serde::{Deserialize, Serialize};
use std::sync::{OnceLock, RwLock};

/// Weight format, which decides which backend can load an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Mlx,
    Gguf,
}

impl Format {
    pub fn id(self) -> &'static str {
        match self {
            Format::Mlx => "mlx",
            Format::Gguf => "gguf",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Generation,
    Embedding,
}

/// What an artifact can do. Roles check these at bind time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Text,
    Vision,
    Embed,
    /// Has a native tool-call format in its chat template.
    Tools,
}

/// The name a GGUF artifact's weights are stored under, inside
/// `<models_dir>/<artifact id>/`. Fixed, so the llama adapter never guesses.
pub const GGUF_MODEL_FILE: &str = "model.gguf";

/// The image projector of a multimodal GGUF artifact, beside `model.gguf`.
/// The llama.cpp adapter starts `llama-server --mmproj` with it when present
/// (the same name is fixed in `estia-llama`).
pub const GGUF_MMPROJ_FILE: &str = "mmproj.gguf";

/// One file of an artifact that lists its files explicitly (every GGUF
/// artifact does: a GGUF repository often holds many quantisations).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ArtifactFile {
    /// Path in the repository at the artifact's `revision`.
    pub remote: &'static str,
    /// Name the store saves it under, inside `<models_dir>/<artifact id>/`.
    pub local: &'static str,
    pub bytes: u64,
    /// SHA-256 of the file, lowercase hex (the Hugging Face LFS `oid`).
    pub sha256: &'static str,
}

/// One downloadable set of weights.
#[derive(Debug, Clone, Serialize)]
pub struct Artifact {
    /// Directory name under the models directory. Stable: it is also the id
    /// stored in every provenance record, so it never changes for an
    /// existing artifact.
    pub id: &'static str,
    /// The family this artifact realises (`gemma4-e4b`). Clients ask by family.
    pub family: &'static str,
    pub label: &'static str,
    pub kind: ModelKind,
    pub format: Format,
    /// Empty for an imported model: it was never downloaded.
    pub repo_id: &'static str,
    /// A commit for GGUF artifacts; `main` for the older MLX ones.
    pub revision: &'static str,
    pub required_disk_bytes: u64,
    pub capabilities: &'static [Capability],
    /// Tokens of context Estia runs the model with. For GGUF artifacts this
    /// is what the llama adapter passes as `-c`: the files declare 128K or
    /// 256K, and llama-server would size its KV cache for that.
    pub context_length: Option<u32>,
    /// SPDX licence identifier or the vendor's licence name. `estia pull` and
    /// `estia setup` print it before a download (with the terms' links for a
    /// vendor licence), and `estia models` and the test client list it.
    pub license: &'static str,
    /// The files to fetch and the names to store them under. Empty means
    /// "the repository's model files" (the MLX artifacts: see
    /// `hf::should_download_hf_file`).
    pub files: &'static [ArtifactFile],
}

impl Artifact {
    pub fn has(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }

    /// The backend that loads this artifact.
    pub fn backend(&self) -> Backend {
        match self.format {
            Format::Mlx => Backend::MlxPython,
            Format::Gguf => Backend::LlamaCpp,
        }
    }

    /// Whether the store fetches exactly [`Artifact::files`] rather than the
    /// repository's model files.
    pub fn has_explicit_files(&self) -> bool {
        !self.files.is_empty()
    }

    /// The file a runner loads, relative to the artifact's directory:
    /// `model.gguf` for GGUF artifacts, `None` for MLX (the runner takes the
    /// directory).
    pub fn model_file(&self) -> Option<&'static str> {
        match self.format {
            Format::Gguf => Some(GGUF_MODEL_FILE),
            Format::Mlx => None,
        }
    }

    /// Bytes of the files the store fetches, when they are listed.
    pub fn listed_bytes(&self) -> Option<u64> {
        self.has_explicit_files().then(|| self.files.iter().map(|f| f.bytes).sum())
    }
}

/// Gemma 4, on either backend: text, tool calls in its chat template's
/// native `<|tool_call>` syntax (the MLX runner renders and parses it;
/// llama-server parses it for the GGUF artifacts), and images: the MLX
/// artifacts carry their vision tower, and every GGUF artifact fetches its
/// `mmproj` projector as `mmproj.gguf`.
const GEMMA4_CAPS: &[Capability] = &[Capability::Text, Capability::Tools, Capability::Vision];

/// Every generation artifact the engine ships knowledge of, in the order a
/// picker should list them. Per family, the first artifact of a format is
/// that format's default.
///
/// `license` is the licence of Google's base model: google/gemma-4-E4B-it,
/// -E2B-it and -12B-it are Apache-2.0 (license_link
/// <https://ai.google.dev/gemma/docs/gemma_4_license>), and a conversion does
/// not change it. The Hugging Face cards of the MLX conversions
/// (mlx-community/gemma-4-e4b-it-4bit, mlx-community/gemma-4-e2b-it-4bit,
/// modelcaddy/gemma-4-12b-it-qat-4bit-mlx) still carry `license: gemma`, the
/// tag of the earlier Gemma Terms of Use; that tag is stale, not a different
/// licence. The GGUF artifacts come from Google's own repositories, whose
/// cards say apache-2.0.
pub const GENERATION_MODELS: &[Artifact] = &[
    Artifact {
        id: "gemma4-e4b-it-4bit-mlx",
        family: "gemma4-e4b",
        label: "Gemma 4 E4B IT 4-bit",
        kind: ModelKind::Generation,
        format: Format::Mlx,
        repo_id: "mlx-community/gemma-4-e4b-it-4bit",
        revision: "main",
        required_disk_bytes: 5_220_000_000,
        capabilities: GEMMA4_CAPS,
        context_length: Some(32_768),
        license: "Apache-2.0",
        files: &[],
    },
    Artifact {
        id: "gemma4-12b-it-qat-4bit-mlx",
        family: "gemma4-12b-qat",
        label: "Gemma 4 12B QAT 4-bit",
        kind: ModelKind::Generation,
        format: Format::Mlx,
        // A uniform-4-bit conversion of Google's QAT checkpoint (see the
        // repo's model card for provenance). Must stay public: downloads are
        // unauthenticated. Loading it needs mlx-vlm >= 0.6.x
        // (`gemma4_unified` architecture); the runtime installer enforces
        // the floor.
        repo_id: "modelcaddy/gemma-4-12b-it-qat-4bit-mlx",
        revision: "main",
        required_disk_bytes: 6_780_000_000,
        capabilities: GEMMA4_CAPS,
        context_length: Some(32_768),
        license: "Apache-2.0",
        files: &[],
    },
    Artifact {
        id: "gemma4-e2b-it-4bit-mlx",
        family: "gemma4-e2b",
        label: "Gemma 4 E2B IT 4-bit",
        kind: ModelKind::Generation,
        format: Format::Mlx,
        repo_id: "mlx-community/gemma-4-e2b-it-4bit",
        revision: "main",
        // 3.58 GB, not the 1.6 GB this used to claim: E2B is a MatFormer
        // slice of a ~5B-parameter model, so the 4-bit weights are 3.55 GB on
        // their own. The old figure made the download look stalled long
        // before it was and understated the disk it needs.
        required_disk_bytes: 3_600_000_000,
        capabilities: GEMMA4_CAPS,
        context_length: Some(32_768),
        license: "Apache-2.0",
        files: &[],
    },
    // Google's own quantisation-aware-trained Q4_0 GGUF files (not gated,
    // Apache-2.0 per the model cards). Revisions are commits; sizes and
    // SHA-256 are from the Hugging Face tree listing at that commit. Each
    // repository also holds the model's image projector, fetched as
    // `mmproj.gguf` (about 1 GB for E2B and E4B, 175 MB for 12B).
    Artifact {
        id: "gemma4-e4b-it-qat-q4_0-gguf",
        family: "gemma4-e4b",
        label: "Gemma 4 E4B IT QAT Q4_0 (GGUF)",
        kind: ModelKind::Generation,
        format: Format::Gguf,
        repo_id: "google/gemma-4-E4B-it-qat-q4_0-gguf",
        revision: "4b4a2c1d584be7264f87aac328a1bc739ce81b6c",
        required_disk_bytes: 6_146_493_536,
        capabilities: GEMMA4_CAPS,
        context_length: Some(32_768),
        license: "Apache-2.0",
        files: &[
            ArtifactFile {
                remote: "gemma-4-E4B_q4_0-it.gguf",
                local: GGUF_MODEL_FILE,
                bytes: 5_154_941_280,
                sha256: "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee",
            },
            ArtifactFile {
                remote: "gemma-4-E4B-it-mmproj.gguf",
                local: GGUF_MMPROJ_FILE,
                bytes: 991_552_256,
                sha256: "7498a37cb619e55f2fcf87eb931f56e99389ed6d432e4c5c66110694c0d65578",
            },
        ],
    },
    Artifact {
        id: "gemma4-12b-it-qat-q4_0-gguf",
        family: "gemma4-12b-qat",
        label: "Gemma 4 12B IT QAT Q4_0 (GGUF)",
        kind: ModelKind::Generation,
        format: Format::Gguf,
        repo_id: "google/gemma-4-12B-it-qat-q4_0-gguf",
        revision: "29d097773436b69ff9feafd636ab4cf873786537",
        required_disk_bytes: 7_150_994_912,
        capabilities: GEMMA4_CAPS,
        context_length: Some(32_768),
        license: "Apache-2.0",
        files: &[
            ArtifactFile {
                remote: "gemma-4-12b-it-qat-q4_0.gguf",
                local: GGUF_MODEL_FILE,
                bytes: 6_975_879_296,
                sha256: "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b",
            },
            ArtifactFile {
                remote: "mmproj-gemma-4-12b-it-qat-q4_0.gguf",
                local: GGUF_MMPROJ_FILE,
                bytes: 175_115_616,
                sha256: "cb018338a7538a9814d994bfe54644c71eb7ed54e31eae2f721e45fd3c260da7",
            },
        ],
    },
    Artifact {
        id: "gemma4-e2b-it-qat-q4_0-gguf",
        family: "gemma4-e2b",
        label: "Gemma 4 E2B IT QAT Q4_0 (GGUF)",
        kind: ModelKind::Generation,
        format: Format::Gguf,
        repo_id: "google/gemma-4-E2B-it-qat-q4_0-gguf",
        revision: "675cff42a74c774d6cb76f76d8eacb49b48c9b93",
        required_disk_bytes: 4_336_349_920,
        capabilities: GEMMA4_CAPS,
        context_length: Some(32_768),
        license: "Apache-2.0",
        files: &[
            ArtifactFile {
                remote: "gemma-4-E2B_q4_0-it.gguf",
                local: GGUF_MODEL_FILE,
                bytes: 3_349_516_256,
                sha256: "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634",
            },
            ArtifactFile {
                remote: "gemma-4-E2B-it-mmproj.gguf",
                local: GGUF_MMPROJ_FILE,
                bytes: 986_833_664,
                sha256: "021059cce659fe7f9170d5599761d7bbaf644b798dab9503aca30dc43e6beb14",
            },
        ],
    },
];

pub const DEFAULT_GENERATION_MODEL_ID: &str = "gemma4-e4b-it-4bit-mlx";

// ── Imported models ───────────────────────────────────────────────────────────
// A model the user imported (`custom::import_gguf`) is registered here at
// startup and then resolves like a built-in. Artifacts are `&'static` all
// over the engine, so each imported one is leaked once: a handful of small
// records per process.

fn custom_generation() -> &'static RwLock<Vec<&'static Artifact>> {
    static SET: OnceLock<RwLock<Vec<&'static Artifact>>> = OnceLock::new();
    SET.get_or_init(Default::default)
}

/// Make an imported generation artifact resolvable. Replaces an earlier
/// registration with the same id; returns the registered record.
pub(crate) fn register_custom_generation(a: Artifact) -> &'static Artifact {
    let mut set = custom_generation().write().unwrap_or_else(|e| e.into_inner());
    if let Some(existing) = set.iter().find(|x| x.id == a.id) {
        if same_artifact(existing, &a) {
            return existing;
        }
    }
    let leaked: &'static Artifact = Box::leak(Box::new(a));
    set.retain(|x| x.id != leaked.id);
    set.push(leaked);
    leaked
}

pub(crate) fn unregister_custom_generation(id: &str) -> bool {
    let mut set = custom_generation().write().unwrap_or_else(|e| e.into_inner());
    let before = set.len();
    set.retain(|x| x.id != id);
    set.len() != before
}

fn same_artifact(a: &Artifact, b: &Artifact) -> bool {
    a.id == b.id
        && a.family == b.family
        && a.label == b.label
        && a.kind == b.kind
        && a.format == b.format
        && a.context_length == b.context_length
        && a.required_disk_bytes == b.required_disk_bytes
        && a.capabilities == b.capabilities
}

/// Imported generation artifacts, in registration order.
pub fn custom_generation_artifacts() -> Vec<&'static Artifact> {
    custom_generation().read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Every generation artifact: the built-ins, then the imported ones.
pub fn generation_artifacts() -> Vec<&'static Artifact> {
    GENERATION_MODELS.iter().chain(custom_generation_artifacts()).collect()
}

/// Is `id` a built-in artifact of any kind (generation or embedding)?
/// Imports may not take these ids.
pub fn is_builtin_id(id: &str) -> bool {
    GENERATION_MODELS.iter().any(|a| a.id == id)
        || super::embed::EMBEDDING_MODELS.iter().any(|m| m.id == id || m.artifacts.iter().any(|a| a.id == id))
}

// ── Lookup ────────────────────────────────────────────────────────────────────

/// Look a generation artifact up by its exact id (built-in or imported).
pub fn find_artifact(id: &str) -> Option<&'static Artifact> {
    GENERATION_MODELS.iter().find(|m| m.id == id).or_else(|| custom_generation_artifacts().into_iter().find(|m| m.id == id))
}

/// Look any artifact up by id: generation (built-in or imported) or one of an
/// embedding model's per-backend artifacts. What the store and downloader
/// use; chat resolution uses [`find_artifact`], which never returns an
/// embedding artifact.
pub fn find_any_artifact(id: &str) -> Option<&'static Artifact> {
    find_artifact(id).or_else(|| super::embed::find_embed_artifact(id).map(|(_, a)| a))
}

/// Does any artifact of `family` exist?
pub fn family_known(family: &str) -> bool {
    generation_artifacts().iter().any(|m| m.family == family)
}

/// Does `family` have `cap` (on any of its artifacts)?
pub fn family_has(family: &str, cap: Capability) -> bool {
    generation_artifacts().iter().any(|m| m.family == family && m.has(cap))
}

/// The first artifact of a family in the given format — what a client that
/// asked by family gets on a host that runs that format. Built-ins come
/// before imports.
pub fn find_family_default(family: &str, format: Format) -> Option<&'static Artifact> {
    generation_artifacts().into_iter().find(|m| m.family == family && m.format == format)
}

/// Why a name did not resolve to an artifact the backend can load.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolveError {
    #[error("unknown model `{0}`")]
    Unknown(String),
    #[error("`{id}` is a {format} artifact and this engine runs {backend}, which loads {wanted}; ask by family or role instead", format = .format.id(), wanted = .backend.format().id())]
    WrongFormat { id: String, format: Format, backend: Backend },
    #[error("model family `{family}` has no {wanted} artifact for {backend}", wanted = .backend.format().id())]
    NoArtifactForBackend { family: String, backend: Backend },
}

/// Artifact id or family → the generation artifact `backend` loads. An exact
/// artifact id of another format is an error rather than a silent swap: the
/// caller named specific weights. Roles are the host's business; resolve them
/// to a family first (`Roles::resolve`).
pub fn resolve_generation(name: &str, backend: Backend) -> Result<&'static Artifact, ResolveError> {
    if let Some(a) = find_artifact(name) {
        if a.format == backend.format() {
            return Ok(a);
        }
        return Err(ResolveError::WrongFormat { id: a.id.to_string(), format: a.format, backend });
    }
    if let Some(a) = find_family_default(name, backend.format()) {
        return Ok(a);
    }
    if family_known(name) {
        return Err(ResolveError::NoArtifactForBackend { family: name.to_string(), backend });
    }
    Err(ResolveError::Unknown(name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_and_families_are_unique_and_the_default_exists() {
        let mut ids: Vec<_> = GENERATION_MODELS.iter().map(|m| m.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), GENERATION_MODELS.len());
        assert!(find_artifact(DEFAULT_GENERATION_MODEL_ID).is_some());
        assert_eq!(find_family_default("gemma4-e2b", Format::Mlx).map(|a| a.id), Some("gemma4-e2b-it-4bit-mlx"));
        assert!(GENERATION_MODELS.iter().all(|m| m.has(Capability::Text)));
    }

    #[test]
    fn gemma4_advertises_what_reaches_the_model() {
        // Text, tools and images reach the model on both backends, and a
        // family's formats agree.
        for a in GENERATION_MODELS {
            assert!(a.has(Capability::Text) && a.has(Capability::Tools) && a.has(Capability::Vision), "{}", a.id);
            for other in GENERATION_MODELS.iter().filter(|o| o.family == a.family) {
                assert_eq!(a.capabilities, other.capabilities, "{} vs {}", a.id, other.id);
            }
        }
        assert!(family_has("gemma4-e4b", Capability::Vision));
        let json = serde_json::to_value(find_artifact("gemma4-e4b-it-4bit-mlx").unwrap().capabilities).unwrap();
        assert_eq!(json, serde_json::json!(["text", "tools", "vision"]));
    }

    #[test]
    fn vision_gguf_artifacts_fetch_their_projector() {
        // A GGUF model reads images only with its projector beside it; the
        // disk it needs counts both files.
        for a in GENERATION_MODELS.iter().filter(|a| a.format == Format::Gguf) {
            let locals: Vec<&str> = a.files.iter().map(|f| f.local).collect();
            assert_eq!(locals, [GGUF_MODEL_FILE, GGUF_MMPROJ_FILE], "{}", a.id);
            assert_eq!(a.required_disk_bytes, a.files.iter().map(|f| f.bytes).sum::<u64>(), "{}", a.id);
            assert!(a.files.iter().all(|f| f.sha256.len() == 64), "{}", a.id);
        }
    }

    #[test]
    fn every_family_has_one_artifact_per_format() {
        for family in ["gemma4-e2b", "gemma4-e4b", "gemma4-12b-qat"] {
            for format in [Format::Mlx, Format::Gguf] {
                let n = GENERATION_MODELS.iter().filter(|a| a.family == family && a.format == format).count();
                assert_eq!(n, 1, "{family} {format:?}");
            }
        }
    }

    #[test]
    fn gguf_artifacts_pin_commits_and_list_their_files() {
        for a in GENERATION_MODELS.iter().filter(|a| a.format == Format::Gguf) {
            assert_eq!(a.revision.len(), 40, "{} must pin a commit", a.id);
            assert!(a.revision.bytes().all(|b| b.is_ascii_hexdigit()), "{}", a.id);
            assert_eq!(a.files.len(), 2, "{}: the weights and the image projector", a.id);
            let (model, projector) = (a.files[0], a.files[1]);
            assert_eq!(model.local, GGUF_MODEL_FILE);
            assert!(model.remote.ends_with(".gguf") && !model.remote.contains("mmproj"), "{}", a.id);
            assert_eq!(projector.local, GGUF_MMPROJ_FILE);
            assert!(projector.remote.ends_with(".gguf") && projector.remote.contains("mmproj"), "{}", a.id);
            for f in a.files {
                assert_eq!(f.sha256.len(), 64);
                assert!(f.sha256.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
            }
            assert_eq!(a.listed_bytes(), Some(a.required_disk_bytes));
            assert_eq!(a.license, "Apache-2.0");
            assert!(a.id.ends_with("-gguf"));
            assert_eq!(a.model_file(), Some("model.gguf"));
            assert_eq!(a.backend(), Backend::LlamaCpp);
        }
        for a in GENERATION_MODELS.iter().filter(|a| a.format == Format::Mlx) {
            assert!(a.files.is_empty() && a.model_file().is_none() && a.listed_bytes().is_none());
            assert_eq!(a.backend(), Backend::MlxPython);
        }
    }

    #[test]
    fn families_resolve_by_backend() {
        let mlx = Backend::MlxPython;
        let llama = Backend::LlamaCpp;
        assert_eq!(resolve_generation("gemma4-e2b", mlx).unwrap().id, "gemma4-e2b-it-4bit-mlx");
        assert_eq!(resolve_generation("gemma4-e2b", llama).unwrap().id, "gemma4-e2b-it-qat-q4_0-gguf");
        assert_eq!(resolve_generation("gemma4-12b-qat", llama).unwrap().id, "gemma4-12b-it-qat-q4_0-gguf");
        assert_eq!(resolve_generation("gemma4-e4b-it-qat-q4_0-gguf", llama).unwrap().id, "gemma4-e4b-it-qat-q4_0-gguf");
        assert_eq!(find_family_default("gemma4-e4b", Format::Gguf).map(|a| a.id), Some("gemma4-e4b-it-qat-q4_0-gguf"));
        // An exact id of the other format is refused, not swapped.
        let err = resolve_generation("gemma4-e2b-it-4bit-mlx", llama).unwrap_err();
        assert!(matches!(err, ResolveError::WrongFormat { format: Format::Mlx, .. }), "{err:?}");
        assert!(err.to_string().contains("ask by family"), "{err}");
        assert!(matches!(resolve_generation("gemma4-e2b-it-qat-q4_0-gguf", mlx), Err(ResolveError::WrongFormat { .. })));
        assert_eq!(resolve_generation("nope", llama).unwrap_err(), ResolveError::Unknown("nope".into()));
        // Embedding artifacts never resolve for chat.
        assert_eq!(
            resolve_generation("embeddinggemma-300m-q8_0-gguf", llama).unwrap_err(),
            ResolveError::Unknown("embeddinggemma-300m-q8_0-gguf".into())
        );
        assert!(find_any_artifact("embeddinggemma-300m-q8_0-gguf").is_some());
    }

    #[test]
    fn imported_artifacts_resolve_like_builtins() {
        let a = Artifact {
            id: "registry-test-import",
            family: "registry-test-family",
            label: "test",
            kind: ModelKind::Generation,
            format: Format::Gguf,
            repo_id: "",
            revision: "",
            required_disk_bytes: 1,
            capabilities: &[Capability::Text],
            context_length: Some(4096),
            license: "unknown",
            files: &[],
        };
        let r1 = register_custom_generation(a.clone());
        let r2 = register_custom_generation(a);
        assert!(std::ptr::eq(r1, r2), "an identical re-registration reuses the record");
        assert_eq!(find_artifact("registry-test-import").map(|a| a.id), Some("registry-test-import"));
        assert!(family_known("registry-test-family"));
        assert!(family_has("registry-test-family", Capability::Text));
        assert_eq!(resolve_generation("registry-test-family", Backend::LlamaCpp).unwrap().id, "registry-test-import");
        assert!(matches!(resolve_generation("registry-test-family", Backend::MlxPython), Err(ResolveError::NoArtifactForBackend { .. })));
        assert!(unregister_custom_generation("registry-test-import"));
        assert!(find_artifact("registry-test-import").is_none());
        assert!(!unregister_custom_generation("registry-test-import"));
    }
}
