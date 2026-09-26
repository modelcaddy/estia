//! Embedding models.
//!
//! One place owns everything that must stay consistent about an embedding
//! model: its id, the **task prefixes** its text needs, the vector width it
//! produces, its pooling, and one artifact per format that a backend can load.
//!
//! The prefixes are the reason this is a registry rather than a set of
//! constants. They are model-specific and not interchangeable — nomic wants
//! `search_query: ` / `search_document: `, e5 wants `query: ` / `passage: `,
//! EmbeddingGemma wants `task: search result | query: ` / `title: none | text: `.
//! Pairing a model with another model's prefixes does not fail, crash, or log:
//! it silently returns slightly worse vectors (measured on a real corpus: correct
//! 0.913 MRR, foreign prefixes 0.905, none 0.903, query/document swapped 0.895).
//! Small and silent is exactly the kind of regression that survives review, so
//! the pair travels with the model.
//!
//! **Model and artifacts.** The model (prefixes, dimensions, pooling, the
//! multilingual flag) is one thing; its weights come per format, as
//! [`EmbedModel::artifacts`]. A model's `id` is its MLX artifact's id for the
//! models that predate the llama.cpp backend, and the `repo_id`, `revision`
//! and `required_disk_bytes` fields describe that MLX artifact, so code and
//! stored fingerprints written before the split keep working.
//!
//! **Fingerprints** are `<artifact id>@<backend id>`: the MLX EmbeddingGemma
//! gives `embeddinggemma-300m-4bit@mlx-python`, the GGUF one
//! `embeddinggemma-300m-q8_0-gguf@llama-cpp`. Different quantisations and
//! kernels give different vector spaces, so an index built on one backend
//! must be re-embedded before the other can search it.
//!
//! Retrieval policy — the confidence floor a host serves results above — is
//! deliberately **not** here. Cosine scales differ wildly between embedders,
//! so a floor is calibrated per model *by the host that owns the index*.

use super::registry::{Artifact, ArtifactFile, Capability, Format, ModelKind, GGUF_MODEL_FILE};
use crate::backend::Backend;
use serde::{Deserialize, Serialize};
use std::sync::{OnceLock, RwLock};

/// What the text is for. Decides the prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbedTask {
    Query,
    Document,
    Clustering,
}

/// Architecture as declared by the model's `config.json` `model_type`. Both
/// MLX backends dispatch on this string, and the Python runner needs it to
/// pick a calling convention (see [`EmbedArch::needs_positional_call`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbedArch {
    Bert,
    ModernBert,
    NomicBert,
    Gemma3Text,
    /// An imported GGUF model whose architecture Estia has no MLX loader for.
    Other,
}

impl EmbedArch {
    pub fn model_type(self) -> &'static str {
        match self {
            EmbedArch::Bert => "bert",
            EmbedArch::ModernBert => "modernbert",
            EmbedArch::NomicBert => "nomic_bert",
            EmbedArch::Gemma3Text => "gemma3_text",
            EmbedArch::Other => "other",
        }
    }

    /// The architecture a GGUF file declares (`general.architecture`).
    pub fn from_gguf(arch: &str) -> EmbedArch {
        match arch {
            "bert" => EmbedArch::Bert,
            "modern-bert" => EmbedArch::ModernBert,
            "nomic-bert" => EmbedArch::NomicBert,
            "gemma-embedding" => EmbedArch::Gemma3Text,
            _ => EmbedArch::Other,
        }
    }

    /// `mlx_embeddings` 0.1.0 cannot call every architecture the same way.
    /// `gemma3_text` and `qwen3` take positional `inputs` and raise
    /// `TypeError: Model.__call__() got an unexpected keyword argument 'input_ids'`
    /// through `generate(...)`; the BERT family goes through `generate()`
    /// normally. The Python runner branches on this — the flag exists here so
    /// the branch is a property of the model, not a guess at the call site.
    pub fn needs_positional_call(self) -> bool {
        matches!(self, EmbedArch::Gemma3Text)
    }

    /// Can mlx-swift's `MLXEmbedders` load this architecture? `modernbert`
    /// cannot: `unsupportedModelType("modernbert")`.
    pub fn loads_in_mlx_swift(self) -> bool {
        !matches!(self, EmbedArch::ModernBert | EmbedArch::Other)
    }
}

/// How token vectors become one text vector. Part of the model: the same
/// weights pooled differently are a different space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    Mean,
    Cls,
    Last,
}

impl Pooling {
    /// llama-server's `--pooling` value.
    pub fn llama_arg(self) -> &'static str {
        match self {
            Pooling::Mean => "mean",
            Pooling::Cls => "cls",
            Pooling::Last => "last",
        }
    }

    /// A GGUF `<arch>.pooling_type` value (llama.cpp's enum: 1 mean, 2 cls,
    /// 3 last). `None` for "none", "rank" or unknown.
    pub fn from_gguf(v: u32) -> Option<Pooling> {
        match v {
            1 => Some(Pooling::Mean),
            2 => Some(Pooling::Cls),
            3 => Some(Pooling::Last),
            _ => None,
        }
    }
}

/// Backend identity strings that become part of an embedding fingerprint. The
/// same model run through two backends yields two different vector spaces
/// (different pooling, different quantised kernels), and cosine between them
/// is meaningless rather than merely worse.
pub const BACKEND_MLX_PYTHON: &str = "mlx-python";
pub const BACKEND_MLX_SWIFT: &str = "mlx-swift";

#[derive(Debug, Clone, Copy, Serialize)]
pub struct EmbedModel {
    /// The model's id: a stored fingerprint's stem for indexes built before
    /// the llama.cpp backend, and the id `find_embed_model` and roles use. For
    /// the built-in models it is also the MLX artifact's id (the directory
    /// the MLX weights live in); for an imported model it is the import's id.
    pub id: &'static str,
    /// User-visible name. Must describe the model honestly, language coverage
    /// included.
    pub label: &'static str,
    /// The MLX artifact's repository, revision and size, kept on the model
    /// for callers written before the split. Per-backend coordinates are in
    /// [`EmbedModel::artifacts`].
    pub repo_id: &'static str,
    pub revision: &'static str,
    pub required_disk_bytes: u64,
    /// Vector width. A change invalidates every stored vector.
    pub dims: usize,
    pub arch: EmbedArch,
    pub pooling: Pooling,
    pub query_prefix: &'static str,
    pub doc_prefix: &'static str,
    pub clustering_prefix: &'static str,
    /// True only with measured cross-lingual retrieval behind it.
    pub multilingual: bool,
    pub license: &'static str,
    /// The weights, one artifact per format (at most one each).
    pub artifacts: &'static [Artifact],
}

impl EmbedModel {
    pub fn prefix(&self, task: EmbedTask) -> &'static str {
        match task {
            EmbedTask::Query => self.query_prefix,
            EmbedTask::Document => self.doc_prefix,
            EmbedTask::Clustering => self.clustering_prefix,
        }
    }

    /// The artifact `backend` loads, if this model has one in its format.
    pub fn artifact_for(&self, backend: Backend) -> Option<&'static Artifact> {
        self.artifacts.iter().find(|a| a.format == backend.format())
    }

    /// Identity to store with every vector and compare on every query:
    /// `<artifact id>@<backend>`. `backend` is a backend id string; for
    /// `mlx-python` and `llama-cpp` the artifact is that backend's own (so
    /// `embeddinggemma-300m-4bit@mlx-python`,
    /// `embeddinggemma-300m-q8_0-gguf@llama-cpp`). Any other string (the
    /// Swift runner's `mlx-swift`) keeps the model id, as before the split.
    pub fn fingerprint_for(&self, backend: &str) -> String {
        let id = backend.parse::<Backend>().ok().and_then(|b| self.artifact_for(b)).map(|a| a.id).unwrap_or(self.id);
        format!("{id}@{backend}")
    }

    /// [`EmbedModel::fingerprint_for`] a typed backend; `None` when this model
    /// has no artifact that backend loads.
    pub fn fingerprint_on(&self, backend: Backend) -> Option<String> {
        self.artifact_for(backend).map(|a| format!("{}@{}", a.id, backend.id()))
    }

    pub fn loads_in_mlx_swift(&self) -> bool {
        self.arch.loads_in_mlx_swift()
    }
}

const EMBED_CAPS: &[Capability] = &[Capability::Embed];

/// EmbeddingGemma-300M, 4-bit MLX. Chosen on recall@k over a real corpus
/// (44 queries, 165 conversations, Greek and English) rather than published
/// benchmarks: cross-lingual r@1 0.82 against 0.45 (e5-small) and 0.32
/// (nomic ModernBERT), English-only r@1 unchanged at 0.90. Loads in both MLX
/// backends (`gemma3_text` is registered in mlx-swift's `EmbedderTypeRegistry`).
///
/// On llama.cpp it is ggml-org's Q8_0 GGUF. Not the QAT Q4_0 file: that one's
/// parameter count is exactly the two dense projection layers (2 × 768 ×
/// 3072) short, so it appears to lack them, and the MLX artifact has them.
pub const EMBEDDING_GEMMA_300M_4BIT: EmbedModel = EmbedModel {
    id: "embeddinggemma-300m-4bit",
    label: "EmbeddingGemma 300M 4-bit (multilingual)",
    repo_id: "mlx-community/embeddinggemma-300m-4bit",
    revision: "main",
    // 203 MB on disk as published; rounded up so the pre-download disk check
    // has slack for a re-quantised revision.
    required_disk_bytes: 250_000_000,
    dims: 768,
    arch: EmbedArch::Gemma3Text,
    pooling: Pooling::Mean,
    // Verbatim from the model's own `config_sentence_transformers.json` prompts.
    query_prefix: "task: search result | query: ",
    doc_prefix: "title: none | text: ",
    clustering_prefix: "task: clustering | query: ",
    multilingual: true,
    license: "Gemma Terms of Use",
    artifacts: &[
        Artifact {
            id: "embeddinggemma-300m-4bit",
            family: "embeddinggemma-300m-4bit",
            label: "EmbeddingGemma 300M 4-bit (MLX)",
            kind: ModelKind::Embedding,
            format: Format::Mlx,
            repo_id: "mlx-community/embeddinggemma-300m-4bit",
            revision: "main",
            required_disk_bytes: 250_000_000,
            capabilities: EMBED_CAPS,
            context_length: None,
            license: "Gemma Terms of Use",
            files: &[],
        },
        Artifact {
            id: "embeddinggemma-300m-q8_0-gguf",
            family: "embeddinggemma-300m-4bit",
            label: "EmbeddingGemma 300M Q8_0 (GGUF)",
            kind: ModelKind::Embedding,
            format: Format::Gguf,
            repo_id: "ggml-org/embeddinggemma-300M-GGUF",
            revision: "0f741b5a6585bd53aeb15cd1372c56f2a0f65e12",
            required_disk_bytes: 333_590_944,
            capabilities: EMBED_CAPS,
            // The model's trained context. Whether to truncate at 512 like
            // the MLX runner is the adapter's call (design, open question 8).
            context_length: Some(2048),
            license: "Gemma Terms of Use",
            files: &[ArtifactFile {
                remote: "embeddinggemma-300M-Q8_0.gguf",
                local: GGUF_MODEL_FILE,
                bytes: 333_590_944,
                sha256: "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63",
            }],
        },
    ],
};

/// Runner-up: half the vector width (384) and the fastest of the candidates,
/// but clearly behind on cross-lingual retrieval (r@1 0.45 vs 0.82). MLX only.
pub const MULTILINGUAL_E5_SMALL: EmbedModel = EmbedModel {
    id: "multilingual-e5-small-mlx",
    label: "Multilingual E5 Small (multilingual)",
    repo_id: "mlx-community/multilingual-e5-small-mlx",
    revision: "main",
    required_disk_bytes: 300_000_000,
    dims: 384,
    arch: EmbedArch::Bert,
    pooling: Pooling::Mean,
    query_prefix: "query: ",
    doc_prefix: "passage: ",
    clustering_prefix: "query: ",
    multilingual: true,
    license: "MIT",
    artifacts: &[Artifact {
        id: "multilingual-e5-small-mlx",
        family: "multilingual-e5-small-mlx",
        label: "Multilingual E5 Small (MLX)",
        kind: ModelKind::Embedding,
        format: Format::Mlx,
        repo_id: "mlx-community/multilingual-e5-small-mlx",
        revision: "main",
        required_disk_bytes: 300_000_000,
        capabilities: EMBED_CAPS,
        context_length: None,
        license: "MIT",
        files: &[],
    }],
};

/// English-only: 0/4 on the cross-lingual probe. Kept so an existing index
/// built with it can still be identified and reindexed. MLX only.
pub const NOMIC_MODERNBERT_6BIT: EmbedModel = EmbedModel {
    id: "nomicai-modernbert-embed-base-6bit",
    label: "Nomic ModernBERT Embed Base 6-bit (English)",
    repo_id: "mlx-community/nomicai-modernbert-embed-base-6bit",
    revision: "main",
    required_disk_bytes: 130_000_000,
    dims: 768,
    arch: EmbedArch::ModernBert,
    pooling: Pooling::Mean,
    query_prefix: "search_query: ",
    doc_prefix: "search_document: ",
    clustering_prefix: "clustering: ",
    multilingual: false,
    license: "Apache-2.0",
    artifacts: &[Artifact {
        id: "nomicai-modernbert-embed-base-6bit",
        family: "nomicai-modernbert-embed-base-6bit",
        label: "Nomic ModernBERT Embed Base 6-bit (MLX)",
        kind: ModelKind::Embedding,
        format: Format::Mlx,
        repo_id: "mlx-community/nomicai-modernbert-embed-base-6bit",
        revision: "main",
        required_disk_bytes: 130_000_000,
        capabilities: EMBED_CAPS,
        context_length: None,
        license: "Apache-2.0",
        files: &[],
    }],
};

/// English-only; loads in mlx-swift where ModernBERT does not. Kept for the
/// same reason as [`NOMIC_MODERNBERT_6BIT`]. On llama.cpp, nomic-ai's own
/// Q8_0 GGUF.
pub const NOMIC_EMBED_TEXT_V1_5: EmbedModel = EmbedModel {
    id: "nomic-embed-text-v1.5",
    label: "Nomic Embed Text v1.5 (English)",
    repo_id: "nomic-ai/nomic-embed-text-v1.5",
    revision: "main",
    required_disk_bytes: 600_000_000,
    dims: 768,
    arch: EmbedArch::NomicBert,
    pooling: Pooling::Mean,
    query_prefix: "search_query: ",
    doc_prefix: "search_document: ",
    clustering_prefix: "clustering: ",
    multilingual: false,
    license: "Apache-2.0",
    artifacts: &[
        Artifact {
            id: "nomic-embed-text-v1.5",
            family: "nomic-embed-text-v1.5",
            label: "Nomic Embed Text v1.5 (MLX)",
            kind: ModelKind::Embedding,
            format: Format::Mlx,
            repo_id: "nomic-ai/nomic-embed-text-v1.5",
            revision: "main",
            required_disk_bytes: 600_000_000,
            capabilities: EMBED_CAPS,
            context_length: None,
            license: "Apache-2.0",
            files: &[],
        },
        Artifact {
            id: "nomic-embed-text-v1.5-q8_0-gguf",
            family: "nomic-embed-text-v1.5",
            label: "Nomic Embed Text v1.5 Q8_0 (GGUF)",
            kind: ModelKind::Embedding,
            format: Format::Gguf,
            repo_id: "nomic-ai/nomic-embed-text-v1.5-GGUF",
            revision: "0188c9bf409793f810680a5a431e7b899c46104c",
            required_disk_bytes: 146_146_432,
            capabilities: EMBED_CAPS,
            context_length: Some(2048),
            license: "Apache-2.0",
            files: &[ArtifactFile {
                remote: "nomic-embed-text-v1.5.Q8_0.gguf",
                local: GGUF_MODEL_FILE,
                bytes: 146_146_432,
                sha256: "3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7",
            }],
        },
    ],
};

/// Every built-in embedding model, active or superseded. Used to turn a
/// fingerprint found in a host's database into something a user can be
/// told. Imported models are in [`embed_models`].
pub const EMBEDDING_MODELS: &[&EmbedModel] =
    &[&EMBEDDING_GEMMA_300M_4BIT, &MULTILINGUAL_E5_SMALL, &NOMIC_MODERNBERT_6BIT, &NOMIC_EMBED_TEXT_V1_5];

// ── Imported embedding models ─────────────────────────────────────────────────

fn custom_embed() -> &'static RwLock<Vec<&'static EmbedModel>> {
    static SET: OnceLock<RwLock<Vec<&'static EmbedModel>>> = OnceLock::new();
    SET.get_or_init(Default::default)
}

/// Make an imported embedding model resolvable (leaked once, like imported
/// generation artifacts). Replaces an earlier registration with the same id.
pub(crate) fn register_custom_embed(m: EmbedModel) -> &'static EmbedModel {
    let mut set = custom_embed().write().unwrap_or_else(|e| e.into_inner());
    if let Some(existing) = set.iter().find(|x| x.id == m.id) {
        if existing.dims == m.dims
            && existing.label == m.label
            && existing.pooling == m.pooling
            && existing.artifacts.first().map(|a| (a.context_length, a.required_disk_bytes))
                == m.artifacts.first().map(|a| (a.context_length, a.required_disk_bytes))
        {
            return existing;
        }
    }
    let leaked: &'static EmbedModel = Box::leak(Box::new(m));
    set.retain(|x| x.id != leaked.id);
    set.push(leaked);
    leaked
}

pub(crate) fn unregister_custom_embed(id: &str) -> bool {
    let mut set = custom_embed().write().unwrap_or_else(|e| e.into_inner());
    let before = set.len();
    set.retain(|x| x.id != id);
    set.len() != before
}

/// Every embedding model: the built-ins, then the imported ones.
pub fn embed_models() -> Vec<&'static EmbedModel> {
    let custom = custom_embed().read().unwrap_or_else(|e| e.into_inner()).clone();
    EMBEDDING_MODELS.iter().copied().chain(custom).collect()
}

/// Look a model up from a stored fingerprint (`<artifact id>@<backend>`), a
/// model id, or any of its artifact ids. Fingerprints written before the
/// backend suffix existed are bare ids, so both forms resolve.
pub fn find_embed_model(fingerprint: &str) -> Option<&'static EmbedModel> {
    let id = fingerprint.split('@').next().unwrap_or(fingerprint);
    embed_models().into_iter().find(|spec| spec.id == id || spec.artifacts.iter().any(|a| a.id == id))
}

/// The model and artifact an artifact id names.
pub fn find_embed_artifact(artifact_id: &str) -> Option<(&'static EmbedModel, &'static Artifact)> {
    embed_models().into_iter().find_map(|m| m.artifacts.iter().find(|a| a.id == artifact_id).map(|a| (m, a)))
}

/// A model id, artifact id or fingerprint → the model and the artifact
/// `backend` loads for it.
pub fn resolve_embed(name: &str, backend: Backend) -> Result<(&'static EmbedModel, &'static Artifact), super::registry::ResolveError> {
    let model = find_embed_model(name).ok_or_else(|| super::registry::ResolveError::Unknown(name.to_string()))?;
    let artifact = model
        .artifact_for(backend)
        .ok_or_else(|| super::registry::ResolveError::NoArtifactForBackend { family: model.id.to_string(), backend })?;
    Ok((model, artifact))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regression this module exists to prevent: a model carrying another
    /// family's prefixes. Each pair is asserted against the family that
    /// documents it, so swapping a model id without swapping its prefixes
    /// fails here.
    #[test]
    fn every_model_carries_its_own_prefix_pair() {
        let expected: &[(&str, &str, &str)] = &[
            ("embeddinggemma-300m-4bit", "task: search result | query: ", "title: none | text: "),
            ("multilingual-e5-small-mlx", "query: ", "passage: "),
            ("nomicai-modernbert-embed-base-6bit", "search_query: ", "search_document: "),
            ("nomic-embed-text-v1.5", "search_query: ", "search_document: "),
        ];
        for (id, query, doc) in expected {
            let spec = find_embed_model(id).unwrap_or_else(|| panic!("{id} missing"));
            assert_eq!(spec.query_prefix, *query, "{id} query prefix");
            assert_eq!(spec.doc_prefix, *doc, "{id} document prefix");
        }
        assert_eq!(EMBEDDING_MODELS.len(), expected.len(), "a model was added without its prefix pair");
    }

    /// A query prefixed as a document (or vice versa) is the worst of the
    /// silent prefix failures. It is only detectable because the two differ.
    #[test]
    fn query_and_document_prefixes_are_distinguishable() {
        for spec in EMBEDDING_MODELS {
            assert_ne!(spec.query_prefix, spec.doc_prefix, "{}", spec.id);
        }
    }

    #[test]
    fn prefix_dispatch_and_fingerprints() {
        let spec = &EMBEDDING_GEMMA_300M_4BIT;
        assert_eq!(spec.prefix(EmbedTask::Query), spec.query_prefix);
        assert_eq!(spec.prefix(EmbedTask::Document), spec.doc_prefix);
        assert_eq!(spec.prefix(EmbedTask::Clustering), spec.clustering_prefix);
        let fp = spec.fingerprint_for(BACKEND_MLX_PYTHON);
        assert_eq!(fp, "embeddinggemma-300m-4bit@mlx-python");
        assert_eq!(find_embed_model(&fp).map(|s| s.id), Some(spec.id));
        assert_eq!(find_embed_model("nomicai-modernbert-embed-base-6bit").map(|s| s.id), Some(NOMIC_MODERNBERT_6BIT.id));
        assert!(find_embed_model("nobody").is_none());
        assert!(!NOMIC_MODERNBERT_6BIT.loads_in_mlx_swift());
        assert!(EMBEDDING_GEMMA_300M_4BIT.loads_in_mlx_swift());
    }

    /// Fingerprints are `<artifact id>@<backend id>`, and an index stamped on
    /// one backend never matches the other.
    #[test]
    fn fingerprints_name_the_backend_artifact() {
        let g = &EMBEDDING_GEMMA_300M_4BIT;
        assert_eq!(g.fingerprint_for("llama-cpp"), "embeddinggemma-300m-q8_0-gguf@llama-cpp");
        assert_eq!(g.fingerprint_on(Backend::LlamaCpp).as_deref(), Some("embeddinggemma-300m-q8_0-gguf@llama-cpp"));
        assert_eq!(g.fingerprint_on(Backend::MlxPython).as_deref(), Some("embeddinggemma-300m-4bit@mlx-python"));
        assert_eq!(g.fingerprint_for(BACKEND_MLX_SWIFT), "embeddinggemma-300m-4bit@mlx-swift");
        assert_ne!(g.fingerprint_on(Backend::LlamaCpp), g.fingerprint_on(Backend::MlxPython));
        // Either fingerprint, and either artifact id, finds the model.
        for name in ["embeddinggemma-300m-q8_0-gguf@llama-cpp", "embeddinggemma-300m-q8_0-gguf", "embeddinggemma-300m-4bit@mlx-python"] {
            assert_eq!(find_embed_model(name).map(|m| m.id), Some("embeddinggemma-300m-4bit"), "{name}");
        }
        assert_eq!(NOMIC_EMBED_TEXT_V1_5.fingerprint_on(Backend::LlamaCpp).as_deref(), Some("nomic-embed-text-v1.5-q8_0-gguf@llama-cpp"));
        // A model with no GGUF artifact has no llama fingerprint.
        assert!(MULTILINGUAL_E5_SMALL.fingerprint_on(Backend::LlamaCpp).is_none());
        assert_eq!(MULTILINGUAL_E5_SMALL.fingerprint_for("llama-cpp"), "multilingual-e5-small-mlx@llama-cpp");
    }

    #[test]
    fn embed_models_resolve_by_backend() {
        let (m, a) = resolve_embed("embeddinggemma-300m-4bit", Backend::LlamaCpp).unwrap();
        assert_eq!((m.id, a.id), ("embeddinggemma-300m-4bit", "embeddinggemma-300m-q8_0-gguf"));
        let (_, a) = resolve_embed("embeddinggemma-300m-q8_0-gguf@llama-cpp", Backend::MlxPython).unwrap();
        assert_eq!(a.id, "embeddinggemma-300m-4bit");
        assert!(resolve_embed("multilingual-e5-small-mlx", Backend::LlamaCpp).is_err());
        assert!(resolve_embed("nobody", Backend::LlamaCpp).is_err());
        assert_eq!(find_embed_artifact("nomic-embed-text-v1.5-q8_0-gguf").map(|(m, _)| m.id), Some("nomic-embed-text-v1.5"));
    }

    /// The fields kept on the model for older callers must agree with its MLX
    /// artifact, and every artifact must be an embedding artifact of its own
    /// format, at most one per format.
    #[test]
    fn model_fields_agree_with_their_artifacts() {
        for m in EMBEDDING_MODELS {
            let mlx = m.artifact_for(Backend::MlxPython).unwrap_or_else(|| panic!("{} has no MLX artifact", m.id));
            assert_eq!((mlx.id, mlx.repo_id, mlx.revision, mlx.required_disk_bytes), (m.id, m.repo_id, m.revision, m.required_disk_bytes));
            for f in [Format::Mlx, Format::Gguf] {
                assert!(m.artifacts.iter().filter(|a| a.format == f).count() <= 1, "{}", m.id);
            }
            for a in m.artifacts {
                assert_eq!(a.kind, ModelKind::Embedding);
                assert!(a.has(Capability::Embed));
                assert_eq!(a.family, m.id);
                if a.format == Format::Gguf {
                    assert_eq!(a.revision.len(), 40, "{} must pin a commit", a.id);
                    assert_eq!(a.listed_bytes(), Some(a.required_disk_bytes));
                    assert_eq!(a.files[0].local, GGUF_MODEL_FILE);
                }
            }
        }
        assert_eq!(EMBEDDING_GEMMA_300M_4BIT.artifact_for(Backend::LlamaCpp).unwrap().license, "Gemma Terms of Use");
    }
}
