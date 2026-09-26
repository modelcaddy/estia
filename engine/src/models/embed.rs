//! Embedding models.
//!
//! One place owns everything that must stay consistent about an embedding
//! model: its id and download coordinates, the **task prefixes** its text
//! needs, the vector width it produces, and which backends can load its
//! architecture.
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
//! Retrieval policy — the confidence floor a host serves results above — is
//! deliberately **not** here. Cosine scales differ wildly between embedders,
//! so a floor is calibrated per model *by the host that owns the index*.

use serde::Serialize;

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
}

impl EmbedArch {
    pub fn model_type(self) -> &'static str {
        match self {
            EmbedArch::Bert => "bert",
            EmbedArch::ModernBert => "modernbert",
            EmbedArch::NomicBert => "nomic_bert",
            EmbedArch::Gemma3Text => "gemma3_text",
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
        !matches!(self, EmbedArch::ModernBert)
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
    /// Directory name under the models directory, and the stem of the stored
    /// fingerprint.
    pub id: &'static str,
    /// User-visible name. Must describe the model honestly, language coverage
    /// included.
    pub label: &'static str,
    pub repo_id: &'static str,
    pub revision: &'static str,
    pub required_disk_bytes: u64,
    /// Vector width. A change invalidates every stored vector.
    pub dims: usize,
    pub arch: EmbedArch,
    pub query_prefix: &'static str,
    pub doc_prefix: &'static str,
    pub clustering_prefix: &'static str,
    /// True only with measured cross-lingual retrieval behind it.
    pub multilingual: bool,
    pub license: &'static str,
}

impl EmbedModel {
    pub fn prefix(&self, task: EmbedTask) -> &'static str {
        match task {
            EmbedTask::Query => self.query_prefix,
            EmbedTask::Document => self.doc_prefix,
            EmbedTask::Clustering => self.clustering_prefix,
        }
    }

    /// Identity to store with every vector and compare on every query: model
    /// **and** backend.
    pub fn fingerprint_for(&self, backend: &str) -> String {
        format!("{}@{}", self.id, backend)
    }

    pub fn loads_in_mlx_swift(&self) -> bool {
        self.arch.loads_in_mlx_swift()
    }
}

/// EmbeddingGemma-300M, 4-bit MLX. Chosen on recall@k over a real corpus
/// (44 queries, 165 conversations, Greek and English) rather than published
/// benchmarks: cross-lingual r@1 0.82 against 0.45 (e5-small) and 0.32
/// (nomic ModernBERT), English-only r@1 unchanged at 0.90. Loads in both MLX
/// backends (`gemma3_text` is registered in mlx-swift's `EmbedderTypeRegistry`).
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
    // Verbatim from the model's own `config_sentence_transformers.json` prompts.
    query_prefix: "task: search result | query: ",
    doc_prefix: "title: none | text: ",
    clustering_prefix: "task: clustering | query: ",
    multilingual: true,
    license: "Gemma Terms of Use",
};

/// Runner-up: half the vector width (384) and the fastest of the candidates,
/// but clearly behind on cross-lingual retrieval (r@1 0.45 vs 0.82).
pub const MULTILINGUAL_E5_SMALL: EmbedModel = EmbedModel {
    id: "multilingual-e5-small-mlx",
    label: "Multilingual E5 Small (multilingual)",
    repo_id: "mlx-community/multilingual-e5-small-mlx",
    revision: "main",
    required_disk_bytes: 300_000_000,
    dims: 384,
    arch: EmbedArch::Bert,
    query_prefix: "query: ",
    doc_prefix: "passage: ",
    clustering_prefix: "query: ",
    multilingual: true,
    license: "MIT",
};

/// English-only: 0/4 on the cross-lingual probe. Kept so an existing index
/// built with it can still be identified and reindexed.
pub const NOMIC_MODERNBERT_6BIT: EmbedModel = EmbedModel {
    id: "nomicai-modernbert-embed-base-6bit",
    label: "Nomic ModernBERT Embed Base 6-bit (English)",
    repo_id: "mlx-community/nomicai-modernbert-embed-base-6bit",
    revision: "main",
    required_disk_bytes: 130_000_000,
    dims: 768,
    arch: EmbedArch::ModernBert,
    query_prefix: "search_query: ",
    doc_prefix: "search_document: ",
    clustering_prefix: "clustering: ",
    multilingual: false,
    license: "Apache-2.0",
};

/// English-only; loads in mlx-swift where ModernBERT does not. Kept for the
/// same reason as [`NOMIC_MODERNBERT_6BIT`].
pub const NOMIC_EMBED_TEXT_V1_5: EmbedModel = EmbedModel {
    id: "nomic-embed-text-v1.5",
    label: "Nomic Embed Text v1.5 (English)",
    repo_id: "nomic-ai/nomic-embed-text-v1.5",
    revision: "main",
    required_disk_bytes: 600_000_000,
    dims: 768,
    arch: EmbedArch::NomicBert,
    query_prefix: "search_query: ",
    doc_prefix: "search_document: ",
    clustering_prefix: "clustering: ",
    multilingual: false,
    license: "Apache-2.0",
};

/// Every embedding model the engine knows how to describe, active or
/// superseded. Used to turn a fingerprint found in a host's database into
/// something a user can be told.
pub const EMBEDDING_MODELS: &[&EmbedModel] =
    &[&EMBEDDING_GEMMA_300M_4BIT, &MULTILINGUAL_E5_SMALL, &NOMIC_MODERNBERT_6BIT, &NOMIC_EMBED_TEXT_V1_5];

/// Look a model up from a stored fingerprint (`id@backend`) or a bare id.
/// Fingerprints written before the backend suffix existed are bare ids, so
/// both forms resolve.
pub fn find_embed_model(fingerprint: &str) -> Option<&'static EmbedModel> {
    let id = fingerprint.split('@').next().unwrap_or(fingerprint);
    EMBEDDING_MODELS.iter().copied().find(|spec| spec.id == id)
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
}
