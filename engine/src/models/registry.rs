//! Built-in artifacts and the families they belong to.

use serde::Serialize;

/// Weight format, which decides which backend can load an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Mlx,
    Gguf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
    pub repo_id: &'static str,
    pub revision: &'static str,
    pub required_disk_bytes: u64,
    pub capabilities: &'static [Capability],
    /// Tokens of context the model was trained for, when known.
    pub context_length: Option<u32>,
    /// SPDX-style licence identifier or the vendor's licence name, shown to
    /// the user before a pull.
    pub license: &'static str,
}

impl Artifact {
    pub fn has(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }
}

const GEMMA4_CAPS: &[Capability] = &[Capability::Text, Capability::Vision];

/// Every generation artifact the engine ships knowledge of, in the order a
/// picker should list them.
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
        license: "Gemma Terms of Use",
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
        license: "Gemma Terms of Use",
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
        license: "Gemma Terms of Use",
    },
];

pub const DEFAULT_GENERATION_MODEL_ID: &str = "gemma4-e4b-it-4bit-mlx";

/// Look an artifact up by its exact id.
pub fn find_artifact(id: &str) -> Option<&'static Artifact> {
    GENERATION_MODELS.iter().find(|m| m.id == id)
}

/// Does any artifact of `family` exist?
pub fn family_known(family: &str) -> bool {
    GENERATION_MODELS.iter().any(|m| m.family == family)
}

/// Does `family` have `cap` (on any of its artifacts)?
pub fn family_has(family: &str, cap: Capability) -> bool {
    GENERATION_MODELS.iter().any(|m| m.family == family && m.has(cap))
}

/// The first artifact of a family in the given format — what a client that
/// asked by family gets on a host that runs that format.
pub fn find_family_default(family: &str, format: Format) -> Option<&'static Artifact> {
    GENERATION_MODELS.iter().find(|m| m.family == family && m.format == format)
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
        assert!(find_family_default("gemma4-e2b", Format::Gguf).is_none());
        assert!(GENERATION_MODELS.iter().all(|m| m.has(Capability::Text)));
    }
}
