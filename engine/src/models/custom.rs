//! Bring your own model: register a local `.gguf` file as an artifact.
//!
//! An import copies (or symlinks) the file into `<models_dir>/<id>/model.gguf`
//! and writes a small manifest beside it, `estia-model.json`. At startup the
//! engine reads every manifest and registers the model, so the rest of the
//! engine resolves it like a built-in: by id, by family, through roles, and,
//! for an embedding model, through `find_embed_model` with the fingerprint
//! `<id>@llama-cpp`.
//!
//! Defaults come from the file's own metadata (see [`super::gguf`]): the
//! architecture, the context length it declares, its embedding width, whether
//! it has a chat template. Anything can be overridden in [`ImportOptions`].
//!
//! The registry is process-wide. Two engines over different model
//! directories in one process see each other's imports; ids are still
//! resolved against each engine's own store on disk.

use super::embed::{self, EmbedArch, EmbedModel, Pooling};
use super::gguf::{self, GgufMetadata};
use super::registry::{self, Artifact, Capability, Format, ModelKind, GGUF_MODEL_FILE};
use super::store::ModelStore;
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The manifest an import writes into its directory.
pub const MANIFEST_FILE: &str = "estia-model.json";

/// The context an imported generation model runs with when the caller does
/// not choose one: the file's declared context, capped here. The built-in
/// artifacts run at 32K too; files often declare 128K or more, and
/// llama-server would size its KV cache for that.
pub const DEFAULT_IMPORT_CONTEXT_CAP: u32 = 32_768;

/// How the file gets into the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportMode {
    /// Copy the file (a clone on filesystems that support it). The import
    /// survives the source being moved or deleted.
    #[default]
    Copy,
    /// Link to the file where it is. Uses no space; breaks if the source
    /// moves. Removing the import removes the link only.
    Symlink,
}

/// What to call the import and how to run it. Every field is optional; the
/// file's metadata fills the rest.
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// Directory name and artifact id. Default: the file's `general.name` or
    /// file stem, lowercased, plus `-gguf`.
    pub id: Option<String>,
    /// Default: embedding when the file looks like an encoder with pooling
    /// and no chat template, generation otherwise.
    pub kind: Option<ModelKind>,
    /// The family roles bind to. Default: the id.
    pub family: Option<String>,
    pub label: Option<String>,
    /// Context Estia runs the model with. Default: the declared context,
    /// capped at [`DEFAULT_IMPORT_CONTEXT_CAP`].
    pub context_length: Option<u32>,
    /// Vector width of an embedding model. Default: the file's
    /// `<arch>.embedding_length`. Set it when the model has a projection head
    /// that changes the width.
    pub embedding_dims: Option<usize>,
    /// Task prefixes of an embedding model. Default: none. Wrong prefixes do
    /// not fail; they quietly give worse vectors (see `models::embed`).
    pub query_prefix: Option<String>,
    pub doc_prefix: Option<String>,
    pub mode: ImportMode,
    /// Replace an earlier import with the same id.
    pub replace: bool,
}

/// The manifest: everything needed to register the import again at startup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportManifest {
    pub id: String,
    pub kind: ModelKind,
    /// Always `gguf` today.
    pub format: Format,
    pub family: String,
    pub label: String,
    /// `general.architecture`.
    #[serde(default)]
    pub architecture: Option<String>,
    /// The context Estia runs the model with (`-c`).
    #[serde(default)]
    pub context_length: Option<u32>,
    /// The context the file declares.
    #[serde(default)]
    pub declared_context_length: Option<u64>,
    /// Vector width, for an embedding model.
    #[serde(default)]
    pub embedding_dims: Option<usize>,
    #[serde(default)]
    pub pooling: Option<Pooling>,
    #[serde(default)]
    pub query_prefix: String,
    #[serde(default)]
    pub doc_prefix: String,
    #[serde(default)]
    pub has_chat_template: bool,
    /// The chat template mentions tools, so the model likely has a tool-call format.
    #[serde(default)]
    pub tools: bool,
    /// Where the file came from (absolute).
    pub source_path: String,
    /// SHA-256 of the stored file, lowercase hex.
    pub sha256: String,
    pub bytes: u64,
    pub mode: ImportMode,
    /// Unix seconds.
    #[serde(default)]
    pub imported_at: u64,
}

impl ImportManifest {
    /// The fingerprint vectors from this model carry on llama.cpp.
    pub fn fingerprint(&self) -> Option<String> {
        (self.kind == ModelKind::Embedding).then(|| format!("{}@{}", self.id, crate::backend::BACKEND_LLAMA_CPP))
    }
}

/// Import the GGUF file at `source` into `store`, register it, and return its
/// manifest. Blocking: it hashes (and by default copies) the whole file.
pub fn import_gguf(store: &ModelStore, source: &Path, opts: ImportOptions) -> Result<ImportManifest> {
    let source = std::fs::canonicalize(source).with_context(|| format!("no model file at {}", source.display()))?;
    if !source.is_file() {
        bail!("{} is not a file", source.display());
    }
    let meta = gguf::read_metadata(&source)?;
    let id = match &opts.id {
        Some(id) => id.clone(),
        None => default_id(&meta, &source),
    };
    validate_id(&id)?;
    if registry::is_builtin_id(&id) {
        bail!("`{id}` is a built-in model id; choose another id for the import");
    }
    // An artifact id is looked up before a role, so an import named `fast`
    // would take the role's requests.
    use crate::roles::{CODE, EMBED, FAST, TEXT, VISION};
    if [TEXT, FAST, VISION, EMBED, CODE].contains(&id.as_str()) {
        bail!("`{id}` is a role name; choose another id for the import");
    }

    let dest = store.path(&id);
    if store.is_override(&id) {
        bail!("`{id}` is pointed at {} by the host; choose another id", dest.display());
    }
    if dest.exists() {
        if !dest.join(MANIFEST_FILE).exists() {
            bail!("{} already exists and is not an import; choose another id", dest.display());
        }
        if !opts.replace {
            bail!("a model named `{id}` is already imported; pass replace to overwrite it");
        }
    }

    let kind = opts.kind.unwrap_or(if meta.looks_like_embedding() { ModelKind::Embedding } else { ModelKind::Generation });
    let declared = meta.context_length();
    let context_length = opts.context_length.or_else(|| declared.map(|c| c.min(DEFAULT_IMPORT_CONTEXT_CAP as u64) as u32));
    let embedding_dims = match kind {
        ModelKind::Embedding => Some(
            opts.embedding_dims
                .or_else(|| meta.embedding_length().map(|n| n as usize))
                .ok_or_else(|| anyhow!("the file does not declare its embedding width; pass embedding_dims"))?,
        ),
        ModelKind::Generation => None,
    };
    let pooling = match kind {
        ModelKind::Embedding => Some(meta.pooling_type().and_then(|p| Pooling::from_gguf(p as u32)).unwrap_or(Pooling::Mean)),
        ModelKind::Generation => None,
    };
    if kind == ModelKind::Generation && !meta.has_chat_template() {
        tracing::warn!(model = %id, "imported generation model has no chat template; llama-server will fall back to a generic one");
    }
    let tools = meta.chat_template().is_some_and(|t| t.contains("tool"));

    // Stage beside the destination and rename into place, so a failed copy
    // never leaves a half-written import that the next startup would load.
    let staging = store.models_dir().join(format!("{id}.import"));
    if staging.exists() {
        std::fs::remove_dir_all(&staging).with_context(|| format!("remove stale {}", staging.display()))?;
    }
    std::fs::create_dir_all(&staging).with_context(|| format!("create {}", staging.display()))?;
    let staged = staging.join(GGUF_MODEL_FILE);
    let result = (|| -> Result<ImportManifest> {
        match opts.mode {
            ImportMode::Copy => {
                std::fs::copy(&source, &staged).with_context(|| format!("copy {} into the model store", source.display()))?;
            }
            ImportMode::Symlink => symlink_file(&source, &staged)?,
        }
        let (sha256, bytes) = sha256_file(&staged)?;
        let manifest = ImportManifest {
            label: opts.label.clone().or_else(|| meta.name().map(str::to_string)).unwrap_or_else(|| id.clone()),
            family: opts.family.clone().unwrap_or_else(|| id.clone()),
            id: id.clone(),
            kind,
            format: Format::Gguf,
            architecture: meta.architecture().map(str::to_string),
            context_length,
            declared_context_length: declared,
            embedding_dims,
            pooling,
            query_prefix: opts.query_prefix.clone().unwrap_or_default(),
            doc_prefix: opts.doc_prefix.clone().unwrap_or_default(),
            has_chat_template: meta.has_chat_template(),
            tools,
            source_path: source.display().to_string(),
            sha256,
            bytes,
            mode: opts.mode,
            imported_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        };
        std::fs::write(staging.join(MANIFEST_FILE), serde_json::to_vec_pretty(&manifest)?).context("write the import manifest")?;
        if dest.exists() {
            std::fs::remove_dir_all(&dest).with_context(|| format!("remove the earlier import at {}", dest.display()))?;
        }
        std::fs::rename(&staging, &dest).with_context(|| format!("move the import into {}", dest.display()))?;
        Ok(manifest)
    })();
    let manifest = match result {
        Ok(m) => m,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(e);
        }
    };
    unregister(&manifest.id);
    register(&manifest);
    tracing::info!(model = %manifest.id, kind = ?manifest.kind, bytes = manifest.bytes, mode = ?manifest.mode, "model imported");
    Ok(manifest)
}

/// Read the manifest of the import in `dir`.
pub fn read_manifest(dir: &Path) -> Result<ImportManifest> {
    let path = dir.join(MANIFEST_FILE);
    let raw = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let m: ImportManifest = serde_json::from_slice(&raw).with_context(|| format!("parse {}", path.display()))?;
    validate_id(&m.id)?;
    if dir.file_name().map(|n| n != m.id.as_str()).unwrap_or(true) {
        bail!("{} names `{}`, but lives in {}", MANIFEST_FILE, m.id, dir.display());
    }
    Ok(m)
}

/// Register every import in `store` (what `Engine::new` runs). A broken
/// manifest is logged and skipped, never fatal: one bad import must not stop
/// the engine. Returns the imports it registered.
pub fn load_imported(store: &ModelStore) -> Vec<ImportManifest> {
    let Ok(entries) = std::fs::read_dir(store.models_dir()) else {
        return Vec::new();
    };
    let mut loaded = Vec::new();
    let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.join(MANIFEST_FILE).is_file()).collect();
    dirs.sort();
    for dir in dirs {
        match read_manifest(&dir) {
            Ok(m) if registry::is_builtin_id(&m.id) => {
                tracing::warn!(model = %m.id, "an import may not use a built-in id; skipped");
            }
            Ok(m) => {
                if !dir.join(GGUF_MODEL_FILE).exists() {
                    tracing::warn!(model = %m.id, source = %m.source_path, "imported model file is missing (moved source?); registered anyway so it can be removed");
                }
                register(&m);
                loaded.push(m);
            }
            Err(e) => tracing::warn!(dir = %dir.display(), error = %e, "skipped an unreadable model import"),
        }
    }
    loaded
}

/// Whether `id` in `store` is an import.
pub fn is_imported(store: &ModelStore, id: &str) -> bool {
    store.path(id).join(MANIFEST_FILE).is_file()
}

/// Make an import resolvable. Returns its generation artifact or, for an
/// embedding model, its embedding artifact.
pub fn register(m: &ImportManifest) -> &'static Artifact {
    let id = leak(&m.id);
    let artifact = Artifact {
        id,
        family: match m.kind {
            // An embedding model's artifacts carry the model id as their family.
            ModelKind::Embedding => id,
            ModelKind::Generation => leak(&m.family),
        },
        label: leak(&m.label),
        kind: m.kind,
        format: m.format,
        repo_id: "",
        revision: "",
        required_disk_bytes: m.bytes,
        capabilities: match (m.kind, m.tools) {
            (ModelKind::Embedding, _) => &[Capability::Embed],
            (ModelKind::Generation, true) => &[Capability::Text, Capability::Tools],
            (ModelKind::Generation, false) => &[Capability::Text],
        },
        context_length: m.context_length,
        license: "unknown (imported)",
        files: &[],
    };
    match m.kind {
        ModelKind::Generation => registry::register_custom_generation(artifact),
        ModelKind::Embedding => {
            let artifacts: &'static [Artifact] = Box::leak(vec![artifact].into_boxed_slice());
            let query = leak(&m.query_prefix);
            let model = EmbedModel {
                id,
                label: artifacts[0].label,
                repo_id: "",
                revision: "",
                required_disk_bytes: m.bytes,
                dims: m.embedding_dims.unwrap_or(0),
                arch: m.architecture.as_deref().map(EmbedArch::from_gguf).unwrap_or(EmbedArch::Other),
                pooling: m.pooling.unwrap_or(Pooling::Mean),
                query_prefix: query,
                doc_prefix: leak(&m.doc_prefix),
                clustering_prefix: query,
                multilingual: false,
                license: "unknown (imported)",
                artifacts,
            };
            &embed::register_custom_embed(model).artifacts[0]
        }
    }
}

/// Forget an import (after its directory is removed). Returns whether it was
/// registered.
pub fn unregister(id: &str) -> bool {
    let g = registry::unregister_custom_generation(id);
    let e = embed::unregister_custom_embed(id);
    g || e
}

fn leak(s: &str) -> &'static str {
    if s.is_empty() {
        return "";
    }
    Box::leak(s.to_string().into_boxed_str())
}

/// Ids are directory names and appear in fingerprints: lowercase ASCII
/// letters, digits, `.`, `_` and `-`, starting with a letter or digit, and
/// never ending in a suffix the store uses for its own staging directories.
pub fn validate_id(id: &str) -> Result<()> {
    let ok_chars = id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-'));
    let ok_start = id.bytes().next().is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
    let reserved = [".download", ".import", ".partial"].iter().any(|s| id.ends_with(s));
    if id.is_empty() || id.len() > 96 || !ok_chars || !ok_start || reserved || id.contains('@') {
        bail!("invalid model id `{id}`: use 1-96 lowercase letters, digits, `.`, `_` or `-`, starting with a letter or digit");
    }
    Ok(())
}

/// `general.name` or the file stem, as an id: lowercase, runs of anything
/// else collapsed to `-`, with `-gguf` appended.
fn default_id(meta: &GgufMetadata, source: &Path) -> String {
    let stem = source.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let base = meta.name().filter(|n| !n.trim().is_empty()).map(str::to_string).unwrap_or(stem);
    let mut out = String::new();
    for c in base.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_' {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let mut out = out.trim_matches(|c| c == '-' || c == '.' || c == '_').to_string();
    out.truncate(80);
    if out.is_empty() {
        out = "imported".into();
    }
    if !out.ends_with("-gguf") {
        out.push_str("-gguf");
    }
    out
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; 1024 * 1024];
    let mut total = 0_u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((super::hf::hex_digest(hasher.finalize()), total))
}

#[cfg(unix)]
fn symlink_file(src: &Path, dst: &Path) -> Result<()> {
    std::os::unix::fs::symlink(src, dst).with_context(|| format!("link {} to {}", dst.display(), src.display()))
}

#[cfg(windows)]
fn symlink_file(src: &Path, dst: &Path) -> Result<()> {
    std::os::windows::fs::symlink_file(src, dst).with_context(|| {
        format!("link {} to {} (Windows needs Developer Mode or admin rights for symlinks; import with copy)", dst.display(), src.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::models::gguf::tests::synthetic_gguf;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("estia-import-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn ids_are_checked_and_derived() {
        for good in ["my-model-gguf", "a", "qwen3.5-7b_q4-gguf"] {
            validate_id(good).unwrap();
        }
        for bad in ["", "My-Model", "../x", "-x", "a/b", "x@llama-cpp", "x.download", "x.import", &"a".repeat(97)] {
            assert!(validate_id(bad).is_err(), "{bad}");
        }
        let meta = gguf::read_from(std::io::Cursor::new(synthetic_gguf("gemma3", 8192, 640, None, None))).unwrap();
        assert_eq!(default_id(&meta, Path::new("/x/whatever.gguf")), "synthetic-test-model-gguf");
    }

    #[test]
    fn import_registers_a_generation_model_and_survives_a_restart() {
        let dir = scratch("gen");
        let src = dir.join("tiny chat.gguf");
        std::fs::write(
            &src,
            synthetic_gguf("gemma3", 131_072, 640, Some("{% for m in messages %}{{ m.content }}{% endfor %}{{ tools }}"), None),
        )
        .unwrap();
        let store = ModelStore::new(dir.join("models"));
        let m = import_gguf(&store, &src, ImportOptions { id: Some("import-test-gen-gguf".into()), ..Default::default() }).unwrap();
        assert_eq!(m.kind, ModelKind::Generation);
        assert_eq!(m.context_length, Some(DEFAULT_IMPORT_CONTEXT_CAP), "the declared 128K is capped");
        assert_eq!(m.declared_context_length, Some(131_072));
        assert_eq!(m.family, "import-test-gen-gguf");
        assert!(m.has_chat_template && m.tools);
        assert_eq!(m.sha256.len(), 64);
        assert!(store.is_installed(&m.id));
        assert_eq!(store.load_path(&m.id), store.path(&m.id).join("model.gguf"));
        assert!(is_imported(&store, &m.id));

        let a = registry::resolve_generation("import-test-gen-gguf", Backend::LlamaCpp).unwrap();
        assert_eq!(a.context_length, Some(DEFAULT_IMPORT_CONTEXT_CAP));
        assert!(a.has(Capability::Tools));
        assert!(matches!(
            registry::resolve_generation("import-test-gen-gguf", Backend::MlxPython),
            Err(registry::ResolveError::WrongFormat { .. })
        ));

        // Same id again: refused without replace, accepted with it.
        assert!(import_gguf(&store, &src, ImportOptions { id: Some(m.id.clone()), ..Default::default() }).is_err());
        import_gguf(
            &store,
            &src,
            ImportOptions { id: Some(m.id.clone()), replace: true, context_length: Some(4096), ..Default::default() },
        )
        .unwrap();
        assert_eq!(registry::find_artifact(&m.id).unwrap().context_length, Some(4096));

        // A restart: forget it, then load the manifests again.
        assert!(unregister(&m.id));
        assert!(registry::find_artifact(&m.id).is_none());
        let loaded = load_imported(&store);
        assert_eq!(loaded.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), vec!["import-test-gen-gguf"]);
        assert_eq!(registry::find_artifact(&m.id).unwrap().context_length, Some(4096));

        // Built-in ids, role names and non-GGUF files are refused.
        assert!(import_gguf(&store, &src, ImportOptions { id: Some("gemma4-e2b-it-qat-q4_0-gguf".into()), ..Default::default() }).is_err());
        assert!(import_gguf(&store, &src, ImportOptions { id: Some("fast".into()), ..Default::default() }).is_err());
        let not_gguf = dir.join("x.gguf");
        std::fs::write(&not_gguf, b"not a model").unwrap();
        assert!(import_gguf(&store, &not_gguf, ImportOptions { id: Some("import-test-bad".into()), ..Default::default() }).is_err());
        assert!(!store.path("import-test-bad").exists() && !store.models_dir().join("import-test-bad.import").exists());

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        assert!(rt.block_on(store.remove(&m.id)).unwrap());
        assert!(registry::find_artifact(&m.id).is_none(), "remove unregisters");
        assert!(src.exists(), "the source is untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn import_registers_an_embedding_model_by_symlink() {
        let dir = scratch("embed");
        let src = dir.join("minilm.gguf");
        std::fs::write(&src, synthetic_gguf("bert", 512, 384, None, Some(1))).unwrap();
        let store = ModelStore::new(dir.join("models"));
        let m = import_gguf(
            &store,
            &src,
            ImportOptions {
                id: Some("import-test-embed-gguf".into()),
                mode: ImportMode::Symlink,
                query_prefix: Some("query: ".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(m.kind, ModelKind::Embedding);
        assert_eq!(m.embedding_dims, Some(384));
        assert_eq!(m.pooling, Some(Pooling::Mean));
        assert_eq!(m.fingerprint().as_deref(), Some("import-test-embed-gguf@llama-cpp"));
        assert!(std::fs::symlink_metadata(store.path(&m.id).join("model.gguf")).unwrap().file_type().is_symlink());

        let model = embed::find_embed_model("import-test-embed-gguf@llama-cpp").unwrap();
        assert_eq!(model.dims, 384);
        assert_eq!(model.arch, EmbedArch::Bert);
        assert_eq!(model.query_prefix, "query: ");
        assert_eq!(model.fingerprint_on(Backend::LlamaCpp).as_deref(), Some("import-test-embed-gguf@llama-cpp"));
        assert!(model.artifact_for(Backend::MlxPython).is_none());
        assert!(registry::find_artifact(&m.id).is_none(), "an embedding import never resolves for chat");
        assert_eq!(registry::find_any_artifact(&m.id).map(|a| a.kind), Some(ModelKind::Embedding));

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        assert!(rt.block_on(store.remove(&m.id)).unwrap());
        assert!(src.exists(), "removing a symlinked import leaves the source");
        assert!(embed::find_embed_model("import-test-embed-gguf").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real test models, when the llama test assets are present.
    #[test]
    fn imports_the_local_test_models() {
        let (Ok(chat), Ok(emb)) = (std::env::var("ESTIA_LLAMA_TEST_MODEL"), std::env::var("ESTIA_LLAMA_TEST_EMBED_MODEL")) else {
            return;
        };
        let dir = scratch("real");
        let store = ModelStore::new(dir.join("models"));
        let c = import_gguf(&store, Path::new(&chat), ImportOptions { mode: ImportMode::Symlink, ..Default::default() }).unwrap();
        assert_eq!(c.kind, ModelKind::Generation);
        assert!(c.has_chat_template);
        let e = import_gguf(&store, Path::new(&emb), ImportOptions { mode: ImportMode::Symlink, ..Default::default() }).unwrap();
        assert_eq!(e.kind, ModelKind::Embedding);
        assert_eq!(e.embedding_dims, Some(384));
        eprintln!("imported {} ({:?}) and {} ({:?})", c.id, c.context_length, e.id, e.fingerprint());
        unregister(&c.id);
        unregister(&e.id);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
