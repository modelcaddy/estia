//! ModelStore on-disk behaviour, plus the live Hugging Face downloader tests
//! (`#[ignore]` — they hit huggingface.co; run with `-- --ignored`). The live
//! tests use a ~0.5 MB public repo that still exercises what matters:
//! tree-listed sizes, LFS SHA256 verification, resume, top-up, stale partials.

use estia_engine::models::hf::{download_client, download_model, hex_digest, hf_repo_files, resolve_url, PARTIAL_REVISION_MARKER};
use estia_engine::models::{DownloadSpec, ModelStore};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const TINY_REPO: &str = "hf-internal-testing/tiny-random-gpt2";
const UA: &str = "estia tests";

fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("estia-dl-test-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn tiny_spec() -> DownloadSpec {
    DownloadSpec { id: "tiny-test-model".into(), repo_id: TINY_REPO.into(), revision: "main".into(), required_disk_bytes: 1_000_000 }
}

#[test]
fn paths_and_overrides() {
    let store = ModelStore::new("/m").with_override("x", "/elsewhere/x");
    assert_eq!(store.path("gemma4-e4b-it-4bit-mlx"), PathBuf::from("/m/gemma4-e4b-it-4bit-mlx"));
    assert_eq!(store.partial_path("gemma4-e4b-it-4bit-mlx"), PathBuf::from("/m/gemma4-e4b-it-4bit-mlx.download"));
    assert_eq!(store.path("x"), PathBuf::from("/elsewhere/x"));
    assert!(store.is_override("x"));
    assert!(!store.is_override("y"));
}

#[tokio::test]
async fn remove_deletes_a_partial_download() {
    let dir = scratch_dir("remove-partial");
    let store = ModelStore::new(dir.join("models"));
    let id = "gemma4-e2b-it-4bit-mlx";
    let partial = store.partial_path(id);
    std::fs::create_dir_all(&partial).unwrap();
    std::fs::write(partial.join("weights.safetensors"), vec![0u8; 1024]).unwrap();
    assert!(!store.is_installed(id));
    assert_eq!(store.partial_bytes_on_disk(id), Some(1024));

    let removed = store.remove(id).await.expect("remove of a partial should succeed");
    assert!(removed, "a partial on disk must count as removed");
    assert!(!partial.exists());
    assert!(!store.path(id).exists());
}

#[tokio::test]
async fn remove_deletes_a_completed_model_and_any_stale_partial() {
    let dir = scratch_dir("remove-both");
    let store = ModelStore::new(dir.join("models"));
    let id = "gemma4-e2b-it-4bit-mlx";
    let destination = store.path(id);
    let partial = store.partial_path(id);
    std::fs::create_dir_all(&destination).unwrap();
    std::fs::write(destination.join("model.safetensors"), b"ok").unwrap();
    std::fs::create_dir_all(&partial).unwrap();
    std::fs::write(partial.join("leftover"), b"stale").unwrap();
    assert!(store.is_installed(id));
    assert_eq!(store.bytes_on_disk(id), Some(2));
    assert_eq!(store.partial_bytes_on_disk(id), None, "installed → no partial reported");

    let removed = store.remove(id).await.expect("remove of an installed model should succeed");
    assert!(removed);
    assert!(!destination.exists());
    assert!(!partial.exists());
}

/// The regression the tree-listing exists for: `total_bytes` must be known
/// from the first update, or the UI can only draw an indeterminate bar for
/// the entire multi-GB download.
#[tokio::test]
#[ignore = "hits huggingface.co"]
async fn model_download_reports_a_real_total_and_completes() {
    let dir = scratch_dir("total");
    let destination = dir.join("model");
    let updates = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = updates.clone();

    let summary = download_model(destination.clone(), &tiny_spec(), UA, move |progress| {
        sink.lock().unwrap().push((progress.phase, progress.bytes_downloaded, progress.total_bytes));
    })
    .await
    .expect("download should succeed");

    let updates = updates.lock().unwrap();
    let downloading: Vec<_> = updates.iter().filter(|u| u.0 == "downloading").collect();
    assert!(!downloading.is_empty(), "no downloading updates emitted");
    for update in &downloading {
        assert!(update.2.is_some(), "downloading update had no total_bytes — the frozen-bar bug");
    }
    let mut previous = 0;
    for update in &downloading {
        assert!(update.1 >= previous, "bytes_downloaded went backwards");
        previous = update.1;
    }
    assert_eq!(updates.last().unwrap().0, "complete");
    assert!(summary.files_downloaded > 0);
    assert!(destination.join("model.safetensors").exists());
    assert!(!destination.with_extension("download").exists());
}

/// A truncated file from an interrupted attempt must be resumed, not
/// refetched, and must still pass the SHA256 check afterwards.
#[tokio::test]
#[ignore = "hits huggingface.co"]
async fn model_download_resumes_a_truncated_file() {
    let dir = scratch_dir("resume");
    let destination = dir.join("model");
    let partial = destination.with_extension("download");

    std::fs::create_dir_all(&partial).unwrap();
    std::fs::write(partial.join(PARTIAL_REVISION_MARKER), "main").unwrap();
    let client = download_client(UA).unwrap();
    let files = hf_repo_files(&client, TINY_REPO, "main").await.unwrap();
    let weights = files.iter().find(|f| f.path == "model.safetensors").expect("test repo should have model.safetensors");
    let whole = client.get(resolve_url(TINY_REPO, "main", &weights.path)).send().await.unwrap().bytes().await.unwrap();
    let cut = whole.len() / 3;
    std::fs::write(partial.join(&weights.path), &whole[..cut]).unwrap();

    download_model(destination.clone(), &tiny_spec(), UA, |_| {}).await.expect("resumed download should succeed and verify");

    let landed = std::fs::read(destination.join(&weights.path)).unwrap();
    assert_eq!(landed.len(), weights.size as usize);
    assert_eq!(landed, whole.to_vec());
    assert_eq!(hex_digest(Sha256::digest(&landed)), *weights.sha256.as_ref().expect("weights are LFS-backed"));
}

/// An install that predates a widening of the file filter gets the missing
/// metadata topped up in place — the alternative is re-downloading 3.5 GB
/// of weights to land a 17 KB template.
#[tokio::test]
#[ignore = "hits huggingface.co"]
async fn installed_model_tops_up_missing_metadata_without_refetching_weights() {
    let dir = scratch_dir("topup");
    let destination = dir.join("model");
    std::fs::create_dir_all(&destination).unwrap();

    let client = download_client(UA).unwrap();
    let files = hf_repo_files(&client, TINY_REPO, "main").await.unwrap();
    let missing = "tokenizer_config.json";
    let sentinel = b"not real weights";
    for file in &files {
        if file.path == missing {
            continue;
        }
        let target = destination.join(&file.path);
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(&target, sentinel).unwrap();
    }
    assert!(!destination.join(missing).exists());

    let summary = download_model(destination.clone(), &tiny_spec(), UA, |_| {}).await.expect("an installed model should report complete");

    assert_eq!(summary.files_downloaded, 0, "must not re-run the download");
    assert!(destination.join(missing).exists(), "missing metadata should have been topped up");
    assert_eq!(std::fs::read(destination.join("model.safetensors")).unwrap(), sentinel);
}

/// `revision` is a branch name, so `main` moves when a repo is re-uploaded
/// — a surviving partial can be a prefix of the superseded object. That must
/// self-heal (discard + refetch), not be reported as tampering.
#[tokio::test]
#[ignore = "hits huggingface.co"]
async fn model_download_discards_a_partial_from_a_superseded_upload() {
    let dir = scratch_dir("stale");
    let destination = dir.join("model");
    let partial = destination.with_extension("download");

    std::fs::create_dir_all(&partial).unwrap();
    std::fs::write(partial.join(PARTIAL_REVISION_MARKER), "main").unwrap();
    let client = download_client(UA).unwrap();
    let files = hf_repo_files(&client, TINY_REPO, "main").await.unwrap();
    let weights = files.iter().find(|f| f.path == "model.safetensors").expect("test repo should have model.safetensors");
    std::fs::write(partial.join(&weights.path), vec![0xAB_u8; weights.size as usize]).unwrap();

    download_model(destination.clone(), &tiny_spec(), UA, |_| {})
        .await
        .expect("a stale partial should be discarded and refetched, not fail");

    let landed = std::fs::read(destination.join(&weights.path)).unwrap();
    assert_ne!(landed, vec![0xAB_u8; weights.size as usize]);
    assert_eq!(hex_digest(Sha256::digest(&landed)), *weights.sha256.as_ref().unwrap());
}
