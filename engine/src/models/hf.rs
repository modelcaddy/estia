//! Hugging Face downloader.
//!
//! - Sizes and LFS SHA-256 come from the tree listing, so progress has a real
//!   total from the first byte.
//! - Files above `PARALLEL_MIN_FILE_SIZE` are fetched as concurrent range
//!   chunks with a `.chunks.json` sidecar, so pause resumes at chunk
//!   granularity. Smaller files stream with `Range` resume.
//! - Every file is verified against its published hash.
//! - A partial from a superseded upload is discarded and refetched rather than
//!   reported as tampering.
//! - An installed model gets small metadata topped up in place instead of a
//!   multi-GB re-download.

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;

use super::store::dir_size;

// ── Pause / cancel ────────────────────────────────────────────────────────────
// A model download is a minutes-long await with no handle to reach it by; a
// caller's Pause button sets a flag here and the download loops poll it.
// Pausing keeps the partial files — the resumable path picks them back up on
// the next download — so pause and cancel are the same mechanism.

/// Error text a paused download fails with. Hosts match on this to show
/// "Paused" instead of an error state.
pub const DOWNLOAD_PAUSED_MARKER: &str = "download paused";

fn download_cancel_set() -> &'static std::sync::Mutex<std::collections::HashSet<String>> {
    static SET: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = std::sync::OnceLock::new();
    SET.get_or_init(Default::default)
}

/// Ask an in-flight download of `model_id` to stop at the next chunk. No-op
/// if nothing is downloading it.
pub fn request_download_cancel(model_id: &str) {
    download_cancel_set().lock().unwrap().insert(model_id.to_string());
}

fn download_cancel_requested(model_id: &str) -> bool {
    download_cancel_set().lock().unwrap().contains(model_id)
}

fn clear_download_cancel(model_id: &str) {
    download_cancel_set().lock().unwrap().remove(model_id);
}

/// What to download: identity plus the repo coordinates.
#[derive(Debug, Clone)]
pub struct DownloadSpec {
    /// Artifact id; also the cancel key and the summary's `model_id`.
    pub id: String,
    pub repo_id: String,
    pub revision: String,
    /// Quoted to the user before the tree listing is known; the listing wins.
    pub required_disk_bytes: u64,
}

impl From<&super::registry::Artifact> for DownloadSpec {
    fn from(a: &super::registry::Artifact) -> Self {
        Self {
            id: a.id.to_string(),
            repo_id: a.repo_id.to_string(),
            revision: a.revision.to_string(),
            required_disk_bytes: a.required_disk_bytes,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadSummary {
    pub model_id: String,
    pub repo_id: String,
    pub revision: String,
    pub files_downloaded: usize,
    pub bytes_downloaded: u64,
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    /// `preparing` · `downloading` · `finalizing` · `complete`.
    pub phase: &'static str,
    pub file_name: Option<String>,
    pub file_index: usize,
    pub file_count: usize,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
}

/// One file we intend to download, resolved from HF's tree API.
///
/// Sizes come from the tree listing on purpose. The model-info endpoint
/// (`/api/models/{id}/revision/{rev}`) returns `siblings` entries carrying only
/// `rfilename` — no `size` — so building the plan from it left `total_bytes`
/// unknown and the download progress bar frozen at its indeterminate width for
/// the whole multi-GB transfer. The tree listing carries size *and* the LFS
/// content hash, so one call answers both questions.
#[derive(Debug, Clone)]
pub struct HfFile {
    pub path: String,
    pub size: u64,
    /// Content SHA256 for LFS-backed files (the multi-GB weights). Plain files
    /// (configs, tokenizer json) publish no hash, so they get the size check only.
    pub sha256: Option<String>,
}

/// Entry from HF's tree API. For LFS-stored files (the multi-GB weights) `lfs`
/// carries the content SHA256 as `oid` — what we verify downloads against.
#[derive(Debug, Deserialize)]
struct HfTreeEntry {
    path: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<HfLfs>,
}

#[derive(Debug, Deserialize)]
struct HfLfs {
    /// Content SHA256 (hex) of the LFS object.
    oid: String,
}

/// The download plan for a repo revision: every file we keep, with its exact
/// byte size and (for LFS objects) its verification hash.
pub async fn hf_repo_files(client: &Client, repo_id: &str, revision: &str) -> Result<Vec<HfFile>> {
    let url = format!("https://huggingface.co/api/models/{}/tree/{}?recursive=true", repo_id, revision);
    let entries = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("list files for {repo_id}@{revision}"))?
        .error_for_status()?
        .json::<Vec<HfTreeEntry>>()
        .await
        .with_context(|| "parse the Hugging Face tree listing")?;

    Ok(entries
        .into_iter()
        .filter(|entry| should_download_hf_file(&entry.path))
        .map(|entry| HfFile { size: entry.size.unwrap_or(0), sha256: entry.lfs.map(|lfs| lfs.oid.to_lowercase()), path: entry.path })
        .collect())
}

/// Model downloads are multi-GB over a link we don't control, so the client is
/// tuned for "long transfer, never hang": no overall deadline (a legitimate
/// 3.5 GB fetch takes minutes), but a connect deadline and a per-read deadline
/// so a silently dropped socket fails fast into the retry loop instead of
/// leaving the UI on a bar that never advances again.
pub fn download_client(user_agent: &str) -> Result<Client> {
    Client::builder()
        .user_agent(user_agent)
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(60))
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .map_err(Into::into)
}

/// Free bytes on the volume holding `path` (nearest existing ancestor), via
/// `df -k`. Best-effort: `None` when the probe fails, so a parse change can
/// never block a download that would have succeeded.
async fn free_space_bytes(path: &Path) -> Option<u64> {
    let mut probe = path;
    while !probe.exists() {
        probe = probe.parent()?;
    }
    let out = tokio::process::Command::new("/bin/df").arg("-k").arg(probe).output().await.ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // Columns: Filesystem 1024-blocks Used Available …
    let available_kb: u64 = text.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
    Some(available_kb.saturating_mul(1024))
}

/// Emitting one message per network chunk is ~100k sends for a 3.5 GB model —
/// enough serialization and UI traffic to measurably slow the download it is
/// reporting on. Coalesce to at most one update per 150 ms or 8 MB; phase
/// changes and file boundaries still emit immediately (`force`).
const PROGRESS_MIN_INTERVAL: Duration = Duration::from_millis(150);
const PROGRESS_MIN_BYTES: u64 = 8 * 1024 * 1024;

struct ProgressPump<F: FnMut(DownloadProgress)> {
    sink: F,
    last_at: Instant,
    last_bytes: u64,
}

impl<F: FnMut(DownloadProgress)> ProgressPump<F> {
    fn new(sink: F) -> Self {
        Self { sink, last_at: Instant::now(), last_bytes: 0 }
    }

    fn send(&mut self, force: bool, progress: DownloadProgress) {
        let bytes = progress.bytes_downloaded;
        if !force && self.last_at.elapsed() < PROGRESS_MIN_INTERVAL && bytes.saturating_sub(self.last_bytes) < PROGRESS_MIN_BYTES {
            return;
        }
        self.last_at = Instant::now();
        self.last_bytes = bytes;
        (self.sink)(progress);
    }
}

pub fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    digest.as_ref().iter().fold(String::with_capacity(64), |mut acc, byte| {
        use std::fmt::Write;
        let _ = write!(acc, "{byte:02x}");
        acc
    })
}

/// Feed an already-downloaded prefix through the hasher so a resumed file ends
/// with the same digest as a single-pass download. Costs one read of the
/// partial bytes — versus re-reading the whole multi-GB file after the fact.
async fn hash_existing_prefix(path: &Path, len: u64, hasher: &mut Sha256) -> Result<()> {
    use tokio::io::AsyncReadExt;
    let mut file = tokio::fs::File::open(path).await?;
    let mut remaining = len;
    let mut buf = vec![0_u8; 1024 * 1024];
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        let read = file.read(&mut buf[..want]).await?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
        remaining = remaining.saturating_sub(read as u64);
    }
    Ok(())
}

// ── Parallel chunked download (large files) ──────────────────────────────────
// One TCP stream to the HF CDN caps at 5-15 MB/s regardless of line speed
// (window/RTT), and a long-lived stream decays further — measured 0.3 MB/s on
// a 37 MB/s line. Files above PARALLEL_MIN_FILE_SIZE are therefore fetched as
// a grid of PARALLEL_CHUNK_SIZE ranges by PARALLEL_STREAMS concurrent workers:
// streams stack toward line rate, and every chunk being a fresh request kills
// the decay mode outright. Completed chunks are recorded in a `.chunks.json`
// sidecar so pause/kill resumes at chunk granularity. The whole file is
// SHA-hashed once at the end (sequential hashing is impossible with
// out-of-order writes).

const PARALLEL_MIN_FILE_SIZE: u64 = 192 * 1024 * 1024;
const PARALLEL_CHUNK_SIZE: u64 = 64 * 1024 * 1024;
const PARALLEL_STREAMS: usize = 5;
const PARALLEL_CHUNK_ATTEMPTS: usize = 3;

#[derive(Serialize, Deserialize)]
struct ChunkSidecar {
    /// Size of the object this grid was cut for — a mismatch (repo moved under
    /// us) invalidates the whole partial.
    expected_size: u64,
    chunk_size: u64,
    done: Vec<bool>,
}

fn chunk_sidecar_path(target: &Path) -> PathBuf {
    let mut name = target.file_name().unwrap_or_default().to_os_string();
    name.push(".chunks.json");
    target.with_file_name(name)
}

async fn load_chunk_sidecar(target: &Path, expected_size: u64) -> Option<ChunkSidecar> {
    let raw = tokio::fs::read(chunk_sidecar_path(target)).await.ok()?;
    let sc: ChunkSidecar = serde_json::from_slice(&raw).ok()?;
    (sc.expected_size == expected_size && sc.chunk_size == PARALLEL_CHUNK_SIZE).then_some(sc)
}

async fn save_chunk_sidecar(target: &Path, sc: &ChunkSidecar) {
    let path = chunk_sidecar_path(target);
    let tmp = path.with_extension("json.tmp");
    if tokio::fs::write(&tmp, serde_json::to_vec(sc).unwrap_or_default()).await.is_ok() {
        let _ = tokio::fs::rename(&tmp, &path).await;
    }
}

fn chunk_len(i: usize, count: usize, total: u64) -> u64 {
    if i + 1 == count {
        total - (i as u64) * PARALLEL_CHUNK_SIZE
    } else {
        PARALLEL_CHUNK_SIZE
    }
}

#[allow(clippy::too_many_arguments)]
async fn fetch_chunk(
    client: &Client,
    url: &str,
    file: &std::fs::File,
    start: u64,
    len: u64,
    progress: &std::sync::atomic::AtomicU64,
    paused: &std::sync::atomic::AtomicBool,
    cancel_id: &str,
) -> Result<()> {
    use std::os::unix::fs::FileExt;
    use std::sync::atomic::Ordering;
    let end = start + len - 1;
    let mut resp = client.get(url).header(reqwest::header::RANGE, format!("bytes={start}-{end}")).send().await?.error_for_status()?;
    if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(anyhow!("server ignored Range request"));
    }
    let mut offset = start;
    let mut written_here: u64 = 0;
    while let Some(chunk) = resp.chunk().await? {
        file.write_all_at(&chunk, offset)?;
        offset += chunk.len() as u64;
        written_here += chunk.len() as u64;
        progress.fetch_add(chunk.len() as u64, Ordering::SeqCst);
        if download_cancel_requested(cancel_id) {
            paused.store(true, Ordering::SeqCst);
            // Roll the counter back — an incomplete chunk is redone whole.
            progress.fetch_sub(written_here, Ordering::SeqCst);
            return Err(anyhow!("{DOWNLOAD_PAUSED_MARKER}"));
        }
    }
    if written_here != len {
        progress.fetch_sub(written_here, Ordering::SeqCst);
        return Err(anyhow!("range returned {written_here} of {len} bytes"));
    }
    Ok(())
}

/// Fetch `url` as concurrent range chunks into `target`. Same contract as
/// `download_file_resumable`: returns (bytes_written, sha256-hex, reused_partial);
/// fails with `DOWNLOAD_PAUSED_MARKER` on pause (sidecar kept for resume).
#[allow(clippy::too_many_arguments)]
async fn download_file_parallel<F>(
    client: &Client,
    url: &str,
    target: &Path,
    expected_size: u64,
    base_bytes: u64,
    file_name: &str,
    file_index: usize,
    file_count: usize,
    total_bytes: Option<u64>,
    pump: &mut ProgressPump<F>,
    cancel_id: &str,
) -> Result<(u64, String, bool)>
where
    F: FnMut(DownloadProgress),
{
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    let chunk_count = expected_size.div_ceil(PARALLEL_CHUNK_SIZE).max(1) as usize;

    // Resume state. Precedence: valid sidecar; else a single-stream prefix
    // from the pre-parallel downloader (contiguous first N bytes → whole
    // leading chunks are done); else fresh.
    let mut sidecar = match load_chunk_sidecar(target, expected_size).await {
        Some(sc) if sc.done.len() == chunk_count => sc,
        _ => {
            let prefix = tokio::fs::metadata(target).await.map(|m| m.len()).unwrap_or(0);
            let whole = if prefix > 0 && prefix <= expected_size { (prefix / PARALLEL_CHUNK_SIZE) as usize } else { 0 };
            ChunkSidecar { expected_size, chunk_size: PARALLEL_CHUNK_SIZE, done: (0..chunk_count).map(|i| i < whole).collect() }
        }
    };
    let reused = sidecar.done.iter().any(|d| *d);
    // Persist the grid BEFORE preallocation: set_len grows the file to full
    // size, so a hard kill between the two would otherwise leave a
    // full-length file with no sidecar — indistinguishable from complete
    // until verification fails.
    save_chunk_sidecar(target, &sidecar).await;

    // Preallocate so out-of-order write_at never grows the file racily.
    {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(target)
            .with_context(|| format!("open {} for parallel download", target.display()))?;
        f.set_len(expected_size).with_context(|| format!("preallocate {expected_size} bytes"))?;
    }
    let file = Arc::new(
        std::fs::OpenOptions::new().write(true).open(target).with_context(|| format!("reopen {} for chunk writes", target.display()))?,
    );

    let done_bytes: u64 = sidecar.done.iter().enumerate().filter(|(_, d)| **d).map(|(i, _)| chunk_len(i, chunk_count, expected_size)).sum();
    let progress_bytes = Arc::new(AtomicU64::new(done_bytes));
    let paused = Arc::new(AtomicBool::new(false));

    // Work queue: indexes of pending chunks, drained by the workers.
    let pending: Vec<usize> = sidecar.done.iter().enumerate().filter(|(_, d)| !**d).map(|(i, _)| i).collect();
    let (done_tx, mut done_rx) = tokio::sync::mpsc::unbounded_channel::<Result<usize>>();
    let queue = Arc::new(std::sync::Mutex::new(pending));

    for _ in 0..PARALLEL_STREAMS.min(chunk_count) {
        let queue = Arc::clone(&queue);
        let file = Arc::clone(&file);
        let progress_bytes = Arc::clone(&progress_bytes);
        let paused = Arc::clone(&paused);
        let done_tx = done_tx.clone();
        let client = client.clone();
        let url = url.to_string();
        let cancel_id = cancel_id.to_string();
        tokio::spawn(async move {
            loop {
                let idx = { queue.lock().unwrap().pop() };
                let Some(idx) = idx else { break };
                let start = (idx as u64) * PARALLEL_CHUNK_SIZE;
                let len = chunk_len(idx, chunk_count, expected_size);
                let mut attempt = 0;
                let result = loop {
                    attempt += 1;
                    match fetch_chunk(&client, &url, &file, start, len, &progress_bytes, &paused, &cancel_id).await {
                        Ok(()) => break Ok(idx),
                        Err(_) if paused.load(Ordering::SeqCst) => break Err(anyhow!("{DOWNLOAD_PAUSED_MARKER}")),
                        Err(e) if attempt >= PARALLEL_CHUNK_ATTEMPTS => break Err(e),
                        Err(e) => {
                            tracing::debug!(model = %cancel_id, chunk = idx, attempt, error = %e, "chunk download failed; retrying");
                            tokio::time::sleep(Duration::from_millis(600 * attempt as u64)).await
                        }
                    }
                };
                let failed = result.is_err();
                let _ = done_tx.send(result);
                if failed {
                    break;
                }
            }
        });
    }
    drop(done_tx);

    // Driver: forward aggregated progress every pump interval; collect results.
    let mut completed = sidecar.done.iter().filter(|d| **d).count();
    let mut failure: Option<anyhow::Error> = None;
    while completed < chunk_count {
        match tokio::time::timeout(Duration::from_millis(150), done_rx.recv()).await {
            Ok(Some(Ok(idx))) => {
                sidecar.done[idx] = true;
                completed += 1;
                save_chunk_sidecar(target, &sidecar).await;
            }
            Ok(Some(Err(e))) => {
                paused.store(true, Ordering::SeqCst);
                failure = Some(e);
                break;
            }
            Ok(None) => break, // all workers exited
            Err(_) => {}       // timeout tick — emit progress below
        }
        pump.send(
            false,
            DownloadProgress {
                phase: "downloading",
                file_name: Some(file_name.to_string()),
                file_index,
                file_count,
                bytes_downloaded: base_bytes.saturating_add(progress_bytes.load(Ordering::SeqCst)),
                total_bytes,
            },
        );
    }
    if let Some(e) = failure {
        save_chunk_sidecar(target, &sidecar).await;
        return Err(e);
    }
    if completed < chunk_count {
        save_chunk_sidecar(target, &sidecar).await;
        return Err(anyhow!("parallel download workers exited early"));
    }

    // Hash the assembled file once (sequential hashing is impossible with
    // out-of-order chunk writes).
    let mut hasher = Sha256::new();
    hash_existing_prefix(target, expected_size, &mut hasher).await?;
    let _ = tokio::fs::remove_file(chunk_sidecar_path(target)).await;
    pump.send(
        true,
        DownloadProgress {
            phase: "downloading",
            file_name: Some(file_name.to_string()),
            file_index,
            file_count,
            bytes_downloaded: base_bytes.saturating_add(expected_size),
            total_bytes,
        },
    );
    Ok((expected_size, hex_digest(hasher.finalize()), reused))
}

/// Download one file into `target`, resuming from whatever a previous attempt
/// left on disk. Returns `(bytes_on_disk, sha256_hex, reused_existing_bytes)` —
/// the digest is computed while streaming, so verification needs no second pass
/// over the file.
///
/// The third field matters because our revisions are branch names, not commit
/// hashes: `main` moves when a repo is re-uploaded, so a partial can be a prefix
/// of the *previous* object. The caller uses it to tell "stale partial, discard
/// and refetch" apart from "fresh download that failed verification", which must
/// stay a hard failure.
#[allow(clippy::too_many_arguments)]
async fn download_file_resumable<F>(
    client: &Client,
    url: &str,
    target: &Path,
    expected_size: u64,
    base_bytes: u64,
    file_name: &str,
    file_index: usize,
    file_count: usize,
    total_bytes: Option<u64>,
    pump: &mut ProgressPump<F>,
    cancel_id: &str,
) -> Result<(u64, String, bool)>
where
    F: FnMut(DownloadProgress),
{
    let on_disk = tokio::fs::metadata(target).await.ok().map(|meta| meta.len()).unwrap_or(0);

    // Already complete from an earlier attempt: hash it and skip the network
    // entirely. Requesting `Range: bytes=<size>-` here would earn a 416, which
    // would fail every retry and strand a resumed download on its first file.
    if expected_size > 0 && on_disk == expected_size {
        let mut hasher = Sha256::new();
        hash_existing_prefix(target, on_disk, &mut hasher).await?;
        pump.send(
            true,
            DownloadProgress {
                phase: "downloading",
                file_name: Some(file_name.to_string()),
                file_index,
                file_count,
                bytes_downloaded: base_bytes.saturating_add(on_disk),
                total_bytes,
            },
        );
        return Ok((on_disk, hex_digest(hasher.finalize()), true));
    }

    // A partial longer than the published size can't be a prefix of it — that
    // file is junk from an interrupted write, so start it over.
    let mut resume_from = if expected_size > 0 && on_disk > expected_size { 0 } else { on_disk };

    let send_request = |from: u64| {
        let mut request = client.get(url);
        if from > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={from}-"));
        }
        request.send()
    };

    let mut response = send_request(resume_from).await?;
    // 416 means the offset is past the end of the object — the partial doesn't
    // belong to this file (size unknown, or it grew stale). Refetch it whole.
    if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && resume_from > 0 {
        resume_from = 0;
        response = send_request(0).await?;
    }
    let mut response = response.error_for_status()?;

    // 200 instead of 206 means the server ignored the Range header and is
    // sending the whole object — drop the prefix rather than concatenating.
    if resume_from > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        resume_from = 0;
    }

    let mut hasher = Sha256::new();
    let mut out = if resume_from > 0 {
        hash_existing_prefix(target, resume_from, &mut hasher).await?;
        tokio::fs::OpenOptions::new().append(true).open(target).await.with_context(|| format!("reopen {} to resume", target.display()))?
    } else {
        tokio::fs::File::create(target).await.with_context(|| format!("create {}", target.display()))?
    };

    let mut written = resume_from;
    let emit = |pump: &mut ProgressPump<F>, force: bool, written: u64| {
        pump.send(
            force,
            DownloadProgress {
                phase: "downloading",
                file_name: Some(file_name.to_string()),
                file_index,
                file_count,
                bytes_downloaded: base_bytes.saturating_add(written),
                total_bytes,
            },
        );
    };
    emit(pump, true, written);

    while let Some(chunk) = response.chunk().await? {
        out.write_all(&chunk).await?;
        hasher.update(&chunk);
        written = written.saturating_add(chunk.len() as u64);
        emit(pump, false, written);
        if download_cancel_requested(cancel_id) {
            // Keep the partial bytes — flush so the resumable path can pick
            // them up when the user presses Download again.
            out.flush().await?;
            return Err(anyhow!("{DOWNLOAD_PAUSED_MARKER}"));
        }
    }
    out.flush().await?;

    Ok((written, hex_digest(hasher.finalize()), resume_from > 0))
}

pub fn should_download_hf_file(name: &str) -> bool {
    if name == ".gitattributes" || name.ends_with(".md") {
        return false;
    }
    name.ends_with(".json")
        || name.ends_with(".txt")
        || name.ends_with(".model")
        || name.ends_with(".safetensors")
        || name.ends_with(".py")
        // Gemma 4 publishes its chat template as a standalone `chat_template.jinja`
        // (17 KB) rather than a `chat_template` key in tokenizer_config.json.
        // Dropping it left `apply_chat_template` returning the prompt unchanged,
        // so the runner's templated fallback collapsed onto the raw prompt.
        || name.ends_with(".jinja")
        || name.contains("tokenizer")
}

pub fn resolve_url(repo_id: &str, revision: &str, filename: &str) -> String {
    format!("https://huggingface.co/{}/resolve/{}/{}?download=true", repo_id, revision, filename)
}

/// Size + integrity check for one downloaded file. LFS-backed files (the
/// weights) publish a content SHA256; plain config/tokenizer files don't, so
/// they're covered by the size check alone.
fn verify_downloaded_file(file: &HfFile, written: u64, actual_sha: &str) -> Result<()> {
    if file.size > 0 && written != file.size {
        return Err(anyhow!("downloaded size mismatch for {}: expected {}, got {}", file.path, file.size, written));
    }
    if let Some(expected_sha) = &file.sha256 {
        if actual_sha != expected_sha {
            return Err(anyhow!(
                "SHA256 mismatch for {} (expected {}, got {}) — the download may be \
                 corrupted or tampered with; the model was not installed",
                file.path,
                expected_sha,
                actual_sha
            ));
        }
    }
    Ok(())
}

/// How many times one file may be re-attempted before the download gives up.
/// Attempts resume from the bytes already on disk, so a retry after 3 GB costs
/// the remaining tail, not another 3 GB.
const MAX_FILE_ATTEMPTS: usize = 4;

/// Marks which revision a partial download directory belongs to, so a stale
/// partial from a different revision is discarded instead of resumed into.
pub const PARTIAL_REVISION_MARKER: &str = ".revision";
/// The marker name earlier versions of this downloader wrote. Read for
/// compatibility so a partial download they left behind can still resume;
/// never written.
const LEGACY_PARTIAL_REVISION_MARKER: &str = ".modelcaddy-revision";

/// Files worth topping up in place on an already-installed model. Bounded to
/// small metadata so this can never turn "the model is already here" into a
/// silent multi-GB re-download; a missing weight file means the install is
/// broken and should be removed and refetched deliberately.
const TOP_UP_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// Fetch metadata files the repo lists but the installed model lacks.
///
/// Exists because widening `should_download_hf_file` (e.g. to pick up
/// `chat_template.jinja`) otherwise only helps people who download a model for
/// the first time — everyone with a 3.5–5 GB model already on disk would need a
/// full re-download for a 17 KB file. Best-effort by design: offline, an API
/// change, or a failed fetch all leave the existing install untouched.
///
/// Returns the names of the files it added.
async fn top_up_installed_model_files(destination: &Path, repo_id: &str, revision: &str, user_agent: &str) -> Vec<String> {
    let Ok(client) = download_client(user_agent) else {
        return Vec::new();
    };
    let Ok(files) = hf_repo_files(&client, repo_id, revision).await else {
        return Vec::new();
    };

    let mut added = Vec::new();
    for file in files {
        if file.size == 0 || file.size > TOP_UP_MAX_BYTES {
            continue;
        }
        let target = destination.join(&file.path);
        if target.exists() {
            continue;
        }
        if let Some(parent) = target.parent() {
            if tokio::fs::create_dir_all(parent).await.is_err() {
                continue;
            }
        }
        // Write to a temp name and rename, so a failure mid-write can't leave a
        // truncated config where a valid one is expected.
        let staging = target.with_extension("topup");
        let Ok(response) = client.get(resolve_url(repo_id, revision, &file.path)).send().await else {
            continue;
        };
        let Ok(response) = response.error_for_status() else {
            continue;
        };
        let Ok(body) = response.bytes().await else {
            continue;
        };
        if body.len() as u64 != file.size {
            continue;
        }
        if let Some(expected_sha) = &file.sha256 {
            if &hex_digest(Sha256::digest(&body)) != expected_sha {
                continue;
            }
        }
        if tokio::fs::write(&staging, &body).await.is_ok() && tokio::fs::rename(&staging, &target).await.is_ok() {
            added.push(file.path);
        } else {
            tokio::fs::remove_file(&staging).await.ok();
        }
    }
    added
}

/// Download `spec` into `destination` (a directory named after the artifact),
/// staging in `<destination>.download` and renaming atomically at the end.
/// An already-installed destination only gets its small metadata topped up.
pub async fn download_model<F>(destination: PathBuf, spec: &DownloadSpec, user_agent: &str, on_progress: F) -> Result<DownloadSummary>
where
    F: FnMut(DownloadProgress),
{
    let model_id = spec.id.as_str();
    let repo_id = spec.repo_id.as_str();
    let revision = spec.revision.as_str();
    let required_disk_bytes = spec.required_disk_bytes;

    let mut pump = ProgressPump::new(on_progress);
    // A stale pause request from a previous run must not kill this download.
    clear_download_cancel(model_id);
    let temp_destination = destination.with_extension("download");
    if destination.exists() {
        // Already installed — but it may predate a widening of
        // `should_download_hf_file`, so fill in any small metadata it's missing
        // before reporting complete. Re-running setup is the reachable trigger
        // for this; it stays cheap because weights are excluded by size.
        top_up_installed_model_files(&destination, repo_id, revision, user_agent).await;

        let bytes = dir_size(&destination).unwrap_or(0);
        pump.send(
            true,
            DownloadProgress {
                phase: "complete",
                file_name: None,
                file_index: 0,
                file_count: 0,
                bytes_downloaded: bytes,
                total_bytes: Some(bytes),
            },
        );
        return Ok(DownloadSummary {
            model_id: model_id.to_string(),
            repo_id: repo_id.to_string(),
            revision: revision.to_string(),
            files_downloaded: 0,
            bytes_downloaded: bytes,
            path: destination.display().to_string(),
        });
    }

    pump.send(
        true,
        DownloadProgress {
            phase: "preparing",
            file_name: None,
            file_index: 0,
            file_count: 0,
            bytes_downloaded: 0,
            total_bytes: Some(required_disk_bytes),
        },
    );

    // Keep any partial from a previous attempt — that is what makes a retry
    // cheap — but only when it belongs to this same revision.
    let marker = temp_destination.join(PARTIAL_REVISION_MARKER);
    let legacy_marker = temp_destination.join(LEGACY_PARTIAL_REVISION_MARKER);
    let mut partial_matches = tokio::fs::read_to_string(&marker).await.map(|text| text.trim() == revision).unwrap_or(false);
    if !partial_matches {
        partial_matches = tokio::fs::read_to_string(&legacy_marker).await.map(|text| text.trim() == revision).unwrap_or(false);
    }
    if temp_destination.exists() && !partial_matches {
        tokio::fs::remove_dir_all(&temp_destination).await.with_context(|| "failed to remove a stale partial model download")?;
    }
    tokio::fs::create_dir_all(&temp_destination).await.with_context(|| "failed to create the model download directory")?;
    tokio::fs::write(&marker, revision).await.ok();
    tokio::fs::remove_file(&legacy_marker).await.ok();

    let client = download_client(user_agent)?;
    let files = hf_repo_files(&client, repo_id, revision).await?;
    if files.is_empty() {
        return Err(anyhow!("Hugging Face repo returned no downloadable model files"));
    }

    let file_count = files.len();
    // Every size comes from the tree listing, so this is always known — the
    // UI can show a real percentage from the first byte.
    let planned_bytes = files.iter().fold(0_u64, |acc, file| acc.saturating_add(file.size));
    let total_bytes = Some(if planned_bytes > 0 { planned_bytes } else { required_disk_bytes });

    // Bytes already on disk from an interrupted attempt count as done, so a
    // resumed download doesn't restart its progress bar at zero.
    let already_on_disk = dir_size(&temp_destination).unwrap_or(0);
    if already_on_disk > 0 {
        tracing::info!(model = %model_id, bytes_on_disk = already_on_disk, "resuming a partial download");
    }
    tracing::debug!(model = %model_id, repo = %repo_id, revision = %revision, files = file_count, bytes = planned_bytes, "download plan");

    // Fail before transferring gigabytes we can't store. `df` is best-effort,
    // so an unreadable probe just skips the check.
    if let Some(free) = free_space_bytes(&temp_destination).await {
        let needed = planned_bytes.saturating_sub(already_on_disk);
        if free < needed {
            return Err(anyhow!(
                "not enough free disk space for {}: needs {:.1} GB, {:.1} GB available",
                model_id,
                needed as f64 / 1_000_000_000.0,
                free as f64 / 1_000_000_000.0
            ));
        }
    }

    pump.send(
        true,
        DownloadProgress { phase: "downloading", file_name: None, file_index: 0, file_count, bytes_downloaded: 0, total_bytes },
    );

    let mut bytes_downloaded = 0_u64;
    let mut files_downloaded = 0_usize;
    for (idx, file) in files.into_iter().enumerate() {
        if download_cancel_requested(model_id) {
            return Err(anyhow!("{DOWNLOAD_PAUSED_MARKER}"));
        }
        let target = temp_destination.join(&file.path);
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let url = resolve_url(repo_id, revision, &file.path);

        let mut attempt = 0_usize;
        let written = loop {
            attempt += 1;
            // Big files go through the parallel chunked path (line-rate,
            // decay-proof); small ones keep the simple streaming path.
            let result = if file.size >= PARALLEL_MIN_FILE_SIZE {
                download_file_parallel(
                    &client,
                    &url,
                    &target,
                    file.size,
                    bytes_downloaded,
                    &file.path,
                    idx + 1,
                    file_count,
                    total_bytes,
                    &mut pump,
                    model_id,
                )
                .await
            } else {
                download_file_resumable(
                    &client,
                    &url,
                    &target,
                    file.size,
                    bytes_downloaded,
                    &file.path,
                    idx + 1,
                    file_count,
                    total_bytes,
                    &mut pump,
                    model_id,
                )
                .await
            };

            let last_attempt = attempt >= MAX_FILE_ATTEMPTS;
            match result {
                Ok((written, actual_sha, reused)) => {
                    match verify_downloaded_file(&file, written, &actual_sha) {
                        Ok(()) => break written,
                        // A resumed file that fails verification is almost always
                        // a prefix of a superseded upload (our revisions are
                        // branch names, so `main` moves under us). Discard it and
                        // refetch clean rather than reporting tampering. A
                        // from-scratch download that fails still hard-errors —
                        // that check is the point of verifying at all.
                        Err(err) if reused && !last_attempt => {
                            tracing::warn!(model = %model_id, file = %file.path, error = %err, "a resumed file failed verification; downloading it again");
                            tokio::fs::remove_file(&target).await.ok();
                            tokio::fs::remove_file(chunk_sidecar_path(&target)).await.ok();
                        }
                        Err(err) => {
                            // Never leave bytes that failed verification behind:
                            // the next run would resume from them and fail the
                            // same way forever.
                            tokio::fs::remove_file(&target).await.ok();
                            tokio::fs::remove_file(chunk_sidecar_path(&target)).await.ok();
                            return Err(err);
                        }
                    }
                }
                Err(err) if !last_attempt => {
                    tracing::warn!(model = %model_id, file = %file.path, attempt, error = %err, "download attempt failed; retrying");
                    // Leave the partial in place on purpose: the next attempt
                    // resumes from it instead of re-fetching what already landed.
                    tokio::time::sleep(Duration::from_secs(2 * attempt as u64)).await;
                }
                Err(err) => return Err(err.context(format!("downloading {} failed after {} attempts", file.path, attempt))),
            }
        };

        files_downloaded += 1;
        bytes_downloaded = bytes_downloaded.saturating_add(written);
    }

    pump.send(
        true,
        DownloadProgress { phase: "finalizing", file_name: None, file_index: file_count, file_count, bytes_downloaded, total_bytes },
    );

    tokio::fs::remove_file(&marker).await.ok();

    if destination.exists() {
        tokio::fs::remove_dir_all(&destination).await?;
    }
    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::rename(&temp_destination, &destination).await?;

    pump.send(
        true,
        DownloadProgress { phase: "complete", file_name: None, file_index: file_count, file_count, bytes_downloaded, total_bytes },
    );

    Ok(DownloadSummary {
        model_id: model_id.to_string(),
        repo_id: repo_id.to_string(),
        revision: revision.to_string(),
        files_downloaded,
        bytes_downloaded,
        path: destination.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma 4 ships its chat template as a standalone `.jinja`, not as a
    /// `chat_template` key in tokenizer_config.json. Dropping it left the
    /// runner's templated candidate collapsing onto the raw prompt.
    #[test]
    fn keeps_the_standalone_chat_template() {
        assert!(should_download_hf_file("chat_template.jinja"));
        assert!(should_download_hf_file("model.safetensors"));
        assert!(should_download_hf_file("tokenizer_config.json"));
        // Still no repo furniture.
        assert!(!should_download_hf_file(".gitattributes"));
        assert!(!should_download_hf_file("README.md"));
    }
}
