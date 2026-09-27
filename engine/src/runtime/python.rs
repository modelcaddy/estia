//! Python runtime for the MLX backend.
//!
//! Lazy install: on first use the engine downloads `python-build-standalone`
//! (Astral's redistributable Python) and `pip install`s the MLX packages into
//! `<root>/python/`. No `pip` on the user's machine, no terminal commands, no
//! Homebrew. Hosts that never pick this backend pay nothing.
//!
//! Layout under `<root>/`:
//! ```text
//! python/                <-- atomically swapped in at the end of install
//!   bin/python3
//!   bin/pip3
//!   lib/python3.12/site-packages/mlx_vlm/...
//! python.partial/        <-- present only mid-install
//! .version               <-- "python=…\nmlx_lm=…" stamp written on success
//! ```
//!
//! The installer itself is behind the `python-mlx` feature: a build that must
//! not download executable code contains none of it, while the status and
//! path helpers stay available so a host can still say "not installed".

#[cfg(feature = "python-mlx")]
use anyhow::Context;
use anyhow::{anyhow, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
#[cfg(feature = "python-mlx")]
use std::{
    io::BufRead,
    process::{Command, Stdio},
    time::{Duration, Instant},
};
#[cfg(feature = "python-mlx")]
use tokio::io::AsyncWriteExt;

/// Minimum gap between download progress messages, matching the model
/// downloader's pump.
#[cfg(feature = "python-mlx")]
const PROGRESS_MIN_INTERVAL: Duration = Duration::from_millis(150);

/// Pinned `python-build-standalone` release. Update consciously; binaries from
/// every release are immutable on GitHub so users never silently get a
/// different Python.
#[cfg(feature = "python-mlx")]
const PBS_RELEASE_DATE: &str = "20241016";
#[cfg(feature = "python-mlx")]
const PBS_PYTHON_VERSION: &str = "3.12.7";
#[cfg(feature = "python-mlx")]
const PBS_FILENAME: &str = "cpython-3.12.7+20241016-aarch64-apple-darwin-install_only.tar.gz";

#[cfg(feature = "python-mlx")]
fn pbs_url() -> String {
    format!("https://github.com/astral-sh/python-build-standalone/releases/download/{}/{}", PBS_RELEASE_DATE, PBS_FILENAME)
}

/// Compile-time pin of the tarball's SHA256 (lowercase hex), authoritative and
/// tamper-resistant — the release asset is immutable, so a pinned hash catches
/// an actively swapped file, not just corruption, with no network dependency.
/// **Must be updated whenever `PBS_RELEASE_DATE`/`PBS_FILENAME` change** — it's
/// the published `<asset>.sha256` value for that exact asset. When `None` (e.g.
/// a release was bumped without re-pinning), we fall back to fetching the
/// sibling `<asset>.sha256`; and if even that can't be obtained we proceed
/// *unverified* rather than blocking a real install (see the call site).
#[cfg(feature = "python-mlx")]
const PBS_SHA256: Option<&str> = Some("4c18852bf9c1a11b56f21bcf0df1946f7e98ee43e9e4c0c5374b2b3765cf9508");

/// The pip requirements of the MLX runner. `mlx-vlm` loads multimodal Gemma
/// models for generation (floor, not pin: >= 0.6.13 is when it gained the
/// `gemma4_unified` architecture the 12B QAT model needs); `mlx-embeddings`
/// loads compact encoder models for embeddings. `mlx-lm` is named explicitly:
/// mlx-vlm 0.6.x pulled it in (`mlx-lm>=0.31.3`) but 0.7 dropped it, and
/// [`verify_install`] imports it and stamps its version, so a clean install
/// that resolved mlx-vlm 0.7 failed verification without it.
///
/// `mlx-embeddings` is GPL-3.0 (0.1.0's PyPI metadata and its repository's
/// LICENSE). Estia does not ship it; this install fetches it from PyPI. It
/// is pinned to an exact version so that neither its code nor its licence
/// changes without a commit here: re-check the licence before a bump. The
/// pin forces no reinstall: [`stack_importable`] checks only that it
/// imports, and pip changes it only when a top-up runs for another reason.
#[cfg(feature = "python-mlx")]
// mlx-vlm is capped below 0.7 until the runner is verified against it: 0.7
// changed the dependency set (see above) and nothing has been run on it yet.
const PIP_PACKAGES: &[&str] = &["mlx-vlm>=0.6.13,<0.7", "mlx-lm>=0.31.3", "mlx-embeddings==0.1.0"];

/// Fetch the release's sibling `<asset>.sha256` and return the 64-char hex
/// digest. GitHub serves it right next to the asset.
#[cfg(feature = "python-mlx")]
async fn fetch_expected_sha256(client: &reqwest::Client, tarball_url: &str) -> Result<String> {
    let url = format!("{tarball_url}.sha256");
    let body = client.get(&url).send().await?.error_for_status()?.text().await?;
    // Sidecar format is "<hex>" or "<hex>  <filename>"; take the first token.
    let hex = body.split_whitespace().next().unwrap_or("").trim().to_lowercase();
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(anyhow!("release SHA256 sidecar was missing or malformed: {url}"));
    }
    Ok(hex)
}

/// Hash the file and compare to `expected_hex`. Errors on mismatch so the
/// caller can wipe the partial install and refuse to execute the binary.
#[cfg(feature = "python-mlx")]
async fn verify_file_sha256(path: &Path, expected_hex: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    use tokio::io::AsyncReadExt;
    let expected = expected_hex.to_lowercase();
    let mut file = tokio::fs::File::open(path).await.with_context(|| format!("open {} for integrity check", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = super::super::models::hf::hex_digest(hasher.finalize());
    if actual != expected {
        return Err(anyhow!(
            "Python download failed its integrity check (expected {expected}, got {actual}). \
             The file may be corrupted or tampered with — nothing was installed."
        ));
    }
    Ok(())
}

/// Phase strings emitted to the host. A UI matches on these to render the
/// right copy and progress affordance.
pub mod phase {
    pub const PREPARING: &str = "runtime/preparing";
    pub const DOWNLOADING_PYTHON: &str = "runtime/downloading-python";
    pub const EXTRACTING_PYTHON: &str = "runtime/extracting-python";
    pub const INSTALLING_MLX_LM: &str = "runtime/installing-mlx-lm";
    pub const VERIFYING: &str = "runtime/verifying";
    pub const COMPLETE: &str = "runtime/complete";
}

#[derive(Debug, Clone, Serialize)]
pub struct SetupProgress {
    pub phase: &'static str,
    pub message: String,
    /// Bytes for download/extract phases. None for indeterminate phases.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_done: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_total: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSummary {
    pub python_version: String,
    pub mlx_lm_version: String,
    pub path: String,
    pub bytes_on_disk: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeState {
    Installed,
    Missing,
}

/// Filesystem state of the runtime. No prose: the host writes the sentence.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatus {
    pub state: RuntimeState,
    pub path: Option<String>,
    pub python_version: Option<String>,
    pub mlx_lm_version: Option<String>,
    pub bytes_on_disk: Option<u64>,
}

/// Roughly what the Python + MLX runtime costs on disk, for surfaces that
/// have to quote a download size before it exists.
pub const RUNTIME_APPROX_BYTES: u64 = 700_000_000;

/// The Python runtime under one root directory.
#[derive(Debug, Clone)]
pub struct PythonRuntime {
    root: PathBuf,
    user_agent: String,
}

impl PythonRuntime {
    /// `root` is the runtime directory itself (the CLI uses
    /// `<data_dir>/runtime`), not its parent.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into(), user_agent: format!("estia/{}", env!("CARGO_PKG_VERSION")) }
    }

    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn install_dir(&self) -> PathBuf {
        self.root.join("python")
    }

    fn partial_dir(&self) -> PathBuf {
        self.root.join("python.partial")
    }

    fn version_stamp_path(&self) -> PathBuf {
        self.root.join(".version")
    }

    /// The installed `python3`, when the runtime is present.
    pub fn python_path(&self) -> Option<PathBuf> {
        let p = self.install_dir().join("bin/python3");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    pub fn is_installed(&self) -> bool {
        self.python_path().is_some()
    }

    pub fn status(&self) -> RuntimeStatus {
        let dir = self.install_dir();
        if !dir.exists() {
            return RuntimeStatus {
                state: RuntimeState::Missing,
                path: None,
                python_version: None,
                mlx_lm_version: None,
                bytes_on_disk: None,
            };
        }
        let (python_version, mlx_lm_version) = self.read_version_stamp();
        RuntimeStatus {
            state: RuntimeState::Installed,
            path: Some(dir.display().to_string()),
            python_version,
            mlx_lm_version,
            bytes_on_disk: super::super::models::dir_size(&dir),
        }
    }

    fn read_version_stamp(&self) -> (Option<String>, Option<String>) {
        let stamp = match std::fs::read_to_string(self.version_stamp_path()) {
            Ok(s) => s,
            Err(_) => return (None, None),
        };
        let mut python = None;
        let mut mlx = None;
        for line in stamp.lines() {
            if let Some(rest) = line.strip_prefix("python=") {
                python = Some(rest.to_string());
            } else if let Some(rest) = line.strip_prefix("mlx_lm=") {
                mlx = Some(rest.to_string());
            }
        }
        (python, mlx)
    }

    #[cfg(feature = "python-mlx")]
    fn write_version_stamp(&self, python_v: &str, mlx_v: &str) -> std::io::Result<()> {
        let body = format!("python={}\nmlx_lm={}\n", python_v, mlx_v);
        std::fs::write(self.version_stamp_path(), body)
    }

    /// Remove the installed runtime (and any half-finished install).
    pub async fn remove(&self) -> Result<bool> {
        let mut removed = false;
        let dir = self.install_dir();
        if dir.exists() {
            tokio::fs::remove_dir_all(&dir).await?;
            removed = true;
        }
        let partial = self.partial_dir();
        if partial.exists() {
            let _ = tokio::fs::remove_dir_all(&partial).await;
        }
        let stamp = self.version_stamp_path();
        if stamp.exists() {
            let _ = tokio::fs::remove_file(&stamp).await;
        }
        Ok(removed)
    }

    /// Refuse an install that cannot finish, before anything is downloaded.
    ///
    /// Running out of disk halfway leaves a `python.partial/` tree and a user
    /// who watched a long download fail for a reason nothing on screen
    /// explained. The headroom is deliberate: unpacking needs room for the
    /// archive *and* its contents at the same time.
    pub fn preflight(&self, needed_bytes: u64) -> Result<()> {
        let probe = if self.root.exists() {
            self.root.clone()
        } else if let Some(parent) = self.root.parent().filter(|p| p.exists()) {
            parent.to_path_buf()
        } else {
            std::env::temp_dir()
        };
        let Some(free) = free_bytes(&probe) else {
            return Ok(()); // Can't tell — don't block a real install on a guess.
        };
        let required = needed_bytes.saturating_mul(2);
        if free < required {
            return Err(anyhow!(
                "Not enough disk space: this needs about {} GB free and there is {} GB.",
                required / 1_000_000_000,
                free / 1_000_000_000,
            ));
        }
        Ok(())
    }

    /// Feature-off stub: a build without `python-mlx` ships no code that
    /// downloads a Python interpreter or pip-installs packages. The signature
    /// is preserved so hosts compile unchanged and get a clean error.
    #[cfg(not(feature = "python-mlx"))]
    pub async fn install<F>(&self, _on_progress: F) -> Result<RuntimeSummary>
    where
        F: FnMut(SetupProgress),
    {
        Err(anyhow!("The downloaded Python (MLX) runtime is not included in this build."))
    }

    /// Install python-build-standalone + the MLX packages into `<root>/python/`.
    ///
    /// Idempotent: if the install dir already exists with a working stack,
    /// returns Ok without re-downloading (topping up packages in place when
    /// an older install lacks one). A crashed install leaves `python.partial/`
    /// behind; the next call wipes it before retrying.
    #[cfg(feature = "python-mlx")]
    pub async fn install<F>(&self, mut on_progress: F) -> Result<RuntimeSummary>
    where
        F: FnMut(SetupProgress),
    {
        // ── 0. Architecture gate ────────────────────────────────────────────
        // MLX is Apple Silicon only.
        if std::env::consts::ARCH != "aarch64" {
            return Err(anyhow!("The MLX runtime requires Apple Silicon (aarch64). This machine is {}.", std::env::consts::ARCH));
        }

        let install_dir = self.install_dir();
        let partial_dir = self.partial_dir();

        // ── 1. Short-circuit if already installed AND the stack imports ─────
        if install_dir.exists() {
            let python_bin = install_dir.join("bin/python3");
            if python_bin.exists() {
                // Earlier installs may lack a package or sit below the version
                // floor. Top up in place rather than re-downloading Python.
                if !stack_importable(&python_bin) {
                    on_progress(SetupProgress {
                        phase: phase::INSTALLING_MLX_LM,
                        message: "Updating the local engine…".into(),
                        bytes_done: None,
                        bytes_total: None,
                    });
                    run_pip_install_with_progress(&python_bin, &mut on_progress)?;
                }

                // (Re-)verify and stamp version. This also picks up the case
                // where the stamp is missing from a previous half-broken install.
                on_progress(SetupProgress {
                    phase: phase::VERIFYING,
                    message: "Checking the local engine…".into(),
                    bytes_done: None,
                    bytes_total: None,
                });
                let (py_v, mlx_v) = verify_install(&python_bin)?;
                self.write_version_stamp(&py_v, &mlx_v)?;

                let bytes = super::super::models::dir_size(&install_dir).unwrap_or(0);
                on_progress(SetupProgress {
                    phase: phase::COMPLETE,
                    message: "Local engine ready.".into(),
                    bytes_done: Some(bytes),
                    bytes_total: Some(bytes),
                });
                return Ok(RuntimeSummary {
                    python_version: py_v,
                    mlx_lm_version: mlx_v,
                    path: install_dir.display().to_string(),
                    bytes_on_disk: bytes,
                });
            }
            // No python3 binary inside → half-broken install. Wipe and rebuild.
            let _ = tokio::fs::remove_dir_all(&install_dir).await;
        }

        // ── 2. Prepare partial dir ─────────────────────────────────────────
        on_progress(SetupProgress {
            phase: phase::PREPARING,
            message: "Preparing the local engine install…".into(),
            bytes_done: None,
            bytes_total: None,
        });

        if partial_dir.exists() {
            tokio::fs::remove_dir_all(&partial_dir).await.with_context(|| "failed to remove previous partial runtime")?;
        }
        tokio::fs::create_dir_all(&partial_dir).await.with_context(|| "failed to create runtime install dir")?;

        // ── 3. Download python-build-standalone ────────────────────────────
        // Deadlines, not an overall timeout: the tarball is ~70 MB over a link
        // we don't control, so cap the connect and the per-read instead — a
        // dropped socket then errors out instead of hanging a dialog forever.
        let client = reqwest::Client::builder()
            .user_agent(&self.user_agent)
            .connect_timeout(Duration::from_secs(20))
            .read_timeout(Duration::from_secs(60))
            .build()?;

        let tarball_path = partial_dir.join("python.tar.gz");
        let url = pbs_url();
        let mut response = client.get(&url).send().await?.error_for_status()?;
        let total = response.content_length();

        let downloading_message = format!("Downloading the local engine (Python {PBS_PYTHON_VERSION} + MLX)…");
        on_progress(SetupProgress {
            phase: phase::DOWNLOADING_PYTHON,
            message: downloading_message.clone(),
            bytes_done: Some(0),
            bytes_total: total,
        });

        let mut out = tokio::fs::File::create(&tarball_path).await?;
        let mut bytes_done: u64 = 0;
        // One message per network chunk is thousands of sends for a download
        // the user reads as a single bar. Coalesce to ~7/s; the loop still
        // ends with an exact final count below.
        let mut last_emit = Instant::now();
        while let Some(chunk) = response.chunk().await? {
            out.write_all(&chunk).await?;
            bytes_done = bytes_done.saturating_add(chunk.len() as u64);
            if last_emit.elapsed() >= PROGRESS_MIN_INTERVAL {
                last_emit = Instant::now();
                on_progress(SetupProgress {
                    phase: phase::DOWNLOADING_PYTHON,
                    message: downloading_message.clone(),
                    bytes_done: Some(bytes_done),
                    bytes_total: total,
                });
            }
        }
        out.flush().await?;
        on_progress(SetupProgress {
            phase: phase::DOWNLOADING_PYTHON,
            message: downloading_message,
            bytes_done: Some(bytes_done),
            bytes_total: total,
        });

        // ── 3b. Integrity check ────────────────────────────────────────────
        // HTTPS alone is not enough for a native interpreter we are about to
        // execute. Verify SHA256 against the release's published digest before
        // extracting; on mismatch, wipe the partial dir and refuse.
        on_progress(SetupProgress {
            phase: phase::VERIFYING,
            message: "Checking the download…".into(),
            bytes_done: None,
            bytes_total: None,
        });
        // Expected digest: the compile-time pin when present (authoritative,
        // no network), else the release's published sidecar. A *missing*
        // expected hash must NOT block a real install — proceed unverified,
        // as before this check existed. Only a genuine *mismatch* aborts.
        let expected_sha: Option<String> = match PBS_SHA256 {
            Some(pinned) => Some(pinned.to_string()),
            None => match fetch_expected_sha256(&client, &url).await {
                Ok(sha) => Some(sha),
                Err(e) => {
                    tracing::warn!(error = %e, "python runtime: no published SHA-256 for the download; installing without an integrity check");
                    None
                }
            },
        };
        if let Some(expected) = &expected_sha {
            if let Err(e) = verify_file_sha256(&tarball_path, expected).await {
                let _ = tokio::fs::remove_dir_all(&partial_dir).await;
                return Err(e);
            }
        }

        // ── 4. Extract tarball ─────────────────────────────────────────────
        on_progress(SetupProgress {
            phase: phase::EXTRACTING_PYTHON,
            message: "Unpacking the local engine…".into(),
            bytes_done: None,
            bytes_total: None,
        });

        let tarball_for_blocking = tarball_path.clone();
        let extract_target = partial_dir.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let file = std::fs::File::open(&tarball_for_blocking).with_context(|| "open python tarball")?;
            let decoder = flate2::read::GzDecoder::new(file);
            let mut archive = tar::Archive::new(decoder);
            archive.set_preserve_permissions(true);
            archive.unpack(&extract_target).with_context(|| "unpack python tarball")?;
            Ok(())
        })
        .await
        .map_err(|e| anyhow!("extract task join error: {e}"))??;

        // python-build-standalone tarballs unpack to `./python/...`. Drop the
        // tarball file once extracted.
        let _ = tokio::fs::remove_file(&tarball_path).await;

        let extracted_python = partial_dir.join("python");
        if !extracted_python.exists() {
            return Err(anyhow!("python-build-standalone tarball did not contain a 'python' directory"));
        }
        strip_quarantine(&extracted_python);

        let python_bin = extracted_python.join("bin/python3");
        if !python_bin.exists() {
            return Err(anyhow!("extracted Python is missing bin/python3 at {}", python_bin.display()));
        }

        // ── 5. pip install the MLX packages ────────────────────────────────
        on_progress(SetupProgress {
            phase: phase::INSTALLING_MLX_LM,
            message: "Installing the model runtime (this can take several minutes)…".into(),
            bytes_done: None,
            bytes_total: None,
        });

        run_pip_install_with_progress(&python_bin, &mut on_progress)?;

        // ── 6. Verify the import works ─────────────────────────────────────
        on_progress(SetupProgress { phase: phase::VERIFYING, message: "Verifying mlx-lm…".into(), bytes_done: None, bytes_total: None });

        let (py_version, mlx_version) = verify_install(&python_bin)?;

        // ── 7. Atomic rename partial → final ───────────────────────────────
        if install_dir.exists() {
            tokio::fs::remove_dir_all(&install_dir).await?;
        }
        if let Some(parent) = install_dir.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::rename(&extracted_python, &install_dir).await?;
        let _ = tokio::fs::remove_dir_all(&partial_dir).await;
        strip_quarantine(&install_dir);

        self.write_version_stamp(&py_version, &mlx_version)?;

        let bytes = super::super::models::dir_size(&install_dir).unwrap_or(0);
        on_progress(SetupProgress {
            phase: phase::COMPLETE,
            message: "Local engine ready.".into(),
            bytes_done: Some(bytes),
            bytes_total: Some(bytes),
        });

        Ok(RuntimeSummary {
            python_version: py_version,
            mlx_lm_version: mlx_version,
            path: install_dir.display().to_string(),
            bytes_on_disk: bytes,
        })
    }
}

/// Strip the `com.apple.quarantine` extended attribute from the extracted
/// tree so launching `python3` doesn't trip Gatekeeper. Files written by
/// `reqwest` + `tar` shouldn't carry the bit, but this is cheap insurance.
#[cfg(all(feature = "python-mlx", target_os = "macos"))]
fn strip_quarantine(path: &Path) {
    let _ = Command::new("/usr/bin/xattr")
        .arg("-rd")
        .arg("com.apple.quarantine")
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(all(feature = "python-mlx", not(target_os = "macos")))]
fn strip_quarantine(_path: &Path) {}

/// Stream `python -m pip install …` and forward each pip status line as a
/// progress message. pip writes "Collecting…", "Downloading…", etc., one per
/// dependency — an indeterminate-but-informative feed.
#[cfg(feature = "python-mlx")]
fn run_pip_install_with_progress<F>(python_bin: &Path, on_progress: &mut F) -> Result<()>
where
    F: FnMut(SetupProgress),
{
    let mut child = Command::new(python_bin)
        .args(["-m", "pip", "install", "--no-input", "--disable-pip-version-check", "--progress-bar", "off"])
        .args(PIP_PACKAGES)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn {} -m pip install", python_bin.display()))?;

    if let Some(stdout) = child.stdout.take() {
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            on_progress(SetupProgress {
                phase: phase::INSTALLING_MLX_LM,
                message: trimmed.chars().take(120).collect(),
                bytes_done: None,
                bytes_total: None,
            });
        }
    }

    let status = child.wait().with_context(|| "wait for pip install")?;
    if !status.success() {
        let mut stderr = String::new();
        if let Some(mut err) = child.stderr.take() {
            use std::io::Read;
            let _ = err.read_to_string(&mut stderr);
        }
        return Err(anyhow!(
            "pip install of the MLX runtime packages failed (exit {}): {}",
            status,
            stderr.lines().rev().take(5).collect::<Vec<_>>().join(" | ")
        ));
    }
    Ok(())
}

/// Cheap probe: true when the generation and embedding packages (and mlx-lm,
/// which [`verify_install`] stamps) import in the given interpreter AND
/// mlx-vlm meets the version floor. The floor
/// matters: the 12B QAT model is a `gemma4_unified` architecture that
/// mlx-vlm < 0.6 cannot load, and installs from before the floor sit on 0.5.x
/// forever unless this probe fails and triggers the top-up.
#[cfg(feature = "python-mlx")]
fn stack_importable(python_bin: &Path) -> bool {
    Command::new(python_bin)
        .args([
            "-c",
            "import sys, mlx_lm, mlx_vlm, mlx_embeddings\n\
             from importlib.metadata import version\n\
             v = tuple(int(x) for x in version('mlx-vlm').split('.')[:3])\n\
             sys.exit(0 if v >= (0, 6, 13) else 1)",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Spawn the installed Python and capture (python_version, mlx_lm_version).
#[cfg(feature = "python-mlx")]
fn verify_install(python_bin: &Path) -> Result<(String, String)> {
    // The "mlx_lm" version is what we stamp as the runtime version (mlx-vlm
    // builds on mlx-lm and bumps less often).
    let output = Command::new(python_bin)
        .args([
            "-c",
            "import sys, mlx_lm, mlx_vlm, mlx_embeddings; \
             print(sys.version.split()[0]); \
             print(getattr(mlx_lm, '__version__', 'unknown'))",
        ])
        .output()
        .with_context(|| "spawn installed python for verification")?;

    if !output.status.success() {
        return Err(anyhow!(
            "installed Python failed to import the MLX runtime packages: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let py = lines.next().ok_or_else(|| anyhow!("verifier returned no Python version"))?.trim().to_string();
    let mlx = lines.next().ok_or_else(|| anyhow!("verifier returned no mlx-lm version"))?.trim().to_string();
    Ok((py, mlx))
}

/// Free bytes on the volume holding `path`, via `df -k`.
fn free_bytes(path: &Path) -> Option<u64> {
    let out = std::process::Command::new("df").arg("-k").arg(path).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    // Second line, fourth column: 1K-blocks available.
    let avail = text.lines().nth(1)?.split_whitespace().nth(3)?;
    avail.parse::<u64>().ok().map(|kb| kb * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_reads_the_stamp_and_reports_missing() {
        let root = std::env::temp_dir().join(format!("estia-py-runtime-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let rt = PythonRuntime::new(&root);
        assert_eq!(rt.status().state, RuntimeState::Missing);
        assert!(!rt.is_installed());
        assert!(rt.python_path().is_none());

        std::fs::create_dir_all(rt.install_dir().join("bin")).unwrap();
        std::fs::write(rt.install_dir().join("bin/python3"), b"#!/bin/sh\n").unwrap();
        std::fs::write(root.join(".version"), "python=3.12.7\nmlx_lm=0.30.0\n").unwrap();
        let st = rt.status();
        assert_eq!(st.state, RuntimeState::Installed);
        assert_eq!(st.python_version.as_deref(), Some("3.12.7"));
        assert_eq!(st.mlx_lm_version.as_deref(), Some("0.30.0"));
        assert!(rt.is_installed());
        assert!(st.bytes_on_disk.unwrap() > 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn remove_clears_install_partial_and_stamp() {
        let root = std::env::temp_dir().join(format!("estia-py-runtime-rm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let rt = PythonRuntime::new(&root);
        std::fs::create_dir_all(rt.install_dir().join("bin")).unwrap();
        std::fs::create_dir_all(rt.partial_dir()).unwrap();
        std::fs::write(root.join(".version"), "python=x\n").unwrap();
        assert!(rt.remove().await.unwrap());
        assert!(!rt.install_dir().exists());
        assert!(!rt.partial_dir().exists());
        assert!(!root.join(".version").exists());
        assert!(!rt.remove().await.unwrap(), "second remove finds nothing");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn preflight_passes_on_a_real_volume_for_a_small_need() {
        let rt = PythonRuntime::new(std::env::temp_dir().join("estia-py-preflight"));
        rt.preflight(1).unwrap();
    }
}
