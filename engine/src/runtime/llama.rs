//! llama.cpp runtime for the `llama-cpp` backend: upstream's prebuilt
//! `llama-server`, pinned to one build ([`llama_pins`]).
//!
//! The installer picks the archive for this OS, CPU and accelerator, checks
//! its SHA-256 against the value compiled into Estia, unpacks it, removes the
//! macOS quarantine attribute, runs `llama-server --version` (the first run
//! of an unsigned binary on macOS can stall for tens of seconds while the
//! system checks it), and swaps it into place atomically.
//!
//! Layout under `<root>/llama/` (`<root>` is the runtime root, the same
//! directory the Python runtime lives in):
//!
//! ```text
//! b11146-metal/              <-- one directory per installed variant of the pinned build
//!   llama-server, lib*, LICENSE
//!   .estia-runtime           <-- stamp: build, variant, archive, sha256, version
//! active                     <-- the variant server_path() uses ("b11146-metal")
//! .partial-b11146-metal/     <-- present only mid-install; keeps downloaded archives for a retry
//! ```
//!
//! The installer is behind the `llama-runtime` feature, for the same reason
//! `python-mlx` exists: a host that must not download executable code (a
//! sandboxed, store-distributed app, say) builds without it and ships its own
//! signed `llama-server`, passed to the engine as a path. Status, removal,
//! [`LlamaRuntime::server_path`] and the accelerator probe stay available
//! without it, so such a host can still report what is there.

use super::llama_pins::{self, LlamaAsset, LLAMA_BUILD};
use super::python::{RuntimeState, SetupProgress};
use anyhow::{anyhow, bail, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

/// Forces the variant the probe would pick (`cpu`, `vulkan`, `cuda-12`,
/// `cuda-13`, `cuda`, `rocm`, `metal`, `auto`).
pub const ENV_LLAMA_VARIANT: &str = "ESTIA_LLAMA_VARIANT";
/// A `llama-server` the user provides instead of the installed one (a
/// distribution package, or a build for a GPU the prebuilt binaries miss).
/// The engine reads it through `LlamaServer::from_env`; the installer never does.
pub const ENV_LLAMA_SERVER: &str = "ESTIA_LLAMA_SERVER";

/// How long the first `llama-server --version` may take. At least 120 s: a
/// freshly downloaded, unsigned binary has been seen to take 37 s on macOS
/// while the system checks it.
pub const FIRST_RUN_TIMEOUT: Duration = Duration::from_secs(180);

const STAMP_FILE: &str = ".estia-runtime";
const ACTIVE_FILE: &str = "active";

/// Phase strings emitted to the host, like `python::phase`.
pub mod phase {
    pub const PREPARING: &str = "llama/preparing";
    pub const DOWNLOADING: &str = "llama/downloading";
    pub const VERIFYING: &str = "llama/verifying";
    pub const EXTRACTING: &str = "llama/extracting";
    pub const CHECKING: &str = "llama/checking";
    pub const COMPLETE: &str = "llama/complete";
}

/// The `llama-server` executable's file name on this OS.
pub fn server_file_name() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

/// This OS in the pins' spelling (`macos`, `linux`, `windows`).
pub fn host_os() -> &'static str {
    std::env::consts::OS
}

/// This CPU in the pins' spelling (`aarch64`, `x86_64`).
pub fn host_arch() -> &'static str {
    std::env::consts::ARCH
}

// ── Accelerator probe and variant choice ─────────────────────────────────────

/// What the cheap accelerator checks found. Tests build this by hand.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct AcceleratorFacts {
    /// `nvidia-smi` ran and succeeded (it fails when no GPU is visible).
    pub nvidia_gpu: bool,
    /// The newest CUDA the NVIDIA driver supports (`CUDA Version` in
    /// `nvidia-smi`), when it could be read.
    pub cuda_driver: Option<(u32, u32)>,
    /// A Vulkan loader and at least one hardware device.
    pub vulkan: bool,
    /// One line per check, for the probe's explanation.
    pub notes: Vec<String>,
}

/// The variant an install would choose, and why.
#[derive(Debug, Clone, Serialize)]
pub struct LlamaProbe {
    pub os: String,
    pub arch: String,
    pub facts: AcceleratorFacts,
    /// `ESTIA_LLAMA_VARIANT`, when set.
    pub requested: Option<String>,
    /// `metal`, `cpu`, `vulkan`, `cuda-12`, `cuda-13` or `rocm`; `None` when
    /// no build fits (the reason says why).
    pub variant: Option<String>,
    /// The archive the installer would download.
    pub asset: Option<String>,
    /// Bytes to download, companion CUDA runtime included.
    pub download_bytes: Option<u64>,
    pub reason: String,
}

/// A chosen archive and the sentence explaining the choice.
#[derive(Debug, Clone)]
pub struct Selection {
    pub asset: &'static LlamaAsset,
    pub reason: String,
}

fn find_asset(os: &str, arch: &str, accel: &str) -> Option<&'static LlamaAsset> {
    llama_pins::assets_for(os, arch).find(|a| a.accel == accel)
}

fn available(os: &str, arch: &str) -> String {
    let v: Vec<&str> = llama_pins::assets_for(os, arch).map(|a| a.accel).collect();
    if v.is_empty() {
        "none".into()
    } else {
        v.join(", ")
    }
}

/// Pick the archive for `os`/`arch` from what the probe found. `requested`
/// (the `--variant` flag or `ESTIA_LLAMA_VARIANT`) wins when a build for it
/// exists and is an error when none does: a named variant never silently
/// becomes another.
///
/// Automatic order: macOS arm64 → Metal (the build includes the CPU backend);
/// macOS x64 → CPU; Linux and Windows → CUDA when `nvidia-smi` works (13 when
/// the driver supports CUDA 13, else 12; 12 when the version is unreadable;
/// none below 12), then Vulkan when a loader and a hardware device are
/// present, then CPU. ROCm is only chosen by name.
pub fn select_variant(os: &str, arch: &str, facts: &AcceleratorFacts, requested: Option<&str>) -> Result<Selection> {
    let requested = requested.map(|r| r.trim().to_ascii_lowercase()).filter(|r| !r.is_empty() && r != "auto");
    if let Some(req) = requested {
        let norm = match req.as_str() {
            "cuda12" | "cuda-12" | "cuda-12.4" | "cuda-12.8" => "cuda-12".to_string(),
            "cuda13" | "cuda-13" | "cuda-13.4" => "cuda-13".to_string(),
            "hip" => "rocm".to_string(),
            other => other.to_string(),
        };
        if norm == "cuda" {
            let order: &[&str] = match facts.cuda_driver {
                Some((maj, _)) if maj < 13 => &["cuda-12"],
                _ => &["cuda-13", "cuda-12"],
            };
            if let Some(a) = order.iter().find_map(|v| find_asset(os, arch, v)) {
                return Ok(Selection { asset: a, reason: format!("asked for cuda; {}", cuda_reason(facts, a.accel)) });
            }
            bail!("no llama.cpp CUDA build for {os} {arch} (available: {})", available(os, arch));
        }
        if norm == "cpu" && os == "macos" && arch == "aarch64" {
            if let Some(a) = find_asset(os, arch, "metal") {
                return Ok(Selection { asset: a, reason: "asked for cpu; the macOS arm64 build is Metal plus CPU".into() });
            }
        }
        return match find_asset(os, arch, &norm) {
            Some(a) => Ok(Selection { asset: a, reason: format!("asked for {norm}") }),
            None => bail!("no llama.cpp `{norm}` build for {os} {arch} (available: {})", available(os, arch)),
        };
    }

    if os == "macos" {
        let (accel, why) = if arch == "aarch64" {
            ("metal", "Apple Silicon: Metal")
        } else {
            ("cpu", "Intel Mac: CPU (upstream's x64 build has Metal off)")
        };
        return find_asset(os, arch, accel)
            .map(|a| Selection { asset: a, reason: why.into() })
            .ok_or_else(|| anyhow!("no llama.cpp build for {os} {arch}"));
    }

    let mut skipped: Vec<String> = Vec::new();
    if facts.nvidia_gpu {
        let order: &[&str] = match facts.cuda_driver {
            Some((maj, _)) if maj >= 13 => &["cuda-13", "cuda-12"],
            Some((12, _)) | None => &["cuda-12"],
            Some(_) => &[],
        };
        if let Some(a) = order.iter().find_map(|v| find_asset(os, arch, v)) {
            return Ok(Selection { asset: a, reason: cuda_reason(facts, a.accel) });
        }
        skipped.push(match facts.cuda_driver {
            Some((maj, min)) if maj < 12 => format!("NVIDIA driver supports CUDA {maj}.{min}; the builds need 12 or newer"),
            _ => format!("NVIDIA GPU found but there is no CUDA build for {os} {arch}"),
        });
    }
    if facts.vulkan {
        if let Some(a) = find_asset(os, arch, "vulkan") {
            let mut reason = String::from("Vulkan loader and a GPU device found");
            if !skipped.is_empty() {
                reason = format!("{}; {reason}", skipped.join("; "));
            }
            return Ok(Selection { asset: a, reason });
        }
        skipped.push(format!("no Vulkan build for {os} {arch}"));
    }
    match find_asset(os, arch, "cpu") {
        Some(a) => {
            let mut reason = String::from("no usable GPU found: CPU");
            if !skipped.is_empty() {
                reason = format!("{}; CPU", skipped.join("; "));
            }
            Ok(Selection { asset: a, reason })
        }
        None => bail!("no llama.cpp build for {os} {arch}"),
    }
}

fn cuda_reason(facts: &AcceleratorFacts, accel: &str) -> String {
    match facts.cuda_driver {
        Some((maj, min)) => format!("NVIDIA driver supports CUDA {maj}.{min}: {accel}"),
        None => format!("NVIDIA GPU found, driver CUDA version unreadable: {accel}"),
    }
}

/// `CUDA Version: 12.4` from `nvidia-smi`'s banner.
pub fn parse_cuda_version(nvidia_smi: &str) -> Option<(u32, u32)> {
    let rest = &nvidia_smi[nvidia_smi.find("CUDA Version")?..];
    let rest = rest.split_once(':')?.1.trim_start();
    let token: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let mut parts = token.split('.');
    let maj = parts.next()?.parse().ok()?;
    let min = parts.next().and_then(|m| m.parse().ok()).unwrap_or(0);
    Some((maj, min))
}

/// Whether `vulkaninfo --summary` lists a hardware device (discrete,
/// integrated or virtual GPU), not only a software one (llvmpipe, lavapipe).
pub fn vulkan_summary_has_gpu(summary: &str) -> bool {
    summary
        .lines()
        .filter(|l| l.contains("deviceType"))
        .any(|l| l.contains("DISCRETE_GPU") || l.contains("INTEGRATED_GPU") || l.contains("VIRTUAL_GPU"))
}

/// Run the cheap accelerator checks for `os`. Never fails: a check that
/// cannot run is a note, not an error.
pub fn detect_accelerators(os: &str) -> AcceleratorFacts {
    let mut f = AcceleratorFacts::default();
    if os == "macos" {
        f.notes.push("macOS: no probe needed".into());
        return f;
    }
    match run_with_timeout(Command::new("nvidia-smi"), Duration::from_secs(15)) {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            f.nvidia_gpu = true;
            f.cuda_driver = parse_cuda_version(&text);
            f.notes.push(match f.cuda_driver {
                Some((a, b)) => format!("nvidia-smi: GPU present, driver CUDA {a}.{b}"),
                None => "nvidia-smi: GPU present, CUDA version not shown".into(),
            });
        }
        Ok(out) => f.notes.push(format!("nvidia-smi failed ({}): {}", out.status, first_line(&out))),
        Err(e) => f.notes.push(format!("nvidia-smi not available: {e}")),
    }
    let mut vulkaninfo = Command::new(if os == "windows" { "vulkaninfo.exe" } else { "vulkaninfo" });
    vulkaninfo.arg("--summary");
    match run_with_timeout(vulkaninfo, Duration::from_secs(20)) {
        Ok(out) if out.status.success() => {
            f.vulkan = vulkan_summary_has_gpu(&String::from_utf8_lossy(&out.stdout));
            f.notes.push(if f.vulkan { "vulkaninfo: hardware device listed".into() } else { "vulkaninfo: only software devices".into() });
        }
        _ => {
            let (loader, device, note) = vulkan_files_probe(os);
            f.vulkan = loader && device;
            f.notes.push(note);
        }
    }
    f
}

/// Without `vulkaninfo`: a loader library, and on Linux a DRM render node
/// plus a hardware driver manifest (anything but the lavapipe software one).
fn vulkan_files_probe(os: &str) -> (bool, bool, String) {
    if os == "windows" {
        let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        let loader = Path::new(&root).join(r"System32\vulkan-1.dll").exists();
        // A GPU driver installs the loader; without vulkaninfo there is no
        // cheaper device check.
        return (loader, loader, format!("vulkan-1.dll {}", if loader { "present" } else { "absent" }));
    }
    const LOADERS: &[&str] = &[
        "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
        "/usr/lib64/libvulkan.so.1",
        "/usr/lib/libvulkan.so.1",
        "/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/lib/aarch64-linux-gnu/libvulkan.so.1",
        "/usr/local/lib/libvulkan.so.1",
    ];
    let loader = LOADERS.iter().any(|p| Path::new(p).exists());
    let render =
        std::fs::read_dir("/dev/dri").map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().starts_with("renderD"))).unwrap_or(false);
    let icd = ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"].iter().any(|dir| {
        std::fs::read_dir(dir)
            .map(|d| {
                d.flatten().any(|e| {
                    let n = e.file_name().to_string_lossy().to_ascii_lowercase();
                    n.ends_with(".json") && !n.starts_with("lvp")
                })
            })
            .unwrap_or(false)
    });
    let note = format!("libvulkan.so.1 {}, render node {}, hardware ICD {}", yes(loader), yes(render), yes(icd));
    (loader, render && icd, note)
}

fn yes(b: bool) -> &'static str {
    if b {
        "found"
    } else {
        "not found"
    }
}

fn first_line(out: &Output) -> String {
    let text = if out.stderr.is_empty() { &out.stdout } else { &out.stderr };
    String::from_utf8_lossy(text).lines().next().unwrap_or("").trim().chars().take(160).collect()
}

/// Run a command with piped output and a deadline; the child is killed when
/// the deadline passes.
pub(crate) fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<Output> {
    use std::io::Read;
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    let mut out_pipe = child.stdout.take();
    let mut err_pipe = child.stderr.take();
    let out_t = std::thread::spawn(move || {
        let mut b = Vec::new();
        if let Some(p) = out_pipe.as_mut() {
            let _ = p.read_to_end(&mut b);
        }
        b
    });
    let err_t = std::thread::spawn(move || {
        let mut b = Vec::new();
        if let Some(p) = err_pipe.as_mut() {
            let _ = p.read_to_end(&mut b);
        }
        b
    });
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break s;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            bail!("timed out after {} s", timeout.as_secs());
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    let stdout = out_t.join().unwrap_or_default();
    let stderr = err_t.join().unwrap_or_default();
    Ok(Output { status, stdout, stderr })
}

/// `llama-server --version`: the `version: …` line (upstream prints it on
/// stderr). `timeout` should be [`FIRST_RUN_TIMEOUT`] for a binary that has
/// never run.
pub fn server_version(server: &Path, timeout: Duration) -> Result<String> {
    let out = run_with_timeout(
        {
            let mut c = Command::new(server);
            c.arg("--version");
            c
        },
        timeout,
    )
    .map_err(|e| anyhow!("`{} --version` did not finish: {e}", server.display()))?;
    let text = format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let line = text.lines().map(str::trim).find(|l| l.starts_with("version:"));
    match (out.status.success(), line) {
        (true, Some(l)) => Ok(l.to_string()),
        _ => bail!("`{} --version` failed ({}): {}", server.display(), out.status, text.trim().lines().last().unwrap_or("")),
    }
}

/// The build number in a `--version` line: `version: 0.5.0-dev (build 11146, commit 7fe450e19)` → 11146.
pub fn parse_build(version_line: &str) -> Option<u32> {
    let rest = &version_line[version_line.find("build ")? + 6..];
    rest.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
}

/// What `--version` says about a `llama-server`, against the pinned build.
#[derive(Debug, Clone, Serialize)]
pub struct LlamaServerInfo {
    pub path: String,
    /// The `version: …` line.
    pub version: String,
    /// The upstream build number in it, when it has one.
    pub build: Option<u32>,
    /// Whether that is the build Estia pins and tests ([`LLAMA_BUILD`]).
    /// A user's own `llama-server` (`ESTIA_LLAMA_SERVER`) may be another
    /// build; it is used, with a warning, because flags and API change
    /// between builds.
    pub pinned: bool,
}

/// Run `server --version` and compare its build with the pin. Logs a warning
/// for a build other than the pinned one.
pub fn inspect_server(server: &Path, timeout: Duration) -> Result<LlamaServerInfo> {
    let version = server_version(server, timeout)?;
    let build = parse_build(&version);
    let pinned = build.is_some() && build == LLAMA_BUILD.trim_start_matches('b').parse().ok();
    if !pinned {
        tracing::warn!(server = %server.display(), version = %version, pinned = LLAMA_BUILD, "llama-server is not the build Estia pins and tests");
    }
    Ok(LlamaServerInfo { path: server.display().to_string(), version, build, pinned })
}

/// `llama-server --list-devices`: what the build can run on (GPU devices, or
/// nothing beyond the CPU). For a host's diagnostics and for a variant
/// fallback that checks a GPU build actually sees a GPU.
pub fn list_devices(server: &Path) -> Result<String> {
    let out = run_with_timeout(
        {
            let mut c = Command::new(server);
            c.arg("--list-devices");
            c
        },
        Duration::from_secs(60),
    )?;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if !out.status.success() {
        bail!("`--list-devices` failed ({}): {}", out.status, text.trim());
    }
    Ok(text)
}

// ── Status ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct LlamaRuntimeStatus {
    pub state: RuntimeState,
    /// The build this Estia pins (`b11146`).
    pub build: String,
    /// The variant `server_path` uses.
    pub variant: Option<String>,
    pub path: Option<String>,
    pub server_path: Option<String>,
    /// The `--version` line recorded at install.
    pub version: Option<String>,
    pub bytes_on_disk: Option<u64>,
    /// Every installed variant of the pinned build.
    pub installed_variants: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlamaRuntimeSummary {
    pub build: String,
    pub variant: String,
    pub path: String,
    pub server_path: String,
    pub version: String,
    pub bytes_on_disk: u64,
    /// Why this variant (the probe's reason, or "asked for …").
    pub reason: String,
}

/// The llama.cpp runtime under one runtime root.
#[derive(Debug, Clone)]
pub struct LlamaRuntime {
    root: PathBuf,
    user_agent: String,
}

impl LlamaRuntime {
    /// `root` is the runtime root (the CLI's `<data_dir>/runtime`, shared
    /// with the Python runtime); llama.cpp lives in `<root>/llama/`.
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

    /// `<root>/llama`.
    pub fn llama_dir(&self) -> PathBuf {
        self.root.join("llama")
    }

    /// `<root>/llama/<build>-<variant>`.
    pub fn install_dir(&self, variant: &str) -> PathBuf {
        self.llama_dir().join(format!("{LLAMA_BUILD}-{variant}"))
    }

    fn partial_dir(&self, variant: &str) -> PathBuf {
        self.llama_dir().join(format!(".partial-{LLAMA_BUILD}-{variant}"))
    }

    fn active_path(&self) -> PathBuf {
        self.llama_dir().join(ACTIVE_FILE)
    }

    /// Installed variants of the pinned build, sorted. Builds from an older
    /// pin are ignored (and removed by [`LlamaRuntime::remove`]).
    pub fn installed_variants(&self) -> Vec<String> {
        let prefix = format!("{LLAMA_BUILD}-");
        let mut v: Vec<String> = std::fs::read_dir(self.llama_dir())
            .map(|d| {
                d.flatten()
                    .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_prefix(&prefix)).map(str::to_string))
                    .filter(|variant| self.install_dir(variant).join(server_file_name()).is_file())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }

    /// The variant in use: the one the last install made active, else the
    /// first installed one.
    pub fn active_variant(&self) -> Option<String> {
        let installed = self.installed_variants();
        let active = std::fs::read_to_string(self.active_path()).ok().map(|s| s.trim().to_string());
        if let Some(v) = active.and_then(|a| a.strip_prefix(&format!("{LLAMA_BUILD}-")).map(str::to_string)) {
            if installed.contains(&v) {
                return Some(v);
            }
        }
        installed.into_iter().next()
    }

    /// The installed `llama-server`, when there is one.
    pub fn server_path(&self) -> Option<PathBuf> {
        self.active_variant().map(|v| self.install_dir(&v).join(server_file_name()))
    }

    pub fn is_installed(&self) -> bool {
        self.server_path().is_some()
    }

    pub fn status(&self) -> LlamaRuntimeStatus {
        let installed = self.installed_variants();
        let Some(variant) = self.active_variant() else {
            return LlamaRuntimeStatus {
                state: RuntimeState::Missing,
                build: LLAMA_BUILD.into(),
                variant: None,
                path: None,
                server_path: None,
                version: None,
                bytes_on_disk: None,
                installed_variants: installed,
            };
        };
        let dir = self.install_dir(&variant);
        let stamp = read_stamp(&dir);
        LlamaRuntimeStatus {
            state: RuntimeState::Installed,
            build: LLAMA_BUILD.into(),
            path: Some(dir.display().to_string()),
            server_path: Some(dir.join(server_file_name()).display().to_string()),
            version: stamp.iter().find(|(k, _)| k == "version").map(|(_, v)| v.clone()),
            bytes_on_disk: crate::models::dir_size(&dir),
            variant: Some(variant),
            installed_variants: installed,
        }
    }

    /// Detect accelerators and choose a variant for this machine, honouring
    /// `ESTIA_LLAMA_VARIANT`.
    pub fn probe() -> LlamaProbe {
        let (os, arch) = (host_os(), host_arch());
        let facts = detect_accelerators(os);
        let requested = std::env::var(ENV_LLAMA_VARIANT).ok().filter(|v| !v.trim().is_empty());
        Self::probe_with(os, arch, facts, requested)
    }

    /// [`LlamaRuntime::probe`] with the facts supplied.
    pub fn probe_with(os: &str, arch: &str, facts: AcceleratorFacts, requested: Option<String>) -> LlamaProbe {
        let sel = select_variant(os, arch, &facts, requested.as_deref());
        let (variant, asset, download_bytes, reason) = match sel {
            Ok(s) => {
                let reason = match &requested {
                    Some(r) if !r.eq_ignore_ascii_case("auto") => format!("{ENV_LLAMA_VARIANT}={r}: {}", s.reason),
                    _ => s.reason,
                };
                (
                    Some(s.asset.accel.to_string()),
                    Some(s.asset.file.to_string()),
                    Some(s.asset.bytes + s.asset.cudart.map(|c| c.1).unwrap_or(0)),
                    reason,
                )
            }
            Err(e) => (None, None, None, e.to_string()),
        };
        LlamaProbe { os: os.into(), arch: arch.into(), facts, requested, variant, asset, download_bytes, reason }
    }

    /// Remove every llama.cpp install under this root (all variants, older
    /// builds, partials). Returns whether anything was there.
    pub async fn remove(&self) -> Result<bool> {
        let dir = self.llama_dir();
        if !dir.exists() {
            return Ok(false);
        }
        tokio::fs::remove_dir_all(&dir).await?;
        Ok(true)
    }

    /// Remove one variant of the pinned build. Returns whether it was there.
    pub async fn remove_variant(&self, variant: &str) -> Result<bool> {
        let mut removed = false;
        for d in [self.install_dir(variant), self.partial_dir(variant)] {
            if d.exists() {
                tokio::fs::remove_dir_all(&d).await?;
                removed = true;
            }
        }
        Ok(removed)
    }

    /// Refuse an install that cannot finish, before downloading. Unpacking
    /// needs room for the archive and its contents at once; CUDA libraries
    /// unpack to about twice their archive size.
    pub fn preflight(&self, download_bytes: u64) -> Result<()> {
        let mut probe = self.root.as_path();
        while !probe.exists() {
            match probe.parent() {
                Some(p) => probe = p,
                None => return Ok(()),
            }
        }
        let Some(free) = free_bytes(probe) else {
            return Ok(());
        };
        let required = download_bytes.saturating_mul(3);
        if free < required {
            bail!(
                "Not enough disk space for llama.cpp: this needs about {} MB free and there is {} MB.",
                required / 1_000_000,
                free / 1_000_000
            );
        }
        Ok(())
    }

    /// Feature-off stub: this build contains no code that downloads
    /// executables. The signature matches, so hosts compile unchanged.
    #[cfg(not(feature = "llama-runtime"))]
    pub async fn install<F>(&self, _variant: Option<&str>, _on_progress: F) -> Result<LlamaRuntimeSummary>
    where
        F: FnMut(SetupProgress),
    {
        bail!("The llama.cpp runtime installer is not included in this build; set {ENV_LLAMA_SERVER} to a llama-server binary.")
    }

    /// Feature-off stub of [`LlamaRuntime::install_from_archive`].
    #[cfg(not(feature = "llama-runtime"))]
    pub async fn install_from_archive<F>(
        &self,
        _variant: &str,
        _archive: &Path,
        _cudart: Option<&Path>,
        _on_progress: F,
    ) -> Result<LlamaRuntimeSummary>
    where
        F: FnMut(SetupProgress),
    {
        bail!("The llama.cpp runtime installer is not included in this build.")
    }
}

fn read_stamp(dir: &Path) -> Vec<(String, String)> {
    std::fs::read_to_string(dir.join(STAMP_FILE))
        .map(|s| s.lines().filter_map(|l| l.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect())
        .unwrap_or_default()
}

/// Free bytes on the volume holding `path`, via `df -k`. `None` when the
/// probe fails (Windows has no `df`), so a guess never blocks an install.
fn free_bytes(path: &Path) -> Option<u64> {
    let out = Command::new("df").arg("-k").arg(path).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let avail = text.lines().nth(1)?.split_whitespace().nth(3)?;
    avail.parse::<u64>().ok().map(|kb| kb * 1024)
}

// ── Install (feature `llama-runtime`) ────────────────────────────────────────

/// What an install checks an archive against. Built from a pinned
/// [`LlamaAsset`]; tests build one for a synthetic archive.
#[cfg(feature = "llama-runtime")]
#[derive(Debug, Clone)]
struct ArchivePlan {
    variant: String,
    file: String,
    bytes: u64,
    sha256: String,
    cudart: Option<(String, u64, String)>,
}

#[cfg(feature = "llama-runtime")]
impl From<&LlamaAsset> for ArchivePlan {
    fn from(a: &LlamaAsset) -> Self {
        Self {
            variant: a.accel.to_string(),
            file: a.file.to_string(),
            bytes: a.bytes,
            sha256: a.sha256.to_string(),
            cudart: a.cudart.map(|(f, b, h)| (f.to_string(), b, h.to_string())),
        }
    }
}

#[cfg(feature = "llama-runtime")]
impl LlamaRuntime {
    /// Download, verify and install the pinned build. `variant` names one
    /// (`cpu`, `vulkan`, `cuda-12`, `cuda-13`, `cuda`, `rocm`, `metal`);
    /// `None` probes this machine (honouring `ESTIA_LLAMA_VARIANT`).
    ///
    /// Idempotent: an installed variant is re-checked with `--version` and
    /// made active, not downloaded again. Downloaded archives are kept in the
    /// partial directory until the install succeeds, so a retry after a
    /// failed check or a dropped connection resumes instead of refetching.
    pub async fn install<F>(&self, variant: Option<&str>, mut on_progress: F) -> Result<LlamaRuntimeSummary>
    where
        F: FnMut(SetupProgress),
    {
        let (os, arch) = (host_os(), host_arch());
        let (asset, reason) = match variant {
            Some(v) => {
                let s = select_variant(os, arch, &facts_for_request(os, v), Some(v))?;
                (s.asset, s.reason)
            }
            None => {
                let p = Self::probe();
                let asset = p.variant.as_deref().and_then(|v| find_asset(os, arch, v)).ok_or_else(|| anyhow!(p.reason.clone()))?;
                (asset, p.reason)
            }
        };
        tracing::info!(build = LLAMA_BUILD, variant = asset.accel, archive = asset.file, reason = %reason, "installing llama.cpp");
        let plan = ArchivePlan::from(asset);

        if let Some(summary) = self.reuse_installed(&plan, &reason, &mut on_progress)? {
            return Ok(summary);
        }

        on_progress(SetupProgress {
            phase: phase::PREPARING,
            message: format!("Preparing llama.cpp {LLAMA_BUILD} ({})…", plan.variant),
            bytes_done: None,
            bytes_total: None,
        });
        let total = plan.bytes + plan.cudart.as_ref().map(|c| c.1).unwrap_or(0);
        self.preflight(total)?;
        let partial = self.partial_dir(&plan.variant);
        tokio::fs::create_dir_all(&partial).await?;

        let client = reqwest::Client::builder()
            .user_agent(&self.user_agent)
            .connect_timeout(Duration::from_secs(20))
            .read_timeout(Duration::from_secs(60))
            .build()?;
        let main_path = partial.join(&plan.file);
        let base = llama_pins::LLAMA_RELEASE_BASE;
        let msg = format!("Downloading llama.cpp {LLAMA_BUILD} ({})…", plan.variant);
        download_resumable(&client, &format!("{base}/{}", plan.file), &main_path, plan.bytes, 0, total, &msg, &mut on_progress).await?;
        let cudart_path = match &plan.cudart {
            Some((file, bytes, _)) => {
                let p = partial.join(file);
                download_resumable(&client, &format!("{base}/{file}"), &p, *bytes, plan.bytes, total, &msg, &mut on_progress).await?;
                Some(p)
            }
            None => None,
        };
        let summary =
            self.install_archives(&plan, &main_path, cudart_path.as_deref(), &reason, Archives::Downloaded, &mut on_progress).await?;
        let _ = tokio::fs::remove_dir_all(&partial).await;
        Ok(summary)
    }

    /// Install from archives already on disk (an offline machine, or a
    /// cache). They must be the pinned archives for `variant` on this OS and
    /// CPU: the SHA-256 check is the same as for a download.
    pub async fn install_from_archive<F>(
        &self,
        variant: &str,
        archive: &Path,
        cudart: Option<&Path>,
        mut on_progress: F,
    ) -> Result<LlamaRuntimeSummary>
    where
        F: FnMut(SetupProgress),
    {
        let s = select_variant(host_os(), host_arch(), &facts_for_request(host_os(), variant), Some(variant))?;
        let plan = ArchivePlan::from(s.asset);
        if plan.cudart.is_some() && cudart.is_none() {
            bail!(
                "the {} build also needs its CUDA runtime archive ({})",
                plan.variant,
                plan.cudart.as_ref().map(|c| c.0.as_str()).unwrap_or("")
            );
        }
        self.install_archives(&plan, archive, cudart, &s.reason, Archives::Provided, &mut on_progress).await
    }

    /// An installed variant whose stamp names this archive: re-check it and
    /// make it active.
    fn reuse_installed<F>(&self, plan: &ArchivePlan, reason: &str, on_progress: &mut F) -> Result<Option<LlamaRuntimeSummary>>
    where
        F: FnMut(SetupProgress),
    {
        let dir = self.install_dir(&plan.variant);
        let server = dir.join(server_file_name());
        let stamp = read_stamp(&dir);
        let same = stamp.iter().any(|(k, v)| k == "sha256" && *v == plan.sha256);
        if !server.is_file() || !same {
            return Ok(None);
        }
        on_progress(SetupProgress {
            phase: phase::CHECKING,
            message: "Checking llama-server…".into(),
            bytes_done: None,
            bytes_total: None,
        });
        let version = match server_version(&server, FIRST_RUN_TIMEOUT) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "installed llama-server failed its check; reinstalling");
                return Ok(None);
            }
        };
        self.set_active(&plan.variant)?;
        let bytes = crate::models::dir_size(&dir).unwrap_or(0);
        on_progress(SetupProgress {
            phase: phase::COMPLETE,
            message: "llama.cpp ready.".into(),
            bytes_done: Some(bytes),
            bytes_total: Some(bytes),
        });
        Ok(Some(LlamaRuntimeSummary {
            build: LLAMA_BUILD.into(),
            variant: plan.variant.clone(),
            path: dir.display().to_string(),
            server_path: server.display().to_string(),
            version,
            bytes_on_disk: bytes,
            reason: reason.to_string(),
        }))
    }

    /// Verify, extract, check and swap in. Shared by the download and the
    /// local-archive paths, and what the tests drive with synthetic archives.
    async fn install_archives<F>(
        &self,
        plan: &ArchivePlan,
        main: &Path,
        cudart: Option<&Path>,
        reason: &str,
        archives: Archives,
        on_progress: &mut F,
    ) -> Result<LlamaRuntimeSummary>
    where
        F: FnMut(SetupProgress),
    {
        on_progress(SetupProgress {
            phase: phase::VERIFYING,
            message: "Checking the download…".into(),
            bytes_done: None,
            bytes_total: None,
        });
        let discard = archives == Archives::Downloaded;
        verify_sha256(main, &plan.sha256, discard).await?;
        if let (Some(path), Some((_, _, sha))) = (cudart, &plan.cudart) {
            verify_sha256(path, sha, discard).await?;
        }

        let partial = self.partial_dir(&plan.variant);
        let unpack = partial.join("unpack");
        let cudart_unpack = partial.join("unpack-cudart");
        let (bin_dir, version) = match self.unpack_and_check(plan, main, cudart, &unpack, &cudart_unpack, on_progress).await {
            Ok(v) => v,
            Err(e) => {
                // Unpacked files are never reused; downloaded archives in the
                // partial directory are, by the next attempt.
                let _ = tokio::fs::remove_dir_all(&unpack).await;
                let _ = tokio::fs::remove_dir_all(&cudart_unpack).await;
                let _ = tokio::fs::remove_dir(&partial).await;
                return Err(e);
            }
        };
        std::fs::write(
            bin_dir.join(STAMP_FILE),
            format!("build={LLAMA_BUILD}\nvariant={}\narchive={}\nsha256={}\nversion={version}\n", plan.variant, plan.file, plan.sha256),
        )?;

        // Swap in: move any earlier install aside, move the new one into
        // place, then delete the old one. The final path never holds a mix
        // of two builds; a crash between the two renames leaves it empty
        // (not installed) and the next install starts clean.
        let final_dir = self.install_dir(&plan.variant);
        let old = self.llama_dir().join(format!(".old-{LLAMA_BUILD}-{}", plan.variant));
        if old.exists() {
            tokio::fs::remove_dir_all(&old).await?;
        }
        if final_dir.exists() {
            tokio::fs::rename(&final_dir, &old).await?;
        }
        tokio::fs::rename(&bin_dir, &final_dir).await?;
        if old.exists() {
            let _ = tokio::fs::remove_dir_all(&old).await;
        }
        let _ = tokio::fs::remove_dir_all(&unpack).await;
        let _ = tokio::fs::remove_dir_all(&cudart_unpack).await;
        // Only removed when empty: a download keeps its archives there until
        // `install` has finished.
        let _ = tokio::fs::remove_dir(&partial).await;
        self.set_active(&plan.variant)?;

        let bytes = crate::models::dir_size(&final_dir).unwrap_or(0);
        on_progress(SetupProgress {
            phase: phase::COMPLETE,
            message: "llama.cpp ready.".into(),
            bytes_done: Some(bytes),
            bytes_total: Some(bytes),
        });
        Ok(LlamaRuntimeSummary {
            build: LLAMA_BUILD.into(),
            variant: plan.variant.clone(),
            path: final_dir.display().to_string(),
            server_path: final_dir.join(server_file_name()).display().to_string(),
            version,
            bytes_on_disk: bytes,
            reason: reason.to_string(),
        })
    }

    /// Unpack the archives into the partial directory, put the CUDA runtime
    /// next to `llama-server`, clear quarantine, and run `--version`.
    /// Returns the directory holding `llama-server` and its version line.
    async fn unpack_and_check<F>(
        &self,
        plan: &ArchivePlan,
        main: &Path,
        cudart: Option<&Path>,
        unpack: &Path,
        cudart_unpack: &Path,
        on_progress: &mut F,
    ) -> Result<(PathBuf, String)>
    where
        F: FnMut(SetupProgress),
    {
        on_progress(SetupProgress {
            phase: phase::EXTRACTING,
            message: "Unpacking llama.cpp…".into(),
            bytes_done: None,
            bytes_total: None,
        });
        for d in [unpack, cudart_unpack] {
            if d.exists() {
                tokio::fs::remove_dir_all(d).await?;
            }
        }
        let (main_owned, unpack_owned) = (main.to_path_buf(), unpack.to_path_buf());
        tokio::task::spawn_blocking(move || extract_archive(&main_owned, &unpack_owned))
            .await
            .map_err(|e| anyhow!("extract task: {e}"))??;
        let bin_dir = locate_server_dir(unpack).ok_or_else(|| anyhow!("the archive has no {}", server_file_name()))?;
        if let Some(path) = cudart {
            let (p, d) = (path.to_path_buf(), cudart_unpack.to_path_buf());
            tokio::task::spawn_blocking(move || extract_archive(&p, &d)).await.map_err(|e| anyhow!("extract task: {e}"))??;
            flatten_into(cudart_unpack, &bin_dir)?;
        }
        strip_quarantine(&bin_dir);

        on_progress(SetupProgress {
            phase: phase::CHECKING,
            message: "Checking llama-server (the first start can take a minute on macOS)…".into(),
            bytes_done: None,
            bytes_total: None,
        });
        let server = bin_dir.join(server_file_name());
        let version = tokio::task::spawn_blocking(move || server_version(&server, FIRST_RUN_TIMEOUT))
            .await
            .map_err(|e| anyhow!("check task: {e}"))??;
        let pinned: Option<u32> = LLAMA_BUILD.trim_start_matches('b').parse().ok();
        if parse_build(&version) != pinned {
            tracing::warn!(version = %version, pinned = LLAMA_BUILD, variant = %plan.variant, "llama-server reports a different build than the pin");
        }
        Ok((bin_dir, version))
    }

    fn set_active(&self, variant: &str) -> Result<()> {
        let path = self.active_path();
        let tmp = path.with_extension("tmp");
        std::fs::create_dir_all(self.llama_dir())?;
        std::fs::write(&tmp, format!("{LLAMA_BUILD}-{variant}\n"))?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }
}

/// A named variant needs no probe, except `cuda`, which picks 12 or 13 by
/// the driver, and `auto` (or nothing), which is the probe.
#[cfg(feature = "llama-runtime")]
fn facts_for_request(os: &str, variant: &str) -> AcceleratorFacts {
    match variant.trim().to_ascii_lowercase().as_str() {
        "cuda" | "auto" | "" => detect_accelerators(os),
        _ => AcceleratorFacts::default(),
    }
}

/// Fetch `url` into `target`, resuming a shorter partial with `Range`.
#[cfg(feature = "llama-runtime")]
#[allow(clippy::too_many_arguments)]
async fn download_resumable<F>(
    client: &reqwest::Client,
    url: &str,
    target: &Path,
    expected: u64,
    base: u64,
    total: u64,
    message: &str,
    on_progress: &mut F,
) -> Result<()>
where
    F: FnMut(SetupProgress),
{
    use tokio::io::AsyncWriteExt;
    let on_disk = tokio::fs::metadata(target).await.map(|m| m.len()).unwrap_or(0);
    if on_disk == expected {
        return Ok(()); // complete from an earlier attempt; the hash check decides
    }
    let mut from = if on_disk < expected { on_disk } else { 0 };
    let mut req = client.get(url);
    if from > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={from}-"));
    }
    let mut resp = req.send().await?.error_for_status()?;
    if from > 0 && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        from = 0;
    }
    let mut out =
        if from > 0 { tokio::fs::OpenOptions::new().append(true).open(target).await? } else { tokio::fs::File::create(target).await? };
    let mut done = from;
    let mut last = Instant::now();
    on_progress(SetupProgress {
        phase: phase::DOWNLOADING,
        message: message.into(),
        bytes_done: Some(base + done),
        bytes_total: Some(total),
    });
    while let Some(chunk) = resp.chunk().await? {
        out.write_all(&chunk).await?;
        done += chunk.len() as u64;
        if last.elapsed() >= Duration::from_millis(150) {
            last = Instant::now();
            on_progress(SetupProgress {
                phase: phase::DOWNLOADING,
                message: message.into(),
                bytes_done: Some(base + done),
                bytes_total: Some(total),
            });
        }
    }
    out.flush().await?;
    on_progress(SetupProgress {
        phase: phase::DOWNLOADING,
        message: message.into(),
        bytes_done: Some(base + done),
        bytes_total: Some(total),
    });
    if done != expected {
        bail!("downloaded {done} of {expected} bytes from {url}");
    }
    Ok(())
}

/// Whose archives an install is unpacking.
#[cfg(feature = "llama-runtime")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Archives {
    /// Fetched by the installer into its partial directory: one that fails
    /// its hash is deleted, so a retry fetches it again.
    Downloaded,
    /// The caller's own files (`install_from_archive`): never deleted.
    Provided,
}

/// Hash `path` and compare. On a mismatch nothing is extracted, and the file
/// is deleted when `discard` is set.
#[cfg(feature = "llama-runtime")]
async fn verify_sha256(path: &Path, expected: &str, discard: bool) -> Result<()> {
    use sha2::{Digest, Sha256};
    use tokio::io::AsyncReadExt;
    let mut f = tokio::fs::File::open(path).await.map_err(|e| anyhow!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = crate::models::hf::hex_digest(hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected) {
        if discard {
            let _ = tokio::fs::remove_file(path).await;
        }
        bail!(
            "{} failed its integrity check (expected {expected}, got {actual}). It may be corrupted or tampered with; nothing was installed.",
            path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
        );
    }
    Ok(())
}

/// Unpack a `.tar.gz` (Rust, no external tool) or a `.zip` (Windows'
/// built-in `tar.exe`, which reads zip; no zip crate needed) into `dest`.
#[cfg(feature = "llama-runtime")]
fn extract_archive(archive: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    let name = archive.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        let file = std::fs::File::open(archive)?;
        let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
        tar.set_preserve_permissions(true);
        // `unpack` refuses entries that would land outside `dest`.
        tar.unpack(dest).map_err(|e| anyhow!("unpack {}: {e}", archive.display()))?;
        return Ok(());
    }
    if name.ends_with(".zip") {
        let tar_exe = if cfg!(windows) {
            PathBuf::from(std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into())).join(r"System32\tar.exe")
        } else {
            PathBuf::from("tar") // bsdtar on macOS reads zip too
        };
        let out = Command::new(&tar_exe).arg("-xf").arg(archive).arg("-C").arg(dest).output()?;
        if !out.status.success() {
            bail!("unpack {}: {}", archive.display(), String::from_utf8_lossy(&out.stderr).trim());
        }
        return Ok(());
    }
    bail!("unknown archive type: {}", archive.display())
}

/// The directory holding `llama-server` inside an unpacked archive: the root
/// or a directory a few levels down (`llama-b11146/`, `build/bin/`).
#[cfg(any(feature = "llama-runtime", test))]
pub(crate) fn locate_server_dir(root: &Path) -> Option<PathBuf> {
    let mut frontier = vec![root.to_path_buf()];
    for _ in 0..4 {
        let mut next = Vec::new();
        for dir in frontier {
            if dir.join(server_file_name()).is_file() {
                return Some(dir);
            }
            if let Ok(rd) = std::fs::read_dir(&dir) {
                let mut subdirs: Vec<PathBuf> =
                    rd.flatten().filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false)).map(|e| e.path()).collect();
                subdirs.sort();
                next.extend(subdirs);
            }
        }
        frontier = next;
    }
    None
}

/// Move every file (and symlink) under `from` into `into`, flattened: the
/// CUDA runtime libraries go next to `llama-server`, where its `$ORIGIN`
/// rpath (or Windows' DLL search) finds them.
#[cfg(feature = "llama-runtime")]
fn flatten_into(from: &Path, into: &Path) -> Result<()> {
    let mut stack = vec![from.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir)?.flatten() {
            let ft = e.file_type()?;
            if ft.is_dir() {
                stack.push(e.path());
            } else {
                let target = into.join(e.file_name());
                if target.symlink_metadata().is_ok() {
                    std::fs::remove_file(&target)?;
                }
                std::fs::rename(e.path(), target)?;
            }
        }
    }
    Ok(())
}

/// Remove `com.apple.quarantine` so the first run is not blocked by
/// Gatekeeper. Files written by reqwest and tar do not carry it, but an
/// archive a user downloaded with a browser for `install_from_archive` does.
#[cfg(all(feature = "llama-runtime", target_os = "macos"))]
fn strip_quarantine(path: &Path) {
    let _ = Command::new("/usr/bin/xattr")
        .arg("-rd")
        .arg("com.apple.quarantine")
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(all(feature = "llama-runtime", not(target_os = "macos")))]
fn strip_quarantine(_path: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(nvidia: bool, cuda: Option<(u32, u32)>, vulkan: bool) -> AcceleratorFacts {
        AcceleratorFacts { nvidia_gpu: nvidia, cuda_driver: cuda, vulkan, notes: vec![] }
    }

    fn pick(os: &str, arch: &str, f: &AcceleratorFacts, req: Option<&str>) -> String {
        select_variant(os, arch, f, req).map(|s| s.asset.accel.to_string()).unwrap_or_else(|e| format!("error: {e}"))
    }

    #[test]
    fn selection_follows_the_probe() {
        let none = facts(false, None, false);
        assert_eq!(pick("macos", "aarch64", &none, None), "metal");
        assert_eq!(pick("macos", "x86_64", &none, None), "cpu");
        assert_eq!(pick("linux", "x86_64", &none, None), "cpu");
        assert_eq!(pick("linux", "aarch64", &none, None), "cpu");
        assert_eq!(pick("windows", "x86_64", &none, None), "cpu");
        assert_eq!(pick("windows", "aarch64", &none, None), "cpu");

        // NVIDIA: CUDA 13 when the driver supports it, 12 otherwise or when
        // the version is unreadable, none below 12.
        assert_eq!(pick("linux", "x86_64", &facts(true, Some((13, 1)), true), None), "cuda-13");
        assert_eq!(pick("linux", "x86_64", &facts(true, Some((12, 4)), true), None), "cuda-12");
        assert_eq!(pick("linux", "x86_64", &facts(true, None, false), None), "cuda-12");
        assert_eq!(pick("windows", "x86_64", &facts(true, Some((13, 0)), false), None), "cuda-13");
        assert_eq!(pick("linux", "x86_64", &facts(true, Some((11, 8)), true), None), "vulkan");
        assert_eq!(pick("linux", "x86_64", &facts(true, Some((11, 8)), false), None), "cpu");
        let s = select_variant("linux", "x86_64", &facts(true, Some((11, 8)), false), None).unwrap();
        assert!(s.reason.contains("CUDA 11.8"), "{}", s.reason);
        // No CUDA build for Linux arm64 in the pins: Vulkan, then CPU.
        assert_eq!(pick("linux", "aarch64", &facts(true, Some((13, 0)), true), None), "vulkan");
        assert_eq!(pick("linux", "aarch64", &facts(true, Some((13, 0)), false), None), "cpu");

        // Vulkan without NVIDIA.
        assert_eq!(pick("linux", "x86_64", &facts(false, None, true), None), "vulkan");
        assert_eq!(pick("windows", "x86_64", &facts(false, None, true), None), "vulkan");
        // Windows arm64 has only a CPU build.
        assert_eq!(pick("windows", "aarch64", &facts(false, None, true), None), "cpu");
        // macOS ignores GPU facts.
        assert_eq!(pick("macos", "aarch64", &facts(true, Some((13, 0)), true), None), "metal");
    }

    #[test]
    fn a_named_variant_wins_or_fails() {
        let none = facts(false, None, false);
        assert_eq!(pick("linux", "x86_64", &none, Some("vulkan")), "vulkan");
        assert_eq!(pick("linux", "x86_64", &none, Some("CUDA12")), "cuda-12");
        assert_eq!(pick("linux", "x86_64", &none, Some("cuda-13.4")), "cuda-13");
        assert_eq!(pick("linux", "x86_64", &none, Some("rocm")), "rocm");
        assert_eq!(pick("linux", "x86_64", &facts(true, Some((12, 8)), false), Some("cuda")), "cuda-12");
        assert_eq!(pick("linux", "x86_64", &facts(true, Some((13, 0)), false), Some("cuda")), "cuda-13");
        assert_eq!(pick("linux", "x86_64", &none, Some("auto")), "cpu");
        assert_eq!(pick("linux", "x86_64", &facts(false, None, true), Some(" ")), "vulkan");
        assert_eq!(pick("macos", "aarch64", &none, Some("cpu")), "metal");
        assert!(pick("linux", "aarch64", &none, Some("cuda")).starts_with("error"));
        assert!(pick("macos", "aarch64", &none, Some("vulkan")).starts_with("error"));
        assert!(pick("linux", "x86_64", &none, Some("sycl")).contains("available: cpu"));
        assert!(pick("freebsd", "x86_64", &none, None).starts_with("error"));

        let p = LlamaRuntime::probe_with("linux", "x86_64", facts(true, Some((12, 8)), false), Some("vulkan".into()));
        assert_eq!(p.variant.as_deref(), Some("vulkan"));
        assert!(p.reason.starts_with("ESTIA_LLAMA_VARIANT=vulkan"), "{}", p.reason);
        let p = LlamaRuntime::probe_with("linux", "x86_64", facts(true, Some((12, 8)), false), None);
        assert_eq!(p.variant.as_deref(), Some("cuda-12"));
        assert_eq!(p.download_bytes, Some(168_920_581 + 594_373_356), "CUDA pairs with its runtime archive");
    }

    #[test]
    fn parses_driver_and_server_output() {
        let smi = "| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4     |";
        assert_eq!(parse_cuda_version(smi), Some((12, 4)));
        assert_eq!(parse_cuda_version("CUDA Version: 13.0 |"), Some((13, 0)));
        assert_eq!(parse_cuda_version("no gpu"), None);
        assert!(vulkan_summary_has_gpu("GPU0:\n\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"));
        assert!(!vulkan_summary_has_gpu("GPU0:\n\tdeviceType = PHYSICAL_DEVICE_TYPE_CPU\n\tdeviceName = llvmpipe"));
        assert_eq!(parse_build("version: 0.5.0-dev (build 11146, commit 7fe450e19)"), Some(11146));
        assert_eq!(parse_build("version: 11146 (7fe450e19)"), None);
    }

    /// A user's own llama-server is compared with the pin. Also checks the
    /// real one when `ESTIA_LLAMA_SERVER` points at the pinned build.
    #[cfg(unix)]
    #[test]
    fn inspects_a_servers_build() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("estia-llama-inspect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("llama-server");
        std::fs::write(&fake, "#!/bin/sh\necho 'version: 0.4.0 (build 10999, commit abc)' >&2\n").unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        let info = inspect_server(&fake, Duration::from_secs(20)).unwrap();
        assert_eq!((info.build, info.pinned), (Some(10999), false));
        assert!(inspect_server(&dir.join("missing"), Duration::from_secs(5)).is_err());
        if let Some(real) = std::env::var_os(ENV_LLAMA_SERVER) {
            let info = inspect_server(Path::new(&real), FIRST_RUN_TIMEOUT).unwrap();
            eprintln!("{info:?}");
            assert!(info.pinned, "{info:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_and_server_path_follow_the_layout() {
        let root = std::env::temp_dir().join(format!("estia-llama-rt-status-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let rt = LlamaRuntime::new(&root);
        assert_eq!(rt.status().state, RuntimeState::Missing);
        assert!(rt.server_path().is_none());
        // An older pin's directory is ignored.
        std::fs::create_dir_all(rt.llama_dir().join("b1-cpu")).unwrap();
        std::fs::write(rt.llama_dir().join("b1-cpu").join(server_file_name()), b"x").unwrap();
        assert!(!rt.is_installed());
        for v in ["cpu", "vulkan"] {
            std::fs::create_dir_all(rt.install_dir(v)).unwrap();
            std::fs::write(rt.install_dir(v).join(server_file_name()), b"x").unwrap();
        }
        std::fs::write(rt.install_dir("vulkan").join(STAMP_FILE), "build=b11146\nversion=version: test\n").unwrap();
        assert_eq!(rt.installed_variants(), vec!["cpu", "vulkan"]);
        assert_eq!(rt.active_variant().as_deref(), Some("cpu"), "no active file: the first installed");
        std::fs::write(rt.llama_dir().join(ACTIVE_FILE), format!("{LLAMA_BUILD}-vulkan\n")).unwrap();
        assert_eq!(rt.server_path(), Some(rt.install_dir("vulkan").join(server_file_name())));
        let st = rt.status();
        assert_eq!(st.state, RuntimeState::Installed);
        assert_eq!(st.variant.as_deref(), Some("vulkan"));
        assert_eq!(st.version.as_deref(), Some("version: test"));
        let rt2 = rt.clone();
        let tokio_rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        assert!(tokio_rt.block_on(rt2.remove_variant("vulkan")).unwrap());
        assert_eq!(rt.active_variant().as_deref(), Some("cpu"), "a stale active file falls back");
        assert!(tokio_rt.block_on(rt.remove()).unwrap());
        assert!(!tokio_rt.block_on(rt.remove()).unwrap());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn locates_the_server_in_nested_layouts() {
        let root = std::env::temp_dir().join(format!("estia-llama-locate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let nested = root.join("llama-b1").join("build").join("bin");
        std::fs::create_dir_all(&nested).unwrap();
        assert!(locate_server_dir(&root).is_none());
        std::fs::write(nested.join(server_file_name()), b"x").unwrap();
        assert_eq!(locate_server_dir(&root), Some(nested));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The installer end to end, without the network, on a synthetic archive
    /// whose `llama-server` is a shell script printing a version line.
    #[cfg(all(feature = "llama-runtime", unix))]
    mod install {
        use super::super::*;
        use std::io::Write;

        fn synthetic_archive(dir: &Path, name: &str, with_cudart_lib: bool) -> PathBuf {
            let path = dir.join(name);
            let gz = flate2::write::GzEncoder::new(std::fs::File::create(&path).unwrap(), flate2::Compression::fast());
            let mut tar = tar::Builder::new(gz);
            let mut add = |p: &str, body: &[u8], mode: u32| {
                let mut h = tar::Header::new_gnu();
                h.set_size(body.len() as u64);
                h.set_mode(mode);
                h.set_cksum();
                tar.append_data(&mut h, p, body).unwrap();
            };
            if with_cudart_lib {
                add("cudart/libcudart.so.12", b"not really a library", 0o644);
            } else {
                add("llama-b1/llama-server", b"#!/bin/sh\necho 'version: 0.0.1-test (build 1, commit abc)' >&2\nexit 0\n", 0o755);
                add("llama-b1/libllama.so", b"lib", 0o644);
                add("llama-b1/LICENSE", b"MIT", 0o644);
            }
            tar.into_inner().unwrap().finish().unwrap().flush().unwrap();
            path
        }

        fn sha(path: &Path) -> String {
            use sha2::{Digest, Sha256};
            crate::models::hf::hex_digest(Sha256::digest(std::fs::read(path).unwrap()))
        }

        fn scratch(name: &str) -> PathBuf {
            let d = std::env::temp_dir().join(format!("estia-llama-install-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            d
        }

        #[tokio::test]
        async fn installs_a_synthetic_archive_and_its_cuda_runtime() {
            let dir = scratch("ok");
            let main = synthetic_archive(&dir, "llama-b1-bin-test.tar.gz", false);
            let cudart = synthetic_archive(&dir, "cudart-test.tar.gz", true);
            let plan = ArchivePlan {
                variant: "cuda-12".into(),
                file: "llama-b1-bin-test.tar.gz".into(),
                bytes: std::fs::metadata(&main).unwrap().len(),
                sha256: sha(&main),
                cudart: Some(("cudart-test.tar.gz".into(), std::fs::metadata(&cudart).unwrap().len(), sha(&cudart))),
            };
            let rt = LlamaRuntime::new(dir.join("runtime"));
            let mut phases = Vec::new();
            let s = rt
                .install_archives(&plan, &main, Some(&cudart), "test", Archives::Provided, &mut |p: SetupProgress| phases.push(p.phase))
                .await
                .unwrap();
            assert_eq!(s.variant, "cuda-12");
            assert_eq!(s.version, "version: 0.0.1-test (build 1, commit abc)");
            let installed = rt.install_dir("cuda-12");
            assert!(installed.join("llama-server").is_file());
            assert!(installed.join("libcudart.so.12").is_file(), "the CUDA runtime sits next to llama-server");
            assert!(installed.join(".estia-runtime").is_file());
            assert_eq!(rt.server_path(), Some(installed.join("llama-server")));
            assert_eq!(rt.status().variant.as_deref(), Some("cuda-12"));
            assert_eq!(phases.first(), Some(&phase::VERIFYING));
            assert_eq!(phases.last(), Some(&phase::COMPLETE));

            // Installing again swaps the directory in place.
            rt.install_archives(&plan, &main, Some(&cudart), "test", Archives::Provided, &mut |_| {}).await.unwrap();
            assert!(installed.join("llama-server").is_file());
            assert!(!rt.llama_dir().join(format!(".old-{LLAMA_BUILD}-cuda-12")).exists());
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[tokio::test]
        async fn refuses_an_archive_with_one_byte_changed() {
            let dir = scratch("corrupt");
            let main = synthetic_archive(&dir, "llama-b1-bin-test.tar.gz", false);
            let plan = ArchivePlan {
                variant: "cpu".into(),
                file: "llama-b1-bin-test.tar.gz".into(),
                bytes: std::fs::metadata(&main).unwrap().len(),
                sha256: sha(&main),
                cudart: None,
            };
            let mut bytes = std::fs::read(&main).unwrap();
            let mid = bytes.len() / 2;
            bytes[mid] ^= 0x01;
            std::fs::write(&main, &bytes).unwrap();
            let rt = LlamaRuntime::new(dir.join("runtime"));
            // The caller's own archive is refused and left alone.
            let err = rt.install_archives(&plan, &main, None, "test", Archives::Provided, &mut |_| {}).await.unwrap_err();
            assert!(err.to_string().contains("integrity check"), "{err}");
            assert!(main.exists(), "a caller's archive is never deleted");
            assert!(!rt.is_installed());
            // A downloaded one is deleted so a retry fetches it again.
            let err = rt.install_archives(&plan, &main, None, "test", Archives::Downloaded, &mut |_| {}).await.unwrap_err();
            assert!(err.to_string().contains("integrity check"), "{err}");
            assert!(!main.exists(), "a download that failed its hash is deleted so a retry refetches it");
            assert!(!rt.is_installed());
            assert!(!rt.install_dir("cpu").exists());
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[tokio::test]
        async fn a_server_that_fails_its_check_is_not_installed() {
            let dir = scratch("badserver");
            let path = dir.join("bad.tar.gz");
            {
                let gz = flate2::write::GzEncoder::new(std::fs::File::create(&path).unwrap(), flate2::Compression::fast());
                let mut tar = tar::Builder::new(gz);
                let body = b"#!/bin/sh\necho 'dyld: Library not loaded' >&2\nexit 1\n";
                let mut h = tar::Header::new_gnu();
                h.set_size(body.len() as u64);
                h.set_mode(0o755);
                h.set_cksum();
                tar.append_data(&mut h, "llama-server", &body[..]).unwrap();
                tar.into_inner().unwrap().finish().unwrap();
            }
            let plan = ArchivePlan { variant: "cpu".into(), file: "bad.tar.gz".into(), bytes: 0, sha256: sha(&path), cudart: None };
            let rt = LlamaRuntime::new(dir.join("runtime"));
            let err = rt.install_archives(&plan, &path, None, "test", Archives::Provided, &mut |_| {}).await.unwrap_err();
            assert!(err.to_string().contains("Library not loaded"), "{err}");
            assert!(!rt.is_installed());
            let _ = std::fs::remove_dir_all(&dir);
        }

        fn live_root(name: &str) -> PathBuf {
            let base = std::env::var_os("ESTIA_LLAMA_INSTALL_ROOT").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
            let root = base.join(format!("estia-llama-live-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            root
        }

        /// The real thing: probe this machine, download the pinned archive
        /// from GitHub (11 to 33 MB for CPU, Metal and Vulkan; hundreds of MB
        /// for CUDA), verify, unpack, run `llama-server --version`.
        #[tokio::test]
        #[ignore = "downloads llama.cpp from github.com"]
        async fn live_install_on_this_machine() {
            let root = live_root("download");
            let rt = LlamaRuntime::new(&root);
            let probe = LlamaRuntime::probe();
            eprintln!("probe: {probe:?}");
            let mut last = String::new();
            let s = rt
                .install(None, |p: SetupProgress| {
                    if p.phase != last {
                        eprintln!("{} {}", p.phase, p.message);
                        last = p.phase.to_string();
                    }
                })
                .await
                .unwrap();
            eprintln!("installed: {s:?}");
            assert_eq!(Some(s.variant.clone()), probe.variant);
            assert_eq!(parse_build(&s.version), LLAMA_BUILD.trim_start_matches('b').parse().ok());
            let server = rt.server_path().unwrap();
            let v = server_version(&server, Duration::from_secs(60)).unwrap();
            eprintln!("{} --version: {v}", server.display());
            eprintln!("devices:\n{}", list_devices(&server).unwrap_or_else(|e| e.to_string()));
            // Idempotent: a second install reuses what is there.
            let again = rt.install(None, |_| {}).await.unwrap();
            assert_eq!(again.path, s.path);
            assert!(rt.remove().await.unwrap());
            assert!(!rt.is_installed());
        }

        /// An archive already on disk (`ESTIA_LLAMA_ARCHIVE`, the pinned one
        /// for this machine) installs through the same checks, offline.
        #[tokio::test]
        async fn installs_the_pinned_archive_from_disk() {
            let Some(archive) = std::env::var_os("ESTIA_LLAMA_ARCHIVE").map(PathBuf::from) else {
                return;
            };
            let variant = std::env::var("ESTIA_LLAMA_ARCHIVE_VARIANT").unwrap_or_else(|_| "metal".into());
            let root = live_root("archive");
            let rt = LlamaRuntime::new(&root);
            let s = rt.install_from_archive(&variant, &archive, None, |_| {}).await.unwrap();
            eprintln!("installed from {}: {s:?}", archive.display());
            assert!(rt.server_path().unwrap().is_file());
            assert_eq!(parse_build(&s.version), LLAMA_BUILD.trim_start_matches('b').parse().ok());
            assert!(archive.exists(), "a verified archive is left where it was");
            assert!(rt.remove().await.unwrap());
        }
    }
}
