//! What this machine can hold, and the memory policy that follows from it.
//!
//! A resident model is gigabytes of unified memory. On a 16 GB fanless
//! MacBook Air, one Gemma 4 runner with MLX's default wired-memory limit
//! (two thirds of RAM, requested on every generation) stalled the whole
//! machine: wired pages cannot be compressed or paged, so every allocation
//! elsewhere took the slow path. [`MachineProfile`] classifies the host into a
//! [`DeviceTier`], and [`MemoryPolicy`] turns the tier into the numbers the
//! rest of the engine reads: how much memory resident models may use in
//! total, how long an idle model stays loaded, and the caps handed to an MLX
//! runner.
//!
//! Detection happens once per process. Two environment variables let you be
//! a small machine on a big one, which is how this policy is tested:
//!
//! ```text
//! ESTIA_DEVICE_TIER=constrained|standard|capable   # the tier, whatever was detected
//! ESTIA_FAKE_RAM_GB=8                              # detected RAM (the tier is re-derived)
//! ESTIA_MEMORY_BUDGET=6GB|off                      # the budget, whatever the tier says
//! ```

use serde::Serialize;
use std::sync::OnceLock;
use std::time::Duration;

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

/// Overrides the detected tier.
pub const DEVICE_TIER_ENV: &str = "ESTIA_DEVICE_TIER";
/// Overrides detected RAM, in whole GB; the tier is derived again from it.
pub const FAKE_RAM_ENV: &str = "ESTIA_FAKE_RAM_GB";
/// Overrides the memory budget: a size (`6GB`, `6144MB`, bytes) or `off`.
pub const MEMORY_BUDGET_ENV: &str = "ESTIA_MEMORY_BUDGET";

/// How much of a footprint the host can absorb.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceTier {
    /// 16 GB of RAM or less, no fan, or 4 cores or fewer: one generation
    /// model at a time, short idle window, tight MLX caps.
    Constrained,
    /// More than 16 GB with a fan.
    Standard,
    /// 32 GB or more with a fan.
    Capable,
}

impl DeviceTier {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceTier::Constrained => "constrained",
            DeviceTier::Standard => "standard",
            DeviceTier::Capable => "capable",
        }
    }
}

impl std::str::FromStr for DeviceTier {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "constrained" => Ok(DeviceTier::Constrained),
            "standard" => Ok(DeviceTier::Standard),
            "capable" => Ok(DeviceTier::Capable),
            other => Err(format!("unknown device tier `{other}` (constrained, standard or capable)")),
        }
    }
}

impl std::fmt::Display for DeviceTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What could be learned about the host.
#[derive(Debug, Clone, Serialize)]
pub struct MachineProfile {
    /// Physical memory in bytes.
    pub ram_bytes: u64,
    pub logical_cores: u32,
    /// `machdep.cpu.brand_string` on macOS, e.g. "Apple M2"; else "unknown".
    pub chip: String,
    /// `hw.model` on macOS, e.g. "Mac14,2"; else "unknown".
    pub model_id: String,
    /// Passively cooled (from an `hw.model` list, see [`is_fanless_model`]).
    pub fanless: bool,
    pub tier: DeviceTier,
    /// Environment overrides that changed this profile, as `NAME=value`.
    pub overrides: Vec<String>,
}

impl MachineProfile {
    /// RAM in whole GB, rounded down.
    pub fn ram_gb(&self) -> u64 {
        self.ram_bytes / GIB
    }
}

/// The tier rule. Constrained wins: a fanless 32 GB Air is constrained,
/// because it throttles under sustained inference whatever its memory.
pub fn classify(ram_bytes: u64, logical_cores: u32, fanless: bool) -> DeviceTier {
    let ram_gb = ram_bytes / GIB;
    if ram_gb <= 16 || fanless || logical_cores <= 4 {
        DeviceTier::Constrained
    } else if ram_gb >= 32 {
        DeviceTier::Capable
    } else {
        DeviceTier::Standard
    }
}

/// Passively cooled Macs by `hw.model`. macOS has no API for "has a fan", so
/// this list needs a line for every fanless Mac Apple ships; a missing one
/// falls back to the RAM and core rule, which puts most Airs in
/// `Constrained` anyway.
pub fn is_fanless_model(model_id: &str) -> bool {
    // Lines where every model is fanless.
    const FAMILIES: &[&str] = &["MacBook8,", "MacBook9,", "MacBook10,", "MacBookAir10,"];
    // Airs inside the `MacN,M` namespace, which they share with fanned models.
    const MODELS: &[&str] = &["Mac14,2", "Mac14,15", "Mac15,12", "Mac15,13", "Mac16,12", "Mac16,13"];
    FAMILIES.iter().any(|f| model_id.starts_with(f)) || MODELS.contains(&model_id)
}

fn detect() -> MachineProfile {
    // When something cannot be read, assume a small machine: under-committing
    // costs a reload, over-committing can stall the host.
    let ram_bytes = sys::ram_bytes().unwrap_or(8 * GIB);
    let logical_cores = sys::logical_cores().or_else(|| std::thread::available_parallelism().ok().map(|n| n.get() as u32)).unwrap_or(4);
    let chip = sys::string("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into());
    let model_id = sys::string("hw.model").unwrap_or_else(|| "unknown".into());
    let fanless = is_fanless_model(&model_id);
    MachineProfile {
        ram_bytes,
        logical_cores,
        chip,
        model_id,
        fanless,
        tier: classify(ram_bytes, logical_cores, fanless),
        overrides: Vec::new(),
    }
}

/// Apply [`FAKE_RAM_ENV`] then [`DEVICE_TIER_ENV`] to a detected profile.
/// Values that do not parse are logged and ignored.
pub fn apply_overrides(mut p: MachineProfile, tier: Option<&str>, fake_ram_gb: Option<&str>) -> MachineProfile {
    if let Some(raw) = fake_ram_gb {
        match raw.trim().parse::<u64>() {
            Ok(gb) if gb > 0 => {
                p.ram_bytes = gb * GIB;
                p.tier = classify(p.ram_bytes, p.logical_cores, p.fanless);
                p.overrides.push(format!("{FAKE_RAM_ENV}={gb}"));
            }
            _ => tracing::warn!(value = %raw, "{FAKE_RAM_ENV} is not a whole number of GB; ignored"),
        }
    }
    if let Some(raw) = tier {
        match raw.parse::<DeviceTier>() {
            Ok(t) => {
                p.tier = t;
                p.overrides.push(format!("{DEVICE_TIER_ENV}={t}"));
            }
            Err(e) => tracing::warn!(value = %raw, "{e}; ignored"),
        }
    }
    p
}

/// This machine, detected once and cached. Reads [`DEVICE_TIER_ENV`] and
/// [`FAKE_RAM_ENV`] the first time it is called.
pub fn profile() -> &'static MachineProfile {
    static CACHE: OnceLock<MachineProfile> = OnceLock::new();
    CACHE.get_or_init(|| {
        let tier = std::env::var(DEVICE_TIER_ENV).ok();
        let ram = std::env::var(FAKE_RAM_ENV).ok();
        apply_overrides(detect(), tier.as_deref(), ram.as_deref())
    })
}

/// How much memory resident models may use, and the limits that follow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemoryPolicy {
    pub tier: DeviceTier,
    /// Total memory resident models may hold, in bytes. Before a model loads,
    /// the least recently used idle models are unloaded until it fits; a
    /// model that cannot fit even alone is refused. `None`: no budget.
    pub budget_bytes: Option<u64>,
    /// Unload a model idle this long. `None`: never.
    #[serde(serialize_with = "secs")]
    pub idle_unload: Option<Duration>,
    /// Generation models resident at once. `None`: as many as the budget holds.
    pub max_generation_models: Option<usize>,
    /// Cap on MLX wired (unpageable) memory, per runner. `None`: MLX's own
    /// default, two thirds of RAM.
    pub mlx_wired_limit_bytes: Option<u64>,
    /// Cap on the freed buffers MLX keeps for reuse, per runner.
    pub mlx_cache_limit_bytes: Option<u64>,
}

fn secs<S: serde::Serializer>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
    match d {
        Some(d) => s.serialize_some(&d.as_secs()),
        None => s.serialize_none(),
    }
}

impl MemoryPolicy {
    /// The policy for a machine. The budget is half of RAM on a constrained
    /// machine, 60 % on a standard one and 70 % on a capable one: the rest is
    /// for macOS, the GPU's other clients and the apps the person has open.
    pub fn for_machine(p: &MachineProfile) -> Self {
        let ram = p.ram_bytes.max(GIB);
        match p.tier {
            DeviceTier::Constrained => MemoryPolicy {
                tier: p.tier,
                budget_bytes: Some(ram / 2),
                idle_unload: Some(Duration::from_secs(3 * 60)),
                max_generation_models: Some(1),
                // A quarter of RAM: 4 GB on a 16 GB Air, not MLX's 10.7 GB.
                mlx_wired_limit_bytes: Some(ram / 4),
                mlx_cache_limit_bytes: Some(256 * MIB),
            },
            DeviceTier::Standard => MemoryPolicy {
                tier: p.tier,
                budget_bytes: Some(ram * 6 / 10),
                idle_unload: Some(Duration::from_secs(15 * 60)),
                max_generation_models: None,
                mlx_wired_limit_bytes: Some(ram / 2),
                mlx_cache_limit_bytes: Some(GIB),
            },
            DeviceTier::Capable => MemoryPolicy {
                tier: p.tier,
                budget_bytes: Some(ram * 7 / 10),
                idle_unload: Some(Duration::from_secs(15 * 60)),
                max_generation_models: None,
                mlx_wired_limit_bytes: None,
                mlx_cache_limit_bytes: Some(GIB),
            },
        }
    }

    /// [`MemoryPolicy::for_machine`] for [`profile`], with
    /// [`MEMORY_BUDGET_ENV`] applied.
    pub fn detect() -> Self {
        let mut p = Self::for_machine(profile());
        if let Ok(raw) = std::env::var(MEMORY_BUDGET_ENV) {
            match parse_budget(&raw) {
                Ok(b) => p.budget_bytes = b,
                Err(e) => tracing::warn!(value = %raw, "{MEMORY_BUDGET_ENV}: {e}; using the default"),
            }
        }
        p
    }

    /// No budget, no slot limit, no idle unload, MLX's own defaults: the
    /// engine as it was before memory policy existed. For tests that must
    /// behave the same on every machine, and hosts that manage memory
    /// themselves.
    pub fn unlimited() -> Self {
        MemoryPolicy {
            tier: profile().tier,
            budget_bytes: None,
            idle_unload: None,
            max_generation_models: None,
            mlx_wired_limit_bytes: None,
            mlx_cache_limit_bytes: None,
        }
    }

    pub fn with_budget(mut self, budget_bytes: Option<u64>) -> Self {
        self.budget_bytes = budget_bytes;
        self
    }

    pub fn with_idle_unload(mut self, idle: Option<Duration>) -> Self {
        self.idle_unload = idle;
        self
    }

    /// The environment an MLX runner is started with (`estia-runner.py`
    /// reads these): its wired and cache caps.
    pub fn mlx_env(&self) -> Vec<(&'static str, String)> {
        let mut env = Vec::new();
        if let Some(b) = self.mlx_wired_limit_bytes {
            env.push(("ESTIA_MLX_WIRED_LIMIT_BYTES", b.to_string()));
        }
        if let Some(b) = self.mlx_cache_limit_bytes {
            env.push(("ESTIA_MLX_CACHE_LIMIT_BYTES", b.to_string()));
        }
        env
    }
}

/// `6GB`, `6.5 GB`, `6144MB`, `6442450944` (bytes) or `off`. GB and MB are
/// binary (GiB, MiB), like Activity Monitor's figures.
pub fn parse_budget(raw: &str) -> Result<Option<u64>, String> {
    let s = raw.trim().to_ascii_lowercase();
    if matches!(s.as_str(), "off" | "none" | "0") {
        return Ok(None);
    }
    let (num, unit) = match s.find(|c: char| c.is_ascii_alphabetic()) {
        Some(i) => (s[..i].trim(), s[i..].trim()),
        None => (s.as_str(), ""),
    };
    let n: f64 = num.parse().map_err(|_| format!("`{raw}` is not a size (for example 8GB, 6144MB or off)"))?;
    let mult = match unit {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => MIB as f64,
        "g" | "gb" | "gib" => GIB as f64,
        other => return Err(format!("unknown unit `{other}` in `{raw}` (use GB or MB)")),
    };
    let bytes = n * mult;
    if !bytes.is_finite() || bytes < (256 * MIB) as f64 {
        return Err(format!("`{raw}` is too small for any model (at least 256MB, or off)"));
    }
    Ok(Some(bytes as u64))
}

/// Bytes as GB with one decimal (`4.2 GB`), for messages.
pub fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / GIB as f64)
}

/// About how much memory a model holds once loaded and warm, from its size
/// on disk. MLX loads lazily, so the Gemma 4 audio and vision towers stay on
/// disk until an image arrives (measured: E2B at 3.58 GB on disk holds 2.62 GB
/// after a text generation); llama.cpp holds the whole file, projector
/// included. Both add room for the KV cache and the runtime. The engine
/// prefers a runner's measured footprint once it has one.
pub fn estimate_resident_bytes(disk_bytes: u64, format: crate::models::Format) -> u64 {
    match format {
        crate::models::Format::Mlx => disk_bytes / 4 * 3 + 256 * MIB,
        crate::models::Format::Gguf => disk_bytes + 256 * MIB,
    }
}

/// One generation family as [`recommend`] weighed it.
#[derive(Debug, Clone, Serialize)]
pub struct FamilyFit {
    pub family: &'static str,
    /// The artifact this backend loads for it.
    pub artifact: &'static str,
    pub estimate_bytes: u64,
    /// Fits the budget beside the embedding model.
    pub fits: bool,
    pub vision: bool,
}

/// Role bindings that fit a memory budget.
#[derive(Debug, Clone, Serialize)]
pub struct Recommendation {
    pub tier: DeviceTier,
    pub budget_bytes: Option<u64>,
    /// Kept free for the embedding model.
    pub embed_reserve_bytes: u64,
    /// Every generation family with an artifact on this backend, smallest first.
    pub families: Vec<FamilyFit>,
    /// `text`: the largest family that fits beside the embedding model (on a
    /// constrained machine, within 60 % of the budget).
    pub text: Option<&'static str>,
    /// `fast`: the smallest family that fits; `text`'s family when only one
    /// generation model may be resident, so switching never reloads.
    pub fast: Option<&'static str>,
    /// `vision`: `text`'s family when it reads images, else the largest
    /// that fits and does.
    pub vision: Option<&'static str>,
}

/// Which families to bind for a machine with `policy`, on `backend`, keeping
/// `embed_reserve` bytes for the embedding model. With no budget, 70 % of RAM
/// stands in for one. When nothing fits, `text` and `fast` fall back to the
/// smallest family (its `fits` says so).
pub fn recommend(policy: &MemoryPolicy, backend: crate::Backend, embed_reserve: u64) -> Recommendation {
    use crate::models::registry::{generation_artifacts, Capability};
    let budget = policy.budget_bytes.unwrap_or(profile().ram_bytes * 7 / 10);
    let mut families: Vec<FamilyFit> = Vec::new();
    for a in generation_artifacts().into_iter().filter(|a| a.format == backend.format() && a.has(Capability::Text)) {
        if families.iter().any(|f| f.family == a.family) {
            continue;
        }
        let estimate_bytes = estimate_resident_bytes(a.required_disk_bytes, a.format);
        families.push(FamilyFit {
            family: a.family,
            artifact: a.id,
            estimate_bytes,
            fits: estimate_bytes + embed_reserve <= budget,
            vision: a.has(Capability::Vision),
        });
    }
    families.sort_by_key(|f| f.estimate_bytes);
    // A constrained machine keeps its default model to 60 % of the budget:
    // the 12B fits a 16 GB Air alone, but a fanless machine runs it slowly
    // and has nothing left for a second model.
    let text_cap = if policy.tier == DeviceTier::Constrained { budget / 10 * 6 } else { budget };
    let largest = families.iter().rev().find(|f| f.fits && f.estimate_bytes <= text_cap);
    let smallest = families.iter().find(|f| f.fits).or(families.first());
    let text = largest.or(smallest).map(|f| f.family);
    let vision = match largest {
        Some(f) if f.vision => Some(f.family),
        _ => families.iter().rev().find(|f| f.fits && f.vision).map(|f| f.family),
    };
    // With one generation model at a time, a separate `fast` model would
    // unload `text` on every switch; share it instead.
    let fast = if policy.max_generation_models == Some(1) { text } else { smallest.map(|f| f.family) };
    Recommendation {
        tier: policy.tier,
        budget_bytes: policy.budget_bytes,
        embed_reserve_bytes: embed_reserve,
        text,
        fast,
        vision,
        families,
    }
}

#[cfg(target_os = "macos")]
mod sys {
    use std::ffi::CString;

    fn raw(name: &str) -> Option<Vec<u8>> {
        let c = CString::new(name).ok()?;
        let mut len: libc::size_t = 0;
        // SAFETY: a null buffer asks for the size only.
        if unsafe { libc::sysctlbyname(c.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) } != 0 || len == 0 {
            return None;
        }
        let mut buf = vec![0u8; len];
        // SAFETY: `buf` is writable for `len` bytes, the size we pass.
        if unsafe { libc::sysctlbyname(c.as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) } != 0 {
            return None;
        }
        buf.truncate(len);
        Some(buf)
    }

    fn int(name: &str) -> Option<u64> {
        let b = raw(name)?;
        match b.len() {
            8 => Some(u64::from_ne_bytes(b.try_into().ok()?)),
            4 => Some(u64::from(u32::from_ne_bytes(b.try_into().ok()?))),
            _ => None,
        }
    }

    pub fn string(name: &str) -> Option<String> {
        let b = raw(name)?;
        let s = String::from_utf8_lossy(&b).trim_end_matches('\0').trim().to_string();
        (!s.is_empty()).then_some(s)
    }

    pub fn ram_bytes() -> Option<u64> {
        int("hw.memsize")
    }

    pub fn logical_cores() -> Option<u32> {
        int("hw.logicalcpu").and_then(|n| u32::try_from(n).ok())
    }
}

#[cfg(not(target_os = "macos"))]
mod sys {
    pub fn string(_name: &str) -> Option<String> {
        None
    }

    /// `MemTotal` from `/proc/meminfo` (Linux); `None` elsewhere.
    pub fn ram_bytes() -> Option<u64> {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 = s.lines().find_map(|l| l.strip_prefix("MemTotal:"))?.split_whitespace().next()?.parse().ok()?;
        kb.checked_mul(1024)
    }

    pub fn logical_cores() -> Option<u32> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(ram_gb: u64, cores: u32, model: &str) -> MachineProfile {
        let fanless = is_fanless_model(model);
        MachineProfile {
            ram_bytes: ram_gb * GIB,
            logical_cores: cores,
            chip: "Apple M2".into(),
            model_id: model.into(),
            fanless,
            tier: classify(ram_gb * GIB, cores, fanless),
            overrides: Vec::new(),
        }
    }

    #[test]
    fn a_16gb_air_is_constrained_and_so_is_a_32gb_one() {
        assert_eq!(machine(16, 8, "Mac14,2").tier, DeviceTier::Constrained);
        assert_eq!(machine(32, 10, "Mac16,13").tier, DeviceTier::Constrained);
        assert_eq!(machine(16, 10, "Mac14,5").tier, DeviceTier::Constrained, "16 GB is constrained with a fan too");
        assert_eq!(machine(24, 10, "Mac14,5").tier, DeviceTier::Standard);
        assert_eq!(machine(32, 10, "MacBookPro18,1").tier, DeviceTier::Capable);
        assert_eq!(machine(64, 4, "Mac14,5").tier, DeviceTier::Constrained, "four cores");
    }

    #[test]
    fn fanless_prefixes_do_not_catch_fanned_models_in_the_same_family() {
        assert!(is_fanless_model("MacBookAir10,1"));
        assert!(is_fanless_model("Mac14,2"));
        assert!(!is_fanless_model("Mac14,5"), "a MacBook Pro");
        assert!(!is_fanless_model("Mac14,20"), "prefix Mac14,2 must not match");
    }

    #[test]
    fn overrides_rederive_then_force() {
        let p = apply_overrides(machine(32, 10, "MacBookPro18,1"), None, Some("8"));
        assert_eq!((p.ram_gb(), p.tier), (8, DeviceTier::Constrained));
        let p = apply_overrides(machine(8, 8, "Mac14,2"), Some("capable"), None);
        assert_eq!(p.tier, DeviceTier::Capable);
        assert_eq!(p.overrides, ["ESTIA_DEVICE_TIER=capable"]);
        let p = apply_overrides(machine(32, 10, "MacBookPro18,1"), Some("tiny"), Some("lots"));
        assert_eq!((p.ram_gb(), p.tier), (32, DeviceTier::Capable), "garbage is ignored");
        assert!(p.overrides.is_empty());
    }

    #[test]
    fn the_constrained_policy_caps_what_froze_the_air() {
        let p = MemoryPolicy::for_machine(&machine(16, 8, "Mac14,2"));
        assert_eq!(p.budget_bytes, Some(8 * GIB));
        assert_eq!(p.mlx_wired_limit_bytes, Some(4 * GIB), "not MLX's two thirds (10.7 GB)");
        assert_eq!(p.max_generation_models, Some(1));
        assert_eq!(p.idle_unload, Some(Duration::from_secs(180)));
        let env = p.mlx_env();
        assert!(env.contains(&("ESTIA_MLX_WIRED_LIMIT_BYTES", (4 * GIB).to_string())), "{env:?}");
        let big = MemoryPolicy::for_machine(&machine(64, 12, "Mac15,9"));
        assert_eq!((big.mlx_wired_limit_bytes, big.max_generation_models), (None, None));
        assert!(big.mlx_env().iter().all(|(k, _)| *k != "ESTIA_MLX_WIRED_LIMIT_BYTES"));
    }

    #[test]
    fn budgets_parse_in_binary_units() {
        assert_eq!(parse_budget("8GB"), Ok(Some(8 * GIB)));
        assert_eq!(parse_budget(" 6.5 gb "), Ok(Some(6 * GIB + GIB / 2)));
        assert_eq!(parse_budget("6144MB"), Ok(Some(6 * GIB)));
        assert_eq!(parse_budget("off"), Ok(None));
        assert_eq!(parse_budget("0"), Ok(None));
        assert!(parse_budget("12 TB").is_err());
        assert!(parse_budget("lots").is_err());
        assert!(parse_budget("10MB").is_err(), "too small to hold a model");
    }

    #[test]
    fn estimates_follow_how_each_backend_loads() {
        use crate::models::Format;
        // E2B MLX: 3.58 GB on disk, 2.62 GB measured after a generation.
        let e2b = estimate_resident_bytes(3_581_101_896, Format::Mlx);
        assert!(e2b > 2_620_000_000 && e2b < 3_300_000_000, "{e2b}");
        assert_eq!(estimate_resident_bytes(GIB, Format::Gguf), GIB + 256 * MIB);
    }

    #[test]
    fn recommendations_follow_the_budget() {
        let small = MemoryPolicy::for_machine(&machine(8, 8, "Mac14,2"));
        let r = recommend(&small, crate::Backend::MlxPython, 400 * MIB);
        assert_eq!((r.text, r.fast), (Some("gemma4-e2b"), Some("gemma4-e2b")), "{r:#?}");
        let air = MemoryPolicy::for_machine(&machine(16, 8, "Mac14,2"));
        let r = recommend(&air, crate::Backend::MlxPython, 400 * MIB);
        assert_eq!(r.text, Some("gemma4-e4b"), "the 12B fits alone but leaves a fanless 16 GB Air nothing: {r:#?}");
        assert_eq!(r.fast, Some("gemma4-e4b"), "one model at a time: fast shares text's");
        assert!(r.families.iter().any(|f| f.family == "gemma4-12b-qat" && f.fits));
        let studio = MemoryPolicy::for_machine(&machine(32, 12, "Mac15,9"));
        assert_eq!(recommend(&studio, crate::Backend::MlxPython, 400 * MIB).text, Some("gemma4-12b-qat"));
        let tight = studio.with_budget(Some(5 * GIB));
        let r = recommend(&tight, crate::Backend::MlxPython, 400 * MIB);
        assert_eq!((r.text, r.vision, r.fast), (Some("gemma4-e4b"), Some("gemma4-e4b"), Some("gemma4-e2b")), "{r:#?}");
        let none = air.with_budget(Some(GIB));
        let r = recommend(&none, crate::Backend::MlxPython, 400 * MIB);
        assert_eq!((r.text, r.vision), (Some("gemma4-e2b"), None), "nothing fits: the smallest, flagged");
        assert!(r.families.iter().all(|f| !f.fits));
    }

    #[test]
    fn this_machine_is_detected() {
        let p = detect();
        assert!(p.ram_bytes >= GIB, "{p:?}");
        assert!(p.logical_cores >= 1);
        if cfg!(target_os = "macos") {
            assert_ne!(p.model_id, "unknown", "hw.model reads on macOS");
        }
    }
}
