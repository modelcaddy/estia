//! Build information for `estia --version`, `estia version` and the `build`
//! object in `/engine/health`: the git commit, the build date, the target and
//! the compiler. See docs/versioning.md.
//!
//! This file is `cli/build.rs`; `server/build.rs` is a symlink to it, so the
//! server reports its own build when another program embeds it.
//!
//! It never fails the build. Anything it cannot find is `unknown`.
//!
//! The commit, from the first of these that gives one:
//!
//! 1. `ESTIA_BUILD_COMMIT` in the environment, for builds from a source
//!    archive that has no `.git` (letters, digits and `.+-_`, at most 64).
//! 2. git, when this crate sits in a checkout of the Estia workspace: the
//!    first 9 hex digits of `HEAD`, plus `+dirty` when a tracked file that is
//!    compiled in (see `SOURCES`) differs from `HEAD`.
//! 3. `.cargo_vcs_info.json`, which `cargo package` writes into a published
//!    crate: its commit, plus `+dirty` if it was packaged with `--allow-dirty`.
//! 4. `unknown`.
//!
//! The date is the UTC day this script ran, `YYYY-MM-DD`, or the day of
//! `SOURCE_DATE_EPOCH` when that is set (reproducible builds).

use std::path::{Path, PathBuf};
use std::process::Command;

/// Paths under the workspace root whose contents are compiled into the
/// binary. A change to one of them reruns this script and marks the build
/// `+dirty`; a change elsewhere (docs, CI) does neither.
const SOURCES: &[&str] = &["Cargo.toml", "Cargo.lock", "proto", "engine", "llama", "server", "cli", "runners"];

/// Hex digits of the commit to keep, as `rustc -V` does.
const SHORT: usize = 9;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=ESTIA_BUILD_COMMIT");
    println!("cargo::rerun-if-env-changed=SOURCE_DATE_EPOCH");

    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let commit = commit_from_env()
        .or_else(|| commit_from_git(&manifest_dir))
        .or_else(|| commit_from_vcs_info(&manifest_dir))
        .unwrap_or_else(|| "unknown".to_string());

    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".into());
    println!("cargo::rustc-env=ESTIA_BUILD_COMMIT={commit}");
    println!("cargo::rustc-env=ESTIA_BUILD_DATE={}", build_date());
    println!("cargo::rustc-env=ESTIA_BUILD_TARGET={target}");
    println!("cargo::rustc-env=ESTIA_BUILD_PROFILE={profile}");
    println!("cargo::rustc-env=ESTIA_BUILD_RUSTC={}", rustc_version());
}

fn commit_from_env() -> Option<String> {
    let v = std::env::var("ESTIA_BUILD_COMMIT").ok()?;
    let v = v.trim();
    let ok = !v.is_empty() && v.len() <= 64 && v.chars().all(|c| c.is_ascii_alphanumeric() || ".+-_".contains(c));
    ok.then(|| v.to_string())
}

/// Run git in `dir`; its trimmed stdout when it succeeds.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_string())
}

fn commit_from_git(manifest_dir: &Path) -> Option<String> {
    // Only a checkout whose top level is this workspace. A crate unpacked
    // under `target/package/` or copied into another repository sits inside
    // some other checkout, whose commit says nothing about this code.
    let root = manifest_dir.parent()?.canonicalize().ok()?;
    let top = PathBuf::from(git(&root, &["rev-parse", "--show-toplevel"])?).canonicalize().ok()?;
    if top != root {
        return None;
    }
    let sha = git(&root, &["rev-parse", "HEAD"])?;
    if sha.len() < SHORT || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }

    // Rerun when HEAD moves (commit, checkout, reset, rebase) or the sources
    // change. A path that does not exist would rerun the script on every
    // build, so only existing ones are named.
    let mut watch: Vec<PathBuf> = ["HEAD", "logs/HEAD", "index", "packed-refs"].iter().filter_map(|p| git_path(&root, p)).collect();
    if let Some(r) = git(&root, &["symbolic-ref", "-q", "HEAD"]) {
        watch.extend(git_path(&root, &r));
    }
    watch.extend(SOURCES.iter().map(|p| root.join(p)));
    for p in watch.iter().filter(|p| p.exists()) {
        println!("cargo::rerun-if-changed={}", p.display());
    }

    // `--no-optional-locks`: do not rewrite the index while reading it, which
    // could race a git command the developer is running.
    let mut args = vec!["--no-optional-locks", "status", "--porcelain", "--untracked-files=no", "--"];
    args.extend(SOURCES);
    let dirty = match git(&root, &args) {
        Some(s) => !s.is_empty(),
        None => true,
    };
    Some(format!("{}{}", &sha[..SHORT], if dirty { "+dirty" } else { "" }))
}

/// Where git keeps `name` (`HEAD`, `index`, a ref), worktrees included.
fn git_path(root: &Path, name: &str) -> Option<PathBuf> {
    let p = PathBuf::from(git(root, &["rev-parse", "--git-path", name])?);
    Some(if p.is_absolute() { p } else { root.join(p) })
}

/// `.cargo_vcs_info.json`: `{"git": {"sha1": "…", "dirty": true}, "path_in_vcs": "cli"}`.
/// Read by hand: a build script with no dependencies stays cheap.
fn commit_from_vcs_info(manifest_dir: &Path) -> Option<String> {
    let path = manifest_dir.join(".cargo_vcs_info.json");
    let text = std::fs::read_to_string(&path).ok()?;
    println!("cargo::rerun-if-changed={}", path.display());
    let rest = &text[text.find("\"sha1\"")? + "\"sha1\"".len()..];
    let rest = &rest[rest.find('"')? + 1..];
    let sha = &rest[..rest.find('"')?];
    if sha.len() < SHORT || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let dirty = text.split_whitespace().collect::<String>().contains("\"dirty\":true");
    Some(format!("{}{}", &sha[..SHORT], if dirty { "+dirty" } else { "" }))
}

fn build_date() -> String {
    let secs = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs() as i64));
    match secs {
        Some(s) => {
            let (y, m, d) = civil_from_days(s.div_euclid(86_400));
            format!("{y:04}-{m:02}-{d:02}")
        }
        None => "unknown".into(),
    }
}

/// Days since 1970-01-01 to a proleptic Gregorian date (Howard Hinnant's
/// `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// `1.89.0 (29483883e 2025-08-04)` from `rustc --version`.
fn rustc_version() -> String {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().trim_start_matches("rustc ").to_string())
        .filter(|s| !s.is_empty() && !s.contains('\n'))
        .unwrap_or_else(|| "unknown".into())
}
