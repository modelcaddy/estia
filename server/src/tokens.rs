//! Bearer tokens. Stored hashed in `tokens.json`; the plaintext is shown once
//! when minted. Scopes gate route groups: `generate`, `embed`, `models:read`,
//! `models:write`, `admin` (everything).
//!
//! The daemon and the `estia token` / `estia pair` commands (other processes)
//! edit the same file. Every change is a read-modify-write under an exclusive
//! lock on the sibling `tokens.lock`, re-reading the file inside the lock, so
//! no writer overwrites another's change.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SCOPE_GENERATE: &str = "generate";
pub const SCOPE_EMBED: &str = "embed";
pub const SCOPE_MODELS_READ: &str = "models:read";
pub const SCOPE_MODELS_WRITE: &str = "models:write";
pub const SCOPE_ADMIN: &str = "admin";

pub const ALL_SCOPES: &[&str] = &[SCOPE_GENERATE, SCOPE_EMBED, SCOPE_MODELS_READ, SCOPE_MODELS_WRITE, SCOPE_ADMIN];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenRecord {
    pub name: String,
    /// SHA-256 of the plaintext, lowercase hex.
    pub sha256: String,
    pub scopes: Vec<String>,
    pub created_unix: u64,
}

impl TokenRecord {
    pub fn allows(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == SCOPE_ADMIN || s == scope)
    }
}

/// [`TokenStore::mint`] refused: names are unique (they are what `revoke`
/// takes), so minting over an existing one needs [`TokenStore::replace`].
/// Callers can `downcast_ref::<TokenExists>()` on the returned error.
#[derive(Debug, Clone, thiserror::Error)]
#[error("a token named `{name}` already exists (scopes {}); revoke it first or replace it", existing.scopes.join(","))]
pub struct TokenExists {
    pub name: String,
    pub existing: TokenRecord,
}

#[derive(Debug)]
pub struct TokenStore {
    path: PathBuf,
    /// Sibling lock file held for every read-modify-write of `path`.
    lock_path: PathBuf,
    records: Mutex<Vec<TokenRecord>>,
    /// Modification time and length of `tokens.json` when `records` was last
    /// read or written. Another process (`estia token revoke`, `estia pair
    /// approve`) edits the file directly; a changed stamp means re-read it.
    stamp: Mutex<Option<(SystemTime, u64)>>,
    /// Serialises this process's writers; the lock file serialises processes.
    writer: Mutex<()>,
}

fn file_stamp(path: &Path) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

/// Create `dir` (and missing parents) owner-only. An existing directory is
/// left as it is.
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    if dir.as_os_str().is_empty() || dir.is_dir() {
        return Ok(());
    }
    let mut b = std::fs::DirBuilder::new();
    b.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        b.mode(0o700);
    }
    b.create(dir)
}

fn private_open_options() -> OpenOptions {
    #[allow(unused_mut)]
    let mut o = OpenOptions::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o
}

/// Write `bytes` to `path` through a temp file and a rename, so a reader in
/// another process never sees half a file, with owner-only permissions.
///
/// The temp file is created exclusively and owner-only from the start (it may
/// hold a pairing token in plaintext), under a name unique to this process and
/// call, so concurrent writers never share one; it is synced before the
/// rename and removed if anything fails.
pub(crate) fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().unwrap_or(Path::new(""));
    create_private_dir(parent)?;
    let base = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "estia".into());
    let (tmp, mut file) = loop {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp = parent.join(format!("{base}.{}.{n}.tmp", std::process::id()));
        match private_open_options().write(true).create_new(true).open(&tmp) {
            Ok(f) => break (tmp, f),
            // A leftover from a crashed process that had this pid: next name.
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };
    let result = (|| {
        // The mode above is filtered by the umask; pin it exactly.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// An exclusive advisory lock on a file, released on drop. Taken on a
/// dedicated lock file, never on the data file (which a rename replaces).
/// Locks taken through separate opens exclude each other within one process
/// too, so two stores on the same file in one process also serialise.
pub(crate) struct FileLock {
    _file: File,
}

pub(crate) fn lock_exclusive(path: &Path) -> std::io::Result<FileLock> {
    if let Some(parent) = path.parent() {
        create_private_dir(parent)?;
    }
    let file = private_open_options().read(true).write(true).create(true).truncate(false).open(path)?;
    file.lock()?;
    Ok(FileLock { _file: file })
}

fn hash(plaintext: &str) -> String {
    let d = Sha256::digest(plaintext.as_bytes());
    d.iter().fold(String::with_capacity(64), |mut acc, b| {
        use std::fmt::Write;
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

fn random_token() -> String {
    let mut bytes = [0u8; 24];
    getrandom::getrandom(&mut bytes).expect("os randomness");
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("estia_{hex}")
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// What is on disk now. Missing (or empty) means no tokens; a file that does
/// not parse is an error, never an empty set a writer would then save over.
fn read_records(path: &Path) -> anyhow::Result<Vec<TokenRecord>> {
    match std::fs::read_to_string(path) {
        Ok(text) if text.trim().is_empty() => Ok(Vec::new()),
        Ok(text) => serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{} does not parse: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(anyhow::anyhow!("read {}: {e}", path.display())),
    }
}

impl TokenStore {
    pub fn open(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let path = path.into();
        let stamp = file_stamp(&path);
        let records = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text)?,
            Err(_) => Vec::new(),
        };
        let lock_path = path.with_extension("lock");
        Ok(Self { path, lock_path, records: Mutex::new(records), stamp: Mutex::new(stamp), writer: Mutex::new(()) })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read-modify-write of `tokens.json` under both locks, starting from what
    /// is on disk (not this instance's possibly stale copy). `f` returns its
    /// result and whether it changed anything; only a change is written.
    fn update<R>(&self, f: impl FnOnce(&mut Vec<TokenRecord>) -> anyhow::Result<(R, bool)>) -> anyhow::Result<R> {
        let _writer = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        let _lock = lock_exclusive(&self.lock_path).map_err(|e| anyhow::anyhow!("lock {}: {e}", self.lock_path.display()))?;
        let mut records = read_records(&self.path)?;
        let (out, changed) = f(&mut records)?;
        if changed {
            write_private(&self.path, serde_json::to_string_pretty(&records)?.as_bytes())?;
        }
        let stamp = file_stamp(&self.path);
        *self.records.lock().unwrap() = records;
        *self.stamp.lock().unwrap() = stamp;
        Ok(out)
    }

    pub fn is_empty(&self) -> bool {
        self.records.lock().unwrap().is_empty()
    }

    pub fn list(&self) -> Vec<TokenRecord> {
        self.records.lock().unwrap().clone()
    }

    /// The record named `name`, as on disk now.
    pub fn get(&self, name: &str) -> Option<TokenRecord> {
        self.reload();
        self.records.lock().unwrap().iter().find(|r| r.name == name).cloned()
    }

    /// Mint a token; returns the plaintext (shown once, never stored). Fails
    /// with [`TokenExists`] when the name is taken — replacing a token is
    /// [`TokenStore::replace`], never a side effect of minting.
    pub fn mint(&self, name: &str, scopes: &[&str]) -> anyhow::Result<String> {
        self.mint_inner(name, scopes, false).map(|(plaintext, _)| plaintext)
    }

    /// Mint a token under `name`, replacing any token of that name (whose
    /// plaintext stops working at once). Returns the new plaintext and the
    /// record it replaced, if there was one.
    pub fn replace(&self, name: &str, scopes: &[&str]) -> anyhow::Result<(String, Option<TokenRecord>)> {
        self.mint_inner(name, scopes, true)
    }

    fn mint_inner(&self, name: &str, scopes: &[&str], replace: bool) -> anyhow::Result<(String, Option<TokenRecord>)> {
        let plaintext = random_token();
        let record = TokenRecord {
            name: name.to_string(),
            sha256: hash(&plaintext),
            scopes: scopes.iter().map(|s| s.to_string()).collect(),
            created_unix: now_unix(),
        };
        self.update(move |records| {
            let replaced = match records.iter().position(|r| r.name == name) {
                Some(i) if !replace => return Err(TokenExists { name: name.to_string(), existing: records[i].clone() }.into()),
                Some(i) => Some(records.remove(i)),
                None => None,
            };
            records.push(record);
            Ok(((plaintext, replaced), true))
        })
    }

    pub fn revoke(&self, name: &str) -> anyhow::Result<bool> {
        self.update(|records| {
            let before = records.len();
            records.retain(|r| r.name != name);
            let removed = records.len() != before;
            Ok((removed, removed))
        })
    }

    /// The record a bearer token belongs to, if any.
    ///
    /// `tokens.json` is the source of truth and other processes edit it:
    /// `estia token new` and `estia pair approve` add tokens, `estia token
    /// revoke` removes them. So every check first compares the file's stamp
    /// with the one last loaded and re-reads on a change — a revoked token
    /// stops working on the next request, not at the next restart. A miss
    /// also re-reads once, for filesystems with coarse modification times.
    pub fn verify(&self, plaintext: &str) -> Option<TokenRecord> {
        let h = hash(plaintext);
        if file_stamp(&self.path) != *self.stamp.lock().unwrap() {
            self.reload();
        }
        if let Some(r) = self.records.lock().unwrap().iter().find(|r| r.sha256 == h).cloned() {
            return Some(r);
        }
        self.reload();
        self.records.lock().unwrap().iter().find(|r| r.sha256 == h).cloned()
    }

    /// Replace the in-memory records with what is on disk. A missing file
    /// means no tokens; a file that does not parse (caught mid-write by a
    /// tool that does not rename) keeps the last good set.
    pub fn reload(&self) {
        let stamp = file_stamp(&self.path);
        match std::fs::read_to_string(&self.path) {
            Ok(text) => {
                if let Ok(records) = serde_json::from_str::<Vec<TokenRecord>>(&text) {
                    *self.records.lock().unwrap() = records;
                    *self.stamp.lock().unwrap() = stamp;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.records.lock().unwrap().clear();
                *self.stamp.lock().unwrap() = None;
            }
            Err(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn scratch(tag: &str) -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("estia-tokens-{tag}-{}-{nanos}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn mint_verify_scopes_revoke() {
        let dir = scratch("basic");
        let store = TokenStore::open(dir.join("tokens.json")).unwrap();
        assert!(store.is_empty());
        let t = store.mint("local", &[SCOPE_GENERATE, SCOPE_EMBED]).unwrap();
        assert!(t.starts_with("estia_"));
        let rec = store.verify(&t).expect("valid token");
        assert!(rec.allows(SCOPE_GENERATE) && !rec.allows(SCOPE_ADMIN) && !rec.allows(SCOPE_MODELS_WRITE));
        assert!(store.verify("estia_nope").is_none());
        let a = store.mint("admin", &[SCOPE_ADMIN]).unwrap();
        assert!(store.verify(&a).unwrap().allows(SCOPE_MODELS_WRITE));
        // Persisted hashed, not plaintext.
        let text = std::fs::read_to_string(dir.join("tokens.json")).unwrap();
        assert!(!text.contains(&t) && text.contains(&hash(&t)));
        let reopened = TokenStore::open(dir.join("tokens.json")).unwrap();
        assert!(reopened.verify(&t).is_some());
        // A token minted by another process is honoured after a reload-on-miss.
        let other = TokenStore::open(dir.join("tokens.json")).unwrap();
        let late = other.mint("late", &[SCOPE_EMBED]).unwrap();
        assert!(store.verify(&late).is_some(), "daemon-side store re-reads the file on a miss");
        assert!(store.revoke("local").unwrap());
        assert!(store.verify(&t).is_none());
        assert!(!store.revoke("local").unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn revoke_from_another_process_takes_effect_without_restart() {
        let dir = scratch("revoke");
        let path = dir.join("tokens.json");
        let daemon = TokenStore::open(&path).unwrap();
        let cli = TokenStore::open(&path).unwrap();
        let t = cli.mint("phone", &[SCOPE_GENERATE]).unwrap();
        assert!(daemon.verify(&t).is_some(), "minted elsewhere, honoured");
        assert!(cli.revoke("phone").unwrap());
        assert!(daemon.verify(&t).is_none(), "revoked elsewhere, refused on the next check");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mint_refuses_an_existing_name_and_replace_rotates() {
        let dir = scratch("dup");
        let store = TokenStore::open(dir.join("tokens.json")).unwrap();
        let first = store.mint("editor", &[SCOPE_GENERATE, SCOPE_EMBED]).unwrap();
        let err = store.mint("editor", &[SCOPE_EMBED]).unwrap_err();
        let exists = err.downcast_ref::<TokenExists>().expect("typed error");
        assert_eq!(exists.name, "editor");
        assert_eq!(exists.existing.scopes, vec!["generate", "embed"]);
        assert!(store.verify(&first).is_some(), "a refused mint leaves the old token working");
        assert_eq!(store.list().len(), 1);
        // A duplicate is also refused when the name was minted by another
        // process since this instance last looked.
        let other = TokenStore::open(dir.join("tokens.json")).unwrap();
        other.mint("late", &[SCOPE_EMBED]).unwrap();
        assert!(store.mint("late", &[SCOPE_EMBED]).unwrap_err().downcast_ref::<TokenExists>().is_some());

        let (second, replaced) = store.replace("editor", &[SCOPE_EMBED]).unwrap();
        assert_eq!(replaced.expect("replaced the old record").scopes, vec!["generate", "embed"]);
        assert!(store.verify(&first).is_none(), "the replaced token stops working");
        assert_eq!(store.verify(&second).unwrap().scopes, vec!["embed"]);
        assert_eq!(store.get("editor").unwrap().scopes, vec!["embed"]);
        let (_, none) = store.replace("fresh", &[SCOPE_GENERATE]).unwrap();
        assert!(none.is_none(), "replace of a new name just mints");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stale_instance_does_not_erase_tokens_minted_elsewhere() {
        let dir = scratch("stale");
        let path = dir.join("tokens.json");
        let daemon = TokenStore::open(&path).unwrap();
        let a = daemon.mint("a", &[SCOPE_GENERATE]).unwrap();
        let cli = TokenStore::open(&path).unwrap();
        let b = cli.mint("b", &[SCOPE_GENERATE]).unwrap();
        // The daemon never re-read (no verify in between); its revoke must
        // start from the file, not from its own copy.
        daemon.revoke("a").unwrap();
        assert!(cli.verify(&b).is_some(), "b survived a revoke from a stale store");
        assert!(cli.verify(&a).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_writers_lose_nothing() {
        let dir = scratch("race");
        let path = dir.join("tokens.json");
        // Each thread has its own store, as separate processes would: only the
        // lock file serialises them.
        let handles: Vec<_> = (0..12)
            .map(|t| {
                let path = path.clone();
                std::thread::spawn(move || {
                    let store = TokenStore::open(&path).unwrap();
                    (0..4).map(|i| store.mint(&format!("t{t}-{i}"), &[SCOPE_GENERATE]).unwrap()).collect::<Vec<_>>()
                })
            })
            .collect();
        let minted: Vec<String> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
        let check = TokenStore::open(&path).unwrap();
        assert_eq!(check.list().len(), 48, "no mint lost");
        assert!(minted.iter().all(|t| check.verify(t).is_some()));
        // One shared store across threads too (the daemon's case).
        let shared = Arc::new(check);
        let handles: Vec<_> = (0..12)
            .map(|t| {
                let s = Arc::clone(&shared);
                std::thread::spawn(move || s.revoke(&format!("t{t}-0")).unwrap())
            })
            .collect();
        assert!(handles.into_iter().all(|h| h.join().unwrap()));
        assert_eq!(TokenStore::open(&path).unwrap().list().len(), 36);
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "no temp files left: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn write_private_never_exposes_the_temp_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = scratch("tmpmode");
        let path = dir.join("secret.json");
        write_private(&path, b"one").unwrap();
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700, "created owner-only");
        // A watcher stats every temp file it can catch while writers run: none
        // may ever be readable by group or others, not even for a moment.
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watcher = {
            let (dir, done) = (dir.clone(), Arc::clone(&done));
            std::thread::spawn(move || {
                let mut seen = Vec::new();
                while !done.load(Ordering::Relaxed) {
                    for e in std::fs::read_dir(&dir).into_iter().flatten().filter_map(|e| e.ok()) {
                        if e.file_name().to_string_lossy().ends_with(".tmp") {
                            if let Ok(m) = std::fs::metadata(e.path()) {
                                seen.push(m.permissions().mode() & 0o777);
                            }
                        }
                    }
                }
                seen
            })
        };
        // Concurrent writers each get their own temp file; every write lands
        // whole and nothing is left behind.
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let path = path.clone();
                std::thread::spawn(move || {
                    for _ in 0..8 {
                        write_private(&path, format!("writer {i}").as_bytes()).unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        done.store(true, Ordering::Relaxed);
        let seen = watcher.join().unwrap();
        if let Some(bad) = seen.iter().find(|m| *m & 0o077 != 0) {
            panic!("a temp file was open to group or others: mode {bad:o}");
        }
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("writer "), "{text}");
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        let names: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name()).collect();
        assert_eq!(names.len(), 1, "{names:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
