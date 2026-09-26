//! Bearer tokens. Stored hashed in `tokens.json`; the plaintext is shown once
//! when minted. Scopes gate route groups: `generate`, `embed`, `models:read`,
//! `models:write`, `admin` (everything).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
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

#[derive(Debug)]
pub struct TokenStore {
    path: PathBuf,
    records: Mutex<Vec<TokenRecord>>,
    /// Modification time and length of `tokens.json` when `records` was last
    /// read or written. Another process (`estia token revoke`, `estia pair
    /// approve`) edits the file directly; a changed stamp means re-read it.
    stamp: Mutex<Option<(SystemTime, u64)>>,
}

fn file_stamp(path: &Path) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

/// Write `bytes` to `path` through a temp file and a rename, so a reader in
/// another process never sees half a file, with owner-only permissions.
pub(crate) fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(&tmp, path)
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

impl TokenStore {
    pub fn open(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let path = path.into();
        let stamp = file_stamp(&path);
        let records = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text)?,
            Err(_) => Vec::new(),
        };
        Ok(Self { path, records: Mutex::new(records), stamp: Mutex::new(stamp) })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn save(&self, records: &[TokenRecord]) -> anyhow::Result<()> {
        write_private(&self.path, serde_json::to_string_pretty(records)?.as_bytes())?;
        *self.stamp.lock().unwrap() = file_stamp(&self.path);
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.records.lock().unwrap().is_empty()
    }

    pub fn list(&self) -> Vec<TokenRecord> {
        self.records.lock().unwrap().clone()
    }

    /// Mint a token; returns the plaintext (shown once, never stored).
    pub fn mint(&self, name: &str, scopes: &[&str]) -> anyhow::Result<String> {
        let plaintext = random_token();
        let record = TokenRecord {
            name: name.to_string(),
            sha256: hash(&plaintext),
            scopes: scopes.iter().map(|s| s.to_string()).collect(),
            created_unix: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        };
        let mut records = self.records.lock().unwrap();
        records.retain(|r| r.name != name);
        records.push(record);
        self.save(&records)?;
        Ok(plaintext)
    }

    pub fn revoke(&self, name: &str) -> anyhow::Result<bool> {
        let mut records = self.records.lock().unwrap();
        let before = records.len();
        records.retain(|r| r.name != name);
        let removed = records.len() != before;
        if removed {
            self.save(&records)?;
        }
        Ok(removed)
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

    #[test]
    fn mint_verify_scopes_revoke() {
        let dir = std::env::temp_dir().join(format!("estia-tokens-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
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
        let dir = std::env::temp_dir().join(format!("estia-tokens-revoke-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
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
}
