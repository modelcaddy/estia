//! One-click pairing for clients on the LAN: a client posts its name and the
//! scopes it wants, the operator approves (in the CLI, or a host UI), the
//! client polls until it receives its token — once.
//!
//! State is a small file in the data directory, `pairings.json`, so the daemon
//! and the `estia pair` command (a different process, run by the operator)
//! share it without an admin token: whoever owns the data directory owns the
//! approvals. Requests older than [`PAIRING_TTL_SECS`] are swept on every
//! read; at most [`MAX_PENDING`] may wait at once so a hostile client cannot
//! flood the operator with cards.

use crate::tokens::{TokenStore, ALL_SCOPES};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const PAIRING_TTL_SECS: u64 = 300;
pub const MAX_PENDING: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairingStatus {
    Pending,
    Approved,
    Denied,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pairing {
    pub id: String,
    pub name: String,
    pub scopes: Vec<String>,
    pub status: PairingStatus,
    pub created_unix: u64,
    /// Where the request came from, for the operator's eyes.
    #[serde(default)]
    pub from: Option<String>,
    /// The minted token, held until the client collects it once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_plain: Option<String>,
    #[serde(default)]
    pub claimed: bool,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub struct PairingStore {
    path: PathBuf,
}

impl PairingStore {
    pub fn new(data_dir: &Path) -> Self {
        Self { path: data_dir.join("pairings.json") }
    }

    fn load(&self) -> Vec<Pairing> {
        let mut v: Vec<Pairing> = std::fs::read_to_string(&self.path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        let cutoff = now().saturating_sub(PAIRING_TTL_SECS);
        // Sweep: pending requests expire; decided ones stay until claimed or expired.
        v.retain(|p| {
            p.created_unix >= cutoff
                || (p.status == PairingStatus::Approved && !p.claimed && p.created_unix >= cutoff.saturating_sub(PAIRING_TTL_SECS))
        });
        v
    }

    /// Owner-only and atomic: an approved, unclaimed pairing holds its token
    /// in plaintext here until the device collects it.
    fn save(&self, v: &[Pairing]) -> anyhow::Result<()> {
        crate::tokens::write_private(&self.path, serde_json::to_string_pretty(v)?.as_bytes())?;
        Ok(())
    }

    pub fn list(&self) -> Vec<Pairing> {
        self.load()
    }

    /// A client asks to be paired.
    pub fn request(&self, name: &str, scopes: &[String], from: Option<String>) -> anyhow::Result<Pairing> {
        for s in scopes {
            if !ALL_SCOPES.contains(&s.as_str()) {
                anyhow::bail!("unknown scope `{s}`");
            }
        }
        let mut v = self.load();
        if v.iter().filter(|p| p.status == PairingStatus::Pending).count() >= MAX_PENDING {
            anyhow::bail!("too many pending pairing requests; try again later");
        }
        let mut bytes = [0u8; 8];
        getrandom::getrandom(&mut bytes).map_err(|e| anyhow::anyhow!("randomness: {e}"))?;
        let id: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        let p = Pairing {
            id,
            name: name.chars().take(64).collect(),
            scopes: scopes.to_vec(),
            status: PairingStatus::Pending,
            created_unix: now(),
            from,
            token_plain: None,
            claimed: false,
        };
        v.push(p.clone());
        self.save(&v)?;
        Ok(p)
    }

    /// The operator approves: a token is minted with the requested scopes and
    /// parked on the record until the client collects it.
    pub fn approve(&self, id: &str, tokens: &TokenStore) -> anyhow::Result<Pairing> {
        let mut v = self.load();
        let p = v.iter_mut().find(|p| p.id == id).ok_or_else(|| anyhow::anyhow!("no pairing `{id}` (expired?)"))?;
        if p.status != PairingStatus::Pending {
            anyhow::bail!("pairing `{id}` is already {:?}", p.status);
        }
        let refs: Vec<&str> = p.scopes.iter().map(String::as_str).collect();
        let token = tokens.mint(&format!("pair:{}:{}", p.name, p.id), &refs)?;
        p.status = PairingStatus::Approved;
        p.token_plain = Some(token);
        let out = p.clone();
        self.save(&v)?;
        Ok(out)
    }

    pub fn deny(&self, id: &str) -> anyhow::Result<Pairing> {
        let mut v = self.load();
        let p = v.iter_mut().find(|p| p.id == id).ok_or_else(|| anyhow::anyhow!("no pairing `{id}` (expired?)"))?;
        p.status = PairingStatus::Denied;
        p.token_plain = None;
        let out = p.clone();
        self.save(&v)?;
        Ok(out)
    }

    /// The client polls. An approved token is handed over exactly once.
    pub fn poll(&self, id: &str) -> anyhow::Result<(PairingStatus, Option<String>)> {
        let mut v = self.load();
        let p = v.iter_mut().find(|p| p.id == id).ok_or_else(|| anyhow::anyhow!("no pairing `{id}` (expired?)"))?;
        let status = p.status;
        let token = if status == PairingStatus::Approved && !p.claimed {
            p.claimed = true;
            p.token_plain.take()
        } else {
            None
        };
        self.save(&v)?;
        Ok((status, token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_approve_poll_once() {
        let dir = std::env::temp_dir().join(format!("estia-pair-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = PairingStore::new(&dir);
        let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
        let p = store.request("phone", &["generate".into(), "embed".into()], Some("10.0.0.7".into())).unwrap();
        assert_eq!(store.poll(&p.id).unwrap(), (PairingStatus::Pending, None));
        assert!(store.request("x", &["bogus".into()], None).is_err());
        let approved = store.approve(&p.id, &tokens).unwrap();
        assert_eq!(approved.status, PairingStatus::Approved);
        let (status, token) = store.poll(&p.id).unwrap();
        assert_eq!(status, PairingStatus::Approved);
        let token = token.expect("token handed over once");
        assert!(tokens.verify(&token).unwrap().allows("embed"));
        assert!(!tokens.verify(&token).unwrap().allows("admin"));
        assert_eq!(store.poll(&p.id).unwrap(), (PairingStatus::Approved, None), "second poll gets no token");
        assert!(store.approve(&p.id, &tokens).is_err(), "cannot approve twice");
        let text = std::fs::read_to_string(dir.join("pairings.json")).unwrap();
        assert!(!text.contains(&token), "claimed plaintext is gone from disk");
        let d = store.request("laptop", &["generate".into()], None).unwrap();
        store.deny(&d.id).unwrap();
        assert_eq!(store.poll(&d.id).unwrap().0, PairingStatus::Denied);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
