//! One-click pairing for clients on the LAN: a client posts its name and the
//! scopes it wants, the operator approves (in the CLI, or a host UI), the
//! client polls until it receives its token — once.
//!
//! State is a small file in the data directory, `pairings.json`, so the daemon
//! and the `estia pair` command (a different process, run by the operator)
//! share it without an admin token: whoever owns the data directory owns the
//! approvals. Every change is a read-modify-write under an in-process mutex
//! and an exclusive lock on the sibling `pairings.lock`, so the daemon's
//! handlers and the CLI never overwrite each other and the token hand-over
//! happens exactly once.
//!
//! Requests older than [`PAIRING_TTL_SECS`] are swept on every read; at most
//! [`MAX_PENDING`] may wait at once, and at most [`MAX_PENDING_PER_SOURCE`]
//! from one address, so a hostile client cannot flood the operator with cards
//! or lock everyone else out. Names are checked here, at the source, because
//! the operator reads them in a terminal: see [`validate_name`].
//!
//! Each step is logged (`info`): requested, approved, denied, token
//! collected, expired. An approved token that expires uncollected is a `warn`,
//! because the token stays valid until revoked.

use crate::tokens::{lock_exclusive, write_private, FileLock, TokenStore, ALL_SCOPES};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

pub const PAIRING_TTL_SECS: u64 = 300;
/// Pending requests at once, from everyone.
pub const MAX_PENDING: usize = 24;
/// Pending requests at once from one source address.
pub const MAX_PENDING_PER_SOURCE: usize = 4;
/// Longest device name, in characters (after trimming).
pub const MAX_NAME_CHARS: usize = 64;

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
    /// Name of the token the approval minted (`pair:<name>:<id>`), so a later
    /// deny can revoke it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_name: Option<String>,
    /// Denied after it was approved: the minted token was revoked.
    #[serde(default)]
    pub revoked: bool,
}

/// Why a pairing operation failed. Handlers map these to HTTP statuses.
#[derive(Debug, thiserror::Error)]
pub enum PairingError {
    /// No such id, or it expired and was swept.
    #[error("no pairing `{id}` (expired?){hint}")]
    NotFound { id: String, hint: &'static str },
    /// A bad name or scope, or a pairing in the wrong state.
    #[error("{0}")]
    Invalid(String),
    /// A pending-request cap is full; try again once some are decided or expire.
    #[error("{0}")]
    TooMany(String),
    /// `pairings.json` or `tokens.json` could not be locked, read or written.
    #[error("pairing store: {0:#}")]
    Storage(anyhow::Error),
}

impl From<anyhow::Error> for PairingError {
    fn from(e: anyhow::Error) -> Self {
        PairingError::Storage(e)
    }
}

impl From<std::io::Error> for PairingError {
    fn from(e: std::io::Error) -> Self {
        PairingError::Storage(e.into())
    }
}

const DENY_HINT: &str = "; if it was approved, its token (pair:<name>:<id>) may still work: find it with `estia token list` and remove it with `estia token revoke`";

/// Check a requested device name and return it trimmed.
///
/// The operator reads pairing names in a terminal (`estia pair list`, the
/// approve and deny messages, `estia token list` through the token name), so a
/// name must not be able to move the cursor, recolour, hide or reorder what is
/// printed after it. Allowed: letters and digits in any script, space (not
/// two in a row), and `. _ - ' ’ ( )`; at most [`MAX_NAME_CHARS`] characters.
/// Everything else is refused, not stripped, so two names cannot silently
/// collapse into one.
pub fn validate_name(raw: &str) -> Result<String, PairingError> {
    const ALLOWED: &str = "letters, digits, single spaces and . _ - ' ( )";
    let name = raw.trim();
    if name.is_empty() {
        return Err(PairingError::Invalid("name is required".into()));
    }
    let count = name.chars().count();
    if count > MAX_NAME_CHARS {
        return Err(PairingError::Invalid(format!("name is {count} characters long; at most {MAX_NAME_CHARS}")));
    }
    let mut prev_space = false;
    for c in name.chars() {
        if c.is_control() || is_format_char(c) {
            return Err(PairingError::Invalid(format!(
                "name contains a control or formatting character ({}); use {ALLOWED}",
                c.escape_unicode()
            )));
        }
        let ok = c.is_alphanumeric() || matches!(c, ' ' | '.' | '_' | '-' | '\'' | '\u{2019}' | '(' | ')');
        if !ok {
            return Err(PairingError::Invalid(format!("name contains {} ({}); use {ALLOWED}", c.escape_debug(), c.escape_unicode())));
        }
        if c == ' ' && prev_space {
            return Err(PairingError::Invalid(format!("name contains consecutive spaces; use {ALLOWED}")));
        }
        prev_space = c == ' ';
    }
    Ok(name.to_string())
}

/// Invisible characters that change how text around them is shown: bidi
/// embeddings, overrides and isolates, zero-width characters, line and
/// paragraph separators, the byte-order mark. (The allowlist in
/// [`validate_name`] refuses them anyway; this names them in the error.)
fn is_format_char(c: char) -> bool {
    matches!(c,
        '\u{00AD}' | '\u{061C}' | '\u{180E}'
        | '\u{200B}'..='\u{200F}'
        | '\u{2028}'..='\u{202E}'
        | '\u{2060}'..='\u{206F}'
        | '\u{FEFF}' | '\u{FFF9}'..='\u{FFFB}')
}

/// Every scope must be one of [`ALL_SCOPES`]; repeats are dropped.
fn validate_scopes(scopes: &[String]) -> Result<Vec<String>, PairingError> {
    let mut out: Vec<String> = Vec::new();
    for s in scopes {
        if !ALL_SCOPES.contains(&s.as_str()) {
            let shown: String = s.chars().take(32).collect();
            return Err(PairingError::Invalid(format!("unknown scope {:?} (one of {})", shown, ALL_SCOPES.join(", "))));
        }
        if !out.contains(s) {
            out.push(s.clone());
        }
    }
    Ok(out)
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn find<'a>(v: &'a mut [Pairing], id: &str, hint: &'static str) -> Result<&'a mut Pairing, PairingError> {
    v.iter_mut().find(|p| p.id == id).ok_or_else(|| PairingError::NotFound { id: id.to_string(), hint })
}

pub struct PairingStore {
    path: PathBuf,
    lock_path: PathBuf,
    /// `tokens.json` beside `pairings.json`, for [`PairingStore::deny`].
    tokens_path: PathBuf,
    /// Serialises this process's writers; the lock file serialises processes.
    guard: Mutex<()>,
    /// Pairings already logged as expired, so the sweep that every read runs
    /// reports each one once.
    expired_seen: Mutex<std::collections::HashSet<String>>,
}

/// Held for a whole load → modify → save.
struct Locked<'a> {
    _guard: MutexGuard<'a, ()>,
    _file: FileLock,
}

impl PairingStore {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            path: data_dir.join("pairings.json"),
            lock_path: data_dir.join("pairings.lock"),
            tokens_path: data_dir.join("tokens.json"),
            guard: Mutex::new(()),
            expired_seen: Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Log pairings the sweep dropped, once each.
    fn note_expired(&self, dropped: Vec<Pairing>) {
        if dropped.is_empty() {
            return;
        }
        let mut seen = self.expired_seen.lock().unwrap_or_else(PoisonError::into_inner);
        if seen.len() > 4096 {
            seen.clear();
        }
        for p in dropped {
            if !seen.insert(p.id.clone()) {
                continue;
            }
            match p.status {
                PairingStatus::Pending => tracing::info!(id = %p.id, name = p.name.as_str(), "pairing request expired undecided"),
                PairingStatus::Approved if !p.claimed => tracing::warn!(
                    id = %p.id,
                    name = p.name.as_str(),
                    token_name = p.token_name.as_deref(),
                    "approved pairing expired before its token was collected; the token stays valid until revoked (estia token revoke)"
                ),
                _ => tracing::debug!(id = %p.id, name = p.name.as_str(), "pairing record dropped"),
            }
        }
    }

    fn lock(&self) -> Result<Locked<'_>, PairingError> {
        let guard = self.guard.lock().unwrap_or_else(PoisonError::into_inner);
        let file = lock_exclusive(&self.lock_path).map_err(|e| anyhow::anyhow!("lock {}: {e}", self.lock_path.display()))?;
        Ok(Locked { _guard: guard, _file: file })
    }

    /// What is on disk, with expired requests swept. Missing means none; a
    /// file that does not parse is an error rather than an empty list that a
    /// writer would then save over every other pairing.
    fn load(&self) -> Result<Vec<Pairing>, PairingError> {
        let mut v: Vec<Pairing> = match std::fs::read_to_string(&self.path) {
            Ok(t) if t.trim().is_empty() => Vec::new(),
            Ok(t) => serde_json::from_str(&t).map_err(|e| anyhow::anyhow!("{} does not parse: {e}", self.path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(anyhow::anyhow!("read {}: {e}", self.path.display()).into()),
        };
        let cutoff = now().saturating_sub(PAIRING_TTL_SECS);
        // Sweep: pending requests expire; decided ones stay until claimed or expired.
        let keep = |p: &Pairing| {
            p.created_unix >= cutoff
                || (p.status == PairingStatus::Approved && !p.claimed && p.created_unix >= cutoff.saturating_sub(PAIRING_TTL_SECS))
        };
        let (kept, dropped): (Vec<Pairing>, Vec<Pairing>) = v.into_iter().partition(keep);
        v = kept;
        self.note_expired(dropped);
        Ok(v)
    }

    /// Owner-only and atomic: an approved, unclaimed pairing holds its token
    /// in plaintext here until the device collects it.
    fn save(&self, v: &[Pairing]) -> Result<(), PairingError> {
        let text = serde_json::to_string_pretty(v).map_err(anyhow::Error::from)?;
        write_private(&self.path, text.as_bytes())?;
        Ok(())
    }

    /// Current requests (an unreadable file reads as none).
    pub fn list(&self) -> Vec<Pairing> {
        self.load().unwrap_or_default()
    }

    /// A client asks to be paired. `from` is its address; it counts against
    /// [`MAX_PENDING_PER_SOURCE`].
    pub fn request(&self, name: &str, scopes: &[String], from: Option<String>) -> Result<Pairing, PairingError> {
        let name = validate_name(name)?;
        let scopes = validate_scopes(scopes)?;
        let mut bytes = [0u8; 8];
        getrandom::getrandom(&mut bytes).map_err(|e| anyhow::anyhow!("randomness: {e}"))?;
        let id: String = bytes.iter().map(|b| format!("{b:02x}")).collect();

        let _lock = self.lock()?;
        let mut v = self.load()?;
        if let Some(src) = from.as_deref() {
            let mine = v.iter().filter(|p| p.status == PairingStatus::Pending && p.from.as_deref() == Some(src)).count();
            if mine >= MAX_PENDING_PER_SOURCE {
                return Err(PairingError::TooMany(format!(
                    "too many pending pairing requests from {src} (at most {MAX_PENDING_PER_SOURCE}); \
                     wait for the operator to decide them or for them to expire ({PAIRING_TTL_SECS} s)"
                )));
            }
        }
        if v.iter().filter(|p| p.status == PairingStatus::Pending).count() >= MAX_PENDING {
            return Err(PairingError::TooMany(format!("too many pending pairing requests (at most {MAX_PENDING}); try again later")));
        }
        let p = Pairing {
            id,
            name,
            scopes,
            status: PairingStatus::Pending,
            created_unix: now(),
            from,
            token_plain: None,
            claimed: false,
            token_name: None,
            revoked: false,
        };
        v.push(p.clone());
        self.save(&v)?;
        tracing::info!(
            id = %p.id,
            name = p.name.as_str(),
            scopes = %p.scopes.join(","),
            from = p.from.as_deref().map(tracing::field::display),
            "pairing requested"
        );
        Ok(p)
    }

    /// The operator approves: a token is minted with the requested scopes and
    /// parked on the record until the client collects it.
    pub fn approve(&self, id: &str, tokens: &TokenStore) -> Result<Pairing, PairingError> {
        let _lock = self.lock()?;
        let mut v = self.load()?;
        let p = find(&mut v, id, "")?;
        if p.status != PairingStatus::Pending {
            return Err(PairingError::Invalid(format!("pairing `{id}` is already {:?}", p.status)));
        }
        let token_name = format!("pair:{}:{}", p.name, p.id);
        let refs: Vec<&str> = p.scopes.iter().map(String::as_str).collect();
        // The name carries this pairing's random id, so a token already under
        // it can only be left over from an earlier approve of this very
        // pairing that failed to save: replacing it is right.
        let (token, _) = tokens.replace(&token_name, &refs)?;
        p.status = PairingStatus::Approved;
        p.token_plain = Some(token);
        p.token_name = Some(token_name.clone());
        let out = p.clone();
        if let Err(e) = self.save(&v) {
            // Not recorded as approved, so nobody will ever collect it.
            let _ = tokens.revoke(&token_name);
            return Err(e);
        }
        tracing::info!(id = %out.id, name = out.name.as_str(), scopes = %out.scopes.join(","), token_name = token_name.as_str(), "pairing approved");
        Ok(out)
    }

    /// The operator refuses. A pending request just becomes denied. An
    /// approved one — whether or not the device has collected its token yet —
    /// also has that token revoked, so it stops working on the next request;
    /// the returned record says so with `revoked`. Denying a denied pairing
    /// changes nothing.
    ///
    /// Revokes through the `tokens.json` beside `pairings.json`; a caller that
    /// already holds that store (the daemon) uses [`PairingStore::deny_with`].
    pub fn deny(&self, id: &str) -> Result<Pairing, PairingError> {
        self.deny_inner(id, None)
    }

    /// [`PairingStore::deny`], revoking through `tokens`.
    pub fn deny_with(&self, id: &str, tokens: &TokenStore) -> Result<Pairing, PairingError> {
        self.deny_inner(id, Some(tokens))
    }

    fn deny_inner(&self, id: &str, tokens: Option<&TokenStore>) -> Result<Pairing, PairingError> {
        let _lock = self.lock()?;
        let mut v = self.load()?;
        let p = find(&mut v, id, DENY_HINT)?;
        match p.status {
            PairingStatus::Denied => return Ok(p.clone()),
            PairingStatus::Pending => {}
            PairingStatus::Approved => {
                let name = p.token_name.clone().unwrap_or_else(|| format!("pair:{}:{}", p.name, p.id));
                // Revoke before recording the denial: if the save then fails,
                // what is left is a dead token behind "approved", never a live
                // one behind "denied".
                match tokens {
                    Some(t) => t.revoke(&name)?,
                    None => TokenStore::open(&self.tokens_path)?.revoke(&name)?,
                };
                p.revoked = true;
            }
        }
        p.status = PairingStatus::Denied;
        p.token_plain = None;
        let out = p.clone();
        self.save(&v)?;
        tracing::info!(id = %out.id, name = out.name.as_str(), revoked = out.revoked, token_name = out.token_name.as_deref(), "pairing denied");
        Ok(out)
    }

    /// The client polls. An approved token is handed over exactly once: only
    /// the poll that records the claim returns it, and only after that record
    /// is on disk.
    pub fn poll(&self, id: &str) -> Result<(PairingStatus, Option<String>), PairingError> {
        // Most polls change nothing (still pending, denied, already
        // collected): answer those from a plain read, so a client polling in
        // a loop never holds the lock or writes the file.
        let collectable = |p: &Pairing| p.status == PairingStatus::Approved && !p.claimed;
        let mut v = self.load()?;
        let p = find(&mut v, id, "")?;
        if !collectable(p) {
            return Ok((p.status, None));
        }
        let _lock = self.lock()?;
        let mut v = self.load()?;
        let p = find(&mut v, id, "")?;
        let status = p.status;
        if !collectable(p) {
            return Ok((status, None));
        }
        p.claimed = true;
        let token = p.token_plain.take();
        let (id, name) = (p.id.clone(), p.name.clone());
        self.save(&v)?;
        tracing::info!(id = %id, name = name.as_str(), "pairing token collected");
        Ok((status, token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn scratch(tag: &str) -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("estia-pair-{tag}-{}-{nanos}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn scopes(s: &[&str]) -> Vec<String> {
        s.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn request_approve_poll_once() {
        let dir = scratch("once");
        let store = PairingStore::new(&dir);
        let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
        let p = store.request("phone", &scopes(&["generate", "embed"]), Some("10.0.0.7".into())).unwrap();
        assert_eq!(store.poll(&p.id).unwrap(), (PairingStatus::Pending, None));
        assert!(store.request("x", &["bogus".into()], None).is_err());
        let approved = store.approve(&p.id, &tokens).unwrap();
        assert_eq!(approved.status, PairingStatus::Approved);
        assert_eq!(approved.token_name.as_deref(), Some(format!("pair:phone:{}", p.id).as_str()));
        let (status, token) = store.poll(&p.id).unwrap();
        assert_eq!(status, PairingStatus::Approved);
        let token = token.expect("token handed over once");
        assert!(tokens.verify(&token).unwrap().allows("embed"));
        assert!(!tokens.verify(&token).unwrap().allows("admin"));
        assert_eq!(store.poll(&p.id).unwrap(), (PairingStatus::Approved, None), "second poll gets no token");
        assert!(store.approve(&p.id, &tokens).is_err(), "cannot approve twice");
        let text = std::fs::read_to_string(dir.join("pairings.json")).unwrap();
        assert!(!text.contains(&token), "claimed plaintext is gone from disk");
        let d = store.request("laptop", &scopes(&["generate"]), None).unwrap();
        let denied = store.deny(&d.id).unwrap();
        assert!(!denied.revoked, "nothing to revoke for a pending request");
        assert_eq!(store.poll(&d.id).unwrap().0, PairingStatus::Denied);
        assert!(matches!(store.poll("0000000000000000"), Err(PairingError::NotFound { .. })));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn names_are_checked_at_the_source() {
        let dir = scratch("names");
        let store = PairingStore::new(&dir);
        let ok = |n: &str| store.request(n, &[], Some(format!("10.1.{}.1", n.len()))).map(|p| p.name);
        assert_eq!(ok("  phone  ").unwrap(), "phone", "trimmed");
        for good in ["George’s iPad", "O'Brien_pc-2 (work)", "Γιώργος laptop", "mac.local", "東京"] {
            assert_eq!(validate_name(good).unwrap(), good);
        }
        let hostile = [
            "ipad\u{1b}[8m",          // SGR conceal
            "ipad\u{1b}[17Cgenerate", // cursor forward
            "ipad\rgenerate",         // carriage return
            "ipad\ngenerate",         // newline
            "ipad\u{7}",              // bell
            "ipad\u{9b}8m",           // C1 CSI
            "ipad\u{202e}nimda",      // bidi override
            "ipad\u{2066}x\u{2069}",  // bidi isolate
            "ip\u{200b}ad",           // zero width space
            "ip\u{200f}ad",           // RTL mark
            "\u{feff}ipad",           // BOM (not trimmed)
            "ipad\u{2028}x",          // line separator
            "ipad  generate",         // column padding
            "ipad/phone",             // outside the set
            "pair:ipad",              // colon: token names split on it
            "📱",                     // outside the set
            "",                       // empty
            "   ",                    // blank
        ];
        for bad in hostile {
            let e = validate_name(bad).expect_err(bad);
            assert!(matches!(e, PairingError::Invalid(_)));
            // The message never carries the raw character back.
            assert!(!e.to_string().chars().any(|c| c.is_control() || is_format_char(c)), "{e}");
            assert!(ok(bad).is_err(), "request refuses {bad:?}");
        }
        let long = "a".repeat(MAX_NAME_CHARS + 1);
        assert!(validate_name(&long).is_err(), "too long is refused, not truncated");
        assert!(validate_name(&"é".repeat(MAX_NAME_CHARS)).is_ok(), "the limit counts characters, not bytes");
        assert_eq!(store.list().len(), 1, "only the good request was stored");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scopes_must_be_known_and_repeats_are_dropped() {
        let dir = scratch("scopes");
        let store = PairingStore::new(&dir);
        let e = store.request("x", &scopes(&["generate", "root"]), None).unwrap_err();
        assert!(matches!(e, PairingError::Invalid(_)) && e.to_string().contains("unknown scope"), "{e}");
        let p = store.request("x", &scopes(&["embed", "embed", "generate", "embed"]), None).unwrap();
        assert_eq!(p.scopes, vec!["embed", "generate"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pending_caps_per_source_and_overall() {
        let dir = scratch("caps");
        let store = PairingStore::new(&dir);
        // Expired requests from the same source do not count.
        let old = now() - PAIRING_TTL_SECS - 10;
        let stale: Vec<Pairing> = (0..MAX_PENDING_PER_SOURCE)
            .map(|i| Pairing {
                id: format!("old{i}"),
                name: "old".into(),
                scopes: vec![],
                status: PairingStatus::Pending,
                created_unix: old,
                from: Some("10.0.0.9".into()),
                token_plain: None,
                claimed: false,
                token_name: None,
                revoked: false,
            })
            .collect();
        store.save(&stale).unwrap();
        for i in 0..MAX_PENDING_PER_SOURCE {
            store.request(&format!("dev {i}"), &[], Some("10.0.0.9".into())).unwrap();
        }
        let e = store.request("one more", &[], Some("10.0.0.9".into())).unwrap_err();
        assert!(matches!(e, PairingError::TooMany(_)) && e.to_string().contains("10.0.0.9"), "{e}");
        // Another address still gets through, up to the overall cap.
        let mut n = MAX_PENDING_PER_SOURCE;
        let mut host = 0;
        while n < MAX_PENDING {
            for _ in 0..MAX_PENDING_PER_SOURCE.min(MAX_PENDING - n) {
                store.request("other", &[], Some(format!("10.0.1.{host}"))).unwrap();
                n += 1;
            }
            host += 1;
        }
        let e = store.request("late", &[], Some("10.0.2.1".into())).unwrap_err();
        assert!(matches!(e, PairingError::TooMany(_)), "{e}");
        assert_eq!(store.list().len(), MAX_PENDING);
        // Deciding one frees a slot.
        let first = store.list()[0].id.clone();
        store.deny(&first).unwrap();
        store.request("late", &[], Some("10.0.2.1".into())).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deny_after_approve_revokes_the_token() {
        let dir = scratch("deny");
        let store = PairingStore::new(&dir);
        let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();

        // Approved and collected, then denied through the daemon's store.
        let p = store.request("laptop", &scopes(&["admin"]), Some("10.0.0.2".into())).unwrap();
        store.approve(&p.id, &tokens).unwrap();
        let token = store.poll(&p.id).unwrap().1.expect("collected");
        assert!(tokens.verify(&token).is_some());
        let denied = store.deny_with(&p.id, &tokens).unwrap();
        assert_eq!(denied.status, PairingStatus::Denied);
        assert!(denied.revoked);
        assert!(tokens.verify(&token).is_none(), "a denied device's token stops working");
        assert_eq!(store.poll(&p.id).unwrap(), (PairingStatus::Denied, None));
        assert!(store.list().iter().any(|q| q.id == p.id && q.revoked));
        // Denying again changes nothing and is not an error.
        assert!(store.deny_with(&p.id, &tokens).unwrap().revoked);

        // Approved but not yet collected, denied through `deny` (the CLI's
        // path, which opens tokens.json beside pairings.json).
        let q = store.request("phone", &scopes(&["generate"]), Some("10.0.0.3".into())).unwrap();
        store.approve(&q.id, &tokens).unwrap();
        assert!(store.deny(&q.id).unwrap().revoked);
        assert_eq!(store.poll(&q.id).unwrap(), (PairingStatus::Denied, None), "never handed over");
        let on_disk = TokenStore::open(dir.join("tokens.json")).unwrap().list();
        assert!(!on_disk.iter().any(|r| r.name.starts_with("pair:")), "no pairing token left: {on_disk:?}");
        assert!(!std::fs::read_to_string(dir.join("pairings.json")).unwrap().contains("estia_"));

        let e = store.deny("ffffffffffffffff").unwrap_err();
        assert!(e.to_string().contains("estia token list"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Pollers race an approval made by another store (the CLI in another
    /// process): the approval is never lost and exactly one poll gets the token.
    #[test]
    fn racing_polls_get_the_token_exactly_once() {
        let dir = scratch("pollrace");
        let shared = Arc::new(PairingStore::new(&dir));
        for trial in 0..6 {
            let p = shared.request(&format!("device {trial}"), &scopes(&["generate"]), Some(format!("10.2.0.{trial}"))).unwrap();
            let pollers = 12;
            let start = Arc::new(Barrier::new(pollers + 1));
            let handles: Vec<_> = (0..pollers)
                .map(|i| {
                    let (dir, shared, start, id) = (dir.clone(), Arc::clone(&shared), Arc::clone(&start), p.id.clone());
                    std::thread::spawn(move || {
                        // Half share the daemon's store, half have their own (as
                        // separate processes would).
                        let own = PairingStore::new(&dir);
                        let store: &PairingStore = if i % 2 == 0 { &shared } else { &own };
                        start.wait();
                        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
                        loop {
                            match store.poll(&id).expect("no poll fails") {
                                (PairingStatus::Approved, token) => return token,
                                (PairingStatus::Pending, _) => assert!(std::time::Instant::now() < deadline, "approval lost"),
                                (other, _) => panic!("unexpected status {other:?}"),
                            }
                        }
                    })
                })
                .collect();
            start.wait();
            std::thread::sleep(std::time::Duration::from_millis(5));
            let cli = PairingStore::new(&dir);
            let cli_tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
            cli.approve(&p.id, &cli_tokens).unwrap();
            let delivered: Vec<String> = handles.into_iter().filter_map(|h| h.join().unwrap()).collect();
            assert_eq!(delivered.len(), 1, "trial {trial}: exactly one poll gets the token");
            assert!(cli_tokens.verify(&delivered[0]).is_some());
            let rec = shared.list().into_iter().find(|q| q.id == p.id).unwrap();
            assert!(rec.status == PairingStatus::Approved && rec.claimed && rec.token_plain.is_none());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Requests and approvals from many threads and stores at once: every
    /// accepted request and every approval is on disk afterwards.
    #[test]
    fn concurrent_requests_and_approvals_lose_nothing() {
        let dir = scratch("rmw");
        let seed = PairingStore::new(&dir);
        let half = MAX_PENDING / 2;
        let early: Vec<String> =
            (0..half).map(|i| seed.request(&format!("early {i}"), &scopes(&["embed"]), Some(format!("10.3.0.{i}"))).unwrap().id).collect();
        let start = Arc::new(Barrier::new(half * 2));
        let mut handles = Vec::new();
        for id in early.clone() {
            let (dir, start) = (dir.clone(), Arc::clone(&start));
            handles.push(std::thread::spawn(move || {
                let store = PairingStore::new(&dir);
                let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
                start.wait();
                store.approve(&id, &tokens).unwrap();
                None
            }));
        }
        for i in 0..half {
            let (dir, start) = (dir.clone(), Arc::clone(&start));
            handles.push(std::thread::spawn(move || {
                let store = PairingStore::new(&dir);
                start.wait();
                Some(store.request(&format!("late {i}"), &scopes(&["generate"]), Some(format!("10.3.1.{i}"))).unwrap().id)
            }));
        }
        let late: Vec<String> = handles.into_iter().filter_map(|h| h.join().unwrap()).collect();
        let all = seed.list();
        assert_eq!(all.len(), MAX_PENDING, "no request lost");
        for id in &early {
            let p = all.iter().find(|p| &p.id == id).expect("early request kept");
            assert_eq!(p.status, PairingStatus::Approved, "no approval lost");
        }
        for id in &late {
            assert_eq!(all.iter().find(|p| &p.id == id).expect("late request kept").status, PairingStatus::Pending);
        }
        let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
        assert_eq!(tokens.list().iter().filter(|r| r.name.starts_with("pair:")).count(), half, "no minted token lost");
        for id in &early {
            let t = seed.poll(id).unwrap().1.expect("token parked for collection");
            assert!(tokens.verify(&t).is_some());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
