//! Access for an app on the same machine, as the same user, without typing.
//!
//! A LAN device pairs and waits for the operator's approval. An app on this
//! Mac, run by the user who runs the engine, can prove more than that: it can
//! read files only that user can read. So on start `serve` writes a fresh
//! secret to [`SECRET_FILE`] in its data directory, readable by its owner
//! only (0600), and `POST /engine/local-token` trades that secret for a token
//! with narrow scopes. The request must also come from loopback.
//!
//! - The secret changes on every start and the file is removed on a clean
//!   exit, so a copy is only good while this engine runs.
//! - Tokens minted this way are named `local-<name>` and may only carry
//!   `generate`, `embed` and `models:read`; asking again replaces the app's
//!   earlier token (an app that lost its token recovers by asking again).
//!   They can never be admin, never take a name a person or paired device
//!   holds, and are listed and revoked like any other token.

use crate::tokens::{SCOPE_EMBED, SCOPE_GENERATE, SCOPE_MODELS_READ};
use std::path::{Path, PathBuf};

/// The secret, in the engine's data directory.
pub const SECRET_FILE: &str = "local-access.secret";

/// Scopes a local app may ask for; the default is `generate` and `embed`.
pub const LOCAL_SCOPES: &[&str] = &[SCOPE_GENERATE, SCOPE_EMBED, SCOPE_MODELS_READ];

/// Prefix of every token minted through this route.
pub const TOKEN_PREFIX: &str = "local-";

pub fn secret_path(data_dir: &Path) -> PathBuf {
    data_dir.join(SECRET_FILE)
}

/// Write a new secret, readable by this user only, and return it. Any older
/// file is replaced, so a copy from an earlier run stops working.
pub fn publish(data_dir: &Path) -> std::io::Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).map_err(|e| std::io::Error::other(e.to_string()))?;
    let secret: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let path = secret_path(data_dir);
    // Created fresh with the mode set at creation: never a moment where the
    // new secret sits in a file others can read.
    let _ = std::fs::remove_file(&path);
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    use std::io::Write;
    opts.open(&path)?.write_all(secret.as_bytes())?;
    Ok(secret)
}

pub fn withdraw(data_dir: &Path) {
    let _ = std::fs::remove_file(secret_path(data_dir));
}

/// Equal without leaking where they differ through timing.
pub fn same_secret(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The app name a local token is minted under: lowercase letters, digits,
/// `.`, `_`, `-`, 1 to 40 of them.
pub fn token_name(app: &str) -> Result<String, String> {
    let ok = !app.is_empty()
        && app.len() <= 40
        && app.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'));
    if ok {
        Ok(format!("{TOKEN_PREFIX}{app}"))
    } else {
        Err("name must be 1 to 40 of a-z, 0-9, `.`, `_`, `-`".to_string())
    }
}

/// The scopes asked for, checked against [`LOCAL_SCOPES`].
pub fn scopes(asked: Option<&[String]>) -> Result<Vec<&'static str>, String> {
    let Some(asked) = asked else { return Ok(vec![SCOPE_GENERATE, SCOPE_EMBED]) };
    if asked.is_empty() {
        return Err("ask for at least one scope".to_string());
    }
    asked
        .iter()
        .map(|s| {
            LOCAL_SCOPES
                .iter()
                .copied()
                .find(|l| l == s)
                .ok_or_else(|| format!("scope `{s}` cannot be granted to a local app; it may have {}", LOCAL_SCOPES.join(", ")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_secret_is_fresh_each_time_and_private() {
        let dir = std::env::temp_dir().join(format!("estia-local-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = publish(&dir).unwrap();
        let b = publish(&dir).unwrap();
        assert_ne!(a, b);
        assert_eq!(std::fs::read_to_string(secret_path(&dir)).unwrap(), b);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(secret_path(&dir)).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        withdraw(&dir);
        assert!(!secret_path(&dir).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn names_and_scopes_are_narrow() {
        assert_eq!(token_name("modelcaddy").unwrap(), "local-modelcaddy");
        assert!(token_name("Admin").is_err() && token_name("").is_err() && token_name("a b").is_err());
        assert_eq!(scopes(None).unwrap(), [SCOPE_GENERATE, SCOPE_EMBED]);
        assert!(scopes(Some(&["admin".to_string()])).is_err());
        assert!(scopes(Some(&["models:write".to_string()])).is_err());
        assert!(scopes(Some(&[])).is_err());
        assert_eq!(scopes(Some(&["embed".to_string(), "models:read".to_string()])).unwrap(), [SCOPE_EMBED, SCOPE_MODELS_READ]);
        assert!(same_secret("abc", "abc") && !same_secret("abc", "abd") && !same_secret("abc", "abcd"));
    }
}
