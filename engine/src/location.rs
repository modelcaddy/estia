//! Where a host's engine runs: in this process, or a daemon somewhere else.

use crate::error::SessionError;
use crate::remote::RemoteEngine;
use crate::tls::Fingerprint;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum EngineLocation {
    /// The engine library inside this process (runners as child processes).
    #[default]
    Local,
    /// A `estia serve` daemon reached over HTTP.
    Remote {
        /// `https://host:port` for an engine on the LAN, `http://127.0.0.1:port`
        /// for the one on this machine; no trailing path.
        base_url: String,
        /// Bearer token from pairing. `None` only for a daemon running with
        /// auth disabled on loopback.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token: Option<String>,
        /// Display name (the host's Bonjour name, or whatever the user typed).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// The engine's certificate fingerprint, from pairing (what
        /// [`crate::PendingPairing::confirm_match`] returns) or confirmed by the
        /// user: connect with [`EngineLocation::connect`]. `None` for plain
        /// HTTP on this machine. JSON `"pin": "<64 hex digits>"`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pin: Option<Fingerprint>,
    },
}

impl EngineLocation {
    /// Read the setting; a missing or unreadable file means local.
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self).map_err(std::io::Error::other)?)
    }

    pub fn is_remote(&self) -> bool {
        matches!(self, EngineLocation::Remote { .. })
    }

    /// The client for a remote location, with its token and its pin
    /// ([`RemoteEngine::connect`]); `None` for [`EngineLocation::Local`].
    /// Prefer it to taking the fields apart: a pattern with `..` drops the pin
    /// without a word, and the https engine then refuses to connect.
    ///
    /// A location saved before 0.5 (`http://` to another machine, a token, no
    /// pin) is an error here: its token would cross the network in clear,
    /// and an engine from 0.5 on refuses plain HTTP from other devices. Pair
    /// again over `https://`.
    pub fn connect(&self) -> Result<Option<RemoteEngine>, SessionError> {
        match self {
            EngineLocation::Local => Ok(None),
            EngineLocation::Remote { base_url, token, pin, .. } => RemoteEngine::connect(base_url.clone(), token.clone(), *pin).map(Some),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_defaults_local() {
        let dir = std::env::temp_dir().join(format!("estia-loc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p = dir.join("engine_location.json");
        assert_eq!(EngineLocation::load(&p), EngineLocation::Local);
        let r = EngineLocation::Remote {
            base_url: "http://127.0.0.1:27200".into(),
            token: Some("estia_x".into()),
            name: Some("studio".into()),
            pin: None,
        };
        r.save(&p).unwrap();
        assert_eq!(EngineLocation::load(&p), r);
        assert!(r.is_remote());
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.contains("\"mode\": \"remote\""));
        assert!(!text.contains("pin"), "no pin, no key: {text}");
        // With a pin: saved as hex, read back as the same fingerprint.
        let pin = Fingerprint::of_certificate(b"engine certificate");
        let r =
            EngineLocation::Remote { base_url: "https://10.0.0.5:27200".into(), token: Some("estia_x".into()), name: None, pin: Some(pin) };
        r.save(&p).unwrap();
        assert!(std::fs::read_to_string(&p).unwrap().contains(&format!("\"pin\": \"{}\"", pin.to_hex())));
        assert_eq!(EngineLocation::load(&p), r);
        // A file written before pins existed still reads, but its plain-http
        // token to another machine is not sent anywhere.
        std::fs::write(&p, r#"{"mode":"remote","base_url":"http://10.0.0.5:27200","token":"estia_x"}"#).unwrap();
        let old = EngineLocation::load(&p);
        assert!(matches!(old, EngineLocation::Remote { pin: None, .. }));
        assert!(old.connect().unwrap_err().to_string().contains("in clear"));
        assert!(EngineLocation::Local.connect().unwrap().is_none());
        let local =
            EngineLocation::Remote { base_url: "http://127.0.0.1:27200".into(), token: Some("estia_x".into()), name: None, pin: None };
        assert!(local.connect().unwrap().unwrap().pin().is_none());
        // The pin is what the client is held to.
        #[cfg(feature = "remote-tls")]
        {
            let pinned = EngineLocation::Remote { base_url: "https://10.0.0.5:27200".into(), token: None, name: None, pin: Some(pin) };
            assert_eq!(pinned.connect().unwrap().unwrap().pin(), Some(pin));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
