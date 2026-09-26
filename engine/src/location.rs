//! Where a host's engine runs: in this process, or a daemon somewhere else.

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
        /// `http://host:port` — no trailing path.
        base_url: String,
        /// Bearer token from pairing. `None` only for a daemon running with
        /// auth disabled on loopback.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        token: Option<String>,
        /// Display name (the host's Bonjour name, or whatever the user typed).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
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
        let r =
            EngineLocation::Remote { base_url: "http://10.0.0.5:27200".into(), token: Some("estia_x".into()), name: Some("studio".into()) };
        r.save(&p).unwrap();
        assert_eq!(EngineLocation::load(&p), r);
        assert!(r.is_remote());
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.contains("\"mode\": \"remote\""));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
