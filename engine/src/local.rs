//! The engine on this machine, as another app finds and joins it.
//!
//! `estia serve` records itself in `engine.json` in its data directory (pid,
//! port, bind) and writes a same-user secret beside it
//! (`local-access.secret`, see the server's `local_access` module). An app run
//! by the same user reads both: [`find_local_engine`] says whether an engine is
//! up and where, [`LocalEngine::claim_token`] trades the secret for a token
//! with `generate` and `embed`, with nothing for the person to type.

use crate::error::SessionError;
use crate::remote::RemoteEngine;
use serde::Deserialize;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Where `estia` keeps its data unless told otherwise (`--data-dir`,
/// `ESTIA_DATA_DIR`): `~/Library/Application Support/estia` on macOS,
/// `~/.local/share/estia` elsewhere.
pub fn default_data_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    if cfg!(target_os = "macos") {
        home.join("Library/Application Support/estia")
    } else {
        home.join(".local/share/estia")
    }
}

/// The data directory `estia` uses when not given one on the command line:
/// `ESTIA_DATA_DIR` when set, else [`default_data_dir`]. What an app should
/// look in to find the engine the user runs.
pub fn configured_data_dir() -> PathBuf {
    std::env::var_os("ESTIA_DATA_DIR").map(PathBuf::from).filter(|p| p.is_absolute()).unwrap_or_else(default_data_dir)
}

/// A running engine found through its data directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalEngine {
    /// `http://127.0.0.1:<port>`: loopback, whatever address it is bound to.
    pub url: String,
    pub data_dir: PathBuf,
    pub pid: u32,
    pub port: u16,
}

#[derive(Deserialize)]
struct Record {
    pid: u32,
    port: u16,
    bind: String,
}

/// The engine recorded in `data_dir` (default: [`configured_data_dir`]), when
/// something answers on its port. `None` when there is no record, or the
/// engine that wrote it has gone.
pub fn find_local_engine(data_dir: Option<&Path>) -> Option<LocalEngine> {
    let data_dir = data_dir.map(Path::to_path_buf).unwrap_or_else(configured_data_dir);
    let text = std::fs::read_to_string(data_dir.join("engine.json")).ok()?;
    let rec: Record = serde_json::from_str(&text).ok()?;
    let ip: IpAddr = rec.bind.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
    // A wildcard or LAN bind still answers on loopback; asking there keeps the
    // handshake on this machine, which the server requires.
    let ip = if ip.is_ipv6() { IpAddr::V6(Ipv6Addr::LOCALHOST) } else { IpAddr::V4(Ipv4Addr::LOCALHOST) };
    let addr = SocketAddr::new(ip, rec.port);
    std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(300)).ok()?;
    let host = if ip.is_ipv6() { format!("[{ip}]") } else { ip.to_string() };
    Some(LocalEngine { url: format!("http://{host}:{}", rec.port), data_dir, pid: rec.pid, port: rec.port })
}

impl LocalEngine {
    /// The same-user secret this engine published when it started.
    pub fn secret(&self) -> std::io::Result<String> {
        std::fs::read_to_string(self.data_dir.join("local-access.secret")).map(|s| s.trim().to_string())
    }

    /// A token for `app` (lowercase letters, digits, `.`, `_`, `-`), minted as
    /// `local-<app>` with `scopes` (default `generate` and `embed`; at most
    /// those and `models:read`). Asking again replaces the app's earlier token.
    pub fn claim_token(&self, app: &str, scopes: Option<&[&str]>) -> Result<String, SessionError> {
        let secret = self.secret().map_err(|e| {
            SessionError::Runner(format!("cannot read the engine's local access secret in {}: {e}", self.data_dir.display()))
        })?;
        RemoteEngine::new(self.url.clone(), None)?.local_token(&secret, app, scopes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_record_or_nothing_listening_means_none() {
        let dir = std::env::temp_dir().join(format!("estia-find-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(find_local_engine(Some(&dir)), None);
        // A record whose engine is gone: nothing answers on its port.
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        std::fs::write(dir.join("engine.json"), format!(r#"{{"pid":1,"port":{port},"bind":"127.0.0.1"}}"#)).unwrap();
        assert_eq!(find_local_engine(Some(&dir)), None);
        // Something listening, bound to every address: reached on loopback.
        let live = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = live.local_addr().unwrap().port();
        std::fs::write(dir.join("engine.json"), format!(r#"{{"pid":7,"port":{port},"bind":"0.0.0.0","started_unix":0,"api_version":1}}"#))
            .unwrap();
        let found = find_local_engine(Some(&dir)).expect("found");
        assert_eq!((found.url, found.pid), (format!("http://127.0.0.1:{port}"), 7));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
