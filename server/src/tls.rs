//! HTTPS on a LAN bind: the engine's own certificate, and how a connection is
//! told apart as TLS or plain HTTP.
//!
//! **The certificate.** On the first start with a non-loopback bind (`estia
//! serve --lan`, or `estia service install`), the engine makes a self-signed
//! ECDSA P-256 certificate and keeps it in `tls/` in the data directory
//! (`key.pem` and `cert.pem`, owner-only). Devices pin its SHA-256 fingerprint
//! when they pair (see `estia_engine::tls`), so it is never replaced on its
//! own: not when the machine's address or name changes (the names in it are
//! cosmetic; pinned clients ignore them), and not when it expires (pinned
//! clients ignore dates too). `estia tls rotate` makes a new one, after which
//! every paired device must pair again.
//!
//! It is valid for [`VALIDITY_DAYS`]: Apple's TLS stack refuses a server
//! certificate valid for more than 825 days even when it is trusted as an
//! anchor, and that covers Safari on the browser page and any Apple app that
//! trusts the certificate rather than pinning it. Those clients, and the
//! others that trust it as an authority (Node's `NODE_EXTRA_CA_CERTS`, curl
//! `--cacert`), see it expire after that; pinned devices do not.
//!
//! Pins name the whole certificate, not its public key. The same SHA-256 is
//! what a browser's certificate viewer and `openssl x509 -fingerprint` show,
//! and a pinned client never relies on parsing the certificate itself. The
//! price: even a certificate re-issued with the same key would break every
//! pin, so the engine never re-issues on its own.
//!
//! **On disk.** Every load, creation and rotation holds an exclusive lock on
//! `tls/.lock`, so two processes (a service starting while `estia tls
//! rotate` runs, two first starts at once) never mix halves of two pairs. A
//! new pair is written whole as `key.pem.new` and `cert.pem.new` before
//! either is renamed into place, and a rename cut short is finished or undone
//! under the lock before the pair is read ([`finish_interrupted`]).
//!
//! **One port, two transports.** The engine still listens on one port.
//! Each connection is sniffed: a first byte of `0x16` is a TLS handshake;
//! anything else is plain HTTP, which is served only to a loopback peer, so an
//! app on the engine's own machine keeps using `http://127.0.0.1:<port>` (on
//! a wildcard bind; an engine bound to one LAN address has nothing on
//! loopback, and the machine itself reaches it at that address over HTTPS,
//! as `estia status` does with the certificate in the data directory). Plain
//! HTTP from any other address, the machine's own LAN address included, is
//! answered (the router's `https_only` step) with a 400 of type
//! `https_required`, or a redirect for the browser page, so an old client is
//! told what to do in its own error envelope. A LAN host cannot send packets
//! with a loopback source address that reach the socket: macOS and Linux drop
//! such martians on other interfaces.

use estia_engine::tls::Fingerprint;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The engine's certificate and key, loaded or just made.
pub struct TlsIdentity {
    cert_der: Vec<u8>,
    key_der: Vec<u8>,
    fingerprint: Fingerprint,
    created_unix: u64,
    /// Made by this call rather than read from disk.
    fresh: bool,
}

impl std::fmt::Debug for TlsIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TlsIdentity").field("fingerprint", &self.fingerprint).field("created_unix", &self.created_unix).finish()
    }
}

impl TlsIdentity {
    /// What devices pin.
    pub fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    /// The certificate, DER.
    pub fn certificate_der(&self) -> &[u8] {
        &self.cert_der
    }

    /// When the certificate file was written (Unix seconds; 0 when unknown).
    pub fn created_unix(&self) -> u64 {
        self.created_unix
    }

    /// True when this call made the certificate rather than reading it.
    pub fn is_new(&self) -> bool {
        self.fresh
    }

    /// curl's `--pinnedpubkey` value: `sha256//` and the base64 SHA-256 of the
    /// certificate's public key. `curl -k --pinnedpubkey <this> https://…`
    /// skips the chain and name checks a self-signed certificate fails, and
    /// still refuses any other key.
    pub fn curl_pin(&self) -> String {
        use rcgen::PublicKeyData;
        use sha2::{Digest, Sha256};
        let spki = rcgen::KeyPair::try_from(self.key_der.as_slice()).map(|k| k.subject_public_key_info()).unwrap_or_default();
        format!("sha256//{}", crate::base64_std(&Sha256::digest(spki)))
    }
}

/// `<data dir>/tls`, owner-only.
pub fn tls_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("tls")
}

pub fn cert_path(data_dir: &Path) -> PathBuf {
    tls_dir(data_dir).join("cert.pem")
}

pub fn key_path(data_dir: &Path) -> PathBuf {
    tls_dir(data_dir).join("key.pem")
}

/// `tls/.lock`: held across every read and write of the pair.
fn lock(data_dir: &Path) -> anyhow::Result<crate::tokens::FileLock> {
    let path = tls_dir(data_dir).join(".lock");
    crate::tokens::lock_exclusive(&path).map_err(|e| anyhow::anyhow!("lock {}: {e}", path.display()))
}

/// Where a new pair's file waits until it is renamed into place.
fn pending(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".new");
    PathBuf::from(name)
}

/// Is there a directory entry at `path`? A symbolic link counts whether or
/// not its target exists: a link that points nowhere (an unmounted volume, a
/// moved file) is a certificate that cannot be read, not one that is absent,
/// and must never be replaced by a new pair. Only "not found" is absence.
fn present(path: &Path) -> anyhow::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(anyhow::anyhow!("{}: {e}", path.display())),
    }
}

/// A new pair goes in as `key.pem.new` and `cert.pem.new`, each written whole,
/// then the key and then the certificate are renamed over the old pair. A
/// crash or a full disk can stop that anywhere; under the lock, before the
/// pair is read, this finishes or undoes it:
/// - both new files there: nothing was renamed yet; drop them, and the old
///   pair (or none) stands, as if the rotation had not started;
/// - only the new certificate: the new key is in place already; rename the
///   certificate after it;
/// - only the new key: its certificate was never written; drop it.
fn finish_interrupted(data_dir: &Path) -> anyhow::Result<()> {
    let (key, cert) = (key_path(data_dir), cert_path(data_dir));
    let (new_key, new_cert) = (pending(&key), pending(&cert));
    let io = |what: &str, p: &Path, e: std::io::Error| anyhow::anyhow!("{what} {}: {e}", p.display());
    match (present(&new_key)?, present(&new_cert)?) {
        (false, false) => return Ok(()),
        (true, true) => {
            std::fs::remove_file(&new_key).map_err(|e| io("remove", &new_key, e))?;
            std::fs::remove_file(&new_cert).map_err(|e| io("remove", &new_cert, e))?;
        }
        (false, true) => std::fs::rename(&new_cert, &cert).map_err(|e| io("rename", &new_cert, e))?,
        (true, false) => std::fs::remove_file(&new_key).map_err(|e| io("remove", &new_key, e))?,
    }
    tracing::warn!(dir = %tls_dir(data_dir).display(), "finished a TLS certificate change that was cut short");
    Ok(())
}

/// The certificate in `data_dir`, when there is one. An error when the files
/// are there but unreadable (a symbolic link that points nowhere included),
/// or do not belong together. Reads under the lock, so it never sees half a
/// rotation (and finishes one that was cut short); it creates nothing when
/// there is no `tls/` yet.
pub fn load(data_dir: &Path) -> anyhow::Result<Option<TlsIdentity>> {
    if !present(&tls_dir(data_dir))? {
        return Ok(None);
    }
    let _lock = lock(data_dir)?;
    load_locked(data_dir)
}

fn load_locked(data_dir: &Path) -> anyhow::Result<Option<TlsIdentity>> {
    finish_interrupted(data_dir)?;
    let (cert, key) = (cert_path(data_dir), key_path(data_dir));
    let (has_cert, has_key) = (present(&cert)?, present(&key)?);
    match (has_cert, has_key) {
        (true, true) => {}
        (false, false) => return Ok(None),
        // One half missing: a new pair is made (an operator removed a file).
        // The half that is there must still be a readable file.
        (true, false) | (false, true) => {
            let there = if has_cert { &cert } else { &key };
            std::fs::metadata(there).map_err(|e| anyhow::anyhow!("{}: {e}", there.display()))?;
            return Ok(None);
        }
    }
    let cert_der = CertificateDer::from_pem_file(&cert).map_err(|e| anyhow::anyhow!("read {}: {e}", cert.display()))?.as_ref().to_vec();
    let key_der = match PrivateKeyDer::from_pem_file(&key).map_err(|e| anyhow::anyhow!("read {}: {e}", key.display()))? {
        PrivateKeyDer::Pkcs8(k) => k.secret_pkcs8_der().to_vec(),
        _ => anyhow::bail!("{} is not a PKCS#8 key; `estia tls rotate` makes a new pair (paired devices must pair again)", key.display()),
    };
    let created_unix = std::fs::metadata(&cert)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let identity = TlsIdentity { fingerprint: Fingerprint::of_certificate(&cert_der), cert_der, key_der, created_unix, fresh: false };
    // Refuses a key that is not the certificate's.
    server_config(&identity).map_err(|_| {
        anyhow::anyhow!(
            "{} and {} do not belong together; `estia tls rotate` makes a new pair (paired devices must pair again)",
            key.display(),
            cert.display()
        )
    })?;
    Ok(Some(identity))
}

/// The certificate in `data_dir`, made now if there is none. Both files
/// absent means none was ever made; one absent (an operator removed it) means
/// a new pair, with a warning, since no device can hold a pin for a
/// certificate that is not there whole. Two files that do not belong
/// together, or that cannot be read, are an error, never silently replaced.
/// Holds the lock throughout, so two processes starting at once make one
/// pair between them.
pub fn load_or_create(data_dir: &Path) -> anyhow::Result<TlsIdentity> {
    let _lock = lock(data_dir)?;
    if let Some(id) = load_locked(data_dir)? {
        return Ok(id);
    }
    if let Some(left) = [cert_path(data_dir), key_path(data_dir)].into_iter().find(|p| p.exists()) {
        tracing::warn!(
            file = %left.display(),
            "half of the TLS key pair is missing; making a new pair: every paired device must pair again"
        );
    }
    create_locked(data_dir)
}

/// A new certificate, replacing the old one. Every device that pinned the
/// old one must pair again; tokens stay valid until revoked. A running engine
/// keeps serving the old one until it restarts.
pub fn rotate(data_dir: &Path) -> anyhow::Result<TlsIdentity> {
    let _lock = lock(data_dir)?;
    finish_interrupted(data_dir)?;
    create_locked(data_dir)
}

/// Write a new pair (the caller holds the lock): see [`finish_interrupted`]
/// for what a crash at any point leaves.
fn create_locked(data_dir: &Path) -> anyhow::Result<TlsIdentity> {
    let (cert_pem, key_pem, cert_der, key_der) = generate()?;
    let (key, cert) = (key_path(data_dir), cert_path(data_dir));
    crate::tokens::write_private(&pending(&key), key_pem.as_bytes())?;
    crate::tokens::write_private(&pending(&cert), cert_pem.as_bytes())?;
    // Both are whole on disk now. The key goes first: the one state between
    // the renames (new key, new certificate still pending) is finished, not
    // undone, by `finish_interrupted`.
    std::fs::rename(pending(&key), &key)?;
    std::fs::rename(pending(&cert), &cert)?;
    let created_unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    Ok(TlsIdentity { fingerprint: Fingerprint::of_certificate(&cert_der), cert_der, key_der, created_unix, fresh: true })
}

/// How long a certificate is valid, from a day before it is made. Apple's TLS
/// stack (Safari, URLSession) refuses a server certificate valid for more than
/// 825 days, even one the user trusts as an anchor; 820 leaves a margin.
/// Pinned clients ignore dates altogether.
pub const VALIDITY_DAYS: u64 = 820;

/// A new self-signed certificate: (cert PEM, key PEM, cert DER, key DER).
///
/// ECDSA P-256, which every client supports and which makes small, fast
/// handshakes. Extended key usage serverAuth (Apple platforms require it of a
/// server certificate); no key usage and no basic constraints, so OpenSSL
/// accepts the certificate as its own trust anchor when a client loads it as
/// one; a random 16-byte serial (Firefox refuses a second certificate with the
/// same issuer and serial, as after a rotation). The names are this machine's
/// at the time and are only for show: a pinned client checks none of them.
fn generate() -> anyhow::Result<(String, String, Vec<u8>, Vec<u8>)> {
    use rcgen::{CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, KeyPair, SerialNumber};
    let key = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)?;
    let mut params = CertificateParams::default();
    let host = crate::hostname();
    let mut dn = DistinguishedName::new();
    dn.push(DnType::CommonName, format!("Estia engine on {host}"));
    params.distinguished_name = dn;
    params.subject_alt_names = subject_alt_names(&host, &crate::machine_hostnames(), &estia_engine::discovery::lan_ipv4_addresses());
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // A day back, for clocks that run a little behind.
    let not_before = now.saturating_sub(86_400);
    params.not_before = rcgen::date_time_ymd(1970, 1, 1) + std::time::Duration::from_secs(not_before);
    params.not_after = rcgen::date_time_ymd(1970, 1, 1) + std::time::Duration::from_secs(not_before + VALIDITY_DAYS * 86_400);
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let mut serial = [0u8; 16];
    getrandom::getrandom(&mut serial).map_err(|e| anyhow::anyhow!("randomness: {e}"))?;
    // Positive, and never shorter than 16 bytes once encoded.
    serial[0] = (serial[0] & 0x7f) | 0x40;
    params.serial_number = Some(SerialNumber::from_slice(&serial));
    let cert = params.self_signed(&key)?;
    Ok((cert.pem(), key.serialize_pem(), cert.der().to_vec(), key.serialize_der()))
}

/// `localhost`, the loopback addresses, this machine's names (short, `.local`,
/// fully qualified) and its LAN IPv4 addresses. A name rcgen cannot carry
/// (not ASCII) is left out.
fn subject_alt_names(host: &str, names: &[String], ips: &[std::net::Ipv4Addr]) -> Vec<rcgen::SanType> {
    use rcgen::SanType;
    let mut dns: Vec<String> = vec!["localhost".into()];
    for n in std::iter::once(host.to_string()).chain(names.iter().cloned()) {
        let n = n.trim().trim_end_matches('.').to_ascii_lowercase();
        if n.is_empty() {
            continue;
        }
        dns.push(n.clone());
        if !n.contains('.') {
            dns.push(format!("{n}.local"));
        }
    }
    let mut out: Vec<SanType> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for n in dns {
        if seen.insert(n.clone()) {
            if let Ok(name) = n.as_str().try_into() {
                out.push(SanType::DnsName(name));
            }
        }
    }
    let loopback = [IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)];
    for ip in loopback.into_iter().chain(ips.iter().map(|ip| IpAddr::V4(*ip))) {
        if seen.insert(ip.to_string()) {
            out.push(SanType::IpAddress(ip));
        }
    }
    out
}

fn server_config(identity: &TlsIdentity) -> Result<rustls::ServerConfig, rustls::Error> {
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(identity.key_der.clone()));
    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
        .with_safe_default_protocol_versions()?
        .with_no_client_auth()
        // Checks that the key is the certificate's.
        .with_single_cert(vec![CertificateDer::from(identity.cert_der.clone())], key)?;
    // The engine speaks HTTP/1.1 only.
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(config)
}

/// How [`crate::serve_router_tls`] serves TLS. Keeps rustls out of the public
/// API: build it from a [`TlsIdentity`].
#[derive(Clone)]
pub struct TlsServe {
    pub(crate) acceptor: tokio_rustls::TlsAcceptor,
    pub(crate) plain_from_loopback: bool,
}

impl std::fmt::Debug for TlsServe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TlsServe").field("plain_from_loopback", &self.plain_from_loopback).finish()
    }
}

impl TlsServe {
    pub fn new(identity: &TlsIdentity) -> anyhow::Result<Self> {
        let config = server_config(identity)?;
        Ok(TlsServe { acceptor: tokio_rustls::TlsAcceptor::from(Arc::new(config)), plain_from_loopback: true })
    }

    /// Serve plain HTTP to loopback peers (the default). Tests turn it off to
    /// play a LAN peer from 127.0.0.1, as with [`crate::ConnLimits::exempt_loopback`].
    pub fn plain_from_loopback(mut self, allow: bool) -> Self {
        self.plain_from_loopback = allow;
        self
    }

    pub(crate) fn plain_allowed(&self, peer: IpAddr) -> bool {
        self.plain_from_loopback && plain_http_allowed(peer)
    }
}

/// How a request's connection arrived, as a request extension. A server
/// without TLS ([`crate::serve_router`]) adds none, which counts as `Plain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// Plain HTTP from this machine.
    Plain,
    /// HTTPS.
    Tls,
    /// Plain HTTP from another device: answered with `https_required`.
    PlainRefused,
}

/// May this peer use plain HTTP? Loopback only, IPv4-mapped loopback included
/// (`::ffff:127.0.0.1`); the unspecified address is not loopback.
pub fn plain_http_allowed(peer: IpAddr) -> bool {
    peer.to_canonical().is_loopback()
}

/// Does a bind to `addr` serve HTTPS? Every bind but loopback does.
pub fn wants_tls(addr: SocketAddr) -> bool {
    !addr.ip().to_canonical().is_loopback()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "estia-tls-{tag}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn plain_http_is_for_loopback_peers_only() {
        for ok in ["127.0.0.1", "127.0.0.2", "::1", "::ffff:127.0.0.1"] {
            assert!(plain_http_allowed(ok.parse().unwrap()), "{ok}");
        }
        for no in ["192.168.1.5", "::ffff:192.168.1.5", "fe80::1", "0.0.0.0", "::", "10.0.0.1"] {
            assert!(!plain_http_allowed(no.parse().unwrap()), "{no}");
        }
    }

    #[test]
    fn tls_is_for_every_bind_but_loopback() {
        for lan in ["0.0.0.0:1", "[::]:1", "192.168.1.20:27200", "[fe80::1]:1"] {
            assert!(wants_tls(lan.parse().unwrap()), "{lan}");
        }
        for local in ["127.0.0.1:27200", "[::1]:27200", "127.0.0.2:1"] {
            assert!(!wants_tls(local.parse().unwrap()), "{local}");
        }
    }

    #[test]
    fn the_certificate_is_made_once_and_kept_private() {
        let dir = scratch("create");
        assert!(load(&dir).unwrap().is_none());
        let first = load_or_create(&dir).unwrap();
        assert!(first.is_new());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&tls_dir(&dir)), 0o700);
            assert_eq!(mode(&key_path(&dir)), 0o600);
            assert_eq!(mode(&cert_path(&dir)), 0o600);
        }
        let again = load_or_create(&dir).unwrap();
        assert!(!again.is_new());
        assert_eq!(again.fingerprint(), first.fingerprint(), "kept, not made again");
        assert_eq!(Fingerprint::of_certificate(again.certificate_der()), again.fingerprint());
        assert!(again.curl_pin().starts_with("sha256//") && again.curl_pin().len() == "sha256//".len() + 44, "{}", again.curl_pin());
        assert_eq!(again.curl_pin(), first.curl_pin());

        // Rotating makes another.
        let rotated = rotate(&dir).unwrap();
        assert_ne!(rotated.fingerprint(), first.fingerprint());
        assert_eq!(load(&dir).unwrap().unwrap().fingerprint(), rotated.fingerprint());

        // A missing certificate (a crash between the writes): a new pair.
        std::fs::remove_file(cert_path(&dir)).unwrap();
        let remade = load_or_create(&dir).unwrap();
        assert!(remade.is_new());
        assert_ne!(remade.fingerprint(), rotated.fingerprint());

        // A key that is not the certificate's: refused, never replaced.
        let other = scratch("other");
        load_or_create(&other).unwrap();
        std::fs::copy(key_path(&other), key_path(&dir)).unwrap();
        let cert_before = std::fs::read(cert_path(&dir)).unwrap();
        let e = load_or_create(&dir).unwrap_err().to_string();
        assert!(e.contains("do not belong together") && e.contains("estia tls rotate"), "{e}");
        assert_eq!(std::fs::read(cert_path(&dir)).unwrap(), cert_before, "the certificate was left alone");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&other);
    }

    /// A change cut short at each point between the writes and the renames
    /// is finished or undone before the pair is read: never a key and a
    /// certificate from two pairs.
    #[test]
    fn a_rotation_cut_short_is_finished_or_undone() {
        let dir = scratch("cut");
        let old = load_or_create(&dir).unwrap().fingerprint();
        let other = scratch("cut-other");
        let new = load_or_create(&other).unwrap().fingerprint();
        let copy = |from: PathBuf, to: PathBuf| std::fs::copy(from, to).unwrap();

        // Both new files written, nothing renamed: undone.
        copy(key_path(&other), pending(&key_path(&dir)));
        copy(cert_path(&other), pending(&cert_path(&dir)));
        assert_eq!(load(&dir).unwrap().unwrap().fingerprint(), old);
        assert!(!pending(&key_path(&dir)).exists() && !pending(&cert_path(&dir)).exists());

        // Only the new key written: undone.
        copy(key_path(&other), pending(&key_path(&dir)));
        assert_eq!(load_or_create(&dir).unwrap().fingerprint(), old);
        assert!(!pending(&key_path(&dir)).exists());

        // The new key renamed, the new certificate not yet: finished.
        copy(key_path(&other), key_path(&dir));
        copy(cert_path(&other), pending(&cert_path(&dir)));
        assert_eq!(load(&dir).unwrap().unwrap().fingerprint(), new);
        assert!(!pending(&cert_path(&dir)).exists());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&other);
    }

    /// Processes that start, read and rotate at once (threads here, each with
    /// its own lock file handle, as processes would have) make one pair
    /// between them and never read a mixed one.
    #[test]
    fn concurrent_creation_and_rotation_never_mix_pairs() {
        let dir = scratch("race");
        let start = std::sync::Arc::new(std::sync::Barrier::new(8));
        let made: Vec<Fingerprint> = (0..8)
            .map(|_| {
                let (dir, start) = (dir.clone(), std::sync::Arc::clone(&start));
                std::thread::spawn(move || {
                    start.wait();
                    load_or_create(&dir).unwrap().fingerprint()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        assert!(made.iter().all(|f| *f == made[0]), "one pair between them: {made:?}");
        assert_eq!(load(&dir).unwrap().unwrap().fingerprint(), made[0]);

        let start = std::sync::Arc::new(std::sync::Barrier::new(6));
        let handles: Vec<_> = (0..6)
            .map(|i| {
                let (dir, start) = (dir.clone(), std::sync::Arc::clone(&start));
                std::thread::spawn(move || {
                    start.wait();
                    for _ in 0..6 {
                        if i % 2 == 0 {
                            rotate(&dir).unwrap();
                        } else {
                            // `load` checks the key belongs to the certificate.
                            assert!(load(&dir).unwrap().is_some());
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(load(&dir).unwrap().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pair kept as symbolic links that point nowhere (an unmounted volume)
    /// is an error, and the links are left alone: never a new pair that
    /// would break every pin.
    #[test]
    #[cfg(unix)]
    fn links_that_point_nowhere_are_an_error_not_an_absence() {
        let dir = scratch("links");
        std::fs::create_dir_all(tls_dir(&dir)).unwrap();
        let gone = dir.join("unmounted/volume");
        std::os::unix::fs::symlink(gone.join("cert.pem"), cert_path(&dir)).unwrap();
        std::os::unix::fs::symlink(gone.join("key.pem"), key_path(&dir)).unwrap();
        assert!(load(&dir).is_err());
        let e = load_or_create(&dir).unwrap_err().to_string();
        assert!(e.contains("cert.pem") || e.contains("key.pem"), "names the file: {e}");
        for p in [cert_path(&dir), key_path(&dir)] {
            assert!(std::fs::symlink_metadata(&p).unwrap().file_type().is_symlink(), "{} was replaced", p.display());
        }
        // One half a dangling link, the other absent: still an error.
        std::fs::remove_file(cert_path(&dir)).unwrap();
        assert!(load_or_create(&dir).is_err());
        assert!(std::fs::symlink_metadata(key_path(&dir)).unwrap().file_type().is_symlink());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Days since 1970-01-01 of a UTCTime `YYMMDDHHMMSSZ` (years 1950 to 2049).
    fn utc_days(t: &[u8]) -> i64 {
        let n = |i: usize| i64::from(t[i] - b'0') * 10 + i64::from(t[i + 1] - b'0');
        let (y, m, d) = (if n(0) < 50 { 2000 + n(0) } else { 1900 + n(0) }, n(2), n(4));
        // Howard Hinnant's days-from-civil.
        let y = if m <= 2 { y - 1 } else { y };
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
        era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
    }

    /// What the certificate carries, read back with a small DER walk: the
    /// names, serverAuth, no basic constraints, a 16-byte positive serial and
    /// a validity Apple accepts (825 days at most).
    #[test]
    fn the_certificate_carries_what_clients_need() {
        let (_, _, der, _) = generate().unwrap();
        let hay = |needle: &[u8]| der.windows(needle.len()).any(|w| w == needle);
        // id-kp-serverAuth 1.3.6.1.5.5.7.3.1
        assert!(hay(&[0x06, 0x08, 0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x01]), "serverAuth");
        // basicConstraints 2.5.29.19 and keyUsage 2.5.29.15 are absent.
        assert!(!hay(&[0x06, 0x03, 0x55, 0x1d, 0x13]), "no basic constraints");
        assert!(!hay(&[0x06, 0x03, 0x55, 0x1d, 0x0f]), "no key usage");
        assert!(hay(b"localhost"));
        assert!(hay(&[0x87, 0x04, 127, 0, 0, 1]), "127.0.0.1 as an IP SAN");
        // Validity ::= SEQUENCE { notBefore UTCTime, notAfter UTCTime }
        let at = der.windows(4).position(|w| w == [0x30, 0x1e, 0x17, 0x0d]).expect("a validity of two UTCTimes");
        let (from, to) = (utc_days(&der[at + 4..at + 17]), utc_days(&der[at + 19..at + 32]));
        assert_eq!(der[at + 17..at + 19], [0x17, 0x0d]);
        assert_eq!(to - from, VALIDITY_DAYS as i64, "valid for {} days", to - from);
        assert!(to - from <= 825, "Apple refuses a server certificate valid for more than 825 days");
        let today = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() / 86_400) as i64;
        assert!((today - 2..=today).contains(&from), "valid from about a day ago: {from} vs today {today}");
        // tbsCertificate: SEQUENCE { [0] version, INTEGER serial, … }
        let serial_at = der.windows(5).position(|w| w == [0xa0, 0x03, 0x02, 0x01, 0x02]).unwrap() + 5;
        assert_eq!(der[serial_at], 0x02, "INTEGER");
        assert_eq!(der[serial_at + 1], 16, "16-byte serial");
        assert!(der[serial_at + 2] & 0x80 == 0, "positive");
        let (_, _, der2, _) = generate().unwrap();
        assert_ne!(der[serial_at + 2..serial_at + 18], der2[serial_at + 2..serial_at + 18], "random serials");
    }

    #[test]
    fn names_skip_what_cannot_be_carried() {
        let sans = subject_alt_names(
            "Studio",
            &["Studio".into(), "studio.example.lan".into(), "Γιώργος".into()],
            &["192.168.1.20".parse().unwrap()],
        );
        let shown: Vec<String> = sans
            .iter()
            .map(|s| match s {
                rcgen::SanType::DnsName(n) => n.as_str().to_string(),
                rcgen::SanType::IpAddress(ip) => ip.to_string(),
                _ => String::new(),
            })
            .collect();
        assert_eq!(shown, ["localhost", "studio", "studio.local", "studio.example.lan", "127.0.0.1", "::1", "192.168.1.20"]);
    }
}
