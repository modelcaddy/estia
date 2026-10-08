//! The engine's TLS certificate as clients see it: its fingerprint, the code
//! people compare when a device pairs, and the check that pins it.
//!
//! On a LAN bind the engine serves HTTPS with a self-signed certificate (the
//! server's `tls` module makes it). Nothing vouches for that certificate, so
//! a device pins it: it trusts exactly the certificate whose SHA-256
//! fingerprint it was given, whatever its names, dates or issuer, and still
//! checks that the server signs the handshake with that certificate's key.
//!
//! How a device learns the fingerprint safely:
//!
//! - **Pairing** proves, in both directions, which certificate each end used,
//!   the way Bluetooth's numeric comparison does. The device commits to a
//!   random nonce ([`pairing_commitment`]) before it sees the engine's nonce;
//!   both sides then compute one 6-digit [`pairing_code`] from the
//!   fingerprint they each used and the two nonces, and each shows its own.
//!   A person checks that the two screens show the same code and says so on
//!   both: the operator on the engine's side (`estia pair approve` asks), so
//!   the engine knows the device saw its certificate; the user on the
//!   device's side, so the device knows the engine saw the same exchange and
//!   pins the certificate only then. Neither end ever sends its code to the
//!   other: something in between would relay it. Something in between that
//!   presented its own certificate makes the two codes differ, except by a
//!   one-in-a-million chance it cannot improve by preparing in advance.
//! - **Pasting** a fingerprint the operator read with `estia tls fingerprint`
//!   ([`UnverifiedFingerprint::matches`]).
//! - **Comparing** the 64-bit short form ([`Fingerprint::short`]) by eye and
//!   confirming it ([`UnverifiedFingerprint::confirmed`]): trust on first use,
//!   for paths with no pairing exchange.
//!
//! A short code taken from the fingerprint alone would not do: the engine's
//! certificate is public and lives for years, so an interceptor could grind,
//! offline, a certificate of its own whose fingerprint shares 20 or 30 bits
//! with it. The per-pairing nonces are what make the 6 digits enough.
//!
//! What a body says about a certificate (`tls.fingerprint` in
//! `/engine/health`, the Bonjour `fp` key) is for display and cross-checks
//! only: an interceptor rewrites bodies, but cannot present the engine's
//! certificate.
//!
//! The pin check itself (the rustls verifier a pinned
//! [`crate::RemoteEngine`] uses) is behind the `remote-tls` feature, which
//! brings rustls with the ring provider. The types and codes here are not.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(feature = "remote-tls")]
use std::sync::{Arc, Mutex};

/// SHA-256 of an engine's end-entity certificate (its DER bytes): what a
/// device pins. Written as 64 lowercase hex digits on the wire and on disk.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fingerprint([u8; 32]);

/// A fingerprint that does not parse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("not a SHA-256 certificate fingerprint ({0}); expected 64 hex digits, as `estia tls fingerprint` prints them")]
pub struct FingerprintError(String);

impl Fingerprint {
    /// The fingerprint of a certificate, from its DER bytes.
    pub fn of_certificate(der: &[u8]) -> Self {
        Fingerprint(Sha256::digest(der).into())
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Fingerprint(bytes)
    }

    /// 64 hex digits in either case. Colons, spaces and dashes are ignored, and
    /// a leading `sha256:`, `sha256//` or `SHA256 Fingerprint=` is dropped, so a
    /// fingerprint pasted from `openssl x509 -fingerprint -sha256` or a
    /// browser's certificate viewer parses as it is.
    pub fn parse(s: &str) -> Result<Self, FingerprintError> {
        let mut t = s.trim();
        for prefix in ["sha256 fingerprint=", "sha256:", "sha-256:", "sha256//", "sha256 "] {
            if t.get(..prefix.len()).is_some_and(|head| head.eq_ignore_ascii_case(prefix)) {
                t = t[prefix.len()..].trim();
                break;
            }
        }
        let digits: Vec<u8> = t.bytes().filter(|b| !matches!(b, b':' | b' ' | b'-' | b'\t')).collect();
        let shown = || s.chars().take(24).collect::<String>().escape_debug().to_string();
        if digits.len() != 64 {
            return Err(FingerprintError(format!("{} hex digits in `{}`", digits.len(), shown())));
        }
        let mut out = [0u8; 32];
        for (i, pair) in digits.chunks(2).enumerate() {
            match (hex_value(pair[0]), hex_value(pair[1])) {
                (Some(hi), Some(lo)) => out[i] = hi << 4 | lo,
                _ => return Err(FingerprintError(format!("`{}` is not hexadecimal", shown()))),
            }
        }
        Ok(Fingerprint(out))
    }

    /// The wire and storage form: 64 lowercase hex digits, no separators.
    pub fn to_hex(&self) -> String {
        to_hex(&self.0)
    }

    /// The first 64 bits in four groups (`3f2a 9c41 7b0e d512`): for people to
    /// compare by eye where there is no pairing exchange. 64 bits is too many
    /// to grind a look-alike certificate for.
    pub fn short(&self) -> String {
        group(&to_hex(&self.0[..8]))
    }

    /// All of it in groups of four hex digits, for comparing in full.
    pub fn grouped(&self) -> String {
        group(&self.to_hex())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl std::fmt::Debug for Fingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fingerprint({})", self.to_hex())
    }
}

impl std::str::FromStr for Fingerprint {
    type Err = FingerprintError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Fingerprint::parse(s)
    }
}

impl Serialize for Fingerprint {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Fingerprint {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Fingerprint::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// A fingerprint seen on the network that nothing has vouched for yet: what
/// [`crate::remote::probe_certificate`] found, or what a Bonjour record
/// claims. It cannot pin anything until a person confirmed it
/// ([`UnverifiedFingerprint::confirmed`]) or it matched a fingerprint pasted
/// from the engine's machine ([`UnverifiedFingerprint::matches`]).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UnverifiedFingerprint(Fingerprint);

impl UnverifiedFingerprint {
    pub(crate) fn new(fp: Fingerprint) -> Self {
        UnverifiedFingerprint(fp)
    }

    /// See [`Fingerprint::short`]: what to show for comparing with
    /// `estia tls fingerprint` on the engine's machine.
    pub fn short(&self) -> String {
        self.0.short()
    }

    pub fn grouped(&self) -> String {
        self.0.grouped()
    }

    pub fn to_hex(&self) -> String {
        self.0.to_hex()
    }

    /// A person compared it with `estia tls fingerprint` on the engine's
    /// machine and it matched: from now on it is the pin.
    pub fn confirmed(self) -> Fingerprint {
        self.0
    }

    /// Is it the fingerprint the user pasted from the engine's machine? When it
    /// is, pin `expected`.
    pub fn matches(&self, expected: &Fingerprint) -> bool {
        self.0 == *expected
    }
}

impl std::fmt::Debug for UnverifiedFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UnverifiedFingerprint({})", self.0.to_hex())
    }
}

impl Serialize for UnverifiedFingerprint {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

/// The 6-digit code both ends of a pairing compute ([`pairing_code`]), each
/// from the certificate it used: the device shows its own, the engine its own
/// (`estia pair list`, `estia pair approve`), and a person checks they are the
/// same. A script approving on the engine's side gives the device's as
/// `estia pair approve <id> --code 482913`. Shown as `482 913`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PairingCode(u32);

impl PairingCode {
    /// Six digits, no space: what `--code` and the HTTP approve route take.
    pub fn digits(&self) -> String {
        format!("{:06}", self.0)
    }

    /// Six digits; spaces and dashes are ignored (`482 913`, `482-913`).
    pub fn parse(s: &str) -> Option<Self> {
        let digits: String = s.chars().filter(|c| !matches!(c, ' ' | '-')).collect();
        if digits.len() != 6 || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        digits.parse().ok().map(PairingCode)
    }
}

impl std::fmt::Display for PairingCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.digits();
        write!(f, "{} {}", &d[..3], &d[3..])
    }
}

impl std::fmt::Debug for PairingCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PairingCode({})", self.digits())
    }
}

impl std::str::FromStr for PairingCode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PairingCode::parse(s).ok_or_else(|| "a pairing code is six digits, such as 482913".to_string())
    }
}

impl Serialize for PairingCode {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.digits())
    }
}

impl<'de> Deserialize<'de> for PairingCode {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        PairingCode::parse(&s).ok_or_else(|| serde::de::Error::custom("a pairing code is six digits"))
    }
}

const COMMITMENT_LABEL: &[u8] = b"estia pairing commitment v1";
const CODE_LABEL: &[u8] = b"estia pairing code v1";

/// What a device sends with its pairing request before it sees the engine's
/// nonce: SHA-256(`"estia pairing commitment v1"` ‖ 0x00 ‖ nonce), in hex.
/// It binds the device to its nonce without revealing it.
pub fn pairing_commitment(device_nonce: &[u8; 32]) -> String {
    let mut h = Sha256::new();
    h.update(COMMITMENT_LABEL);
    h.update([0]);
    h.update(device_nonce);
    to_hex(&h.finalize())
}

/// The pairing code: the first 8 bytes of SHA-256(`"estia pairing code v1"`
/// ‖ 0x00 ‖ fingerprint ‖ device nonce ‖ engine nonce) as a big-endian
/// integer, modulo 1 000 000. The device computes it with the fingerprint of
/// the certificate it saw, the engine with its own; they agree only when the
/// two are the same certificate. Each end shows its own and never sends it to
/// the other: a person compares the two screens.
pub fn pairing_code(fp: &Fingerprint, device_nonce: &[u8; 32], engine_nonce: &[u8; 32]) -> PairingCode {
    let mut h = Sha256::new();
    h.update(CODE_LABEL);
    h.update([0]);
    h.update(fp.as_bytes());
    h.update(device_nonce);
    h.update(engine_nonce);
    let d = h.finalize();
    let mut first = [0u8; 8];
    first.copy_from_slice(&d[..8]);
    PairingCode((u64::from_be_bytes(first) % 1_000_000) as u32)
}

/// 32 bytes from the operating system's random source, for a pairing nonce.
pub fn random_nonce() -> std::io::Result<[u8; 32]> {
    let mut n = [0u8; 32];
    getrandom::getrandom(&mut n).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(n)
}

/// A nonce or commitment as 64 hex digits.
pub fn nonce_hex(nonce: &[u8; 32]) -> String {
    to_hex(nonce)
}

/// 64 hex digits back to 32 bytes; `None` for anything else.
pub fn parse_nonce(s: &str) -> Option<[u8; 32]> {
    let s = s.trim().as_bytes();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, pair) in s.chunks(2).enumerate() {
        out[i] = hex_value(pair[0])? << 4 | hex_value(pair[1])?;
    }
    Some(out)
}

/// The letters a pairing id is made of: lowercase, without the ones that pass
/// for a digit (`b g i l o q s z`). A device shows its request's id next to
/// its pairing code, and whatever answers the request chooses the id, so an
/// id must not be readable as a code: an interceptor would otherwise answer
/// with an id such as `771204`, the code the engine shows for a request of
/// its own, and put that code on the device's screen too.
pub const PAIRING_ID_ALPHABET: &str = "acdefhjkmnprtuvwxy";

/// Letters in a pairing id: 16 of 18, about 66 bits.
pub const PAIRING_ID_LEN: usize = 16;

/// A new pairing id: [`PAIRING_ID_LEN`] letters of [`PAIRING_ID_ALPHABET`],
/// drawn without bias from the operating system's random source.
pub fn random_pairing_id() -> std::io::Result<String> {
    let alphabet = PAIRING_ID_ALPHABET.as_bytes();
    // The largest multiple of the alphabet's size that fits in a byte: bytes
    // at or above it are drawn again, so every letter is equally likely.
    let limit = 256 - 256 % alphabet.len();
    let mut id = String::with_capacity(PAIRING_ID_LEN);
    while id.len() < PAIRING_ID_LEN {
        let mut bytes = [0u8; 32];
        getrandom::getrandom(&mut bytes).map_err(|e| std::io::Error::other(e.to_string()))?;
        for b in bytes.into_iter().map(usize::from).filter(|b| *b < limit) {
            if id.len() == PAIRING_ID_LEN {
                break;
            }
            id.push(char::from(alphabet[b % alphabet.len()]));
        }
    }
    Ok(id)
}

/// Is `s` an id an engine makes ([`random_pairing_id`])? A device pairing
/// over HTTPS refuses any other before it goes on, so no id it shows can pass
/// for a code.
pub fn is_pairing_id(s: &str) -> bool {
    s.len() == PAIRING_ID_LEN && s.bytes().all(|b| PAIRING_ID_ALPHABET.as_bytes().contains(&b))
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn group(hex: &str) -> String {
    hex.as_bytes().chunks(4).map(|c| std::str::from_utf8(c).unwrap_or("")).collect::<Vec<_>>().join(" ")
}

#[cfg(feature = "remote-tls")]
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The certificate check of a pinned client: accept exactly the certificate
/// whose fingerprint is the pin, and verify the handshake signature with its
/// key. Chain, dates and names are not checked, on purpose: the certificate is
/// self-signed and its names are whatever the engine's machine was called
/// when it was made. The signature is what proves the server holds the
/// pinned certificate's private key, so those two methods are real checks.
///
/// Learning mode (no pin yet) is the probe of [`crate::remote::probe_certificate`]
/// and [`crate::remote::begin_pairing`]: the first certificate whose handshake
/// signature verifies becomes the pin, and every later connection of the same
/// client must present it, so a certificate swapped mid-pairing is refused.
#[cfg(feature = "remote-tls")]
#[derive(Debug)]
pub(crate) struct PinVerifier {
    pin: Mutex<Option<Fingerprint>>,
    learn: bool,
    /// The last certificate refused, so the client can say which one it was
    /// (`SessionError::CertificateMismatch`).
    refused: Mutex<Option<Fingerprint>>,
    algs: rustls::crypto::WebPkiSupportedAlgorithms,
}

#[cfg(feature = "remote-tls")]
impl PinVerifier {
    pub(crate) fn pinned(pin: Fingerprint) -> Arc<Self> {
        Self::make(Some(pin), false)
    }

    pub(crate) fn learning() -> Arc<Self> {
        Self::make(None, true)
    }

    fn make(pin: Option<Fingerprint>, learn: bool) -> Arc<Self> {
        let algs = provider().signature_verification_algorithms;
        Arc::new(PinVerifier { pin: Mutex::new(pin), learn, refused: Mutex::new(None), algs })
    }

    /// The pin: the one given, or in learning mode the first certificate seen.
    pub(crate) fn pin(&self) -> Option<Fingerprint> {
        *lock(&self.pin)
    }

    /// The certificate refused since the last call, if any.
    pub(crate) fn take_refused(&self) -> Option<Fingerprint> {
        lock(&self.refused).take()
    }

    fn refuse(&self, got: Fingerprint) -> rustls::Error {
        *lock(&self.refused) = Some(got);
        rustls::Error::InvalidCertificate(rustls::CertificateError::ApplicationVerificationFailure)
    }

    /// After a handshake signature verified: in learning mode the first such
    /// certificate becomes the pin; any other certificate is refused (two
    /// handshakes racing in learning mode cannot pin two certificates).
    fn signed_by(&self, cert: &rustls::pki_types::CertificateDer<'_>) -> Result<(), rustls::Error> {
        let got = Fingerprint::of_certificate(cert);
        let mut pin = lock(&self.pin);
        match *pin {
            Some(p) if p == got => Ok(()),
            None if self.learn => {
                *pin = Some(got);
                Ok(())
            }
            _ => {
                drop(pin);
                Err(self.refuse(got))
            }
        }
    }
}

#[cfg(feature = "remote-tls")]
impl rustls::client::danger::ServerCertVerifier for PinVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let got = Fingerprint::of_certificate(end_entity);
        match self.pin() {
            Some(p) if p == got => Ok(rustls::client::danger::ServerCertVerified::assertion()),
            // Learning: accepted here, pinned once its signature verifies.
            None if self.learn => Ok(rustls::client::danger::ServerCertVerified::assertion()),
            _ => Err(self.refuse(got)),
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        let valid = rustls::crypto::verify_tls12_signature(message, cert, dss, &self.algs)?;
        self.signed_by(cert)?;
        Ok(valid)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        let valid = rustls::crypto::verify_tls13_signature(message, cert, dss, &self.algs)?;
        self.signed_by(cert)?;
        Ok(valid)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.algs.supported_schemes()
    }
}

/// Without the `remote-tls` feature there is no pinned client: this type has
/// no values, so a `RemoteEngine`'s verifier is always `None`.
#[cfg(not(feature = "remote-tls"))]
#[derive(Debug)]
pub(crate) enum PinVerifier {}

#[cfg(not(feature = "remote-tls"))]
impl PinVerifier {
    pub(crate) fn pin(&self) -> Option<Fingerprint> {
        match *self {}
    }

    pub(crate) fn take_refused(&self) -> Option<Fingerprint> {
        match *self {}
    }
}

/// The ring provider, always passed explicitly: relying on a process-wide
/// default would panic in a host that links rustls with another provider too.
#[cfg(feature = "remote-tls")]
pub(crate) fn provider() -> rustls::crypto::CryptoProvider {
    rustls::crypto::ring::default_provider()
}

/// The rustls configuration a pinned or probing client hands reqwest: safe
/// default protocol versions, the [`PinVerifier`], HTTP/1.1 only (all the
/// engine speaks).
#[cfg(feature = "remote-tls")]
pub(crate) fn client_config(verifier: Arc<PinVerifier>) -> Result<rustls::ClientConfig, rustls::Error> {
    let mut config = rustls::ClientConfig::builder_with_provider(Arc::new(provider()))
        .with_safe_default_protocol_versions()?
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_no_client_auth();
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(start: u8) -> [u8; 32] {
        std::array::from_fn(|i| start + i as u8)
    }

    /// The known-answer vector in docs/api.md, computed independently with
    /// Python's hashlib. examples/python/estia_tls.py checks the same one.
    #[test]
    fn pairing_vector() {
        let fp = Fingerprint::from_bytes(seq(0x00));
        let (device, engine) = (seq(0x20), seq(0x40));
        assert_eq!(pairing_commitment(&device), "71f9884a2523f984d667e49fff85b931cdaff466dc51fe953ca46436319974dd");
        let code = pairing_code(&fp, &device, &engine);
        assert_eq!(code.digits(), "865695");
        assert_eq!(code.to_string(), "865 695");
        assert_eq!(fp.short(), "0001 0203 0405 0607");
        assert_eq!(fp.to_hex(), "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
        // Every input matters.
        assert_ne!(pairing_code(&Fingerprint::from_bytes(seq(0x01)), &device, &engine), code);
        assert_ne!(pairing_code(&fp, &seq(0x21), &engine), code);
        assert_ne!(pairing_code(&fp, &device, &seq(0x41)), code);
    }

    /// Pairing ids cannot pass for a code: letters only, none that reads as
    /// a digit; and a device refuses anything else.
    #[test]
    fn pairing_ids_hold_no_digit_and_nothing_like_one() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let id = random_pairing_id().unwrap();
            assert!(is_pairing_id(&id), "{id}");
            assert!(!id.chars().any(|c| c.is_numeric() || "bgilqosz".contains(c)), "{id}");
            seen.insert(id);
        }
        assert_eq!(seen.len(), 200, "ids do not repeat");
        assert!(PAIRING_ID_ALPHABET.chars().all(|c| c.is_ascii_lowercase()));
        for not_an_id in ["771204", "6de47a30928f4f15", "", "acdefhjkmnprtuv", "acdefhjkmnprtuvwx", "ACDEFHJKMNPRTUVW", "acdefhjkmnprtuv1"]
        {
            assert!(!is_pairing_id(not_an_id), "{not_an_id}");
        }
    }

    #[test]
    fn fingerprints_parse_in_the_forms_people_paste() {
        let fp = Fingerprint::of_certificate(b"not really a certificate");
        let hex = fp.to_hex();
        assert_eq!(hex.len(), 64);
        assert_eq!(Fingerprint::parse(&hex).unwrap(), fp);
        assert_eq!(hex.parse::<Fingerprint>().unwrap(), fp);
        let openssl: String =
            hex.to_uppercase().as_bytes().chunks(2).map(|c| std::str::from_utf8(c).unwrap()).collect::<Vec<_>>().join(":");
        assert_eq!(Fingerprint::parse(&format!("SHA256 Fingerprint={openssl}")).unwrap(), fp, "openssl x509 -fingerprint -sha256");
        assert_eq!(Fingerprint::parse(&format!("sha256:{hex}")).unwrap(), fp);
        assert_eq!(Fingerprint::parse(&fp.grouped()).unwrap(), fp, "the grouped form round trips");
        assert_eq!(Fingerprint::parse(&openssl.replace(':', " ")).unwrap(), fp, "a browser's spaced form");
        for bad in ["", "abcd", &hex[..62], &format!("{hex}00"), &hex.replace('a', "g")] {
            assert!(Fingerprint::parse(bad).is_err(), "{bad:?}");
        }
        let json = serde_json::to_string(&fp).unwrap();
        assert_eq!(json, format!("\"{hex}\""));
        assert_eq!(serde_json::from_str::<Fingerprint>(&json).unwrap(), fp);
        assert_eq!(fp.short().len(), 19);
        assert_eq!(fp.grouped().split(' ').count(), 16);
    }

    #[test]
    fn unverified_fingerprints_pin_only_once_vouched_for() {
        let fp = Fingerprint::of_certificate(b"x");
        let seen = UnverifiedFingerprint::new(fp);
        assert!(seen.matches(&fp));
        assert!(!seen.matches(&Fingerprint::of_certificate(b"y")));
        assert_eq!(seen.short(), fp.short());
        assert_eq!(seen.confirmed(), fp);
    }

    #[test]
    fn pairing_codes_display_and_parse() {
        let c = PairingCode(7);
        assert_eq!((c.digits(), c.to_string()), ("000007".to_string(), "000 007".to_string()));
        assert_eq!(PairingCode::parse("482 913"), Some(PairingCode(482_913)));
        assert_eq!(PairingCode::parse("482-913"), Some(PairingCode(482_913)));
        assert_eq!("000007".parse::<PairingCode>().unwrap(), c);
        for bad in ["", "12345", "1234567", "48291a", "+48291"] {
            assert_eq!(PairingCode::parse(bad), None, "{bad:?}");
        }
        assert_eq!(serde_json::to_string(&c).unwrap(), "\"000007\"");
        assert_eq!(parse_nonce(&nonce_hex(&seq(9))), Some(seq(9)));
        assert_eq!(parse_nonce("00"), None);
        assert_ne!(random_nonce().unwrap(), random_nonce().unwrap());
    }
}
