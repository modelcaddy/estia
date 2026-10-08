//! A daemon somewhere else, used through the same shapes as a local session.
//!
//! Blocking HTTP on purpose: hosts call sessions from blocking contexts
//! (`spawn_blocking`, worker threads), and a `Session` is blocking too. The
//! native `/engine/*` routes are used, not `/v1/*`, so the caller gets the
//! engine's own facts back: fingerprint, family, cached tokens, repairs.
//!
//! Safe from any thread, a Tokio worker included. reqwest's blocking client
//! panics when it is built, used or dropped inside an async runtime ("Cannot
//! drop a runtime in a context where blocking is not allowed"), and a host
//! cannot always tell where a handle will be created or dropped: ModelCaddy hit
//! it creating a handle in an async command and describing a picture inside an
//! async digest pass. So every step that touches the client runs on a plain
//! thread when the caller is inside a runtime ([`off_reactor`]); the caller
//! still blocks, as it would on a local session.
//!
//! # Three ways to trust the engine
//!
//! - `http://` ([`RemoteEngine::new`]): no TLS. Right for the engine on this
//!   machine (`http://127.0.0.1:27200`, what [`crate::find_local_engine`]
//!   returns). To another machine the token, prompts and outputs would cross
//!   the network in clear, so [`RemoteEngine::new`] refuses a token there
//!   (an engine from 0.5 on refuses plain HTTP from other devices anyway);
//!   [`RemoteEngine::cleartext`] is the explicit way to reach an older one.
//! - `https://` with a pin ([`RemoteEngine::with_pin`], [`RemoteEngine::connect`]):
//!   how a device talks to an engine on the LAN. Exactly the certificate with
//!   that SHA-256 fingerprint is trusted, whatever its names or dates. The pin
//!   comes from pairing ([`begin_pairing`], then [`PendingPairing::confirm`]
//!   with the confirmation code the operator read), or from `estia tls
//!   fingerprint` on the engine's machine, pasted into the app and checked
//!   against [`probe_certificate`]. A different certificate is
//!   [`SessionError::CertificateMismatch`], and the request is never sent.
//!   Needs the `remote-tls` feature.
//! - `https://` without a pin ([`RemoteEngine::new`]): the system's trust
//!   store, for an engine behind a TLS-terminating proxy with a real
//!   certificate. An engine's own self-signed certificate is refused.

use crate::error::SessionError;
use crate::proto::{GenerationMeta, Message};
use crate::session::{CancelToken, Priority};
use crate::tls::{self, Fingerprint, PairingCode, PinVerifier, UnverifiedFingerprint};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::time::Duration;

type Result<T> = std::result::Result<T, SessionError>;

fn err(msg: impl Into<String>) -> SessionError {
    SessionError::Runner(msg.into())
}

/// Connection to one daemon.
#[derive(Debug)]
pub struct RemoteEngine {
    base_url: String,
    token: Option<String>,
    /// The certificate check of a pinned or probing client; `None` for plain
    /// HTTP and for `https://` through the system's trust store.
    verifier: Option<Arc<PinVerifier>>,
    /// Always `Some` until [`Drop`], which may have to move it to another thread.
    client: Option<reqwest::blocking::Client>,
}

/// Run `f` where reqwest's blocking client may run: here, unless this thread
/// is inside a Tokio runtime, in which case on a scoped plain thread that this
/// one waits for. A panic in `f` is re-raised on the caller.
fn off_reactor<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    if tokio::runtime::Handle::try_current().is_err() {
        return f();
    }
    std::thread::scope(|s| match s.spawn(f).join() {
        Ok(v) => v,
        Err(panic) => std::panic::resume_unwind(panic),
    })
}

/// The URL as reqwest reads it, and so as it will dial it: the WHATWG rules
/// of the `url` crate, under which a `\` ends the host of an http(s) URL and
/// `127.1` is 127.0.0.1. A check made on any other reading could pass a URL
/// that connects somewhere else. `None` when it does not parse.
fn parse_url(url: &str) -> Option<reqwest::Url> {
    reqwest::Url::parse(url.trim()).ok()
}

/// A user name or password in the URL (`https://user:pass@host`).
fn has_userinfo(url: &reqwest::Url) -> bool {
    !url.username().is_empty() || url.password().is_some()
}

/// Does `url` name this machine: `localhost`, `*.localhost`, or a loopback
/// address? Only those may use plain `http://` for a token. A URL with a
/// user name or password in it never counts.
fn is_loopback_url(url: &str) -> bool {
    let Some(u) = parse_url(url) else { return false };
    if has_userinfo(&u) {
        return false;
    }
    let Some(host) = u.host_str() else { return false };
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }
    // IPv6 hosts come bracketed.
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    bare.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.to_canonical().is_loopback())
}

fn scheme_of(url: &str) -> Option<String> {
    parse_url(url).map(|u| u.scheme().to_string())
}

/// reqwest would send a user name and password from the URL as Basic auth,
/// to whatever host the URL names: never what a caller means. The URL is not
/// repeated in the error, since it holds a password.
fn refuse_userinfo(url: &str) -> Result<()> {
    match parse_url(url) {
        Some(u) if has_userinfo(&u) => Err(err("an engine URL cannot carry a user name or password; pass the token separately")),
        _ => Ok(()),
    }
}

/// What a pinned or probing call says in a build without the `remote-tls`
/// feature.
#[cfg(not(feature = "remote-tls"))]
const NO_REMOTE_TLS: &str = "this build of estia-engine has no client for an engine's own certificate: \
                             enable the `remote-tls` feature of estia-engine";

/// How a client checks the engine's certificate.
// Without remote-tls a pin is refused unread.
#[cfg_attr(not(feature = "remote-tls"), allow(dead_code))]
enum Trust {
    /// `http://`, or `https://` through the system's trust store.
    System,
    /// Exactly the certificate with this fingerprint.
    Pinned(Fingerprint),
    /// The first certificate seen, and from then on only that one.
    Learning,
}

/// An error and its causes on one line: reqwest's own text ("error sending
/// request for url …") hides the cause a person needs (refused, timed out, a
/// certificate problem).
fn with_causes(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut source = e.source();
    while let Some(s) = source {
        let text = s.to_string();
        if !out.contains(&text) {
            out.push_str(": ");
            out.push_str(&text);
        }
        source = s.source();
    }
    out
}

impl Drop for RemoteEngine {
    fn drop(&mut self) {
        if let Some(client) = self.client.take() {
            if tokio::runtime::Handle::try_current().is_ok() {
                // Dropping it here would shut its runtime down inside ours.
                std::thread::spawn(move || drop(client));
            }
        }
    }
}

fn prio_str(p: Priority) -> &'static str {
    match p {
        Priority::Interactive => "interactive",
        Priority::Background => "background",
    }
}

impl RemoteEngine {
    /// `base_url` like `http://127.0.0.1:27200`, no trailing path.
    ///
    /// `http://` sends everything in clear: right for the engine on this
    /// machine (`localhost` or a loopback address), and refused with a token
    /// for any other host, whose token would cross the network readable by
    /// anyone on it (use `https://` and a pin, or [`RemoteEngine::cleartext`]
    /// for an engine older than 0.5). `https://` without a pin trusts what the
    /// system's trust store trusts, which an Estia engine's own certificate is
    /// not: use [`RemoteEngine::with_pin`] for that.
    pub fn new(base_url: impl Into<String>, token: Option<String>) -> Result<Self> {
        let base_url = base_url.into();
        refuse_userinfo(&base_url)?;
        if scheme_of(&base_url).as_deref() == Some("http") && token.is_some() && !is_loopback_url(&base_url) {
            return Err(err(format!(
                "`{base_url}` is plain http to another machine: the token, prompts and outputs would cross the network in \
                 clear. An engine from 0.5 on serves the LAN over https: pair with it again (begin_pairing) and connect \
                 with the pin (RemoteEngine::connect). For an older engine on a network you trust, RemoteEngine::cleartext"
            )));
        }
        Self::build(base_url, token, Trust::System)
    }

    /// Plain `http://` to any machine, the token included: only for an engine
    /// older than 0.5, which serves the LAN over plain HTTP, on a network you
    /// trust. Anyone who can watch that network reads the token, the prompts
    /// and the outputs, and can reuse the token. Logs a warning. An engine
    /// from 0.5 on refuses plain HTTP from other devices (`https_required`).
    pub fn cleartext(base_url: impl Into<String>, token: Option<String>) -> Result<Self> {
        let base_url = base_url.into();
        if scheme_of(&base_url).as_deref() == Some("http") && token.is_some() && !is_loopback_url(&base_url) {
            tracing::warn!(
                url = %base_url,
                "connecting to another machine over plain http: the token, prompts and outputs cross the network in clear"
            );
        }
        Self::build(base_url, token, Trust::System)
    }

    /// An `https://` engine whose certificate has this fingerprint, whatever
    /// its names, dates or issuer: the pin from pairing, or from `estia tls
    /// fingerprint`. Any other certificate is refused before a request is sent
    /// ([`SessionError::CertificateMismatch`]). Needs the `remote-tls` feature.
    pub fn with_pin(base_url: impl Into<String>, token: Option<String>, pin: Fingerprint) -> Result<Self> {
        let base_url = base_url.into();
        if scheme_of(&base_url).as_deref() != Some("https") {
            return Err(err(format!("a pinned certificate needs an https:// URL, not `{base_url}`")));
        }
        Self::build(base_url, token, Trust::Pinned(pin))
    }

    /// [`RemoteEngine::with_pin`] when there is a pin, else [`RemoteEngine::new`]:
    /// what a host calls with a stored [`crate::EngineLocation::Remote`] (or
    /// [`crate::EngineLocation::connect`], which cannot drop the pin). A pin
    /// with an `http://` URL is an error, and so is a token with `http://` to
    /// another machine.
    pub fn connect(base_url: impl Into<String>, token: Option<String>, pin: Option<Fingerprint>) -> Result<Self> {
        match pin {
            Some(p) => Self::with_pin(base_url, token, p),
            None => Self::new(base_url, token),
        }
    }

    /// Learning mode: the first certificate seen becomes the pin of this
    /// client (see [`PinVerifier`]). For [`probe_certificate`] and [`begin_pairing`].
    fn probing(base_url: &str) -> Result<Self> {
        if scheme_of(base_url).as_deref() != Some("https") {
            return Err(err(format!("only an https:// engine has a certificate to check, not `{base_url}`")));
        }
        Self::build(base_url.to_string(), None, Trust::Learning)
    }

    fn build(base_url: String, token: Option<String>, trust: Trust) -> Result<Self> {
        refuse_userinfo(&base_url)?;
        #[cfg(feature = "remote-tls")]
        let verifier = match trust {
            Trust::System => None,
            Trust::Pinned(pin) => Some(PinVerifier::pinned(pin)),
            Trust::Learning => Some(PinVerifier::learning()),
        };
        #[cfg(feature = "remote-tls")]
        let config = match &verifier {
            Some(v) => Some(tls::client_config(Arc::clone(v)).map_err(|e| err(format!("tls configuration: {e}")))?),
            None => None,
        };
        #[cfg(not(feature = "remote-tls"))]
        let verifier: Option<Arc<PinVerifier>> = match trust {
            Trust::System => None,
            Trust::Pinned(_) | Trust::Learning => return Err(err(NO_REMOTE_TLS)),
        };
        let client = off_reactor(|| {
            #[allow(unused_mut)] // Only the remote-tls build changes it.
            let mut b = reqwest::blocking::Client::builder()
                .connect_timeout(Duration::from_secs(5))
                // Generations run for minutes; this is the whole-request ceiling.
                .timeout(Duration::from_secs(600));
            #[cfg(feature = "remote-tls")]
            if let Some(config) = config {
                // A pinned client follows no redirect: a redirect could only
                // lead somewhere the pin does not cover.
                b = b.use_preconfigured_tls(config).http1_only().redirect(reqwest::redirect::Policy::none());
            }
            b.build()
        })
        .map_err(|e| err(format!("http client: {}", with_causes(&e))))?;
        Ok(Self { base_url: base_url.trim_end_matches('/').to_string(), token, verifier, client: Some(client) })
    }

    fn client(&self) -> &reqwest::blocking::Client {
        self.client.as_ref().expect("client lives until drop")
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The certificate this client trusts: the pin it was given, or for a
    /// probing client the one it learned. `None` without TLS pinning.
    pub fn pin(&self) -> Option<Fingerprint> {
        self.verifier.as_ref().and_then(|v| v.pin())
    }

    /// A transport failure as a [`SessionError`]: a refused certificate is
    /// [`SessionError::CertificateMismatch`]; anything else says what failed,
    /// with its causes.
    fn transport_err(&self, what: &str, e: reqwest::Error) -> SessionError {
        let causes = with_causes(&e);
        if let Some(v) = &self.verifier {
            // The verifier noted the certificate it refused; the error text
            // confirms this failure is that refusal and not a later one.
            if let (Some(presented), Some(pinned)) = (v.take_refused(), v.pin()) {
                if causes.contains("certificate") {
                    return SessionError::CertificateMismatch { url: self.base_url.clone(), pinned, presented };
                }
            }
        }
        let mut text = format!("{what}: {causes}");
        if self.verifier.is_none() && scheme_of(&self.base_url).as_deref() == Some("https") && text.to_lowercase().contains("certif") {
            text.push_str(
                " (an Estia engine serves its own certificate, which the system does not trust: pair with it, \
                 or pin it with RemoteEngine::with_pin and the fingerprint `estia tls fingerprint` prints)",
            );
        }
        err(text)
    }

    fn post(&self, path: &str, body: &Value) -> reqwest::blocking::RequestBuilder {
        let mut r = self.client().post(format!("{}{path}", self.base_url)).json(body);
        if let Some(t) = &self.token {
            r = r.bearer_auth(t);
        }
        r
    }

    fn get(&self, path: &str) -> reqwest::blocking::RequestBuilder {
        let mut r = self.client().get(format!("{}{path}", self.base_url));
        if let Some(t) = &self.token {
            r = r.bearer_auth(t);
        }
        r
    }

    fn check(resp: reqwest::blocking::Response) -> Result<reqwest::blocking::Response> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        let text = resp.text().unwrap_or_default();
        let message =
            serde_json::from_str::<Value>(&text).ok().and_then(|v| v["error"]["message"].as_str().map(str::to_string)).unwrap_or(text);
        Err(err(format!("remote engine {status}: {message}")))
    }

    /// `GET /engine/health` — reachable, and which API version.
    pub fn health(&self) -> Result<Value> {
        off_reactor(|| {
            let resp = self
                .client()
                .get(format!("{}/engine/health", self.base_url))
                .timeout(Duration::from_secs(5))
                .send()
                .map_err(|e| self.transport_err("remote engine unreachable", e))?;
            Self::check(resp)?.json().map_err(|e| err(format!("health: {e}")))
        })
    }

    /// Non-streaming generation of a raw prompt.
    pub fn generate(&self, model: &str, prompt: &str, max_tokens: Option<u32>, temperature: Option<f32>, prio: Priority) -> Result<String> {
        off_reactor(|| {
            let body =
                json!({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "priority": prio_str(prio)});
            let resp = Self::check(self.post("/engine/generate", &body).send().map_err(|e| self.transport_err("remote generate", e))?)?;
            let v: Value = resp.json().map_err(|e| err(format!("remote generate: {e}")))?;
            v["text"].as_str().map(str::to_string).ok_or_else(|| err("remote generate: no text in response"))
        })
    }

    /// Streaming generation; `on_token` per token line from the daemon's SSE.
    /// Dropping the connection (a flipped cancel token) cancels on the host.
    // Public API; mirrors `chat_stream` below. A params struct would break callers.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        prio: Priority,
        cancel: Option<&CancelToken>,
        on_token: impl FnMut(&str),
    ) -> Result<(String, Option<GenerationMeta>)> {
        let body = json!({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "priority": prio_str(prio), "stream": true});
        self.stream(body, cancel, on_token)
    }

    /// Streaming chat through the model's template on the host.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
        cache_key: Option<&str>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        prio: Priority,
        cancel: Option<&CancelToken>,
        on_token: impl FnMut(&str),
    ) -> Result<(String, Option<GenerationMeta>)> {
        let body = json!({"model": model, "messages": messages, "tools": tools, "cache_key": cache_key, "max_tokens": max_tokens, "temperature": temperature, "priority": prio_str(prio), "stream": true});
        self.stream(body, cancel, on_token)
    }

    fn stream(
        &self,
        body: Value,
        cancel: Option<&CancelToken>,
        mut on_token: impl FnMut(&str),
    ) -> Result<(String, Option<GenerationMeta>)> {
        // The request and the reading run on a plain thread (see
        // [`off_reactor`]); tokens come back over a channel, so `on_token`
        // runs here, on the caller's thread, and needs no `Send`.
        enum Line {
            Token(String),
            Done(Option<String>, Option<GenerationMeta>),
            Failed(SessionError),
        }
        std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::channel::<Line>();
            let body = &body;
            let reader = scope.spawn(move || {
                let resp = match self.post("/engine/generate", body).send() {
                    Ok(r) => r,
                    Err(e) => return drop(tx.send(Line::Failed(self.transport_err("remote stream", e)))),
                };
                let resp = match Self::check(resp) {
                    Ok(r) => r,
                    Err(e) => return drop(tx.send(Line::Failed(e))),
                };
                for line in BufReader::new(resp).lines() {
                    let line = match line {
                        Ok(l) => l,
                        Err(e) => return drop(tx.send(Line::Failed(err(format!("remote stream read: {e}"))))),
                    };
                    let Some(data) = line.strip_prefix("data: ") else { continue };
                    let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
                    let msg = if let Some(e) = v.get("error") {
                        Line::Failed(err(format!("remote engine: {}", e.as_str().unwrap_or(&e.to_string()))))
                    } else if let Some(t) = v.get("token").and_then(Value::as_str) {
                        Line::Token(t.to_string())
                    } else if v.get("done").and_then(Value::as_bool) == Some(true) {
                        let meta = v.get("meta").cloned().and_then(|m| serde_json::from_value(m).ok());
                        Line::Done(v.get("text").and_then(Value::as_str).map(str::to_string), meta)
                    } else {
                        continue;
                    };
                    let last = !matches!(msg, Line::Token(_));
                    // A closed channel means the caller cancelled: returning
                    // drops the response, which closes the connection, and
                    // the daemon cancels the runner.
                    if tx.send(msg).is_err() || last {
                        return;
                    }
                }
                let _ = tx.send(Line::Failed(err("remote stream ended without a done line")));
            });
            let mut full = String::new();
            let outcome = loop {
                if cancel.map(CancelToken::is_cancelled).unwrap_or(false) {
                    break Err(SessionError::Cancelled { partial: std::mem::take(&mut full) });
                }
                match rx.recv_timeout(Duration::from_millis(200)) {
                    Ok(Line::Token(t)) => {
                        full.push_str(&t);
                        on_token(&t);
                    }
                    // Prefer the host's assembled text (identical bytes, no drift).
                    Ok(Line::Done(text, meta)) => break Ok((text.unwrap_or(std::mem::take(&mut full)), meta)),
                    Ok(Line::Failed(e)) => break Err(e),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break Err(err("remote stream ended without a done line")),
                }
            };
            drop(rx);
            if reader.join().is_err() {
                return Err(err("remote stream reader panicked"));
            }
            outcome
        })
    }

    /// Trade the same-user secret (`local-access.secret`) for a token named
    /// `local-<app>`. The daemon accepts this from loopback only; see
    /// [`crate::local::LocalEngine::claim_token`].
    pub fn local_token(&self, secret: &str, app: &str, scopes: Option<&[&str]>) -> Result<String> {
        off_reactor(|| {
            let body = json!({"secret": secret, "name": app, "scopes": scopes});
            let resp = self.client().post(format!("{}/engine/local-token", self.base_url)).json(&body).send();
            let resp = resp.map_err(|e| self.transport_err("local token", e))?;
            let v: Value = Self::check(resp)?.json().map_err(|e| err(format!("local token: {e}")))?;
            v["token"].as_str().map(str::to_string).ok_or_else(|| err("local token: no token in the answer"))
        })
    }

    /// Ask a daemon for a token: the first half of pairing. Needs no token.
    /// Returns the pairing id to poll with [`RemoteEngine::pair_poll`].
    ///
    /// It sends no commitment, so over `https://` the request has no code
    /// (`check: none`): the operator can approve it only explicitly without
    /// one, and the device learns nothing about the certificate it pins. Use
    /// [`begin_pairing`], which checks the certificate both ways.
    #[deprecated(since = "0.5.0", note = "use begin_pairing, which checks the engine's certificate with a code both ways")]
    pub fn pair_request(&self, name: &str, scopes: &[&str]) -> Result<String> {
        off_reactor(|| {
            let resp = self
                .client()
                .post(format!("{}/engine/pair", self.base_url))
                .json(&json!({"name": name, "scopes": scopes}))
                .send()
                .map_err(|e| self.transport_err("pair request", e))?;
            let v: Value = Self::check(resp)?.json().map_err(|e| err(format!("pair request: {e}")))?;
            v["id"].as_str().map(str::to_string).ok_or_else(|| err("pair request: no id"))
        })
    }

    /// Poll a pairing: `pending`, `approved` (with the token, once) or `denied`.
    #[deprecated(since = "0.5.0", note = "use begin_pairing and PendingPairing::poll")]
    pub fn pair_poll(&self, id: &str) -> Result<(String, Option<String>)> {
        off_reactor(|| {
            let resp =
                self.client().get(format!("{}/engine/pair/{id}", self.base_url)).send().map_err(|e| self.transport_err("pair poll", e))?;
            let v: Value = Self::check(resp)?.json().map_err(|e| err(format!("pair poll: {e}")))?;
            Ok((v["status"].as_str().unwrap_or("pending").to_string(), v["token"].as_str().map(str::to_string)))
        })
    }

    /// Admin: pending and recent pairings on the daemon. Each row carries
    /// `check` (`code`, `waiting`, `none` or `plain`), `code`, the code the
    /// engine computed for it (see [`crate::tls`]), and once approved
    /// `confirm`, the confirmation code for the user to type into the device.
    pub fn pairings(&self) -> Result<Value> {
        self.get_json("/engine/pairings")
    }

    /// `GET` a route with this client's token and return its JSON body: for
    /// routes without a method here (`/engine/stats`, `/engine/models`).
    pub fn get_json(&self, path: &str) -> Result<Value> {
        off_reactor(|| {
            let resp = self.get(path).send().map_err(|e| self.transport_err(path, e))?;
            Self::check(resp)?.json().map_err(|e| err(format!("{path}: {e}")))
        })
    }

    /// Admin: approve or deny a pairing from wherever this client is. An
    /// approval sends no code, so an engine from 0.5 on accepts it only for a
    /// request over plain HTTP from its own machine (`check: plain`). It
    /// refuses it for a request with a code (use
    /// [`RemoteEngine::approve_pairing`] with the code the device shows) and
    /// for a request over HTTPS without one (`check: none`, see
    /// [`RemoteEngine::approve_pairing_without_code`]).
    pub fn decide_pairing(&self, id: &str, approve: bool) -> Result<Value> {
        if approve {
            return self.approve_pairing(id, None);
        }
        off_reactor(|| {
            let resp =
                self.post(&format!("/engine/pairings/{id}/deny"), &json!({})).send().map_err(|e| self.transport_err("pairing deny", e))?;
            Self::check(resp)?.json().map_err(|e| err(format!("pairing deny: {e}")))
        })
    }

    /// Admin: approve a pairing with the code the device shows (`482913`, or
    /// `482 913`). The engine compares it with the code it computed from its
    /// own certificate and refuses a mismatch: the device then saw another
    /// certificate, and the request should be denied. The answer carries
    /// `confirm`, the confirmation code to show the user, who types it into
    /// the device ([`PendingPairing::confirm`]).
    ///
    /// `None` only for a request over plain HTTP from the engine's own
    /// machine (`check: plain`), or an engine older than 0.5.
    pub fn approve_pairing(&self, id: &str, code: Option<&str>) -> Result<Value> {
        let body = match code {
            Some(c) => json!({"code": c}),
            None => json!({}),
        };
        self.send_approval(id, &body)
    }

    /// Admin: approve a request over HTTPS that has no code (`check: none`):
    /// a browser or curl, which cannot show one. Only after the person at the
    /// device compared the certificate fingerprint it shows with `estia tls
    /// fingerprint`: nothing else checks it. If the device in front of you
    /// shows a 6-digit code, deny the request instead: something between it
    /// and the engine removed the code from its request.
    pub fn approve_pairing_without_code(&self, id: &str) -> Result<Value> {
        self.send_approval(id, &json!({"no_code": true}))
    }

    fn send_approval(&self, id: &str, body: &Value) -> Result<Value> {
        off_reactor(|| {
            let resp =
                self.post(&format!("/engine/pairings/{id}/approve"), body).send().map_err(|e| self.transport_err("pairing approve", e))?;
            Self::check(resp)?.json().map_err(|e| err(format!("pairing approve: {e}")))
        })
    }

    /// Embed already-prefixed inputs (`task: none`) and check the host serves
    /// the fingerprint the caller's index was built with.
    pub fn embed_batch(
        &self,
        model: &str,
        inputs: &[String],
        expect_fingerprint: Option<&str>,
        prio: Priority,
    ) -> Result<(Vec<Vec<f32>>, String)> {
        off_reactor(|| {
            let body = json!({"model": model, "inputs": inputs, "task": "none", "expect_fingerprint": expect_fingerprint, "priority": prio_str(prio)});
            let resp = Self::check(self.post("/engine/embed", &body).send().map_err(|e| self.transport_err("remote embed", e))?)?;
            #[derive(Deserialize)]
            struct EmbedResp {
                vectors: Vec<Vec<f32>>,
                fingerprint: String,
            }
            let r: EmbedResp = resp.json().map_err(|e| err(format!("remote embed: {e}")))?;
            Ok((r.vectors, r.fingerprint))
        })
    }
}

/// Connect to an `https://` engine without trusting it, and say which
/// certificate it presented. Nothing is pinned: a person must compare the
/// result with `estia tls fingerprint` on the engine's machine
/// ([`UnverifiedFingerprint::confirmed`]), or it must match a fingerprint the
/// user pasted from there ([`UnverifiedFingerprint::matches`]), before it can
/// pin anything. For the pasted-token path; pairing checks the certificate
/// itself ([`begin_pairing`]). An error for `http://`, which has no
/// certificate. Needs the `remote-tls` feature.
pub fn probe_certificate(base_url: &str) -> Result<UnverifiedFingerprint> {
    let probe = RemoteEngine::probing(base_url)?;
    off_reactor(|| {
        // Any answer will do, an error status included: the handshake is done.
        let sent = probe.client().get(format!("{}/engine/health", probe.base_url)).timeout(Duration::from_secs(5)).send();
        match (sent, probe.pin()) {
            (_, Some(fp)) => Ok(UnverifiedFingerprint::new(fp)),
            (Err(e), None) => Err(probe.transport_err("remote engine unreachable", e)),
            (Ok(_), None) => Err(err("the engine answered without a certificate")),
        }
    })
}

/// The device side of pairing, up to the point where the operator decides.
///
/// Over `https://` it connects without trusting the engine yet, commits to a
/// random nonce, asks for a token, reveals the nonce once the engine has
/// answered with its own, and computes two 6-digit codes from the certificate
/// it saw and the two nonces:
///
/// - the **pairing code** ([`PendingPairing::code`]): show it. The operator
///   types it into `estia pair approve <id> --code <code>`, and the engine
///   approves only when it computed the same code from its own certificate.
///   That tells the engine the device saw its certificate.
/// - the **confirmation code**, which the device keeps: `estia pair approve`
///   prints the engine's, and the user types it into the device
///   ([`PendingPairing::confirm`]). That tells the device the approval came
///   from the engine. Without it a device cannot tell: whatever answered its
///   request could also answer its polls with "approved" and a token of its
///   own making.
///
/// The certificate seen here is the one every later call of this pairing must
/// present (the polls included), and the pin once confirmed.
///
/// `http://` is allowed for this machine only (`localhost`, a loopback
/// address), with no codes: to another machine the token would cross the
/// network in clear.
///
/// Blocking. `name` is how the operator sees the device; ask for the least
/// `scopes` the app needs (`generate`, `embed`, `models:read`). Over
/// `https://` it needs the `remote-tls` feature.
pub fn begin_pairing(base_url: &str, name: &str, scopes: &[&str]) -> Result<PendingPairing> {
    let base_url = base_url.trim().trim_end_matches('/').to_string();
    match scheme_of(&base_url).as_deref() {
        Some("https") => {}
        Some("http") if is_loopback_url(&base_url) => {
            let engine = RemoteEngine::new(base_url.clone(), None)?;
            let (id, expires_in) = off_reactor(|| request_pairing(&engine, name, scopes, None))?.0;
            return Ok(PendingPairing::new(base_url, id, None, expires_in));
        }
        Some("http") => {
            return Err(err(format!(
                "`{base_url}` is plain http to another machine: the token would cross the network in clear. \
                 Use https:// (an engine from 0.5 on serves the LAN over https)"
            )))
        }
        _ => return Err(err(format!("`{base_url}` is not an http:// or https:// URL"))),
    }
    let engine = Arc::new(RemoteEngine::probing(&base_url)?);
    let nonce = tls::random_nonce().map_err(|e| err(format!("randomness: {e}")))?;
    let ((id, expires_in), answer) = off_reactor(|| request_pairing(&engine, name, scopes, Some(&tls::pairing_commitment(&nonce))))?;
    let seen = engine.pin().ok_or_else(|| err("pair request: the engine presented no certificate"))?;
    check_stated(&answer, seen)?;
    let engine_nonce = answer["engine_nonce"].as_str().and_then(tls::parse_nonce).ok_or_else(|| {
        err("the engine did not answer the commitment with a nonce: it is older than 0.5, or did not see this request arrive over https")
    })?;
    off_reactor(|| {
        let resp = engine
            .client()
            .post(format!("{}/engine/pair/{id}/reveal", engine.base_url))
            .json(&json!({"nonce": tls::nonce_hex(&nonce)}))
            .send()
            .map_err(|e| engine.transport_err("pair reveal", e))?;
        RemoteEngine::check(resp).map(drop)
    })?;
    let codes = Codes {
        certificate: seen,
        code: tls::pairing_code(&seen, &nonce, &engine_nonce),
        confirmation: tls::pairing_confirmation(&seen, &nonce, &engine_nonce),
    };
    let pending = PendingPairing::new(base_url, id, Some(codes), expires_in);
    let _ = pending.client.set(engine);
    Ok(pending)
}

/// `POST /engine/pair`; returns (id, expires_in) and the whole answer.
fn request_pairing(engine: &RemoteEngine, name: &str, scopes: &[&str], commitment: Option<&str>) -> Result<((String, u64), Value)> {
    let mut body = json!({"name": name, "scopes": scopes});
    if let Some(c) = commitment {
        body["commitment"] = json!(c);
    }
    let resp = engine
        .client()
        .post(format!("{}/engine/pair", engine.base_url))
        .json(&body)
        .send()
        .map_err(|e| engine.transport_err("pair request", e))?;
    let v: Value = RemoteEngine::check(resp)?.json().map_err(|e| err(format!("pair request: {e}")))?;
    let id = v["id"].as_str().filter(|id| !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    let id = id.ok_or_else(|| err("pair request: no usable pairing id in the answer"))?.to_string();
    let expires_in = v["expires_in"].as_u64().unwrap_or(300);
    Ok(((id, expires_in), v))
}

/// The certificate a pairing answer names must be the one this connection
/// saw. Not the security mechanism (whatever rewrote the certificate could
/// rewrite the body too) but a clear early error for a confused setup.
fn check_stated(answer: &Value, seen: Fingerprint) -> Result<()> {
    match answer["tls"]["fingerprint"].as_str().map(Fingerprint::parse) {
        Some(Ok(stated)) if stated != seen => Err(err(format!(
            "the engine says its certificate is {}, but this connection presented {}: something between this device and \
             the engine answered with its own certificate",
            stated.short(),
            seen.short()
        ))),
        _ => Ok(()),
    }
}

/// What an `https://` pairing computed from the certificate it saw.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Codes {
    /// The certificate this device saw: the pin once confirmed.
    certificate: Fingerprint,
    /// The pairing code, shown to the operator.
    code: PairingCode,
    /// The confirmation code the engine's operator will read out; kept here
    /// and never shown.
    confirmation: PairingCode,
}

/// A pairing in progress on the device: what [`begin_pairing`] returns. It
/// holds no secret (an interceptor could compute every code in it from what
/// crossed the network), so a host can keep it between calls (it serialises)
/// and poll from wherever. It must never show the confirmation code it
/// expects: the user types the one the engine's operator reads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingPairing {
    base_url: String,
    id: String,
    /// `None` for a loopback `http://` pairing, which has no certificate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    codes: Option<Codes>,
    expires_unix: u64,
    /// The connection the request went out on, reused for polls: a host that
    /// restored this from storage gets a new one, held to the same certificate.
    #[serde(skip)]
    client: std::sync::OnceLock<Arc<RemoteEngine>>,
}

/// What a poll of a [`PendingPairing`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingOutcome {
    /// The operator has not decided yet. Poll again in a couple of seconds.
    Pending,
    /// The operator refused.
    Denied,
    /// An answer of "approved", with the token (handed over once: store it
    /// now). Over `https://` this alone proves nothing: whatever answered the
    /// device's request could answer its polls too, with a token of its own.
    /// Before using the token, ask the user for the confirmation code `estia
    /// pair approve` printed on the engine's machine and pass it to
    /// [`PendingPairing::confirm`], which returns the pin. A loopback
    /// `http://` pairing has nothing to confirm
    /// ([`PendingPairing::needs_confirmation`] is false).
    Approved { token: String },
    /// Approved, but an earlier poll collected the token. Ask the operator to
    /// deny this pairing (which revokes that token) and pair again.
    AlreadyCollected,
    /// The engine no longer knows the request: it drops an undecided one 300 s
    /// after it was made. Pair again.
    Expired,
}

impl PendingPairing {
    fn new(base_url: String, id: String, codes: Option<Codes>, expires_in: u64) -> Self {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        PendingPairing { base_url, id, codes, expires_unix: now + expires_in, client: std::sync::OnceLock::new() }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The pairing code to show, large: the operator types it on the engine's
    /// machine. `None` for a loopback `http://` pairing.
    pub fn code(&self) -> Option<&PairingCode> {
        self.codes.as_ref().map(|c| &c.code)
    }

    /// The certificate this device saw. Nothing has vouched for it until
    /// [`PendingPairing::confirm`] matches; that returns it as the pin.
    /// `None` for a loopback `http://` pairing.
    pub fn certificate(&self) -> Option<UnverifiedFingerprint> {
        self.codes.map(|c| UnverifiedFingerprint::new(c.certificate))
    }

    /// Does this pairing end with a confirmation code typed on the device?
    /// True over `https://`; false for a loopback `http://` pairing.
    pub fn needs_confirmation(&self) -> bool {
        self.codes.is_some()
    }

    /// The device's half of the check: `typed` is the confirmation code that
    /// `estia pair approve` printed on the engine's machine (`452618`, or
    /// `452 618`). When it equals the one this device computed from the
    /// certificate it saw, the engine approved this very request having seen
    /// that certificate's code, and the certificate is returned: the pin, to
    /// store with the token. Any other code is an error: either a typo (ask
    /// again), or the approval did not come from the engine, and the token
    /// must not be used.
    ///
    /// Read the code only from the engine's machine (the operator, `estia
    /// pair approve`, an admin app), never from anything this device received.
    pub fn confirm(&self, typed: &str) -> Result<Fingerprint> {
        let Some(codes) = self.codes else {
            return Err(err("this pairing went over plain http to this machine: it has no certificate and nothing to confirm"));
        };
        let given = PairingCode::parse(typed).ok_or_else(|| err("a confirmation code is six digits, such as 452618"))?;
        if given != codes.confirmation {
            return Err(err(format!(
                "that is not the confirmation code this device expects. Check it is the one `estia pair approve {}` printed \
                 on the engine's machine. If it is, the approval did not come from that engine: something between this \
                 device and the engine answered in its place. Do not use the token; deny the pairing there and pair again",
                self.id
            )));
        }
        Ok(codes.certificate)
    }

    /// When the engine drops the request if nobody decides (Unix seconds).
    pub fn expires_unix(&self) -> u64 {
        self.expires_unix
    }

    /// What the operator runs: `estia pair approve <id> --code 482913`.
    pub fn approve_command(&self) -> String {
        match self.code() {
            Some(c) => format!("estia pair approve {} --code {}", self.id, c.digits()),
            None => format!("estia pair approve {}", self.id),
        }
    }

    fn engine(&self) -> Result<Arc<RemoteEngine>> {
        if let Some(e) = self.client.get() {
            return Ok(Arc::clone(e));
        }
        let e = Arc::new(RemoteEngine::connect(self.base_url.clone(), None, self.codes.map(|c| c.certificate))?);
        Ok(Arc::clone(self.client.get_or_init(|| e)))
    }

    /// Ask the engine once, over the connection held to the certificate this
    /// device saw. Errors are worth another poll (the engine restarting, Wi-Fi
    /// dropping) except [`SessionError::CertificateMismatch`]: then cancel,
    /// and if the engine refused the operator's code, tell the user so too.
    pub fn poll(&self) -> Result<PairingOutcome> {
        let engine = self.engine()?;
        let v: Value = off_reactor(|| {
            let resp = engine
                .client()
                .get(format!("{}/engine/pair/{}", engine.base_url, self.id))
                .timeout(Duration::from_secs(10))
                .send()
                .map_err(|e| engine.transport_err("pair poll", e))?;
            if resp.status() == reqwest::StatusCode::NOT_FOUND {
                return Ok(Value::Null);
            }
            RemoteEngine::check(resp)?.json().map_err(|e| err(format!("pair poll: {e}")))
        })?;
        if v.is_null() {
            return Ok(PairingOutcome::Expired);
        }
        if let Some(c) = self.codes {
            check_stated(&v, c.certificate)?;
        }
        Ok(match (v["status"].as_str(), v["token"].as_str()) {
            (Some("approved"), Some(token)) => PairingOutcome::Approved { token: token.to_string() },
            (Some("approved"), None) => PairingOutcome::AlreadyCollected,
            (Some("denied"), _) => PairingOutcome::Denied,
            _ => PairingOutcome::Pending,
        })
    }
}

/// A remote generation model under the local session's shape.
pub struct RemoteGen {
    engine: Arc<RemoteEngine>,
    model_id: String,
}

impl RemoteGen {
    pub fn new(engine: Arc<RemoteEngine>, model_id: impl Into<String>) -> Self {
        Self { engine, model_id: model_id.into() }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn engine(&self) -> &RemoteEngine {
        &self.engine
    }

    pub fn generate_with(&self, prompt: &str, max_tokens: Option<u32>, temperature: Option<f32>, prio: Priority) -> Result<String> {
        self.engine.generate(&self.model_id, prompt, max_tokens, temperature, prio)
    }

    pub fn generate_stream_with(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        prio: Priority,
        cancel: Option<&CancelToken>,
        on_token: impl FnMut(&str),
    ) -> Result<String> {
        self.engine.generate_stream(&self.model_id, prompt, max_tokens, temperature, prio, cancel, on_token).map(|(t, _)| t)
    }
}

/// A remote embedding model under the local session's shape. `fingerprint` is
/// what the caller expects; every call asks the host to refuse a mismatch.
pub struct RemoteEmbed {
    engine: Arc<RemoteEngine>,
    model_id: String,
    fingerprint: String,
}

impl RemoteEmbed {
    pub fn new(engine: Arc<RemoteEngine>, model_id: impl Into<String>, fingerprint: impl Into<String>) -> Self {
        Self { engine, model_id: model_id.into(), fingerprint: fingerprint.into() }
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn engine(&self) -> &RemoteEngine {
        &self.engine
    }

    pub fn embed_batch_with(&self, inputs: &[String], prio: Priority) -> Result<Vec<Vec<f32>>> {
        let (vectors, served) = self.engine.embed_batch(&self.model_id, inputs, Some(&self.fingerprint), prio)?;
        if served != self.fingerprint {
            return Err(err(format!("remote engine served fingerprint `{served}`, expected `{}`", self.fingerprint)));
        }
        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    /// Answer one HTTP/1.1 request on `conn`, whatever carries it (TCP or
    /// TLS): `/engine/health` answers JSON, anything else streams two tokens
    /// and a done line, as `/engine/generate` does.
    fn answer<S: Read + Write>(mut conn: S) {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        // Headers, then the body its Content-Length announces.
        let (head_end, len) = loop {
            let n = conn.read(&mut chunk).unwrap_or(0);
            if n == 0 {
                return;
            }
            buf.extend_from_slice(&chunk[..n]);
            if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..i]).to_lowercase();
                let len =
                    head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
                break (i + 4, len);
            }
        };
        while buf.len() < head_end + len {
            let n = conn.read(&mut chunk).unwrap_or(0);
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
        }
        let first = String::from_utf8_lossy(&buf).lines().next().unwrap_or_default().to_string();
        let (ctype, body) = if first.contains("/engine/health") {
            ("application/json", r#"{"version":"test","backend":"fake"}"#.to_string())
        } else {
            (
                "text/event-stream",
                "data: {\"token\":\"a\"}\n\ndata: {\"token\":\"b\"}\n\ndata: {\"done\":true,\"text\":\"ab\"}\n\n".to_string(),
            )
        };
        let _ =
            write!(conn, "HTTP/1.1 200 OK\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len());
        let _ = conn.flush();
    }

    /// A one-file HTTP daemon (see [`answer`]).
    fn fake_daemon() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(conn) = conn else { continue };
                std::thread::spawn(move || answer(conn));
            }
        });
        format!("http://{addr}")
    }

    /// A self-signed certificate and its key (DER), as an engine has.
    #[cfg(feature = "remote-tls")]
    fn certificate() -> (Vec<u8>, Vec<u8>) {
        let c = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        (c.cert.der().to_vec(), c.signing_key.serialize_der())
    }

    /// Always presents one certificate and signs with one key, matching or not.
    #[cfg(feature = "remote-tls")]
    #[derive(Debug)]
    struct Fixed(Arc<rustls::sign::CertifiedKey>);

    #[cfg(feature = "remote-tls")]
    impl rustls::server::ResolvesServerCert for Fixed {
        fn resolve(&self, _: rustls::server::ClientHello<'_>) -> Option<Arc<rustls::sign::CertifiedKey>> {
            Some(Arc::clone(&self.0))
        }
    }

    /// The fake daemon over TLS, presenting `cert` and signing the handshake
    /// with `key`. Passing another certificate's key is the impostor that
    /// copied the engine's (public) certificate but does not hold its key.
    #[cfg(feature = "remote-tls")]
    fn tls_daemon(cert: &[u8], key: &[u8]) -> String {
        use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
        let provider = Arc::new(crate::tls::provider());
        let signing = provider.key_provider.load_private_key(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.to_vec()))).unwrap();
        let certified = rustls::sign::CertifiedKey::new(vec![CertificateDer::from(cert.to_vec())], signing);
        let config = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_cert_resolver(Arc::new(Fixed(Arc::new(certified))));
        let config = Arc::new(config);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(tcp) = conn else { continue };
                let config = Arc::clone(&config);
                std::thread::spawn(move || {
                    let Ok(server) = rustls::ServerConnection::new(config) else { return };
                    answer(rustls::StreamOwned::new(server, tcp));
                });
            }
        });
        format!("https://{addr}")
    }

    /// The pinned certificate is accepted, through the whole client: health
    /// and a stream.
    #[test]
    #[cfg(feature = "remote-tls")]
    fn a_pinned_certificate_is_trusted_whatever_its_issuer() {
        let (cert, key) = certificate();
        let url = tls_daemon(&cert, &key);
        let pin = Fingerprint::of_certificate(&cert);
        let engine = RemoteEngine::with_pin(&url, Some("t".into()), pin).unwrap();
        assert_eq!(engine.pin(), Some(pin));
        assert_eq!(engine.health().unwrap()["backend"], "fake");
        let (text, _) =
            engine.chat_stream("m", &[Message::new("user", "hi")], None, None, None, None, Priority::Background, None, |_| {}).unwrap();
        assert_eq!(text, "ab");
        assert!(probe_certificate(&url).unwrap().matches(&pin), "the probe sees the same certificate");
        assert!(RemoteEngine::connect(&url, None, Some(pin)).unwrap().health().is_ok());
    }

    #[test]
    #[cfg(feature = "remote-tls")]
    fn another_certificate_is_refused_before_the_request() {
        let (cert, key) = certificate();
        let url = tls_daemon(&cert, &key);
        let pinned = Fingerprint::of_certificate(b"the certificate this device paired with");
        let engine = RemoteEngine::with_pin(&url, Some("estia_secret".into()), pinned).unwrap();
        match engine.health() {
            Err(SessionError::CertificateMismatch { url: u, pinned: p, presented }) => {
                assert_eq!((u.as_str(), p, presented), (url.as_str(), pinned, Fingerprint::of_certificate(&cert)));
            }
            other => panic!("expected a certificate mismatch, got {other:?}"),
        }
        let e = engine.generate("m", "p", None, None, Priority::Background).unwrap_err();
        assert!(matches!(e, SessionError::CertificateMismatch { .. }), "{e}");
        assert!(e.to_string().contains("estia tls rotate"), "{e}");
    }

    /// The impostor has the engine's certificate (anyone who connected once
    /// does) but not its key: the handshake signature check refuses it, so the
    /// signature methods of the verifier are real checks.
    #[test]
    #[cfg(feature = "remote-tls")]
    fn the_right_certificate_with_the_wrong_key_is_refused() {
        let (cert, _) = certificate();
        let (_, other_key) = certificate();
        let url = tls_daemon(&cert, &other_key);
        let pin = Fingerprint::of_certificate(&cert);
        let e = RemoteEngine::with_pin(&url, None, pin).unwrap().health().unwrap_err();
        assert!(!matches!(e, SessionError::CertificateMismatch { .. }), "a bad signature, not another certificate: {e}");
        assert!(e.to_string().contains("BadSignature"), "refused by the signature check: {e}");
        assert!(probe_certificate(&url).is_err(), "nothing is learned from a handshake that does not verify");
    }

    #[test]
    #[cfg(feature = "remote-tls")]
    fn https_without_a_pin_uses_the_system_trust_store() {
        let (cert, key) = certificate();
        let url = tls_daemon(&cert, &key);
        let e = RemoteEngine::new(&url, None).unwrap().health().unwrap_err();
        assert!(e.to_string().contains("with_pin"), "says how to trust the engine: {e}");
    }

    #[test]
    fn pins_need_https_and_tokens_need_tls_off_this_machine() {
        let pin = Fingerprint::of_certificate(b"x");
        assert!(RemoteEngine::with_pin("http://127.0.0.1:1", None, pin).is_err());
        assert!(RemoteEngine::connect("http://192.168.1.2:27200", None, Some(pin)).is_err());
        assert!(RemoteEngine::connect("http://127.0.0.1:1", None, None).is_ok(), "plain loopback, no pin");
        assert!(RemoteEngine::connect("http://127.0.0.1:1", Some("estia_t".into()), None).is_ok(), "a token over loopback");
        assert!(probe_certificate("http://127.0.0.1:1").is_err(), "http has no certificate");
        // A token over plain http to another machine is refused before any
        // connection, `connect` with a stored pre-0.5 location included;
        // `cleartext` is the explicit way to an old engine.
        for url in ["http://192.168.1.2:27200", "http://studio.local:27200", "http://studio.local\\@127.0.0.1:27200"] {
            let e = RemoteEngine::connect(url, Some("estia_t".into()), None).unwrap_err();
            assert!(e.to_string().contains("in clear") && e.to_string().contains("cleartext"), "{url}: {e}");
            assert!(RemoteEngine::new(url, None).is_ok(), "{url}: no token, nothing to leak");
        }
        assert!(RemoteEngine::cleartext("http://192.168.1.2:27200", Some("estia_t".into())).is_ok());
        // A URL carrying a user name or password is refused whatever its host.
        for url in ["http://user@127.0.0.1:1", "https://u:p@studio.local:27200", "http://127.0.0.1:1@studio.local"] {
            assert!(RemoteEngine::new(url, None).unwrap_err().to_string().contains("user name"), "{url}");
        }
        // Refused before any connection: port 1 on a LAN address would hang.
        let t0 = std::time::Instant::now();
        let e = begin_pairing("http://192.168.1.2:1", "laptop", &["generate"]).unwrap_err();
        assert!(e.to_string().contains("in clear"), "{e}");
        // reqwest reads this as the host studio.local, path /@127.0.0.1:1.
        let e = begin_pairing("http://studio.local\\@127.0.0.1:1", "laptop", &["generate"]).unwrap_err();
        assert!(e.to_string().contains("in clear"), "{e}");
        assert!(t0.elapsed() < Duration::from_secs(1));
        assert!(begin_pairing("ftp://x", "laptop", &[]).is_err());
    }

    /// Without the `remote-tls` feature every pinned or probing call says
    /// which feature it needs, and plain loopback still works.
    #[test]
    #[cfg(not(feature = "remote-tls"))]
    fn pinned_calls_name_the_feature_they_need() {
        let pin = Fingerprint::of_certificate(b"x");
        for e in [
            RemoteEngine::with_pin("https://10.0.0.5:27200", None, pin).unwrap_err(),
            probe_certificate("https://10.0.0.5:27200").unwrap_err(),
            begin_pairing("https://10.0.0.5:27200", "laptop", &["generate"]).unwrap_err(),
        ] {
            assert!(e.to_string().contains("remote-tls"), "{e}");
        }
        assert!(RemoteEngine::connect("http://127.0.0.1:1", Some("t".into()), None).is_ok());
    }

    #[test]
    fn loopback_urls() {
        for yes in [
            "http://127.0.0.1:27200",
            "http://localhost",
            "http://LOCALHOST:1/x",
            "http://localhost.:1",
            "http://[::1]:5",
            "http://[::ffff:127.0.0.1]:5",
            "http://a.localhost:1",
            "http://127.1.2.3",
            "http://127.1:1",
        ] {
            assert!(is_loopback_url(yes), "{yes}");
        }
        for no in [
            "http://192.168.1.2:27200",
            "http://localhost.evil.example",
            "http://[fe80::1]:1",
            "http://studio.local",
            "127.0.0.1",
            "http://0.0.0.0:1",
            // What reqwest dials, not what a naive split reads: a `\` ends
            // the host, and the text after the last `@` is not the host.
            "http://studio.local\\@127.0.0.1:27200",
            "http://studio.local\\@localhost",
            "http://user@127.0.0.1:1",
            "http://127.0.0.1:1@studio.local",
        ] {
            assert!(!is_loopback_url(no), "{no}");
        }
        assert_eq!(scheme_of("HTTPS://Studio.Local.:27200/x").as_deref(), Some("https"));
        assert_eq!(scheme_of("not a url"), None);
    }

    fn codes(certificate: Fingerprint, code: &str, confirmation: &str) -> Codes {
        Codes { certificate, code: PairingCode::parse(code).unwrap(), confirmation: PairingCode::parse(confirmation).unwrap() }
    }

    /// A pairing in progress survives a round trip through a host's storage,
    /// and says what the operator runs.
    #[test]
    fn pending_pairings_serialise_and_say_what_to_run() {
        let pin = Fingerprint::of_certificate(b"engine");
        let p = PendingPairing::new("https://10.0.0.5:27200".into(), "6de47a30928f4f15".into(), Some(codes(pin, "042913", "452618")), 300);
        assert_eq!(p.approve_command(), "estia pair approve 6de47a30928f4f15 --code 042913");
        assert_eq!(p.code().unwrap().to_string(), "042 913");
        assert!(p.needs_confirmation());
        assert!(p.certificate().unwrap().matches(&pin));
        let back: PendingPairing = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!((back.id(), back.code().copied()), (p.id(), p.code().copied()));
        assert_eq!(back.confirm("452 618").unwrap(), pin, "the restored pairing confirms the same way");
        let plain = PendingPairing::new("http://127.0.0.1:1".into(), "ab".into(), None, 300);
        assert_eq!(plain.approve_command(), "estia pair approve ab");
        assert!(!plain.needs_confirmation() && plain.certificate().is_none());
        assert!(plain.confirm("452618").is_err(), "nothing to confirm over loopback http");
    }

    /// The device pins only after the confirmation code the engine's operator
    /// read matches its own: an "approved" answer alone proves nothing.
    #[test]
    fn the_device_pins_only_the_confirmed_certificate() {
        let seen = Fingerprint::of_certificate(b"the certificate this device saw");
        let (nonce_d, nonce_e) = ([1u8; 32], [2u8; 32]);
        let p = PendingPairing::new(
            "https://10.0.0.5:27200".into(),
            "6de47a30928f4f15".into(),
            Some(Codes {
                certificate: seen,
                code: tls::pairing_code(&seen, &nonce_d, &nonce_e),
                confirmation: tls::pairing_confirmation(&seen, &nonce_d, &nonce_e),
            }),
            300,
        );
        // What the real engine prints when its certificate is another one
        // (the device met an interceptor): refused, with what to do.
        let engines = tls::pairing_confirmation(&Fingerprint::of_certificate(b"the engine's certificate"), &nonce_d, &nonce_e);
        let e = p.confirm(&engines.digits()).unwrap_err().to_string();
        assert!(e.contains("not the confirmation code") && e.contains("Do not use the token"), "{e}");
        // The pairing code is not the confirmation code.
        assert!(p.confirm(&p.code().unwrap().digits()).is_err());
        assert!(p.confirm("12345").unwrap_err().to_string().contains("six digits"));
        // The engine's own code, from the certificate the device saw.
        let right = tls::pairing_confirmation(&seen, &nonce_d, &nonce_e);
        assert_eq!(p.confirm(&right.to_string()).unwrap(), seen);
    }

    fn use_it(base: &str) {
        let engine = RemoteEngine::new(base, Some("t".into())).unwrap();
        assert_eq!(engine.health().unwrap()["backend"], "fake");
        let mut seen = String::new();
        let (text, _) = engine
            .chat_stream("m", &[Message::new("user", "hi")], None, None, Some(8), None, Priority::Background, None, |t| seen.push_str(t))
            .unwrap();
        assert_eq!((text.as_str(), seen.as_str()), ("ab", "ab"));
        assert!(RemoteEngine::new("http://127.0.0.1:9", None).unwrap().health().is_err(), "unreachable is an error");
        // `engine` drops here, inside whatever runtime the caller is in.
    }

    /// The bug this guards: reqwest's blocking client panicked when it was
    /// built, used or dropped on a Tokio worker ("Cannot drop a runtime in a
    /// context where blocking is not allowed"), which is where a host's async
    /// commands call it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn works_on_a_multi_thread_runtime_worker() {
        use_it(&fake_daemon());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn works_on_a_current_thread_runtime() {
        use_it(&fake_daemon());
    }

    #[test]
    fn works_outside_any_runtime() {
        use_it(&fake_daemon());
    }
}
