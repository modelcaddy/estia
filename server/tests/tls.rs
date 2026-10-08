//! HTTPS on a LAN bind, against the exit checks of ROADMAP item 4:
//!
//! - the engine answers HTTPS with its own certificate, and a pinned client
//!   and a probe see that certificate;
//! - a pairing and a chat from a paired device cross the wire encrypted: a
//!   recording relay between them never sees the token, the prompt or the
//!   output;
//! - a device that pinned one engine refuses another certificate at the same
//!   address, before it sends anything;
//! - something in between that answers a pairing with its own certificate is
//!   caught both ways: the engine refuses the device's code, an approval
//!   without a code needs the approver's explicit word, and the device refuses
//!   an "approved" that did not come with the engine's confirmation code;
//! - the Python examples pair, pin and chat over HTTPS, computing the same
//!   pairing code as the engine;
//! - plain HTTP from another device is refused (`https_required`), loopback
//!   stays plain, and stalled handshakes are cut off by the usual caps.
//!
//! The chat tests drive the stdlib-only fake runner; they skip without python3.

use estia_engine::models::ModelStore;
use estia_engine::runtime::PythonRuntime;
use estia_engine::tls::Fingerprint;
use estia_engine::{Engine, EngineConfig, PairingOutcome, Priority, RemoteEngine, SessionError};
use estia_server::tls::{self, TlsServe};
use estia_server::tokens::{TokenStore, SCOPE_ADMIN};
use estia_server::{router, serve_router, serve_router_tls, AppState, ConnLimits};
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// python3 3.9 or newer, which the examples need (3.9 is the oldest run, the
/// macOS system Python); older ones skip like a missing python3.
fn python3_available() -> bool {
    std::process::Command::new("python3")
        .args(["-c", "import sys; sys.exit(sys.version_info < (3, 9))"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The chat part of the fake runner in api.rs: it echoes the roles and the
/// last user turn, streamed in two pieces.
const FAKE: &str = r#"import sys, json
def out(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    t = req.get("type")
    if t == "ping":
        out({"ok": True})
    elif t == "hello":
        out({"ok": True, "runner": "fake", "version": "0", "protocol": 2,
             "capabilities": {"generate": True, "stream": True, "embed": True, "cancel": True, "load": True, "chat": True}})
    elif t == "load":
        out({"ok": True, "loaded": True, "ms": 1})
    elif t in ("chat", "chat_stream"):
        msgs = req["messages"]
        text = "roles:" + ",".join(m["role"] for m in msgs) + "|" + msgs[-1]["content"]
        if t == "chat":
            out({"text": text, "meta": {"prompt_tokens": 3, "generation_tokens": 2}})
        else:
            for piece in [text[:5], text[5:]]:
                out({"type": "token", "text": piece})
            out({"type": "meta", "prompt_tokens": 3, "generation_tokens": 2})
            out({"done": True})
    else:
        out({"error": "unknown type %r" % (t,)})
"#;

struct Harness {
    addr: SocketAddr,
    admin: String,
    state: Arc<AppState>,
    fingerprint: Fingerprint,
    dir: std::path::PathBuf,
}

impl Harness {
    fn url(&self) -> String {
        format!("https://{}", self.addr)
    }
}

fn scratch_dir() -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "estia-tls-test-{}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// An engine serving HTTPS on 127.0.0.1 with its own certificate, as `serve`
/// does on a LAN bind. `plain_from_loopback: false` makes this test process
/// a LAN device as far as plain HTTP goes.
async fn start_tls(limits: ConnLimits, plain_from_loopback: bool) -> Harness {
    let dir = scratch_dir();
    std::fs::create_dir_all(dir.join("models/gemma4-e2b-it-4bit-mlx")).unwrap();
    let runner = dir.join("fake_runner.py");
    std::fs::write(&runner, FAKE).unwrap();
    let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), &runner)
        .with_python("python3")
        .with_memory(estia_engine::MemoryPolicy::unlimited());
    let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
    let admin = tokens.mint("admin", &[SCOPE_ADMIN]).unwrap();
    let identity = tls::load_or_create(&dir).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(AppState::new(Arc::new(Engine::new(cfg)), tokens, true, addr));
    state.set_tls(Some(identity.fingerprint()));
    let serve = TlsServe::new(&identity).unwrap().plain_from_loopback(plain_from_loopback);
    let app = router(Arc::clone(&state));
    tokio::spawn(async move {
        serve_router_tls(listener, app, limits, serve, std::future::pending()).await.unwrap();
    });
    Harness { addr, admin, state, fingerprint: identity.fingerprint(), dir }
}

/// A TCP relay that records every byte it forwards, the way a packet capture
/// on the LAN would see them. Its target can be switched, so the client keeps
/// one address while another engine answers behind it.
struct Relay {
    addr: SocketAddr,
    target: Arc<Mutex<SocketAddr>>,
    /// Client → engine, every connection.
    up: Arc<Mutex<Vec<u8>>>,
    /// Engine → client.
    down: Arc<Mutex<Vec<u8>>>,
    /// The first byte of each connection from the client.
    firsts: Arc<Mutex<Vec<u8>>>,
    /// Connections in flight, closed by [`Relay::retarget`].
    live: Arc<Mutex<Vec<tokio::task::AbortHandle>>>,
}

impl Relay {
    async fn start(target: SocketAddr) -> Relay {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let relay = Relay {
            addr: listener.local_addr().unwrap(),
            target: Arc::new(Mutex::new(target)),
            up: Arc::default(),
            down: Arc::default(),
            firsts: Arc::default(),
            live: Arc::default(),
        };
        let (target, up, down, firsts) =
            (Arc::clone(&relay.target), Arc::clone(&relay.up), Arc::clone(&relay.down), Arc::clone(&relay.firsts));
        let live = Arc::clone(&relay.live);
        tokio::spawn(async move {
            while let Ok((client, _)) = listener.accept().await {
                let to = *target.lock().unwrap();
                let (up, down, firsts) = (Arc::clone(&up), Arc::clone(&down), Arc::clone(&firsts));
                let task = tokio::spawn(async move {
                    let Ok(server) = tokio::net::TcpStream::connect(to).await else { return };
                    let (mut cr, mut cw) = client.into_split();
                    let (mut sr, mut sw) = server.into_split();
                    tokio::join!(pump(&mut cr, &mut sw, up, Some(firsts)), pump(&mut sr, &mut cw, down, None));
                });
                live.lock().unwrap().push(task.abort_handle());
            }
        });
        relay
    }

    fn url(&self) -> String {
        format!("https://{}", self.addr)
    }

    /// Send new connections to `to`, and close the ones in flight: the
    /// client's next request needs a new connection, and with it a handshake.
    fn retarget(&self, to: SocketAddr) {
        *self.target.lock().unwrap() = to;
        for t in self.live.lock().unwrap().drain(..) {
            t.abort();
        }
    }

    fn clear(&self) {
        self.up.lock().unwrap().clear();
        self.down.lock().unwrap().clear();
    }

    /// Every recorded byte, both directions.
    fn seen(&self) -> Vec<u8> {
        let mut all = self.up.lock().unwrap().clone();
        all.extend_from_slice(&self.down.lock().unwrap());
        all
    }
}

async fn pump(
    from: &mut tokio::net::tcp::OwnedReadHalf,
    to: &mut tokio::net::tcp::OwnedWriteHalf,
    log: Arc<Mutex<Vec<u8>>>,
    firsts: Option<Arc<Mutex<Vec<u8>>>>,
) {
    let mut buf = [0u8; 16 * 1024];
    let mut first = true;
    loop {
        let n = match from.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        if std::mem::take(&mut first) {
            if let Some(f) = &firsts {
                f.lock().unwrap().push(buf[0]);
            }
        }
        log.lock().unwrap().extend_from_slice(&buf[..n]);
        if to.write_all(&buf[..n]).await.is_err() {
            break;
        }
    }
    let _ = to.shutdown().await;
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

/// Run blocking client code (RemoteEngine, begin_pairing) off the test's runtime.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(f).await.unwrap()
}

/// One raw HTTP/1.1 exchange over plain TCP: (status, headers, body).
async fn raw_http(addr: SocketAddr, request: String) -> (u16, String, String) {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(request.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), s.read_to_end(&mut out)).await;
    let text = String::from_utf8_lossy(&out).to_string();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    (status, head.to_lowercase(), body.to_string())
}

/// "`estia serve --lan` answers HTTPS on the LAN address": the engine's
/// certificate is what a pinned client trusts and what a probe reports, and
/// health names it. A loopback client still gets plain HTTP on the same port.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn serves_https_with_its_certificate() {
    let h = start_tls(ConnLimits::default(), true).await;
    let (url, fp) = (h.url(), h.fingerprint);
    let (health, probed) = blocking(move || {
        let engine = RemoteEngine::with_pin(&url, None, fp).unwrap();
        (engine.health().unwrap(), estia_engine::probe_certificate(&url).unwrap())
    })
    .await;
    assert_eq!(health["tls"]["fingerprint"], fp.to_hex(), "{health}");
    assert_eq!(health["tls"]["short"], fp.short());
    assert!(probed.matches(&fp));
    assert_eq!(Some(fp), tls::load(&h.dir).unwrap().map(|i| i.fingerprint()), "the certificate on disk");

    // Loopback stays plain HTTP, on the same port.
    let r = reqwest::get(format!("http://{}/engine/health", h.addr)).await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["tls"]["fingerprint"], fp.to_hex(), "a local app can read the fingerprint to show it");

    // An engine without TLS says so.
    let dir = scratch_dir();
    let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), dir.join("r.py"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(AppState::new(Arc::new(Engine::new(cfg)), TokenStore::open(dir.join("tokens.json")).unwrap(), true, addr));
    let app = router(state);
    tokio::spawn(async move { serve_router(listener, app, ConnLimits::default(), std::future::pending()).await.unwrap() });
    let v: Value = reqwest::get(format!("http://{addr}/engine/health")).await.unwrap().json().await.unwrap();
    assert!(v["tls"].is_null(), "{v}");
}

/// "A packet capture of a chat from a paired device shows no token, prompt or
/// output in clear text": the whole pairing (request, reveal, approval with
/// the code, collecting the token) and a streamed chat go through a relay
/// that records every byte.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pairing_and_chat_cross_the_wire_encrypted() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start_tls(ConnLimits::default(), true).await;
    let relay = Relay::start(h.addr).await;
    let url = relay.url();

    let u = url.clone();
    let pending = blocking(move || estia_engine::begin_pairing(&u, "phone", &["generate"]).unwrap()).await;
    let code = *pending.code().expect("a pairing over https has a code");
    assert!(pending.certificate().is_some_and(|c| c.matches(&h.fingerprint)), "the device saw the engine's certificate");
    assert_eq!(pending.approve_command(), format!("estia pair approve {} --code {}", pending.id(), code.digits()));
    // The row the operator sees carries the same code, computed from the
    // engine's own certificate.
    let row = h.state.pairings.list().into_iter().find(|p| p.id == pending.id()).unwrap();
    assert_eq!(row.code.as_deref(), Some(code.digits().as_str()));
    let approved = h.state.pairings.approve_with_code(pending.id(), &h.state.tokens, Some(&code.to_string()), false).unwrap();
    let confirmation = approved.confirm_shown().expect("the approval gives the confirmation code to read to the user");
    let p = pending.clone();
    let outcome = blocking(move || p.poll().unwrap()).await;
    let PairingOutcome::Approved { token } = outcome else { panic!("not approved: {outcome:?}") };
    assert!(token.starts_with("estia_"));
    // The user types the confirmation code into the device, which pins the
    // certificate only now.
    let pin = pending.confirm(&confirmation).expect("the engine's confirmation code matches the device's");
    assert_eq!(pin, h.fingerprint);

    let prompt = format!("PROMPT-{}", std::process::id() * 7919);
    let (u, t, pr) = (url.clone(), token.clone(), prompt.clone());
    let text = blocking(move || {
        let engine = RemoteEngine::with_pin(&u, Some(t), pin).unwrap();
        let msgs = [estia_engine::proto::Message::new("user", pr)];
        engine.chat_stream("fast", &msgs, None, None, Some(16), None, Priority::Interactive, None, |_| {}).unwrap().0
    })
    .await;
    assert_eq!(text, format!("roles:user|{prompt}"), "the chat really happened");

    let seen = relay.seen();
    assert!(!relay.up.lock().unwrap().is_empty() && !relay.down.lock().unwrap().is_empty(), "the relay carried both directions");
    let firsts = relay.firsts.lock().unwrap().clone();
    assert!(!firsts.is_empty() && firsts.iter().all(|b| *b == 0x16), "every connection opened with a TLS handshake: {firsts:?}");
    for clear in [token.as_bytes(), prompt.as_bytes(), b"roles:user", b"estia_", b"/engine/", b"Authorization", b"authorization"] {
        assert!(!contains(&seen, clear), "`{}` crossed the wire in clear", String::from_utf8_lossy(clear));
    }
}

/// "A device that paired with one engine refuses a server with a different
/// certificate at the same address": the relay keeps the address and switches
/// the engine behind it. The device refuses in the handshake, so its token
/// never leaves it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pinned_device_refuses_another_certificate() {
    let a = start_tls(ConnLimits::default(), true).await;
    let b = start_tls(ConnLimits::default(), true).await;
    assert_ne!(a.fingerprint, b.fingerprint);
    let relay = Relay::start(a.addr).await;
    let (url, token, pin) = (relay.url(), a.admin.clone(), a.fingerprint);
    let engine = Arc::new(blocking(move || RemoteEngine::with_pin(url, Some(token), pin).unwrap()).await);
    let e = Arc::clone(&engine);
    assert!(blocking(move || e.health()).await.is_ok(), "the paired engine");

    relay.retarget(b.addr);
    // Let the client's pool notice its connection closed.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    relay.clear();
    let e = Arc::clone(&engine);
    let (health, models) = blocking(move || (e.health(), e.get_json("/engine/models"))).await;
    for r in [health.map(drop), models.map(drop)] {
        match r {
            Err(SessionError::CertificateMismatch { pinned, presented, .. }) => {
                assert_eq!((pinned, presented), (a.fingerprint, b.fingerprint));
            }
            other => panic!("expected a certificate mismatch, got {other:?}"),
        }
    }
    let up = relay.up.lock().unwrap().clone();
    assert!(!up.is_empty(), "the device did reach the other engine");
    assert!(!contains(&up, a.admin.as_bytes()) && !contains(&up, b"uthorization"), "the token never left the device");
    let url = relay.url();
    let probed = blocking(move || estia_engine::probe_certificate(&url).unwrap()).await;
    assert!(probed.matches(&b.fingerprint), "a probe now reports the other certificate");
}

/// Plain HTTP from another device is answered with `https_required` before
/// any handler runs (nothing is acted on), the browser page redirects, and
/// TLS on the same port still works.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn plain_http_from_another_device_is_refused() {
    let h = start_tls(ConnLimits::default(), false).await;
    let body = r#"{"name":"old client","scopes":["generate"]}"#;
    let (status, head, text) = raw_http(
        h.addr,
        format!(
            "POST /engine/pair HTTP/1.1\r\nHost: {}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            h.addr,
            h.admin,
            body.len()
        ),
    )
    .await;
    assert_eq!(status, 400, "{head}\n{text}");
    assert!(head.contains("connection: close"), "{head}");
    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    assert_eq!(v["error"]["type"], "https_required", "{text}");
    assert!(v["error"]["request_id"].is_string(), "{text}");
    assert!(v["error"]["message"].as_str().unwrap().contains(&format!("https://{}", h.addr)), "{text}");
    assert!(h.state.pairings.list().is_empty(), "the refused request filed nothing");

    let (status, head, _) = raw_http(h.addr, format!("GET /client HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", h.addr)).await;
    assert_eq!(status, 307, "{head}");
    assert!(head.contains(&format!("location: https://{}/client", h.addr)), "{head}");
    let (status, _, text) = raw_http(h.addr, format!("GET /v1/models HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", h.addr)).await;
    assert_eq!(status, 400, "every route, open or not: {text}");

    let (url, fp, token) = (h.url(), h.fingerprint, h.admin.clone());
    let v = blocking(move || RemoteEngine::with_pin(url, Some(token), fp).unwrap().get_json("/engine/pairings").unwrap()).await;
    assert!(v["pairings"].as_array().unwrap().is_empty(), "TLS on the same port is served: {v}");
}

/// Slowloris during the handshake: connections that start a TLS handshake
/// and stop hold their peer's share only until the header timeout, the cap
/// still closes the one over it at once, and a pinned client gets through
/// afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stalled_tls_handshakes_are_cut_off() {
    let limits =
        ConnLimits { header_read_timeout: std::time::Duration::from_millis(800), per_peer: 4, exempt_loopback: false, total: Some(64) };
    let h = start_tls(limits, true).await;
    async fn stalled(addr: SocketAddr) -> tokio::net::TcpStream {
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        s.write_all(&[0x16, 0x03, 0x01]).await.unwrap();
        s
    }
    async fn closed_within(s: &mut tokio::net::TcpStream, ms: u64) -> bool {
        let mut buf = [0u8; 512];
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(ms);
        loop {
            match tokio::time::timeout_at(deadline, s.read(&mut buf)).await {
                Err(_) => return false,
                Ok(Ok(0)) | Ok(Err(_)) => return true,
                Ok(Ok(_)) => continue,
            }
        }
    }
    let mut held = Vec::new();
    for _ in 0..4 {
        held.push(stalled(h.addr).await);
    }
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let mut fifth = stalled(h.addr).await;
    assert!(closed_within(&mut fifth, 300).await, "over the per-peer cap: closed at once");
    for s in held.iter_mut() {
        assert!(closed_within(s, 2_000).await, "a handshake that never ends is cut off");
    }
    let mut silent = tokio::net::TcpStream::connect(h.addr).await.unwrap();
    assert!(closed_within(&mut silent, 2_000).await, "a connection that sends nothing is cut off too");
    let (url, fp) = (h.url(), h.fingerprint);
    assert!(blocking(move || RemoteEngine::with_pin(url, None, fp).unwrap().health()).await.is_ok(), "the share is back");
}

/// "`estia remote-check`, remote_client.rs and the Python examples work with
/// the pinned certificate": `pair.py` pairs over HTTPS (recording the
/// certificate, committing, revealing) and prints the code it computed; the
/// engine accepts that code, which proves the Python code computes the same
/// one as Rust; then `no_sdk.py` chats with the token and the pin, and refuses
/// a wrong pin.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn python_examples_pair_and_chat_over_tls() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start_tls(ConnLimits::default(), true).await;
    let examples = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/python");
    let token_file = h.dir.join("token");
    let mut pair = tokio::process::Command::new("python3")
        .arg("pair.py")
        .args(["--name", "ci", "--scopes", "generate", "--out"])
        .arg(&token_file)
        .current_dir(&examples)
        .env("ESTIA_URL", h.url())
        .env_remove("ESTIA_PIN")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    // Read until the approve command, which carries the id and the code.
    let mut out = pair.stdout.take().unwrap();
    let mut printed = String::new();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
    let (id, code) = loop {
        let mut buf = [0u8; 1024];
        let n = tokio::time::timeout_at(deadline, out.read(&mut buf)).await.expect("pair.py printed the approve command").unwrap();
        assert!(n > 0, "pair.py ended early:\n{printed}");
        printed.push_str(&String::from_utf8_lossy(&buf[..n]));
        let found = printed.lines().find_map(|l| {
            let rest = l.trim().strip_prefix("estia pair approve ")?;
            let (id, code) = rest.split_once(" --code ")?;
            Some((id.trim().to_string(), code.trim().to_string()))
        });
        if let Some(f) = found {
            break f;
        }
    };
    let approved =
        h.state.pairings.approve_with_code(&id, &h.state.tokens, Some(&code), false).expect("the engine accepts the code pair.py computed");
    // What `estia pair approve` prints, typed into pair.py: it keeps the
    // token only once this matches the code it computed itself.
    let confirmation = approved.confirm_shown().expect("a confirmation code");
    let mut stdin = pair.stdin.take().unwrap();
    stdin.write_all(format!("{confirmation}\n").as_bytes()).await.unwrap();
    drop(stdin);
    let status = tokio::time::timeout(std::time::Duration::from_secs(20), pair.wait()).await.expect("pair.py finished").unwrap();
    let mut rest = String::new();
    out.read_to_string(&mut rest).await.unwrap();
    printed.push_str(&rest);
    assert!(status.success(), "pair.py failed:\n{printed}");
    let token = std::fs::read_to_string(&token_file).unwrap().trim().to_string();
    assert!(token.starts_with("estia_"));
    assert!(printed.contains(&format!("ESTIA_PIN={}", h.fingerprint.to_hex())), "pair.py prints the pin to export:\n{printed}");

    let chat = |pin: String| {
        let mut c = tokio::process::Command::new("python3");
        c.arg("no_sdk.py")
            .current_dir(&examples)
            .env("ESTIA_URL", h.url())
            .env("ESTIA_TOKEN", &token)
            .env("ESTIA_PIN", pin)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        c
    };
    for (pin, works) in [(h.fingerprint.to_hex(), true), (Fingerprint::of_certificate(b"another engine").to_hex(), false)] {
        let mut child = chat(pin).spawn().unwrap();
        child.stdin.take().unwrap().write_all(b"hello over tls\n/quit\n").await.unwrap();
        let o =
            tokio::time::timeout(std::time::Duration::from_secs(30), child.wait_with_output()).await.expect("no_sdk.py finished").unwrap();
        let (stdout, stderr) = (String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
        if works {
            assert!(o.status.success(), "{stdout}\n{stderr}");
            assert!(stdout.contains("roles:system,user|hello over tls"), "{stdout}\n{stderr}");
        } else {
            assert!(!o.status.success(), "a wrong pin must fail:\n{stdout}\n{stderr}");
            // As a sentence, not the tuple ssl.SSLError prints by default.
            assert!(stderr.contains("but this app pinned") && !stderr.contains("('"), "says why: {stderr}");
        }
    }
}

/// What the interceptor in [`an_interceptor_is_caught_both_ways`] does.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mitm {
    /// Pass the device's commitment and nonce on to the engine unchanged.
    Forward,
    /// Remove the commitment on the way, so the engine sees a request with no
    /// code, and make up an engine nonce for the device.
    Strip,
    /// Answer the device's polls itself: "approved", with a token of its own.
    FakeApproval,
}

struct Interceptor {
    engine: Arc<AppState>,
    engine_fp: Fingerprint,
    own_fp: Fingerprint,
    mode: Mutex<Mitm>,
}

impl Interceptor {
    fn mode(&self) -> Mitm {
        *self.mode.lock().unwrap()
    }

    /// What bodies say about the certificate: its own, as a careful
    /// interceptor would, so the device's cross-check of the body passes.
    fn tls(&self) -> Value {
        serde_json::json!({"fingerprint": self.own_fp.to_hex(), "short": self.own_fp.short()})
    }
}

/// Something on the path between a device and the engine (ARP spoofing on
/// home Wi-Fi): it ends the device's TLS with a certificate of its own
/// (`identity`), which a first connection cannot tell from the engine's, and
/// relays the pairing to the engine (here by calling the engine's pairing
/// store as the engine's own routes do, with the engine's fingerprint, which
/// is what relaying each request over TLS amounts to). Returns its URL.
async fn start_interceptor(m: Arc<Interceptor>, identity: &tls::TlsIdentity) -> String {
    use axum::extract::{Path, State};
    use axum::routing::{get, post};
    use axum::Json;
    type St = State<Arc<Interceptor>>;
    async fn pair(State(m): St, Json(body): Json<Value>) -> (axum::http::StatusCode, Json<Value>) {
        let name = body["name"].as_str().unwrap_or("?").to_string();
        let scopes: Vec<String> = body["scopes"].as_array().into_iter().flatten().filter_map(|s| s.as_str().map(String::from)).collect();
        let strip = m.mode() == Mitm::Strip;
        let commitment = body["commitment"].as_str().filter(|_| !strip).map(String::from);
        let st = Arc::clone(&m.engine);
        let p =
            tokio::task::spawn_blocking(move || st.pairings.request_tls(&name, &scopes, Some("10.0.0.66".into()), commitment.as_deref()))
                .await
                .unwrap()
                .unwrap();
        let engine_nonce = match strip {
            true => estia_engine::tls::nonce_hex(&estia_engine::tls::random_nonce().unwrap()),
            false => p.engine_nonce.clone().unwrap(),
        };
        let answer = serde_json::json!({"id": p.id, "status": "pending", "expires_in": 300, "engine_nonce": engine_nonce, "tls": m.tls()});
        (axum::http::StatusCode::ACCEPTED, Json(answer))
    }
    async fn reveal(State(m): St, Path(id): Path<String>, Json(body): Json<Value>) -> Json<Value> {
        if m.mode() != Mitm::Strip {
            let (st, fp, nonce, i) = (Arc::clone(&m.engine), m.engine_fp, body["nonce"].as_str().unwrap_or("").to_string(), id.clone());
            tokio::task::spawn_blocking(move || st.pairings.reveal(&i, &nonce, &fp)).await.unwrap().unwrap();
        }
        Json(serde_json::json!({"id": id, "status": "pending", "check": "code"}))
    }
    async fn poll(State(m): St, Path(id): Path<String>) -> Json<Value> {
        if m.mode() == Mitm::FakeApproval {
            return Json(serde_json::json!({"id": id, "status": "approved", "token": "estia_made_up_by_the_interceptor", "tls": m.tls()}));
        }
        let (st, i) = (Arc::clone(&m.engine), id.clone());
        let (status, token) = tokio::task::spawn_blocking(move || st.pairings.poll(&i)).await.unwrap().unwrap();
        Json(serde_json::json!({"id": id, "status": status, "token": token, "tls": m.tls()}))
    }
    let app = axum::Router::new()
        .route("/engine/pair", post(pair))
        .route("/engine/pair/:id/reveal", post(reveal))
        .route("/engine/pair/:id", get(poll))
        .with_state(m);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let serve = TlsServe::new(identity).unwrap();
    tokio::spawn(async move { serve_router_tls(listener, app, ConnLimits::default(), serve, std::future::pending()).await.unwrap() });
    format!("https://{addr}")
}

/// A pairing intercepted by something that presents its own certificate is
/// caught both ways, whatever the interceptor does:
/// - it relays the codes: the engine refuses the code the device shows;
/// - it answers the device's polls itself: the device refuses, because no
///   confirmation code the engine's operator can read matches its own;
/// - it strips the commitment, so the engine sees a request with no code:
///   no approval path takes that without the approver's explicit word (the
///   HTTP route's `no_code`, the CLI's `--no-code`), and even then the device
///   has no confirmation code to accept, so it never pins the interceptor.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_interceptor_is_caught_both_ways() {
    use estia_server::pairing::{PairingCheck, PairingStatus};
    let h = start_tls(ConnLimits::default(), true).await;
    let identity = tls::load_or_create(&scratch_dir()).unwrap();
    let m = Arc::new(Interceptor {
        engine: Arc::clone(&h.state),
        engine_fp: h.fingerprint,
        own_fp: identity.fingerprint(),
        mode: Mutex::new(Mitm::Forward),
    });
    let url = start_interceptor(Arc::clone(&m), &identity).await;
    let row = |id: &str| h.state.pairings.list().into_iter().find(|p| p.id == id).unwrap();
    let minted = || TokenStore::open(h.dir.join("tokens.json")).unwrap().list().into_iter().filter(|r| r.name.starts_with("pair:")).count();

    // 1. Relayed: the device's code comes from the interceptor's certificate,
    //    the engine's from its own. The operator types the device's: refused.
    let u = url.clone();
    let device = blocking(move || estia_engine::begin_pairing(&u, "phone", &["generate"]).unwrap()).await;
    assert!(device.certificate().unwrap().matches(&m.own_fp), "the device met the interceptor's certificate");
    assert_eq!(row(device.id()).check(), PairingCheck::Code);
    let shown = device.code().unwrap().digits();
    let e = h.state.pairings.approve_with_code(device.id(), &h.state.tokens, Some(&shown), false).unwrap_err();
    assert!(e.to_string().contains("does not match"), "{e}");
    assert_eq!((row(device.id()).status, minted()), (PairingStatus::Pending, 0), "nothing minted");

    // 2. The interceptor answers the device's polls itself. The device hears
    //    "approved", but no code from the engine's side confirms it: not the
    //    confirmation code the engine holds for this request, not any other.
    *m.mode.lock().unwrap() = Mitm::FakeApproval;
    let d = device.clone();
    let outcome = blocking(move || d.poll().unwrap()).await;
    assert_eq!(outcome, PairingOutcome::Approved { token: "estia_made_up_by_the_interceptor".into() });
    let engines = row(device.id()).confirm.expect("the engine computed a confirmation code on reveal");
    for typed in [engines.as_str(), "000000", shown.as_str()] {
        let e = device.confirm(typed).unwrap_err().to_string();
        assert!(e.contains("not the confirmation code"), "{typed}: {e}");
    }

    // 3. Stripped: the engine sees a request over https with no code, while
    //    the device shows one, made from a nonce the interceptor made up.
    *m.mode.lock().unwrap() = Mitm::Strip;
    let u = url.clone();
    let device = blocking(move || estia_engine::begin_pairing(&u, "laptop", &["generate"]).unwrap()).await;
    assert!(device.code().is_some());
    let id = device.id().to_string();
    assert_eq!(row(&id).check(), PairingCheck::Uncoded);
    // No approval without the approver's explicit word: not the store's
    // `approve`, not the HTTP route with `{}` (`decide_pairing(id, true)`,
    // `approve_pairing(id, None)`) or no body at all, and not the code the
    // device shows either.
    assert!(h.state.pairings.approve(&id, &h.state.tokens).is_err());
    let (e_url, admin, fp) = (h.url(), h.admin.clone(), h.fingerprint);
    let admin_client = Arc::new(blocking(move || RemoteEngine::with_pin(e_url, Some(admin), fp).unwrap()).await);
    let (a, i, c) = (Arc::clone(&admin_client), id.clone(), device.code().unwrap().digits());
    let (decided, without_body, with_code) =
        blocking(move || (a.decide_pairing(&i, true), a.approve_pairing(&i, None), a.approve_pairing(&i, Some(&c)))).await;
    for r in [decided, without_body] {
        let e = r.unwrap_err().to_string();
        assert!(e.contains("no code") && e.contains("deny this request") && e.contains("no_code"), "{e}");
    }
    assert!(with_code.unwrap_err().to_string().contains("different request"));
    // Raw HTTP: no body, and a code with no_code, are refused too.
    let raw = reqwest::Client::builder().danger_accept_invalid_certs(true).build().unwrap();
    let approve_url = format!("{}/engine/pairings/{id}/approve", h.url());
    for body in [None, Some(r#"{"code":"123456","no_code":true}"#)] {
        let mut r = raw.post(&approve_url).bearer_auth(&h.admin);
        if let Some(b) = body {
            r = r.header("content-type", "application/json").body(b);
        }
        let r = r.send().await.unwrap();
        assert_eq!(r.status().as_u16(), 400, "{body:?}");
    }
    assert_eq!((row(&id).status, minted()), (PairingStatus::Pending, 0), "nothing minted");
    // An approver who says explicitly there is no code gets what they asked
    // for, but the engine has no confirmation code to give, so the device,
    // which shows a code, cannot be told one and never pins the interceptor.
    let (a, i) = (Arc::clone(&admin_client), id.clone());
    let v = blocking(move || a.approve_pairing_without_code(&i).unwrap()).await;
    assert!(v["confirm"].is_null(), "{v}");
    *m.mode.lock().unwrap() = Mitm::Forward;
    let d = device.clone();
    assert!(matches!(blocking(move || d.poll().unwrap()).await, PairingOutcome::Approved { .. }));
    assert!(device.confirm("000000").is_err());
}

/// estia_tls.py's own checks: the known-answer vectors for both codes, a
/// pinned context that refuses to start without a pin, and plain http only to
/// this machine.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn python_helper_checks_itself() {
    if !python3_available() {
        eprintln!("skip: python3 3.9 or newer not available");
        return;
    }
    let examples = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/python");
    let o =
        tokio::process::Command::new("python3").arg("estia_tls.py").current_dir(&examples).env_remove("ESTIA_PIN").output().await.unwrap();
    assert!(o.status.success(), "{}\n{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
}
