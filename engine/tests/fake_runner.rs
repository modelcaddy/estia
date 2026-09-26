//! Session behaviour against stdlib-only Python fake runners (no MLX). Skipped
//! when `python3` is not on PATH.

use estia_engine::proto::{EmbedBatchResp, GenerateResp, PingResp, Request};
use estia_engine::{Launch, NoopObserver, Session, SessionConfig, SessionError, SessionObserver};
use std::io::Write as _;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn python3_available() -> bool {
    Command::new("python3").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn write_runner(tag: &str, body: &str) -> String {
    let src = format!(
        r#"import sys, json, time, os
def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            t = req.get("type")
{body}
        except Exception as e:
            sys.stdout.write(json.dumps({{"error": str(e)}}) + "\n")
        sys.stdout.flush()
if __name__ == "__main__":
    main()
"#
    );
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("estia_fake_{tag}_{}_{n}.py", std::process::id()));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(src.as_bytes()).unwrap();
    path.to_string_lossy().into_owned()
}

fn spawn(path: &str, timeout: Option<Duration>) -> Session {
    spawn_with(path, timeout, Arc::new(NoopObserver))
}

fn spawn_with(path: &str, timeout: Option<Duration>, obs: Arc<dyn SessionObserver>) -> Session {
    let mut cfg = SessionConfig::default();
    if let Some(t) = timeout {
        cfg.call_timeout = t;
    }
    Session::spawn(Launch::new("python3").arg(path), cfg, obs).unwrap()
}

const ECHO: &str = r#"
            if t == "ping":
                resp = {"ok": True}
            elif t == "embed_batch":
                resp = {"embeddings": [[1.0, 0.0] for _ in req["inputs"]]}
            elif t == "embed":
                resp = {"embedding": [1.0, 0.0]}
            elif t == "generate":
                resp = {"text": "hello"}
            elif t == "generate_stream":
                sys.stdout.write(json.dumps({"type": "keepalive"}) + "\n")
                for tok in ["hel", "lo ", "world"]:
                    sys.stdout.write(json.dumps({"type": "token", "text": tok}) + "\n")
                sys.stdout.write("loading chatter that is not json\n")
                sys.stdout.write(json.dumps({"done": True}) + "\n")
                sys.stdout.flush()
                continue
            else:
                resp = {"error": "unknown type %r" % (t,)}
            sys.stdout.write(json.dumps(resp) + "\n")
"#;

#[test]
fn ping_embed_generate_and_stream() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let path = write_runner("echo", ECHO);
    let s = spawn(&path, None);

    let ping: PingResp = serde_json::from_value(s.call_unobserved(&Request::Ping).unwrap()).unwrap();
    assert!(ping.ok);
    assert_eq!(s.in_flight(), 0);

    let inputs = vec!["a".to_string(), "b".to_string()];
    let e: EmbedBatchResp = s.call_typed(&Request::EmbedBatch { model_path: "/m", inputs: &inputs }).unwrap();
    assert_eq!(e.embeddings, vec![vec![1.0, 0.0], vec![1.0, 0.0]]);

    let g: GenerateResp = s
        .call_typed(&Request::Generate { model_path: "/m", prompt: "p", max_tokens: Some(8), temperature: Some(0.0), json: None })
        .unwrap();
    assert_eq!(g.text, "hello");

    let mut toks = Vec::new();
    let full = s
        .stream(&Request::GenerateStream { model_path: "/m", prompt: "p", max_tokens: Some(8), temperature: Some(0.0) }, |t| {
            toks.push(t.to_string())
        })
        .unwrap();
    assert_eq!(toks, vec!["hel", "lo ", "world"]);
    assert_eq!(full, "hello world");
    // Keepalive and non-JSON chatter were skipped; the session is reusable.
    let full2 =
        s.stream(&Request::GenerateStream { model_path: "/m", prompt: "p", max_tokens: Some(8), temperature: Some(0.0) }, |_| {}).unwrap();
    assert_eq!(full2, "hello world");
    assert_eq!(s.in_flight(), 0);

    // Unknown request type → runner error envelope → typed error.
    let err = s.call(&Request::AppleHealth).unwrap_err();
    assert!(matches!(err, SessionError::Runner(_)), "{err}");

    let _ = std::fs::remove_file(&path);
}

const LEGACY_STREAM: &str = r#"
            if t == "ping":
                resp = {"ok": True}
            elif t == "generate_stream":
                resp = {"text": "all at once"}
            else:
                resp = {"error": "unknown"}
            sys.stdout.write(json.dumps(resp) + "\n")
"#;

#[test]
fn legacy_single_envelope_stream_is_one_token() {
    if !python3_available() {
        return;
    }
    let path = write_runner("legacy", LEGACY_STREAM);
    let s = spawn(&path, None);
    let mut toks = Vec::new();
    let full = s
        .stream(&Request::GenerateStream { model_path: "/m", prompt: "p", max_tokens: None, temperature: None }, |t| {
            toks.push(t.to_string())
        })
        .unwrap();
    assert_eq!(toks, vec!["all at once"]);
    assert_eq!(full, "all at once");
    let _ = std::fs::remove_file(&path);
}

const MIDFAIL: &str = r#"
            if t == "ping":
                resp = {"ok": True}
            elif t == "generate_stream":
                sys.stdout.write(json.dumps({"type": "token", "text": "par"}) + "\n")
                sys.stdout.write(json.dumps({"error": "gpu fell over"}) + "\n")
                sys.stdout.flush()
                continue
            else:
                resp = {"error": "unknown"}
            sys.stdout.write(json.dumps(resp) + "\n")
"#;

#[test]
fn midstream_error_surfaces_without_retry() {
    if !python3_available() {
        return;
    }
    let path = write_runner("midfail", MIDFAIL);
    let s = spawn(&path, None);
    let pid = s.pid();
    let mut toks = Vec::new();
    let res = s.stream(&Request::GenerateStream { model_path: "/m", prompt: "p", max_tokens: None, temperature: None }, |t| {
        toks.push(t.to_string())
    });
    assert_eq!(toks, vec!["par"], "tokens before the failure are delivered");
    assert!(matches!(res, Err(SessionError::Runner(ref m)) if m == "gpu fell over"), "{res:?}");
    assert_eq!(s.pid(), pid, "a mid-stream error must not respawn");
    assert_eq!(s.in_flight(), 0);
    let _ = std::fs::remove_file(&path);
}

const HANG: &str = r#"
            if t == "ping":
                resp = {"ok": True}
            elif t in ("embed_batch", "generate", "generate_stream"):
                time.sleep(3)
                resp = {"text": "late"}
            else:
                resp = {"error": "unknown"}
            sys.stdout.write(json.dumps(resp) + "\n")
"#;

#[test]
fn wedged_child_trips_the_deadline_and_is_respawned() {
    if !python3_available() {
        return;
    }
    let path = write_runner("hang", HANG);
    let s = spawn(&path, Some(Duration::from_millis(300)));
    s.call_unobserved(&Request::Ping).unwrap(); // fast path unaffected
    let pid_before = s.pid();

    let start = Instant::now();
    let res = s.call(&Request::Generate { model_path: "/m", prompt: "p", max_tokens: None, temperature: None, json: None });
    let elapsed = start.elapsed();
    assert!(res.is_err(), "a wedged runner must error, not hang");
    assert!(matches!(res, Err(SessionError::AfterRespawn(_))), "{res:?}");
    // ~2×300 ms plus a respawn; the bound only has to tell "deadline fired" from
    // "hung", with room for a slow CI machine starting Python.
    assert!(elapsed < Duration::from_secs(5), "deadline (~2×300ms incl. respawn), got {elapsed:?}");
    assert_ne!(s.pid(), pid_before, "the wedged child was replaced");

    // Streams use the same deadline as a silence timeout.
    let start = Instant::now();
    let res = s.stream(&Request::GenerateStream { model_path: "/m", prompt: "p", max_tokens: None, temperature: None }, |_| {});
    assert!(matches!(res, Err(SessionError::StreamSilence { .. })), "{res:?}");
    assert!(start.elapsed() < Duration::from_secs(5));
    assert_eq!(s.in_flight(), 0);
    let _ = std::fs::remove_file(&path);
}

const DIE_AFTER_ONE: &str = r#"
            if t == "ping":
                resp = {"ok": True}
            elif t == "generate":
                sys.stdout.write(json.dumps({"text": "once"}) + "\n")
                sys.stdout.flush()
                os._exit(0)
            else:
                resp = {"error": "unknown"}
            sys.stdout.write(json.dumps(resp) + "\n")
"#;

struct CountingObserver {
    spawns: AtomicUsize,
    starts: AtomicUsize,
    ends: AtomicUsize,
}
impl SessionObserver for CountingObserver {
    fn on_spawn(&self, _pid: u32) {
        self.spawns.fetch_add(1, Ordering::SeqCst);
    }
    fn on_call_start(&self) -> Option<Box<dyn std::any::Any + Send>> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        None
    }
    fn on_call_end(&self, _elapsed: Duration) {
        self.ends.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn dead_child_is_respawned_once_and_observer_sees_it() {
    if !python3_available() {
        return;
    }
    let path = write_runner("die", DIE_AFTER_ONE);
    let obs = Arc::new(CountingObserver { spawns: AtomicUsize::new(0), starts: AtomicUsize::new(0), ends: AtomicUsize::new(0) });
    let s = spawn_with(&path, None, obs.clone());
    assert_eq!(obs.spawns.load(Ordering::SeqCst), 1);

    let req = Request::Generate { model_path: "/m", prompt: "p", max_tokens: None, temperature: None, json: None };
    let g: GenerateResp = s.call_typed(&req).unwrap();
    assert_eq!(g.text, "once");
    // The child exited right after answering. The next call finds a broken
    // pipe or EOF, respawns once, and succeeds.
    let g: GenerateResp = s.call_typed(&req).unwrap();
    assert_eq!(g.text, "once");
    assert_eq!(obs.spawns.load(Ordering::SeqCst), 2, "exactly one respawn");
    assert_eq!(obs.starts.load(Ordering::SeqCst), 2);
    assert_eq!(obs.ends.load(Ordering::SeqCst), 2);

    // Idle teardown is gated and then works; the next call respawns again.
    assert!(!s.is_idle(Duration::from_secs(3600)));
    assert!(s.maybe_shutdown(Duration::from_millis(0)));
    let g: GenerateResp = s.call_typed(&req).unwrap();
    assert_eq!(g.text, "once");
    assert_eq!(obs.spawns.load(Ordering::SeqCst), 3);
    let _ = std::fs::remove_file(&path);
}

/// A runner that answers every line with `{"ok": true}`. On SIGTERM it writes
/// "stopped" to `marker` and exits, or, with `ignore_term`, carries on.
#[cfg(unix)]
fn write_term_runner(tag: &str, marker: &std::path::Path, ignore_term: bool) -> String {
    let on_term = if ignore_term { "signal.SIG_IGN" } else { "on_term" };
    let src = format!(
        r#"import json, os, signal, sys
def on_term(sig, frame):
    with open({marker:?}, "w") as f:
        f.write("stopped")
    os._exit(0)
signal.signal(signal.SIGTERM, {on_term})
for line in sys.stdin:
    sys.stdout.write(json.dumps({{"ok": True}}) + "\n")
    sys.stdout.flush()
"#,
        marker = marker.to_string_lossy()
    );
    let path = std::env::temp_dir().join(format!("estia_fake_{tag}_{}.py", std::process::id()));
    std::fs::write(&path, src).unwrap();
    path.to_string_lossy().into_owned()
}

/// Dropping a session sends the runner SIGTERM first, so it can clean up
/// after itself (the llama adapter stops its llama-server and removes its
/// socket), and kills it only if it is still there after a grace period.
#[cfg(unix)]
#[test]
fn drop_asks_the_runner_to_stop_then_kills_it() {
    if !python3_available() {
        return;
    }
    let marker = std::env::temp_dir().join(format!("estia_fake_term_{}.marker", std::process::id()));
    let _ = std::fs::remove_file(&marker);
    let path = write_term_runner("term", &marker, false);
    let s = spawn(&path, None);
    // The answer means the handler is installed.
    s.call_unobserved(&Request::Ping).unwrap();
    let t = Instant::now();
    drop(s);
    assert_eq!(std::fs::read_to_string(&marker).ok().as_deref(), Some("stopped"), "the runner saw SIGTERM");
    assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());
    let _ = std::fs::remove_file(&marker);
    let _ = std::fs::remove_file(&path);

    // A runner that ignores SIGTERM is killed after the grace period.
    let path = write_term_runner("noterm", &marker, true);
    let s = spawn(&path, None);
    s.call_unobserved(&Request::Ping).unwrap();
    let t = Instant::now();
    drop(s);
    let took = t.elapsed();
    assert!(took >= Duration::from_secs(3) && took < Duration::from_secs(15), "{took:?}");
    assert!(!marker.exists());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn spawn_failure_is_typed() {
    let err = Session::spawn(Launch::new("/definitely/not/a/program"), SessionConfig::default(), Arc::new(NoopObserver)).err().unwrap();
    assert!(matches!(err, SessionError::Spawn { .. }), "{err}");
}

// ── Priority and cancel ──────────────────────────────────────────────────────

/// Slow generate (250 ms); streams four tokens 300 ms apart; honours cancel by
/// reading stdin on a thread and ending the stream with done+cancelled.
fn write_slow_cancellable_runner() -> String {
    let src = r#"import sys, json, time, threading, queue
q = queue.Queue()
cancel = threading.Event()
def reader():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        if req.get("type") == "cancel":
            cancel.set()
        else:
            q.put(req)
    q.put(None)
threading.Thread(target=reader, daemon=True).start()
def out(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
while True:
    req = q.get()
    if req is None:
        break
    cancel.clear()
    t = req.get("type")
    if t == "ping":
        out({"ok": True})
    elif t == "generate":
        time.sleep(0.25)
        out({"text": "slow:" + req.get("prompt", "")})
    elif t == "generate_stream":
        for tok in ["one ", "two ", "three ", "four "]:
            for _ in range(6):
                time.sleep(0.05)
                if cancel.is_set():
                    break
            if cancel.is_set():
                out({"done": True, "cancelled": True})
                break
            out({"type": "token", "text": tok})
        else:
            out({"done": True})
    else:
        out({"error": "unknown"})
"#;
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("estia_fake_slowcancel_{}_{n}.py", std::process::id()));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(src.as_bytes()).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn interactive_calls_jump_the_background_queue() {
    if !python3_available() {
        return;
    }
    let path = write_slow_cancellable_runner();
    let s = Arc::new(spawn(&path, None));
    let order = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));

    // One background call occupies the child (250 ms). While it runs, queue
    // two more background calls, then one interactive call. The interactive
    // call must complete before either queued background call.
    //
    // Each call is started only once the previous one is inside the session
    // (`in_flight` counts the running call plus the queued ones), so the queue
    // order is fixed by the test, not by how a busy CI machine schedules
    // threads — fixed sleeps let two background calls swap places on CI.
    let mut handles = Vec::new();
    for (i, prio) in [
        (0, estia_engine::Priority::Background),
        (1, estia_engine::Priority::Background),
        (2, estia_engine::Priority::Background),
        (3, estia_engine::Priority::Interactive),
    ] {
        let s2 = Arc::clone(&s);
        let order = Arc::clone(&order);
        handles.push(std::thread::spawn(move || {
            let prompt = format!("{i}");
            let g: GenerateResp = s2
                .call_typed_with(
                    &Request::Generate { model_path: "/m", prompt: &prompt, max_tokens: None, temperature: None, json: None },
                    prio,
                )
                .unwrap();
            order.lock().unwrap().push(g.text);
        }));
        let deadline = Instant::now() + Duration::from_secs(5);
        while s.in_flight() < i + 1 {
            assert!(Instant::now() < deadline, "call {i} never entered the session (in_flight {})", s.in_flight());
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    for h in handles {
        h.join().unwrap();
    }
    let order = order.lock().unwrap().clone();
    assert_eq!(order[0], "slow:0", "the running call finishes first: {order:?}");
    assert_eq!(order[1], "slow:3", "the interactive call is served next, ahead of queued background: {order:?}");
    assert_eq!(&order[2..], &["slow:1".to_string(), "slow:2".to_string()], "background keeps FIFO: {order:?}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cancel_token_stops_a_stream_and_keeps_the_session_usable() {
    if !python3_available() {
        return;
    }
    let path = write_slow_cancellable_runner();
    let s = spawn(&path, None);
    let token = estia_engine::CancelToken::new();
    let flip = token.clone();
    // Cancel after the first token has landed. Time from the cancel, not from
    // the call: the call includes starting Python, which a cold CI runner can
    // stretch past any fixed bound.
    let mut toks = Vec::new();
    let mut cancelled_at: Option<Instant> = None;
    let res = s.stream_with(
        &Request::GenerateStream { model_path: "/m", prompt: "p", max_tokens: None, temperature: None },
        estia_engine::Priority::Interactive,
        Some(&token),
        |t| {
            toks.push(t.to_string());
            cancelled_at.get_or_insert_with(Instant::now);
            flip.cancel();
        },
    );
    let elapsed = cancelled_at.expect("a token arrived before the cancel").elapsed();
    match res {
        Err(SessionError::Cancelled { partial }) => {
            assert_eq!(partial, "one ", "partial text is what was delivered");
        }
        other => panic!("expected Cancelled, got {other:?}"),
    }
    assert_eq!(toks, vec!["one "]);
    // The rest of the stream would take 900 ms; the runner checks for cancel every 50 ms.
    assert!(elapsed < Duration::from_millis(600), "cancel must not wait for the full stream: {elapsed:?}");

    // The session is not desynced: the next call gets its own answer.
    let g: GenerateResp =
        s.call_typed(&Request::Generate { model_path: "/m", prompt: "after", max_tokens: None, temperature: None, json: None }).unwrap();
    assert_eq!(g.text, "slow:after");
    assert_eq!(s.in_flight(), 0);
    let _ = std::fs::remove_file(&path);
}

/// A runner that does not know `cancel` finishes the stream anyway and then
/// answers the cancel with an error line; the client must drain it so the
/// next call is not handed a stale error.
#[test]
fn cancel_against_an_old_runner_drains_the_stray_reply() {
    if !python3_available() {
        return;
    }
    let path = write_runner("oldcancel", ECHO);
    let s = spawn(&path, None);
    let token = estia_engine::CancelToken::new();
    token.cancel(); // already cancelled before the call: the cancel goes out at once
    let res = s.stream_with(
        &Request::GenerateStream { model_path: "/m", prompt: "p", max_tokens: None, temperature: None },
        estia_engine::Priority::Background,
        Some(&token),
        |_| {},
    );
    assert!(matches!(res, Err(SessionError::Cancelled { .. })), "{res:?}");
    // ECHO answers `cancel` with {"error":"unknown type 'cancel'"} after the
    // stream; that line must not surface here:
    let g: GenerateResp =
        s.call_typed(&Request::Generate { model_path: "/m", prompt: "p", max_tokens: None, temperature: None, json: None }).unwrap();
    assert_eq!(g.text, "hello");
    let _ = std::fs::remove_file(&path);
}

// ── Protocol v2 ───────────────────────────────────────────────────────────────

const V2: &str = r#"
            if t == "ping":
                resp = {"ok": True}
            elif t == "hello":
                resp = {"ok": True, "runner": "fake", "version": "0", "protocol": 2,
                        "capabilities": {"generate": True, "stream": True, "embed": True, "cancel": False, "load": True, "structured": []}}
            elif t == "load":
                time.sleep(0.05)
                resp = {"ok": True, "loaded": True, "ms": 50}
            elif t == "unload":
                resp = {"ok": True, "unloaded": True}
            elif t == "generate":
                resp = {"text": "hello"}
            else:
                resp = {"error": "unknown type %r" % (t,)}
            sys.stdout.write(json.dumps(resp) + "\n")
"#;

#[test]
fn hello_distinguishes_v2_from_v1_and_load_is_a_noop_on_v1() {
    if !python3_available() {
        return;
    }
    use estia_engine::{EmbedSession, GenSession};
    let v2 = write_runner("v2", V2);
    let s = spawn(&v2, None);
    let h = s.hello().unwrap().expect("v2 runner answers hello");
    assert_eq!(h.protocol, 2);
    assert!(h.capabilities.load && h.capabilities.stream && !h.capabilities.cancel);

    let g = GenSession::spawn(Launch::new("python3").arg(&v2), SessionConfig::default(), Arc::new(NoopObserver), "/m", "fake-id").unwrap();
    assert!(g.capabilities().load);
    assert_eq!(g.load().unwrap(), Some(Duration::from_millis(50)));
    assert!(g.unload().unwrap());
    assert_eq!(g.generate("p", None, None).unwrap(), "hello");

    // A v1 runner (ECHO knows no `hello`): None, empty capabilities, load no-op.
    let v1 = write_runner("v1", ECHO);
    let s1 = spawn(&v1, None);
    assert!(s1.hello().unwrap().is_none());
    let e = EmbedSession::spawn(Launch::new("python3").arg(&v1), SessionConfig::default(), Arc::new(NoopObserver), "/m", "fp").unwrap();
    assert!(e.hello().is_none());
    assert_eq!(e.capabilities(), estia_engine::proto::Capabilities::default());
    assert_eq!(e.load().unwrap(), None);
    assert!(!e.unload().unwrap());
    // …and the failed hello did not desync the session.
    assert_eq!(e.embed_batch(&["a".into()]).unwrap(), vec![vec![1.0, 0.0]]);

    let _ = std::fs::remove_file(&v2);
    let _ = std::fs::remove_file(&v1);
}

// ── Protocol v2: chat, meta, count_tokens ─────────────────────────────────────

const V2_CHAT: &str = r#"
            if t == "ping":
                resp = {"ok": True}
            elif t == "hello":
                resp = {"ok": True, "runner": "fake", "version": "0", "protocol": 2,
                        "capabilities": {"generate": True, "stream": True, "chat": True, "tools": True, "prompt_cache": True, "count_tokens": True}}
            elif t == "chat_stream":
                roles = ",".join(m["role"] for m in req["messages"])
                sys.stdout.write(json.dumps({"type": "token", "text": "roles:" + roles}) + "\n")
                cached = 7 if req.get("cache_key") else 0
                sys.stdout.write(json.dumps({"type": "meta", "prompt_tokens": 12, "cached_tokens": cached, "generation_tokens": 3, "template": "native", "tools_seen": len(req.get("tools") or [])}) + "\n")
                sys.stdout.write(json.dumps({"done": True}) + "\n")
                sys.stdout.flush()
                continue
            elif t == "chat":
                resp = {"text": "chat:" + req["messages"][-1]["content"], "meta": {"prompt_tokens": 5, "cached_tokens": 0, "generation_tokens": 2}}
            elif t == "count_tokens":
                resp = {"tokens": len(req["text"].split())}
            else:
                resp = {"error": "unknown type %r" % (t,)}
            sys.stdout.write(json.dumps(resp) + "\n")
"#;

#[test]
fn chat_stream_carries_meta_and_count_tokens_works() {
    if !python3_available() {
        return;
    }
    use estia_engine::proto::Message;
    use estia_engine::GenSession;
    let path = write_runner("v2chat", V2_CHAT);
    let g =
        GenSession::spawn(Launch::new("python3").arg(&path), SessionConfig::default(), Arc::new(NoopObserver), "/m", "fake-id").unwrap();
    assert!(g.capabilities().chat && g.capabilities().prompt_cache);

    let msgs =
        vec![Message::new("system", "terse"), Message::new("user", "hi"), Message::new("assistant", "yo"), Message::new("user", "again")];
    let tools = vec![serde_json::json!({"type": "function", "function": {"name": "f"}})];
    let mut toks = Vec::new();
    let out = g
        .chat_stream_with(&msgs, Some(&tools), Some("conv-1"), None, Some(16), Some(0.0), estia_engine::Priority::Interactive, None, |t| {
            toks.push(t.to_string())
        })
        .unwrap();
    assert_eq!(out.text, "roles:system,user,assistant,user");
    assert_eq!(toks, vec!["roles:system,user,assistant,user"]);
    assert_eq!(out.meta.prompt_tokens, Some(12));
    assert_eq!(out.meta.cached_tokens, Some(7), "cache_key was passed through");
    assert_eq!(out.meta.template.as_deref(), Some("native"));

    let out = g.chat_with(&msgs, None, None, None, Some(16), Some(0.0), estia_engine::Priority::Background).unwrap();
    assert_eq!(out.text, "chat:again");
    assert_eq!(out.meta.generation_tokens, Some(2));

    assert_eq!(g.count_tokens("one two three").unwrap(), Some(3));

    // A v1 runner: chat refused clearly, count_tokens None.
    let v1 = write_runner("v1chat", ECHO);
    let g1 = GenSession::spawn(Launch::new("python3").arg(&v1), SessionConfig::default(), Arc::new(NoopObserver), "/m", "fake-id").unwrap();
    assert!(matches!(g1.chat_with(&msgs, None, None, None, None, None, estia_engine::Priority::Background), Err(SessionError::Runner(_))));
    assert_eq!(g1.count_tokens("x").unwrap(), None);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&v1);
}
