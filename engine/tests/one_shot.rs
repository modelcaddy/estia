//! OneShot behaviour against a stdlib-only Python fake that reads stdin to EOF
//! and answers with `{ok, …}` envelopes — the older runners' shape.

use estia_engine::proto::{GenerateResp, HealthResp, Request};
use estia_engine::{Launch, NoopObserver, OneShot, OneShotConfig, SessionError};
use serde_json::json;
use std::io::Write as _;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn python3_available() -> bool {
    Command::new("python3").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn write_fake(tag: &str) -> String {
    let src = r#"import sys, json, time
def out(o):
    sys.stdout.write(json.dumps(o))
    sys.stdout.flush()
def line(o):
    sys.stdout.write(json.dumps(o) + "\n")
    sys.stdout.flush()
req = json.loads(sys.stdin.read())
t = req.get("type")
if t == "health":
    out({"ok": True, "mlx_available": True, "version": "fake", "detail": None})
elif t == "generate":
    out({"ok": True, "text": "hello " + str(req.get("json"))})
elif t == "generate_stream":
    sys.stderr.write("loading weights\n")
    line({"type": "token", "text": "hel"})
    line({"type": "token", "text": "lo"})
    line({"ok": True, "done": True})
elif t == "legacy_stream":
    out({"ok": True, "text": "all at once"})
elif t == "midfail_stream":
    line({"type": "token", "text": "par"})
    line({"ok": False, "error": "gpu fell over"})
elif t == "fail":
    out({"ok": False, "error": "nope"})
elif t == "crash":
    sys.stderr.write("boom\n")
    sys.exit(3)
elif t == "silent":
    pass
elif t == "hang":
    time.sleep(3)
    out({"ok": True, "text": "late"})
else:
    out({"ok": False, "error": "unknown type %r" % (t,)})
"#;
    // Tag + pid + counter: a timestamp alone collided across tests running in
    // the same nanosecond, and the first test to finish deleted the other's runner.
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("estia_oneshot_fake_{tag}_{}_{n}.py", std::process::id()));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(src.as_bytes()).unwrap();
    path.to_string_lossy().into_owned()
}

fn one_shot(path: &str, timeout: Option<Duration>) -> OneShot {
    let mut cfg = OneShotConfig::default();
    if let Some(t) = timeout {
        cfg.call_timeout = t;
    }
    OneShot::new(Launch::new("python3").arg(path), cfg, Arc::new(NoopObserver))
}

#[test]
fn envelopes_round_trip() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let path = write_fake("envelopes");
    let os = one_shot(&path, None);

    let h: HealthResp = os.call_typed(&Request::Health).unwrap();
    assert!(h.mlx_available);
    assert_eq!(h.version.as_deref(), Some("fake"));

    // The one-shot `json` flag reaches the runner (it echoes it back).
    let g: GenerateResp =
        os.call_typed(&Request::Generate { model_path: "/m", prompt: "p", max_tokens: None, temperature: None, json: Some(true) }).unwrap();
    assert_eq!(g.text, "hello True");

    let err = os.call(&json!({"type": "fail"})).unwrap_err();
    assert!(matches!(err, SessionError::Runner(ref m) if m == "nope"), "{err}");

    let err = os.call(&json!({"type": "crash"})).unwrap_err();
    assert!(matches!(err, SessionError::Exited { ref stderr, .. } if stderr == "boom"), "{err}");

    let err = os.call(&json!({"type": "silent"})).unwrap_err();
    assert!(matches!(err, SessionError::Parse(_)), "{err}");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn streams_tokens_then_reaps() {
    if !python3_available() {
        return;
    }
    let path = write_fake("stream");
    let os = one_shot(&path, None);

    let mut toks = Vec::new();
    let full = os
        .stream(&Request::GenerateStream { model_path: "/m", prompt: "p", max_tokens: None, temperature: None }, |t| {
            toks.push(t.to_string())
        })
        .unwrap();
    assert_eq!(toks, vec!["hel", "lo"]);
    assert_eq!(full, "hello");

    let mut toks = Vec::new();
    let full = os.stream(&json!({"type": "legacy_stream"}), |t| toks.push(t.to_string())).unwrap();
    assert_eq!(toks, vec!["all at once"]);
    assert_eq!(full, "all at once");

    let mut toks = Vec::new();
    let err = os.stream(&json!({"type": "midfail_stream"}), |t| toks.push(t.to_string())).unwrap_err();
    assert_eq!(toks, vec!["par"]);
    assert!(matches!(err, SessionError::Runner(ref m) if m == "gpu fell over"), "{err}");

    let err = os.stream(&json!({"type": "silent"}), |_| {}).unwrap_err();
    assert!(matches!(err, SessionError::NoOutput { .. }), "{err}");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn deadline_kills_a_wedged_child() {
    if !python3_available() {
        return;
    }
    let path = write_fake("deadline");
    let os = one_shot(&path, Some(Duration::from_millis(300)));

    let start = Instant::now();
    let err = os.call(&json!({"type": "hang"})).unwrap_err();
    assert!(matches!(err, SessionError::Timeout { .. }), "{err}");
    assert!(start.elapsed() < Duration::from_secs(5), "{:?}", start.elapsed());

    let start = Instant::now();
    let err = os.stream(&json!({"type": "hang"}), |_| {}).unwrap_err();
    assert!(matches!(err, SessionError::Timeout { .. }), "{err}");
    assert!(start.elapsed() < Duration::from_secs(5), "{:?}", start.elapsed());

    let _ = std::fs::remove_file(&path);
}
