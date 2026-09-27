//! The adapter binary against a real llama-server, through protocol v2.
//!
//! Skipped (with a note) unless these point at local files:
//!
//! - `ESTIA_LLAMA_SERVER`: a `llama-server` executable (Estia pins build b11146)
//! - `ESTIA_LLAMA_TEST_MODEL`: a small chat GGUF with a chat template
//!   (CI uses ggml-org/tinygemma3-GGUF, 47 MB; its answers are gibberish, so
//!   these tests check mechanics, not quality)
//! - `ESTIA_LLAMA_TEST_EMBED_MODEL`: an embedding GGUF with 384 dimensions
//!   (CI uses all-MiniLM-L6-v2 Q8_0)
//! - `ESTIA_LLAMA_TEST_MMPROJ` (optional, for the image test): the test chat
//!   model's image projector (CI uses tinygemma3's `mmproj-tinygemma3.gguf`, 1 MB)
//!
//! `ESTIA_LLAMA_TEST_VERBOSE=1` echoes the adapter's stderr.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const LOAD: Duration = Duration::from_secs(300);
const CALL: Duration = Duration::from_secs(120);
const LEAKED_KEY: &str = "key-from-the-environment";

/// Status code of one request straight to llama-server's socket.
#[cfg(unix)]
fn status_of(sock: &Path, method: &str, path: &str, key: Option<&str>) -> u16 {
    use std::io::Read;
    let mut s = std::os::unix::net::UnixStream::connect(sock).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let auth = key.map(|k| format!("Authorization: Bearer {k}\r\n")).unwrap_or_default();
    let body = r#"{"content":"hi"}"#;
    write!(s, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len())
        .unwrap();
    let mut head = [0u8; 12];
    s.read_exact(&mut head).unwrap();
    String::from_utf8_lossy(&head[9..12]).parse().unwrap()
}

struct Assets {
    server: PathBuf,
    model: PathBuf,
    embed: PathBuf,
}

fn assets(test: &str) -> Option<Assets> {
    let get = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    match (get("ESTIA_LLAMA_SERVER"), get("ESTIA_LLAMA_TEST_MODEL"), get("ESTIA_LLAMA_TEST_EMBED_MODEL")) {
        (Some(server), Some(model), Some(embed)) => Some(Assets { server, model, embed }),
        _ => {
            eprintln!("skipping {test}: set ESTIA_LLAMA_SERVER, ESTIA_LLAMA_TEST_MODEL and ESTIA_LLAMA_TEST_EMBED_MODEL to run it");
            None
        }
    }
}

/// A short private scratch directory (socket paths must stay short).
fn scratch(tag: &str) -> PathBuf {
    let base = std::env::var_os("ESTIA_LLAMA_TEST_TMP").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp"));
    let d = base.join(format!("ellt-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Whether `pid` is a live (not zombie) process.
fn alive(pid: u32) -> bool {
    let out = Command::new("ps").args(["-o", "stat=", "-p", &pid.to_string()]).output().expect("ps");
    let stat = String::from_utf8_lossy(&out.stdout);
    let stat = stat.trim();
    !stat.is_empty() && !stat.starts_with('Z')
}

fn wait_until(what: &str, within: Duration, mut f: impl FnMut() -> bool) {
    let t = Instant::now();
    while !f() {
        assert!(t.elapsed() < within, "timed out after {within:?} waiting for: {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

struct Adapter {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    run_dir: PathBuf,
}

impl Adapter {
    fn start(a: &Assets, run_dir: &Path) -> Adapter {
        let mut child = Command::new(env!("CARGO_BIN_EXE_estia-llama"))
            .arg("--server")
            .arg(&a.server)
            .arg("--run-dir")
            .arg(run_dir)
            .args(["--ctx", "4096"])
            // Would be one more key llama-server accepts if passed through.
            .env("LLAMA_API_KEY", LEAKED_KEY)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn estia-llama");
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, lines) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let log = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&log);
        let verbose = std::env::var_os("ESTIA_LLAMA_TEST_VERBOSE").is_some();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                if verbose {
                    eprintln!("{line}");
                }
                sink.lock().unwrap().push(line);
            }
        });
        Adapter { child, stdin, lines, stderr: log, run_dir: run_dir.to_path_buf() }
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn stderr_tail(&self) -> String {
        let l = self.stderr.lock().unwrap();
        l[l.len().saturating_sub(30)..].join("\n")
    }

    fn send(&mut self, v: &Value) {
        let w = self.stdin.as_mut().expect("stdin open");
        writeln!(w, "{v}").unwrap();
        w.flush().unwrap();
    }

    fn recv(&self, within: Duration) -> Value {
        match self.lines.recv_timeout(within) {
            Ok(line) => serde_json::from_str(&line).unwrap_or_else(|e| panic!("stdout line is not JSON ({e}): {line}")),
            Err(RecvTimeoutError::Timeout) => panic!("no answer in {within:?}; stderr:\n{}", self.stderr_tail()),
            Err(RecvTimeoutError::Disconnected) => panic!("adapter stdout closed; stderr:\n{}", self.stderr_tail()),
        }
    }

    /// The next line that is not a keepalive. A cold model load (a slow CI
    /// machine, or macOS checking a freshly unpacked binary) sends keepalives
    /// before the first token.
    fn recv_event(&self, within: Duration) -> Value {
        loop {
            let v = self.recv(within);
            if v["type"] != "keepalive" {
                return v;
            }
        }
    }

    fn call(&mut self, v: Value) -> Value {
        let slow = matches!(v["type"].as_str(), Some("load" | "embed_batch" | "embed" | "count_tokens"));
        self.send(&v);
        self.recv(if slow { LOAD } else { CALL })
    }

    fn ok(&mut self, v: Value) -> Value {
        let r = self.call(v.clone());
        assert!(r.get("error").is_none(), "{v} failed: {r}\nstderr:\n{}", self.stderr_tail());
        r
    }

    /// Token texts, the meta line, and the terminal line of a stream.
    fn stream(&mut self, v: Value) -> (Vec<String>, Option<Value>, Value) {
        self.send(&v);
        let mut tokens = Vec::new();
        let mut meta = None;
        loop {
            let line = self.recv(LOAD);
            match line["type"].as_str() {
                Some("token") => tokens.push(line["text"].as_str().unwrap().to_string()),
                Some("keepalive") => {}
                Some("meta") => meta = Some(line),
                _ => return (tokens, meta, line),
            }
        }
    }

    fn record(&self) -> Value {
        let p = self.run_dir.join(format!("llama-{}.json", self.pid()));
        let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        serde_json::from_str(&text).unwrap()
    }

    fn server_pid(&self) -> u32 {
        self.record()["server_pid"].as_u64().unwrap() as u32
    }

    /// Close stdin and wait for a clean exit.
    fn close(mut self) -> std::process::ExitStatus {
        drop(self.stdin.take());
        let t = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(t.elapsed() < Duration::from_secs(15), "adapter did not exit after stdin closed:\n{}", self.stderr_tail());
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Adapter {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn user(text: &str) -> Value {
    json!([{ "role": "user", "content": text }])
}

fn chat(model: &Path, key: Option<&str>, max_tokens: u32) -> Value {
    let mut v = json!({
        "type": "chat",
        "model_path": model,
        "messages": user("Tell me about the sea."),
        "max_tokens": max_tokens,
        "temperature": null,
    });
    if let Some(k) = key {
        v["cache_key"] = json!(k);
    }
    v
}

#[test]
fn generation_through_the_protocol() {
    let Some(a) = assets("generation_through_the_protocol") else { return };
    let dir = scratch("gen");
    let mut ad = Adapter::start(&a, &dir.join("run"));

    let h = ad.ok(json!({"type": "hello"}));
    assert_eq!((h["runner"].as_str(), h["protocol"].as_u64()), (Some("estia-llama"), Some(2)));
    let caps: estia_proto::Capabilities = serde_json::from_value(h["capabilities"].clone()).unwrap();
    assert!(caps.chat && caps.stream && caps.cancel && caps.load && caps.tools && caps.prompt_cache && caps.count_tokens);
    assert!(caps.embed && caps.generate && caps.parses_tool_calls);
    assert_eq!(caps.structured, ["json", "json_schema"]);
    assert_eq!(caps.backend.as_deref(), Some("llama-cpp"));
    assert_eq!(ad.ok(json!({"type": "ping"})), json!({"ok": true}));

    let r = ad.ok(json!({"type": "load", "model_path": a.model, "kind": "generation"}));
    assert_eq!(r["loaded"], true);
    assert!(r["ms"].as_u64().unwrap() > 0, "{r}");
    let server = ad.server_pid();
    assert!(alive(server));
    let sock = PathBuf::from(ad.record()["socket"].as_str().unwrap());
    assert!(sock.exists());
    // Only the adapter can talk to it: a private directory, a key it never
    // wrote down past start-up, and no key taken from the environment.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&ad.run_dir).unwrap().permissions().mode() & 0o777, 0o700);
        assert!(!ad.run_dir.join(format!("llama-{}.key", ad.pid())).exists(), "the key file outlived start-up");
        assert_eq!(status_of(&sock, "GET", "/health", None), 200);
        assert_eq!(status_of(&sock, "POST", "/tokenize", None), 401);
        assert_eq!(status_of(&sock, "POST", "/tokenize", Some(LEAKED_KEY)), 401);
    }
    // Loading it again is free.
    assert_eq!(ad.ok(json!({"type": "load", "model_path": a.model, "kind": "generation"}))["ms"], 0);
    assert_eq!(ad.ok(json!({"type": "ping"})), json!({"ok": true}));

    // Stream: tokens, then meta with counts, then done.
    let mut req = chat(&a.model, Some("conv-a"), 12);
    req["type"] = json!("chat_stream");
    let (tokens, meta, end) = ad.stream(req);
    assert_eq!(end, json!({"done": true}));
    assert!(!tokens.is_empty());
    let meta = meta.expect("a meta line");
    let g: estia_proto::GenerationMeta = serde_json::from_value(meta.clone()).unwrap();
    assert!(g.prompt_tokens.unwrap() > 0, "{meta}");
    assert_eq!(g.cached_tokens, Some(0), "a new conversation starts cold: {meta}");
    assert!(g.generation_tokens.unwrap() > 0, "{meta}");
    assert!(g.generation_tps.unwrap() > 0.0, "{meta}");
    assert_eq!(g.template.as_deref(), Some("native"));

    // The same conversation reuses the slot; another one, or none, does not.
    let r = ad.ok(chat(&a.model, Some("conv-a"), 4));
    assert!(r["meta"]["cached_tokens"].as_u64().unwrap() > 0, "same key should hit the cache: {r}");
    assert!(r["text"].is_string());
    let r = ad.ok(chat(&a.model, Some("conv-b"), 4));
    assert_eq!(r["meta"]["cached_tokens"], 0, "{r}");
    let r = ad.ok(chat(&a.model, None, 4));
    assert_eq!(r["meta"]["cached_tokens"], 0, "{r}");
    let r = ad.ok(chat(&a.model, None, 4));
    assert_eq!(r["meta"]["cached_tokens"], 0, "no key never reuses: {r}");
    // A request that fails leaves the slot to nobody. Here conversation B's
    // first request is refused by the chat template (two user turns in a
    // row), so the slot still holds A's prompt; B's next request must not
    // be told how much of it matches.
    ad.ok(chat(&a.model, Some("iso-a"), 4));
    let mut bad = chat(&a.model, Some("iso-b"), 4);
    bad["messages"] = json!([{"role": "user", "content": "Tell me about the sea."}, {"role": "user", "content": "Tell me about the sea."}]);
    let r = ad.call(bad);
    assert!(r["error"].is_string(), "the tiny model's template should refuse this: {r}");
    let r = ad.ok(chat(&a.model, Some("iso-b"), 4));
    assert_eq!(r["meta"]["cached_tokens"], 0, "another conversation's prefix was reused: {r}");

    let n = ad.ok(json!({"type": "count_tokens", "model_path": a.model, "text": "hello world, how are you?"}));
    assert!(n["tokens"].as_u64().unwrap() > 0, "{n}");

    // Output constrained to a JSON Schema parses and matches it.
    let schema = json!({
        "type": "object",
        "properties": {
            "color": {"type": "string", "enum": ["red", "green", "blue"]},
            "ok": {"type": "boolean"}
        },
        "required": ["color", "ok"],
        "additionalProperties": false
    });
    let mut req = chat(&a.model, None, 512);
    req["messages"] = user("Pick a color. Answer in JSON.");
    req["format"] = json!({"type": "json_schema", "schema": schema});
    let r = ad.ok(req);
    let text = r["text"].as_str().unwrap();
    let v: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("not JSON ({e}): {text:?} {r}"));
    let obj = v.as_object().unwrap_or_else(|| panic!("not an object: {v}"));
    assert_eq!(obj.len(), 2, "{v}");
    assert!(["red", "green", "blue"].contains(&obj["color"].as_str().unwrap()), "{v}");
    assert!(obj["ok"].is_boolean(), "{v}");
    // Any JSON.
    let mut req = chat(&a.model, None, 256);
    req["format"] = json!({"type": "json"});
    let r = ad.ok(req);
    assert!(r["text"].is_string());

    // v1 generate: the prompt as one user turn.
    let r = ad.ok(json!({"type": "generate", "model_path": a.model, "prompt": "Name a sea.", "max_tokens": 4, "temperature": null}));
    assert!(r["text"].is_string(), "{r}");
    let (tokens, _, end) =
        ad.stream(json!({"type": "generate_stream", "model_path": a.model, "prompt": "Name a sea.", "max_tokens": 4, "temperature": 0.0}));
    assert_eq!(end, json!({"done": true}));
    assert!(!tokens.is_empty());

    // Unload stops the server and removes its files.
    assert_eq!(ad.ok(json!({"type": "unload", "model_path": a.model})), json!({"ok": true, "unloaded": true}));
    wait_until("llama-server to stop after unload", Duration::from_secs(10), || !alive(server));
    assert!(!sock.exists(), "socket left behind");
    assert!(!ad.run_dir.join(format!("llama-{}.json", ad.pid())).exists(), "pid record left behind");
    assert_eq!(ad.ok(json!({"type": "unload", "model_path": a.model})), json!({"ok": true, "unloaded": false}));

    // A request for a model that is not loaded loads it first.
    let r = ad.ok(chat(&a.model, None, 2));
    assert!(r["text"].is_string());
    let server = ad.server_pid();

    // Closing stdin ends the adapter and its server.
    let status = ad.close();
    assert!(status.success(), "{status}");
    wait_until("llama-server to stop after stdin closed", Duration::from_secs(10), || !alive(server));
    let left: Vec<_> = std::fs::read_dir(dir.join("run")).unwrap().flatten().map(|e| e.file_name()).collect();
    assert!(left.is_empty(), "files left in the run dir: {left:?}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn cancel_ends_a_long_stream_and_sigterm_cleans_up() {
    let Some(a) = assets("cancel_ends_a_long_stream_and_sigterm_cleans_up") else { return };
    let dir = scratch("cancel");
    let mut ad = Adapter::start(&a, &dir.join("run"));
    ad.ok(json!({"type": "load", "model_path": a.model, "kind": "generation"}));
    let server = ad.server_pid();

    let mut req = chat(&a.model, Some("long"), 4096);
    req["type"] = json!("chat_stream");
    req["messages"] = user("Count from one to one thousand, in words.");
    ad.send(&req);
    let first = ad.recv_event(LOAD);
    assert_eq!(first["type"], "token", "{first}");
    ad.send(&json!({"type": "cancel"}));
    let cancelled_at = Instant::now();
    let mut tokens = 1;
    let end = loop {
        let line = ad.recv(CALL);
        match line["type"].as_str() {
            Some("token") => tokens += 1,
            Some("keepalive") | Some("meta") => {}
            _ => break line,
        }
    };
    assert_eq!(end, json!({"done": true, "cancelled": true}));
    let took = cancelled_at.elapsed();
    assert!(took < Duration::from_secs(5), "cancel took {took:?}");
    assert!(tokens < 1000, "{tokens} tokens arrived after the cancel");
    // llama-server saw the connection close and cancelled its task.
    let log = Arc::clone(&ad.stderr);
    wait_until("llama-server to log the cancel", Duration::from_secs(5), || {
        log.lock().unwrap().iter().any(|l| l.contains(&format!("[llama-server {server}]")) && l.contains("cancel task"))
    });

    // Still in step: a cancel between requests is dropped, the next call works.
    ad.send(&json!({"type": "cancel"}));
    let r = ad.ok(chat(&a.model, Some("long"), 4));
    assert!(r["text"].is_string(), "{r}");
    assert_eq!(r["meta"]["cached_tokens"], 0, "a cancelled request leaves the slot to nobody: {r}");

    // SIGTERM: the adapter stops its server and exits.
    let status = Command::new("kill").args(["-TERM", &ad.pid().to_string()]).status().unwrap();
    assert!(status.success());
    let t = Instant::now();
    let exit = loop {
        if let Some(s) = ad.child.try_wait().unwrap() {
            break s;
        }
        assert!(t.elapsed() < Duration::from_secs(10), "adapter ignored SIGTERM:\n{}", ad.stderr_tail());
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(!exit.success(), "a signalled adapter exits non-zero: {exit}");
    wait_until("llama-server to stop after SIGTERM", Duration::from_secs(10), || !alive(server));
    let left: Vec<_> = std::fs::read_dir(dir.join("run")).unwrap().flatten().map(|e| e.file_name()).collect();
    assert!(left.is_empty(), "files left in the run dir: {left:?}");
    let _ = std::fs::remove_dir_all(dir);
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn vector(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn embeddings_in_order_and_switching_models() {
    let Some(a) = assets("embeddings_in_order_and_switching_models") else { return };
    let dir = scratch("embed");
    let mut ad = Adapter::start(&a, &dir.join("run"));

    // A chat model first; asking for an embedding model restarts the server.
    ad.ok(json!({"type": "count_tokens", "model_path": a.model, "text": "hi"}));
    let chat_server = ad.server_pid();
    let r = ad.ok(json!({"type": "load", "model_path": a.embed, "kind": "embedding"}));
    assert!(r["ms"].as_u64().unwrap() > 0);
    let embed_server = ad.server_pid();
    assert_ne!(chat_server, embed_server);
    wait_until("the chat server to stop", Duration::from_secs(10), || !alive(chat_server));

    let inputs: Vec<String> =
        (0..256).map(|i| format!("Sentence {i}: the sea is {} today.", ["calm", "rough", "grey", "blue"][i % 4])).collect();
    let t = Instant::now();
    let r = ad.ok(json!({"type": "embed_batch", "model_path": a.embed, "inputs": inputs}));
    eprintln!("embed_batch of 256 took {:?}", t.elapsed());
    let vecs = r["embeddings"].as_array().unwrap();
    assert_eq!(vecs.len(), 256);
    for v in vecs {
        let v = vector(v);
        assert_eq!(v.len(), 384);
        assert!((norm(&v) - 1.0).abs() < 1e-3, "not unit length: {}", norm(&v));
    }
    // Order: each vector is the one its input gets alone.
    for i in [0usize, 1, 2, 255] {
        let one = ad.ok(json!({"type": "embed", "model_path": a.embed, "input": inputs[i]}));
        let one = vector(&one["embedding"]);
        let dot: f64 = one.iter().zip(vector(&vecs[i])).map(|(x, y)| x * y).sum();
        assert!(dot > 0.999, "input {i}: cosine {dot}");
    }

    // An input far past the model's 512-token window is cut, not refused.
    let long = "the quick brown fox jumps over the lazy dog ".repeat(300);
    let r = ad.ok(json!({"type": "embed_batch", "model_path": a.embed, "inputs": [long, "short", ""]}));
    assert_eq!(r["embeddings"].as_array().unwrap().len(), 3);
    assert_eq!(ad.ok(json!({"type": "embed_batch", "model_path": a.embed, "inputs": []})), json!({"embeddings": []}));
    assert_eq!(ad.server_pid(), embed_server, "no restarts in between");

    assert!(ad.close().success());
    wait_until("the embedding server to stop", Duration::from_secs(10), || !alive(embed_server));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn stale_socket_bad_model_and_a_killed_adapter() {
    let Some(a) = assets("stale_socket_bad_model_and_a_killed_adapter") else { return };
    let dir = scratch("stale");
    let run = dir.join("run");
    std::fs::create_dir_all(&run).unwrap();
    let mut ad = Adapter::start(&a, &run);

    // A model llama-server cannot load: an error line, and the adapter lives on.
    let bad = dir.join("bad.gguf");
    std::fs::write(&bad, b"this is not a gguf file").unwrap();
    let r = ad.call(json!({"type": "load", "model_path": bad, "kind": "generation"}));
    assert!(r["error"].as_str().unwrap_or("").contains("could not load"), "{r}");
    assert_eq!(ad.ok(json!({"type": "ping"})), json!({"ok": true}));

    // The socket a previous llama-server left behind (it never removes it).
    #[cfg(unix)]
    {
        let sock = run.join(format!("llama-{}.sock", ad.pid()));
        let _ = std::fs::remove_file(&sock);
        drop(std::os::unix::net::UnixListener::bind(&sock).unwrap());
        assert!(sock.exists());
    }
    ad.ok(json!({"type": "load", "model_path": a.model, "kind": "generation"}));
    let server = ad.server_pid();

    // Killed outright, the adapter cannot clean up; llama-server must still
    // go (PR_SET_PDEATHSIG on Linux, the guard process on macOS).
    ad.child.kill().unwrap();
    ad.child.wait().unwrap();
    wait_until("llama-server to stop after the adapter was killed", Duration::from_secs(10), || !alive(server));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_dead_llama_server_fails_the_call_and_ends_the_adapter() {
    let Some(a) = assets("a_dead_llama_server_fails_the_call_and_ends_the_adapter") else { return };
    let dir = scratch("crash");
    let mut ad = Adapter::start(&a, &dir.join("run"));
    ad.ok(json!({"type": "load", "model_path": a.model, "kind": "generation"}));
    let server = ad.server_pid();
    assert!(Command::new("kill").args(["-KILL", &server.to_string()]).status().unwrap().success());
    wait_until("llama-server to die", Duration::from_secs(5), || !alive(server));

    // The call in flight fails, then the adapter exits so Session respawns it.
    let r = ad.call(chat(&a.model, None, 4));
    assert!(r["error"].as_str().unwrap_or("").contains("exited"), "{r}");
    let t = Instant::now();
    let status = loop {
        if let Some(s) = ad.child.try_wait().unwrap() {
            break s;
        }
        assert!(t.elapsed() < Duration::from_secs(10), "adapter kept running:\n{}", ad.stderr_tail());
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(!status.success(), "{status}");
    let left: Vec<_> = std::fs::read_dir(dir.join("run")).unwrap().flatten().map(|e| e.file_name()).collect();
    assert!(left.is_empty(), "files left in the run dir: {left:?}");
    let _ = std::fs::remove_dir_all(dir);
}

#[cfg(unix)]
#[test]
fn an_orphaned_adapter_stops_its_server() {
    let Some(a) = assets("an_orphaned_adapter_stops_its_server") else { return };
    let dir = scratch("orphan");
    let run = dir.join("run");
    // A shell starts the adapter in the background, sharing its own stdin and
    // stdout (fd 3 keeps sh from pointing the job's stdin at /dev/null), and
    // prints its pid. Killing the shell orphans the adapter while its stdin
    // stays open here, so only the parent-death watchdog can end it.
    let script = r#"exec 3<&0; "$0" --server "$1" --run-dir "$2" <&3 & echo "pid $!"; wait"#;
    let mut sh = Command::new("/bin/sh")
        .args(["-c", script])
        .arg(env!("CARGO_BIN_EXE_estia-llama"))
        .arg(&a.server)
        .arg(&run)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = sh.stdin.take().unwrap();
    let mut out = BufReader::new(sh.stdout.take().unwrap());
    let mut line = String::new();
    out.read_line(&mut line).unwrap();
    let adapter: u32 = line.trim().strip_prefix("pid ").and_then(|p| p.parse().ok()).unwrap_or_else(|| panic!("{line:?}"));
    writeln!(stdin, "{}", json!({"type": "load", "model_path": a.model, "kind": "generation"})).unwrap();
    stdin.flush().unwrap();
    line.clear();
    out.read_line(&mut line).unwrap();
    assert!(line.contains("\"loaded\":true"), "{line}");
    let record: Value = serde_json::from_str(&std::fs::read_to_string(run.join(format!("llama-{adapter}.json"))).unwrap()).unwrap();
    let server = record["server_pid"].as_u64().unwrap() as u32;
    assert!(alive(adapter) && alive(server));

    sh.kill().unwrap();
    sh.wait().unwrap();
    wait_until("the orphaned adapter to exit", Duration::from_secs(10), || !alive(adapter));
    wait_until("its llama-server to stop", Duration::from_secs(10), || !alive(server));
    let left: Vec<_> = std::fs::read_dir(&run).unwrap().flatten().map(|e| e.file_name()).collect();
    assert!(left.is_empty(), "files left in the run dir: {left:?}");
    drop(stdin);
    let _ = std::fs::remove_dir_all(dir);
}

/// A valid 1×1 PNG.
const TINY_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

#[test]
fn images_reach_a_model_with_a_projector_and_are_refused_without() {
    let Some(a) = assets("images_reach_a_model_with_a_projector_and_are_refused_without") else { return };
    let Some(mmproj) = std::env::var_os("ESTIA_LLAMA_TEST_MMPROJ").filter(|v| !v.is_empty()).map(PathBuf::from) else {
        eprintln!("skipping images_reach_a_model_with_a_projector_and_are_refused_without: set ESTIA_LLAMA_TEST_MMPROJ to run it");
        return;
    };
    let dir = scratch("img");
    // The store's layout: model.gguf with mmproj.gguf beside it.
    let with = dir.join("with");
    std::fs::create_dir_all(&with).unwrap();
    std::os::unix::fs::symlink(std::fs::canonicalize(&a.model).unwrap(), with.join("model.gguf")).unwrap();
    std::os::unix::fs::symlink(std::fs::canonicalize(&mmproj).unwrap(), with.join("mmproj.gguf")).unwrap();
    let mut ad = Adapter::start(&a, &dir.join("run"));
    let image = json!({"mime": "image/png", "data": TINY_PNG});
    let turn = |content: &str, with_image: bool| {
        let mut m = json!({"role": "user", "content": content});
        if with_image {
            m["images"] = json!([image]);
        }
        m
    };

    // Without a projector: a clear error, not an answer that ignores the image.
    let r = ad
        .call(json!({"type": "chat", "model_path": a.model, "messages": [turn("What is this?", true)], "max_tokens": 4, "cache_key": "k"}));
    let err = r["error"].as_str().unwrap_or_else(|| panic!("expected an error: {r}"));
    assert!(err.contains("cannot read images"), "{err}");

    // With one: llama-server starts with --mmproj, and the image adds tokens.
    let text = ad.ok(json!({"type": "chat", "model_path": with, "messages": [turn("What is this?", false)], "max_tokens": 4}));
    let text_prompt = text["meta"]["prompt_tokens"].as_u64().unwrap();
    let r =
        ad.ok(json!({"type": "chat", "model_path": with, "messages": [turn("What is this?", true)], "max_tokens": 4, "cache_key": "img"}));
    let img_prompt = r["meta"]["prompt_tokens"].as_u64().unwrap();
    assert!(img_prompt > text_prompt, "the image should add tokens: {img_prompt} vs {text_prompt}");
    assert_eq!(r["meta"]["cached_tokens"], 0, "an image turn is never served from the cache: {r}");
    // Streamed too.
    let mut req = json!({"type": "chat_stream", "model_path": with, "messages": [turn("Describe.", true)], "max_tokens": 4});
    req["cache_key"] = json!("img");
    let (_tokens, meta, end) = ad.stream(req);
    assert_eq!(end, json!({"done": true}));
    assert_eq!(meta.expect("meta")["cached_tokens"], 0);
    let cmd = std::fs::read_to_string(format!("/proc/{}/cmdline", ad.server_pid())).ok();
    if let Some(cmd) = cmd {
        assert!(cmd.contains("--mmproj"), "{cmd}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
