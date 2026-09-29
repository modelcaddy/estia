//! Details of the HTTP contract the 0.4.0 acceptance run found wrong, each
//! against a stdlib-only fake runner: one load per model however many
//! requests arrive for it at once, `finish_reason: "length"`, base64
//! embeddings, the request-body limit and its JSON 413, empty embed input,
//! unique completion and tool-call ids, `load_ms`, error messages that name
//! no paths on the engine's machine, and per-runner pid and memory in
//! `/engine/stats`. Skipped without python3.

use estia_engine::models::ModelStore;
use estia_engine::runtime::PythonRuntime;
use estia_engine::{Engine, EngineConfig, MemoryPolicy};
use estia_server::tokens::{TokenStore, SCOPE_ADMIN};
use estia_server::{router, serve_router, AppState, ConnLimits, DEFAULT_MAX_BODY_BYTES};
use serde_json::{json, Value};
use std::sync::Arc;

fn python3_available() -> bool {
    std::process::Command::new("python3").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// A v2 fake that logs every process start to `spawns.log` and every load to
/// `loads.log` beside the script, and takes 600 ms to load, so requests that
/// arrive together overlap the load.
///
/// Chat answers by the last user message: `LEN` produces exactly `max_tokens`
/// tokens and says nothing about why it stopped; `RUNNER_LENGTH` produces two
/// and says `length`; `RUNNER_STOP` produces `max_tokens` and says `stop`;
/// anything else produces two tokens and says nothing. With tools it answers
/// with a Gemma tool call. A streamed `PREFILL` spends 5 s "prefilling"
/// before its first token, stops at a cancel and logs it to `cancels.log`.
/// Embeddings are a fixed 4-dim vector.
const FAKE: &str = r#"import sys, json, os, time, select
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "spawns.log"), "a") as f:
    f.write("%d\n" % os.getpid())
def out(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
VEC = [0.1, -2.5, 3.0e-5, 12345.678]
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
             "capabilities": {"generate": True, "stream": True, "embed": True, "cancel": True, "load": True, "chat": True,
                              "tools": True, "prompt_cache": True, "count_tokens": True}})
    elif t == "load":
        with open(os.path.join(HERE, "loads.log"), "a") as f:
            f.write(json.dumps({"kind": req.get("kind"), "model": os.path.basename(req.get("model_path", ""))}) + "\n")
        time.sleep(0.6)
        out({"ok": True, "loaded": True, "ms": 600})
    elif t in ("chat", "chat_stream"):
        last = req["messages"][-1]["content"]
        cap = req.get("max_tokens") or 1024
        meta = {"prompt_tokens": 5, "cached_tokens": 0, "template": "native"}
        if t == "chat_stream" and "PREFILL" in last:
            # A long prompt: 5 s of prefill before the first token, cancellable
            # between chunks as the MLX runner is. A cancel is logged, with how
            # long into the prefill it came, to cancels.log.
            t0 = time.time()
            cancelled = False
            while time.time() - t0 < 5 and not cancelled:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    if json.loads(sys.stdin.readline()).get("type") == "cancel":
                        cancelled = True
            if cancelled:
                with open(os.path.join(HERE, "cancels.log"), "a") as f:
                    f.write("%.2f\n" % (time.time() - t0))
                out({"done": True, "cancelled": True})
                continue
            pieces = ["late ", "tokens"]
        elif req.get("tools"):
            pieces = ['<|tool_call>call:get_weather{city:<|"|>Athens<|"|>}<tool_call|>']
        elif "RUNNER_LENGTH" in last:
            pieces = ["a ", "b"]
            meta["finish_reason"] = "length"
        elif "RUNNER_STOP" in last:
            pieces = ["w "] * cap
            meta["finish_reason"] = "stop"
        elif "LEN" in last:
            pieces = ["w "] * cap
        else:
            pieces = ["ok ", "done"]
        meta["generation_tokens"] = len(pieces)
        if t == "chat":
            out({"text": "".join(pieces), "meta": meta})
        else:
            for p in pieces:
                out({"type": "token", "text": p})
            out(dict(type="meta", **meta))
            out({"done": True})
    elif t == "embed_batch":
        out({"embeddings": [VEC for _ in req["inputs"]]})
    elif t == "generate":
        out({"text": "gen:" + req["prompt"]})
    else:
        out({"error": "unknown type %r" % (t,)})
"#;

const VEC: [f32; 4] = [0.1, -2.5, 3.0e-5, 12345.678];

struct Harness {
    base: String,
    admin: String,
    dir: std::path::PathBuf,
}

fn scratch_dir() -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "estia-contract-test-{}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// An MLX engine over the fake with `fast` (gemma4-e2b) and the embedding
/// model installed, and `text` (gemma4-e4b) not. `configure` runs on the
/// state before the router is built.
async fn start_with(configure: impl FnOnce(&AppState)) -> Harness {
    let dir = scratch_dir();
    std::fs::create_dir_all(dir.join("models/gemma4-e2b-it-4bit-mlx")).unwrap();
    std::fs::create_dir_all(dir.join("models/embeddinggemma-300m-4bit")).unwrap();
    let runner = dir.join("fake_runner.py");
    std::fs::write(&runner, FAKE).unwrap();
    let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), &runner)
        .with_python("python3")
        // The same on every machine: no budget and no slot limit unless a test sets them.
        .with_memory(MemoryPolicy::unlimited());
    let engine = Arc::new(Engine::new(cfg));
    let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
    let admin = tokens.mint("admin", &[SCOPE_ADMIN]).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(AppState::new(engine, tokens, true, addr));
    configure(&state);
    let app = router(Arc::clone(&state));
    tokio::spawn(async move {
        serve_router(listener, app, ConnLimits::default(), std::future::pending()).await.unwrap();
    });
    Harness { base: format!("http://{addr}"), admin, dir }
}

async fn start() -> Harness {
    start_with(|_| {}).await
}

async fn post(h: &Harness, path: &str, body: Value) -> (u16, Value) {
    let r = reqwest::Client::new().post(format!("{}{path}", h.base)).bearer_auth(&h.admin).json(&body).send().await.unwrap();
    (r.status().as_u16(), r.json().await.unwrap_or(Value::Null))
}

async fn get(h: &Harness, path: &str) -> (u16, Value) {
    let r = reqwest::Client::new().get(format!("{}{path}", h.base)).bearer_auth(&h.admin).send().await.unwrap();
    (r.status().as_u16(), r.json().await.unwrap_or(Value::Null))
}

/// The data events of an SSE body, `[DONE]` left out.
async fn sse(h: &Harness, path: &str, body: Value) -> Vec<Value> {
    let r = reqwest::Client::new().post(format!("{}{path}", h.base)).bearer_auth(&h.admin).json(&body).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let text = r.text().await.unwrap();
    text.lines().filter_map(|l| l.strip_prefix("data: ")).filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect()
}

fn log_lines(h: &Harness, name: &str) -> Vec<String> {
    std::fs::read_to_string(h.dir.join(name)).unwrap_or_default().lines().map(str::to_string).collect()
}

fn chat(content: &str, max_tokens: u32) -> Value {
    json!({"model": "fast", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens})
}

/// perf-F1: requests that arrive together for a model that is not loaded
/// start one runner and load the weights once; the others wait for that load
/// and then run on the same session. Each of them reports the wait as
/// `load_ms`; the next request, on a resident model, reports none.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_first_requests_share_one_load() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    let mut calls = Vec::new();
    for i in 0..4 {
        let h = Harness { base: h.base.clone(), admin: h.admin.clone(), dir: h.dir.clone() };
        calls.push(tokio::spawn(async move { post(&h, "/v1/chat/completions", chat(&format!("hello {i}"), 16)).await }));
    }
    for i in 0..3 {
        let h = Harness { base: h.base.clone(), admin: h.admin.clone(), dir: h.dir.clone() };
        calls.push(tokio::spawn(async move {
            post(&h, "/v1/embeddings", json!({"model": "embed", "input": [format!("doc {i}")], "encoding_format": "float"})).await
        }));
    }
    for c in calls {
        let (s, v) = c.await.unwrap();
        assert_eq!(s, 200, "{v}");
        let load_ms = v["x_estia"]["load_ms"].as_u64();
        assert!(load_ms.is_some_and(|ms| ms >= 300), "every first request waited for the load: {v}");
    }
    let spawns = log_lines(&h, "spawns.log");
    let loads = log_lines(&h, "loads.log");
    assert_eq!(spawns.len(), 2, "one runner per model, not one per request: {spawns:?} {loads:?}");
    assert_eq!(loads.len(), 2, "{loads:?}");
    assert!(loads.iter().any(|l| l.contains("generation")) && loads.iter().any(|l| l.contains("embedding")), "{loads:?}");

    // Resident now: no load, and none reported.
    let (s, v) = post(&h, "/v1/chat/completions", chat("again", 16)).await;
    assert_eq!(s, 200, "{v}");
    assert!(v["x_estia"]["load_ms"].is_null(), "{v}");
    assert!(v["x_estia"]["ms"].as_u64().is_some_and(|ms| ms < 600), "ms is generation time, not the load: {v}");
    assert_eq!(log_lines(&h, "spawns.log").len(), 2);

    // A load that fails is not retried by every waiter behind it, and the
    // error reaches them all.
    let mut calls = Vec::new();
    for _ in 0..3 {
        let h = Harness { base: h.base.clone(), admin: h.admin.clone(), dir: h.dir.clone() };
        calls.push(tokio::spawn(async move {
            post(&h, "/v1/chat/completions", json!({"model": "text", "messages": [{"role": "user", "content": "x"}]})).await
        }));
    }
    for c in calls {
        let (s, v) = c.await.unwrap();
        assert_eq!(s, 404, "{v}");
    }
}

/// api-F2: `finish_reason` is `length` when the budget ran out, from the
/// runner's own `meta.finish_reason` when it gives one and from the token
/// count when it does not: /v1 streamed and not, and /engine/generate.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn finish_reason_says_length_when_the_budget_ran_out() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    // (last user message, max_tokens, expected finish_reason)
    let cases = [("LEN", 8, "length"), ("RUNNER_LENGTH", 8, "length"), ("RUNNER_STOP", 8, "stop"), ("short", 8, "stop")];
    for (content, max, want) in cases {
        let (s, v) = post(&h, "/v1/chat/completions", chat(content, max)).await;
        assert_eq!(s, 200, "{v}");
        assert_eq!(v["choices"][0]["finish_reason"], want, "{content}: {v}");

        let mut body = chat(content, max);
        body["stream"] = json!(true);
        let chunks = sse(&h, "/v1/chat/completions", body).await;
        let last = chunks.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], want, "{content} streamed: {last}");
        assert!(chunks[..chunks.len() - 1].iter().all(|c| c["choices"][0]["finish_reason"].is_null()));

        let (s, v) =
            post(&h, "/engine/generate", json!({"model": "fast", "messages": [{"role": "user", "content": content}], "max_tokens": max}))
                .await;
        assert_eq!(s, 200, "{v}");
        assert_eq!(v["finish_reason"], want, "{content} /engine/generate: {v}");

        let events = sse(
            &h,
            "/engine/generate",
            json!({"model": "fast", "messages": [{"role": "user", "content": content}], "max_tokens": max, "stream": true}),
        )
        .await;
        let done = events.iter().find(|e| e["done"] == true).expect("done event");
        assert_eq!(done["finish_reason"], want, "{content} /engine/generate streamed: {done}");
    }
    // The default budget counts too: 1024 tokens with no max_tokens given.
    let (_, v) = post(&h, "/v1/chat/completions", json!({"model": "fast", "messages": [{"role": "user", "content": "LEN"}]})).await;
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{}", v["usage"]);
    // Tool calls still say tool_calls.
    let (_, v) = post(
        &h,
        "/v1/chat/completions",
        json!({"model": "fast", "messages": [{"role": "user", "content": "weather?"}], "max_tokens": 8,
               "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]}),
    )
    .await;
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls", "{v}");
    // A raw prompt has no accounting to go by.
    let (_, v) = post(&h, "/engine/generate", json!({"model": "fast", "prompt": "hi"})).await;
    assert!(v["finish_reason"].is_null(), "{v}");
}

/// Standard base64 → bytes, for checking what the server sent.
fn b64_decode(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            other => panic!("not standard base64: {:?}", other as char),
        }
    };
    assert_eq!(s.len() % 4, 0, "padded to a multiple of 4: {s}");
    let mut out = Vec::new();
    for q in s.as_bytes().chunks(4) {
        let pad = q.iter().filter(|c| **c == b'=').count();
        let n = q.iter().take(4 - pad).enumerate().fold(0u32, |n, (i, c)| n | val(*c) << (18 - 6 * i));
        out.extend_from_slice(&n.to_be_bytes()[1..4 - pad]);
    }
    out
}

/// examples-1: `encoding_format: "base64"` (what OpenAI's JS SDK asks for
/// by default) returns each vector as its little-endian f32 bytes in standard
/// base64, which decode to exactly the floats `float` returns.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn embeddings_in_base64_decode_to_the_floats() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    let (s, floats) = post(&h, "/v1/embeddings", json!({"model": "embed", "input": ["a", "b"], "encoding_format": "float"})).await;
    assert_eq!(s, 200, "{floats}");
    let (s, v) = post(&h, "/v1/embeddings", json!({"model": "embed", "input": ["a", "b"]})).await;
    assert_eq!(s, 200);
    assert_eq!(v["data"], floats["data"], "float is the default");
    let (s, b64) = post(&h, "/v1/embeddings", json!({"model": "embed", "input": ["a", "b"], "encoding_format": "base64"})).await;
    assert_eq!(s, 200, "{b64}");
    for (i, item) in b64["data"].as_array().unwrap().iter().enumerate() {
        assert_eq!(item["index"], i);
        let bytes = b64_decode(item["embedding"].as_str().expect("a base64 string"));
        let decoded: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        assert_eq!(decoded, VEC, "round trip");
        let as_floats: Vec<f32> = floats["data"][i]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        assert_eq!(decoded, as_floats, "the same vector either way");
    }
    let (s, v) = post(&h, "/v1/embeddings", json!({"model": "embed", "input": "a", "encoding_format": "int8"})).await;
    assert_eq!(s, 400, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("encoding_format"), "{v}");
}

/// api-F4: a full embed batch of large inputs fits the default body limit
/// (axum's own default, 2 MB, refused it); a body over the configured limit
/// gets a JSON 413 with `type`, `message` and `request_id`, and so does any
/// other request axum's extractors refuse.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn body_limit_is_generous_and_its_413_is_json() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    assert_eq!(DEFAULT_MAX_BODY_BYTES, 32 * 1024 * 1024);
    let h = start().await;
    // 256 inputs of ~100 KB: ~25 MB of JSON.
    let input: Vec<String> = (0..256).map(|i| format!("{i} {}", "x".repeat(100_000))).collect();
    let (s, v) = post(&h, "/v1/embeddings", json!({"model": "embed", "input": input})).await;
    assert_eq!(s, 200, "{}", v["error"]);
    assert_eq!(v["data"].as_array().unwrap().len(), 256);

    let small = start_with(|st| st.set_max_body_bytes(64 * 1024)).await;
    let big = json!({"model": "embed", "input": ["y".repeat(100_000)]});
    let r = reqwest::Client::new()
        .post(format!("{}/v1/embeddings", small.base))
        .bearer_auth(&small.admin)
        .header("x-request-id", "big-body-1")
        .json(&big)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 413);
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("application/json"), "{:?}", r.headers());
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    assert_eq!(v["error"]["code"], 413, "{v}");
    assert_eq!(v["error"]["request_id"], "big-body-1", "{v}");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("65536") && msg.contains("--max-body-bytes") && msg.contains("ESTIA_MAX_BODY_BYTES"), "{msg}");
    // Under the limit it still works.
    let (s, _) = post(&small, "/v1/embeddings", json!({"model": "embed", "input": "small"})).await;
    assert_eq!(s, 200);

    // Malformed JSON and a missing field: the same envelope, not text/plain.
    let r = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth(&h.admin)
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 400);
    let v: Value = r.json().await.expect("a JSON error body");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    assert!(v["error"]["request_id"].is_string(), "{v}");
    let (s, v) = post(&h, "/v1/chat/completions", json!({"messages": []})).await;
    assert_eq!(s, 422);
    assert!(v["error"]["message"].as_str().unwrap().contains("model"), "{v}");
}

/// api-F5, api-F6, api-F9: empty embed input is a 400 on /engine/embed as on
/// /v1; completion and tool-call ids never repeat; the model-not-installed
/// 404 names no path on the engine's machine and says how to install it; an
/// unknown name says it could be a model or a role.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ids_errors_and_empty_input() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    let (s, v) = post(&h, "/engine/embed", json!({"inputs": []})).await;
    assert_eq!(s, 400, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("empty"), "{v}");
    let (s, v) = post(&h, "/v1/embeddings", json!({"model": "embed", "input": []})).await;
    assert_eq!(s, 400, "{v}");
    assert_eq!(log_lines(&h, "spawns.log").len(), 0, "an empty request loads nothing");

    let hex24 = |id: &str, prefix: &str| {
        id.strip_prefix(prefix).is_some_and(|rest| rest.len() == 24 && rest.chars().all(|c| c.is_ascii_hexdigit()))
    };
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]);
    let mut completion_ids = std::collections::HashSet::new();
    let mut call_ids = std::collections::HashSet::new();
    for _ in 0..3 {
        let (s, v) =
            post(&h, "/v1/chat/completions", json!({"model": "fast", "messages": [{"role": "user", "content": "w?"}], "tools": tools}))
                .await;
        assert_eq!(s, 200, "{v}");
        let id = v["id"].as_str().unwrap().to_string();
        assert!(hex24(&id, "chatcmpl-"), "{id}");
        assert!(completion_ids.insert(id.clone()), "{id} repeated");
        let call = v["choices"][0]["message"]["tool_calls"][0]["id"].as_str().unwrap().to_string();
        assert!(hex24(&call, "call_"), "{call}");
        assert!(call_ids.insert(call.clone()), "{call} repeated");
    }
    // Streamed: every chunk of one response carries the same id; the call id is new.
    let chunks = sse(
        &h,
        "/v1/chat/completions",
        json!({"model": "fast", "messages": [{"role": "user", "content": "w?"}], "tools": tools, "stream": true}),
    )
    .await;
    let id = chunks[0]["id"].as_str().unwrap();
    assert!(chunks.iter().all(|c| c["id"] == id), "one id per response");
    assert!(completion_ids.insert(id.to_string()));
    let call = chunks.last().unwrap()["choices"][0]["delta"]["tool_calls"][0]["id"].as_str().unwrap();
    assert!(hex24(call, "call_") && call_ids.insert(call.to_string()), "{call}");
    // /engine/generate hands out fresh call ids too.
    let (_, v) =
        post(&h, "/engine/generate", json!({"model": "fast", "messages": [{"role": "user", "content": "w?"}], "tools": tools})).await;
    let call = v["tool_calls"][0]["id"].as_str().unwrap();
    assert!(hex24(call, "call_") && call_ids.insert(call.to_string()), "{call}");

    // `text` is gemma4-e4b, which this engine has not downloaded.
    let (s, v) = post(&h, "/v1/chat/completions", json!({"model": "text", "messages": [{"role": "user", "content": "x"}]})).await;
    assert_eq!(s, 404, "{v}");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("gemma4-e4b-it-4bit-mlx") && msg.contains("not installed") && msg.contains("estia pull gemma4-e4b-it-4bit-mlx"),
        "{msg}"
    );
    let data_dir = h.dir.to_string_lossy();
    // Before: "model `…` is not downloaded at /Users/…/models/gemma4-e4b-it-4bit-mlx".
    assert!(!msg.contains(&*data_dir) && !msg.contains("/models/gemma4") && !msg.contains(" at /"), "no local paths: {msg}");
    let (s, v) = post(&h, "/engine/generate", json!({"model": "text", "prompt": "x"})).await;
    assert_eq!(s, 404);
    assert!(!v["error"]["message"].as_str().unwrap().contains(&*data_dir), "{v}");

    let (s, v) = post(&h, "/v1/chat/completions", json!({"model": "writer", "messages": [{"role": "user", "content": "x"}]})).await;
    assert_eq!(s, 404, "{v}");
    assert_eq!(v["error"]["message"], "unknown model or role `writer`", "{v}");
    let (s, v) = post(&h, "/v1/embeddings", json!({"model": "no-such-embedder", "input": "x"})).await;
    assert_eq!(s, 404, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().starts_with("unknown model or role"), "{v}");
}

/// /engine/stats: each loaded model with its runner's pid and the physical
/// memory that runner holds; `loaded` stays the list of ids.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stats_name_each_runner_pid_and_memory() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    let (s, v) = get(&h, "/engine/stats").await;
    assert_eq!(s, 200);
    assert_eq!(v["models"], json!([]), "{v}");
    let (s, _) = post(&h, "/v1/chat/completions", chat("hi", 8)).await;
    assert_eq!(s, 200);
    let (s, _) = post(&h, "/engine/embed", json!({"inputs": ["x"]})).await;
    assert_eq!(s, 200);
    let (s, v) = get(&h, "/engine/stats").await;
    assert_eq!(s, 200);
    assert_eq!(v["loaded"], json!(["embeddinggemma-300m-4bit", "gemma4-e2b-it-4bit-mlx"]), "{v}");
    let spawned: Vec<u64> = log_lines(&h, "spawns.log").iter().map(|l| l.parse().unwrap()).collect();
    let models = v["models"].as_array().unwrap();
    assert_eq!(models.len(), 2, "{v}");
    for m in models {
        let pid = m["pid"].as_u64().unwrap_or_else(|| panic!("a pid: {m}"));
        assert!(spawned.contains(&pid), "the runner's own pid {pid}, one of {spawned:?}");
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let bytes = m["memory_bytes"].as_u64().unwrap_or_else(|| panic!("memory: {m}"));
            assert!(bytes > 1024 * 1024, "a python process holds more than 1 MiB: {m}");
        }
    }
    let kinds: Vec<&str> = models.iter().filter_map(|m| m["kind"].as_str()).collect();
    assert!(kinds.contains(&"generation") && kinds.contains(&"embedding"), "{kinds:?}");
}

/// api-F3: a client that goes away while a long prompt is prefilled (no
/// token sent yet) cancels the generation within about a second, on /v1 and
/// on /engine/generate, and the next request on the model is answered at
/// once. Before, the cancel waited for the first token to fail to send: the
/// whole prefill.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_that_leaves_during_prefill_cancels_it() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    let (s, _) = post(&h, "/v1/chat/completions", chat("warm up", 8)).await;
    assert_eq!(s, 200);
    let generate = json!({"model": "fast", "messages": [{"role": "user", "content": "PREFILL"}], "max_tokens": 16, "stream": true});
    let mut v1 = chat("PREFILL", 16);
    v1["stream"] = json!(true);
    for (n, (path, body)) in [("/v1/chat/completions", v1), ("/engine/generate", generate)].into_iter().enumerate() {
        let mut r = reqwest::Client::new().post(format!("{}{path}", h.base)).bearer_auth(&h.admin).json(&body).send().await.unwrap();
        assert_eq!(r.status().as_u16(), 200);
        if path.starts_with("/v1") {
            // The role chunk goes out before the prefill starts.
            let first = r.chunk().await.unwrap().unwrap();
            assert!(String::from_utf8_lossy(&first).contains("assistant"));
        }
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let left = std::time::Instant::now();
        drop(r);
        let cancelled_after = loop {
            let lines = log_lines(&h, "cancels.log");
            if lines.len() > n {
                break Some(left.elapsed());
            }
            if left.elapsed() > std::time::Duration::from_secs(4) {
                break None;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        };
        let after = cancelled_after.unwrap_or_else(|| panic!("{path}: no cancel reached the runner during the prefill"));
        assert!(after < std::time::Duration::from_millis(1500), "{path}: cancel reached the runner {after:?} after the client left");
        let t = std::time::Instant::now();
        let (s, v) = post(&h, "/v1/chat/completions", chat("next", 8)).await;
        assert_eq!(s, 200, "{v}");
        assert!(t.elapsed() < std::time::Duration::from_secs(2), "{path}: the next request waited {:?}", t.elapsed());
    }
}

/// api-F8: models list what reaches them through the API. Images reach the
/// Gemma 4 models, so each says `vision` on /v1/models and /engine/models,
/// and the `vision` role binds to one of them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn models_claim_vision_only_when_images_reach_them() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    let (s, v) = get(&h, "/v1/models").await;
    assert_eq!(s, 200);
    let gen: Vec<&Value> =
        v["data"].as_array().unwrap().iter().filter(|m| m["x_estia"]["family"].is_string() && m["x_estia"]["role"].is_null()).collect();
    assert!(!gen.is_empty(), "{v}");
    for m in gen {
        assert_eq!(m["x_estia"]["capabilities"], json!(["text", "tools", "vision"]), "{m}");
    }
    let (s, v) = get(&h, "/engine/models").await;
    assert_eq!(s, 200);
    for m in v["generation"].as_array().unwrap() {
        assert!(m["capabilities"].as_array().unwrap().contains(&json!("vision")), "{m}");
    }
    let (s, defaults) = get(&h, "/engine/defaults").await;
    assert_eq!(s, 200);
    let r = reqwest::Client::new().put(format!("{}/engine/defaults", h.base)).bearer_auth(&h.admin).json(&defaults).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200, "the default table, vision included, passes its own check");
}

fn loaded(v: &Value) -> Vec<String> {
    v["loaded"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect()
}

const GIB: u64 = 1024 * 1024 * 1024;

/// mem-F1: before a model loads, idle models are unloaded, least recently
/// used first, until it fits the memory budget; `/engine/stats` reports the
/// budget and what the resident models use.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_load_unloads_the_least_recently_used_model_to_fit_the_budget() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    // E2B (~2.8 GB estimated) and the embedding model fit 5 GB together;
    // E4B (~3.9 GB) fits only once E2B, the older of the two, is gone.
    let h = start_with(|s| s.set_memory_policy(MemoryPolicy::unlimited().with_budget(Some(5 * GIB)))).await;
    std::fs::create_dir_all(h.dir.join("models/gemma4-e4b-it-4bit-mlx")).unwrap();
    assert_eq!(post(&h, "/v1/chat/completions", chat("hi", 8)).await.0, 200);
    assert_eq!(post(&h, "/engine/embed", json!({"inputs": ["x"]})).await.0, 200);
    let (_, v) = get(&h, "/engine/stats").await;
    assert_eq!(loaded(&v), ["embeddinggemma-300m-4bit", "gemma4-e2b-it-4bit-mlx"], "{v}");
    assert_eq!(v["memory"]["budget_bytes"], json!(5 * GIB), "{v}");
    assert!(v["memory"]["used_bytes"].as_u64().unwrap() > 2 * GIB, "estimates count: {v}");
    let text = json!({"model": "text", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8});
    let (s, body) = post(&h, "/v1/chat/completions", text).await;
    assert_eq!(s, 200, "{body}");
    let (_, v) = get(&h, "/engine/stats").await;
    assert_eq!(loaded(&v), ["embeddinggemma-300m-4bit", "gemma4-e4b-it-4bit-mlx"], "E2B made room: {v}");
    assert!(v["memory"]["used_bytes"].as_u64().unwrap() <= 5 * GIB, "{v}");
}

/// mem-F2: a model larger than the whole budget is refused with a 503 that
/// names it and says what to do, and nothing is started or unloaded for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_model_larger_than_the_budget_is_refused() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start_with(|s| s.set_memory_policy(MemoryPolicy::unlimited().with_budget(Some(3 * GIB)))).await;
    std::fs::create_dir_all(h.dir.join("models/gemma4-e4b-it-4bit-mlx")).unwrap();
    assert_eq!(post(&h, "/v1/chat/completions", chat("hi", 8)).await.0, 200);
    let spawns = log_lines(&h, "spawns.log").len();
    let text = json!({"model": "text", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8});
    let (s, v) = post(&h, "/v1/chat/completions", text).await;
    assert_eq!(s, 503, "{v}");
    assert_eq!(v["error"]["type"], "insufficient_memory", "{v}");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("gemma4-e4b-it-4bit-mlx") && msg.contains("--memory-budget"), "{msg}");
    assert_eq!(log_lines(&h, "spawns.log").len(), spawns, "no runner started for it");
    let (_, v) = get(&h, "/engine/stats").await;
    assert_eq!(loaded(&v), ["gemma4-e2b-it-4bit-mlx"], "nothing unloaded for a load that cannot fit: {v}");
}

/// mem-F3: with one generation model at a time (a constrained machine),
/// switching models unloads the other one; the embedding model stays.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_generation_model_at_a_time_when_the_policy_says_so() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start_with(|s| s.set_memory_policy(MemoryPolicy { max_generation_models: Some(1), ..MemoryPolicy::unlimited() })).await;
    std::fs::create_dir_all(h.dir.join("models/gemma4-e4b-it-4bit-mlx")).unwrap();
    assert_eq!(post(&h, "/engine/embed", json!({"inputs": ["x"]})).await.0, 200);
    assert_eq!(post(&h, "/v1/chat/completions", chat("hi", 8)).await.0, 200);
    let text = json!({"model": "text", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8});
    assert_eq!(post(&h, "/v1/chat/completions", text).await.0, 200);
    let (_, v) = get(&h, "/engine/stats").await;
    assert_eq!(loaded(&v), ["embeddinggemma-300m-4bit", "gemma4-e4b-it-4bit-mlx"], "{v}");
    assert_eq!(v["memory"]["max_generation_models"], 1, "{v}");
}

/// local-F1: an app on this machine trades the same-user secret for a token
/// named `local-<app>` with narrow scopes; a wrong secret, a bad name or a
/// wider scope is refused, and asking again replaces the earlier token.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_local_app_trades_the_secret_for_a_narrow_token() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start_with(|s| s.set_local_secret(Some("s3cret".into()))).await;
    let client = reqwest::Client::new();
    let ask = |body: Value| {
        let (client, url) = (client.clone(), format!("{}/engine/local-token", h.base));
        async move {
            let r = client.post(url).json(&body).send().await.unwrap();
            (r.status().as_u16(), r.json::<Value>().await.unwrap_or(Value::Null))
        }
    };
    let (s, v) = ask(json!({"secret": "nope", "name": "modelcaddy"})).await;
    assert_eq!(s, 403, "{v}");
    let (s, _) = ask(json!({"secret": "s3cret", "name": "Model Caddy"})).await;
    assert_eq!(s, 400);
    let (s, _) = ask(json!({"secret": "s3cret", "name": "modelcaddy", "scopes": ["admin"]})).await;
    assert_eq!(s, 400);
    let (s, v) = ask(json!({"secret": "s3cret", "name": "modelcaddy"})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["name"], "local-modelcaddy");
    assert_eq!(v["scopes"], json!(["generate", "embed"]));
    let token = v["token"].as_str().unwrap().to_string();
    let embed = client.post(format!("{}/engine/embed", h.base)).bearer_auth(&token).json(&json!({"inputs": ["x"]})).send().await.unwrap();
    assert_eq!(embed.status().as_u16(), 200, "the token embeds");
    let admin = client.get(format!("{}/engine/pairings", h.base)).bearer_auth(&token).send().await.unwrap();
    assert_eq!(admin.status().as_u16(), 403, "and is not admin");
    // Asking again replaces it: the first token stops working.
    let (s, v) = ask(json!({"secret": "s3cret", "name": "modelcaddy"})).await;
    assert_eq!(s, 200);
    assert_ne!(v["token"].as_str().unwrap(), token);
    let old = client.post(format!("{}/engine/embed", h.base)).bearer_auth(&token).json(&json!({"inputs": ["x"]})).send().await.unwrap();
    assert_eq!(old.status().as_u16(), 401);
}
