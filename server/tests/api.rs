//! The HTTP surface against stdlib-only fake runners: auth, models, chat
//! completions (tool calls, response_format), embeddings, fingerprint checks,
//! native generate, the Host/Origin guard, per-token prompt caches, work caps
//! and unknown routes on an MLX engine; and on a llama.cpp engine, backend
//! resolution, runner-parsed tool calls, `format` forwarded to a runner that
//! constrains decoding, and imported models. Skipped without python3.

use estia_engine::models::ModelStore;
use estia_engine::runtime::PythonRuntime;
use estia_engine::{Engine, EngineConfig, LlamaLaunch, LlamaServer};
use estia_server::engine_api::{MAX_EMBED_INPUTS, MAX_TOKENS_CEILING};
use estia_server::tokens::{TokenStore, SCOPE_ADMIN, SCOPE_EMBED, SCOPE_GENERATE};
use estia_server::{router, serve_router, AppState, ConnLimits};
use serde_json::{json, Value};
use std::io::Write as _;
use std::sync::Arc;

fn python3_available() -> bool {
    std::process::Command::new("python3").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// A v2 fake: chat echoes the roles and the last user turn, or answers a
/// Gemma-style tool call when tools are declared; embed returns 2-dim vectors.
/// Every chat request's `cache_key` and `max_tokens` are appended to
/// `chat_log.jsonl` beside the script, so tests can see what reached the runner.
const FAKE: &str = r#"import sys, json, os
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_log.jsonl")
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
             "capabilities": {"generate": True, "stream": True, "embed": True, "cancel": True, "load": True, "chat": True, "tools": True, "prompt_cache": True, "count_tokens": True}})
    elif t == "load":
        out({"ok": True, "loaded": True, "ms": 1})
    elif t in ("chat", "chat_stream"):
        with open(LOG, "a") as f:
            f.write(json.dumps({"cache_key": req.get("cache_key"), "max_tokens": req.get("max_tokens"), "format": req.get("format"),
                                "system": [m["content"] for m in req["messages"] if m["role"] == "system"]}) + "\n")
        msgs = req["messages"]
        last = msgs[-1]["content"]
        if req.get("tools"):
            text = '<|tool_call>call:get_weather{city:<|"|>Athens<|"|>}<tool_call|>'
        elif "json" in last.lower():
            text = '```json\n{"title": "T", "facts": ["a"]}\n```'
        else:
            text = "roles:" + ",".join(m["role"] for m in msgs) + "|" + last
        meta = {"type": "meta", "prompt_tokens": 11, "cached_tokens": 3 if req.get("cache_key") else 0, "generation_tokens": 4, "template": "native"}
        if t == "chat":
            out({"text": text, "meta": {k: v for k, v in meta.items() if k != "type"}})
        else:
            for piece in [text[:5], text[5:]]:
                out({"type": "token", "text": piece})
            out(meta)
            out({"done": True})
    elif t == "embed_batch":
        out({"embeddings": [[1.0, 0.0] for _ in req["inputs"]]})
    elif t == "generate":
        out({"text": "gen:" + req["prompt"]})
    else:
        out({"error": "unknown type %r" % (t,)})
"#;

struct Harness {
    base: String,
    addr: std::net::SocketAddr,
    admin: String,
    embed_only: String,
    state: Arc<AppState>,
    _dir: std::path::PathBuf,
}

async fn start() -> Harness {
    start_with(ConnLimits::default()).await
}

/// A fresh scratch directory per harness.
fn scratch_dir() -> std::path::PathBuf {
    // Tests run in parallel and the clock is coarse (µs on macOS): a counter
    // keeps two harnesses from sharing a directory and its tokens.json.
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "estia-server-test-{}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

async fn start_with(limits: ConnLimits) -> Harness {
    let dir = scratch_dir();
    std::fs::create_dir_all(dir.join("models/gemma4-e2b-it-4bit-mlx")).unwrap();
    std::fs::create_dir_all(dir.join("models/gemma4-e4b-it-4bit-mlx")).unwrap();
    std::fs::create_dir_all(dir.join("models/embeddinggemma-300m-4bit")).unwrap();
    let runner = dir.join("fake_runner.py");
    std::fs::File::create(&runner).unwrap().write_all(FAKE.as_bytes()).unwrap();
    let cfg =
        EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), &runner).with_python("python3");
    serve_engine(dir, cfg, limits).await
}

async fn serve_engine(dir: std::path::PathBuf, cfg: EngineConfig, limits: ConnLimits) -> Harness {
    let engine = Arc::new(Engine::new(cfg));
    let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
    let admin = tokens.mint("admin", &[SCOPE_ADMIN]).unwrap();
    let embed_only = tokens.mint("embedder", &[SCOPE_EMBED]).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(AppState::new(engine, tokens, true, addr));
    let app = router(Arc::clone(&state));
    tokio::spawn(async move {
        serve_router(listener, app, limits, std::future::pending()).await.unwrap();
    });
    Harness { base: format!("http://{addr}"), addr, admin, embed_only, state, _dir: dir }
}

/// The last chat request the fake runner saw: (cache_key, max_tokens).
fn last_chat(h: &Harness) -> (Option<String>, Option<u64>) {
    let log = std::fs::read_to_string(h._dir.join("chat_log.jsonl")).unwrap_or_default();
    let v: Value = serde_json::from_str(log.lines().last().expect("a chat reached the runner")).unwrap();
    (v["cache_key"].as_str().map(str::to_string), v["max_tokens"].as_u64())
}

/// The last line of the fake runner's request log, whole.
fn last_logged(h: &Harness) -> Value {
    let log = std::fs::read_to_string(h._dir.join("chat_log.jsonl")).unwrap_or_default();
    serde_json::from_str(log.lines().last().expect("a chat reached the runner")).unwrap()
}

/// A request with explicit headers (a `Host` of our choosing, an `Origin`).
async fn send(
    h: &Harness,
    method: reqwest::Method,
    path: &str,
    token: Option<&str>,
    headers: &[(&str, &str)],
    body: Option<Value>,
) -> (u16, Value) {
    let mut req = reqwest::Client::new().request(method, format!("{}{path}", h.base));
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    if let Some(b) = body {
        req = req.json(&b);
    }
    let r = req.send().await.unwrap();
    (r.status().as_u16(), r.json().await.unwrap_or(Value::Null))
}

async fn post(h: &Harness, token: &str, path: &str, body: Value) -> (u16, Value) {
    let r = reqwest::Client::new().post(format!("{}{path}", h.base)).bearer_auth(token).json(&body).send().await.unwrap();
    let status = r.status().as_u16();
    let v: Value = r.json().await.unwrap_or(Value::Null);
    (status, v)
}

async fn get(h: &Harness, token: Option<&str>, path: &str) -> (u16, Value) {
    let mut req = reqwest::Client::new().get(format!("{}{path}", h.base));
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let r = req.send().await.unwrap();
    (r.status().as_u16(), r.json().await.unwrap_or(Value::Null))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn health_auth_models_chat_embed() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;

    // Health is open; everything else needs a token with the right scope.
    let (s, v) = get(&h, None, "/engine/health").await;
    assert_eq!(s, 200);
    assert_eq!(v["api_version"], 1);
    let (s, v) = get(&h, None, "/v1/models").await;
    assert_eq!(s, 401, "{v}");
    let (s, _) = get(&h, Some("estia_bogus"), "/v1/models").await;
    assert_eq!(s, 401);
    let (s, v) =
        post(&h, &h.embed_only, "/v1/chat/completions", json!({"model": "text", "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(s, 403, "embed-only token cannot generate: {v}");

    // Models list roles and artifacts.
    let (s, v) = get(&h, Some(&h.admin), "/v1/models").await;
    assert_eq!(s, 200);
    let ids: Vec<&str> = v["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"text") && ids.contains(&"gemma4-e2b-it-4bit-mlx") && ids.contains(&"embeddinggemma-300m-4bit"), "{ids:?}");

    // Chat completion by role, template-rendered, usage from meta.
    let (s, v) = post(&h, &h.admin, "/v1/chat/completions", json!({"model": "fast", "messages": [{"role": "system", "content": "terse"}, {"role": "user", "content": [{"type": "text", "text": "hello there"}]}], "user": "conv-1"})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "gemma4-e2b-it-4bit-mlx");
    assert_eq!(v["choices"][0]["message"]["content"], "roles:system,user|hello there");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 11);
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 3, "cache_key from `user` reached the runner");
    assert_eq!(v["x_estia"]["family"], "gemma4-e2b");

    // Tools → OpenAI tool_calls parsed from Gemma's syntax.
    let (s, v) = post(
        &h,
        &h.admin,
        "/v1/chat/completions",
        json!({"model": "text", "messages": [{"role": "user", "content": "weather in Athens?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]}),
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let call = &v["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(), json!({"city": "Athens"}));

    // response_format json_schema: fenced output repaired and validated.
    let (s, v) = post(&h, &h.admin, "/v1/chat/completions", json!({"model": "text", "messages": [{"role": "user", "content": "give me json"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "d", "schema": {"type": "object", "required": ["title", "facts"]}}}})).await;
    assert_eq!(s, 200, "{v}");
    let content: Value = serde_json::from_str(v["choices"][0]["message"]["content"].as_str().unwrap()).unwrap();
    assert_eq!(content["title"], "T");
    assert_eq!(v["x_estia"]["repaired"], true);
    assert_eq!(v["x_estia"]["backend"], "mlx-python");
    let logged = last_logged(&h);
    assert!(logged["format"].is_null(), "a runner without `structured` is not sent a format");
    // Instead the schema is shown to the model, in a system message.
    let system = logged["system"][0].as_str().unwrap_or_default();
    assert!(system.contains("JSON Schema") && system.contains(r#""required":["title","facts"]"#), "{logged}");

    // Streaming: role chunk, content deltas, final chunk with usage, [DONE].
    let r = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth(&h.admin)
        .json(&json!({"model": "text", "messages": [{"role": "user", "content": "stream me"}], "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let body = r.text().await.unwrap();
    let datas: Vec<&str> = body.lines().filter_map(|l| l.strip_prefix("data: ")).collect();
    assert_eq!(datas.last().copied(), Some("[DONE]"), "{body}");
    let chunks: Vec<Value> = datas.iter().filter(|d| **d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let text: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(text, "roles:user|stream me");
    assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "stop");
    assert_eq!(chunks.last().unwrap()["usage"]["prompt_tokens"], 11);

    // Embeddings: prefix applied engine-side, fingerprint reported, mismatch refused.
    let (s, v) = post(&h, &h.embed_only, "/v1/embeddings", json!({"model": "embeddinggemma-300m-4bit", "input": ["a", "b"]})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["data"].as_array().unwrap().len(), 2);
    assert_eq!(v["x_estia"]["fingerprint"], "embeddinggemma-300m-4bit@mlx-python");
    let (s, v) =
        post(&h, &h.admin, "/engine/embed", json!({"inputs": "x", "expect_fingerprint": "nomic-embed-text-v1.5@mlx-python"})).await;
    assert_eq!(s, 422, "{v}");
    let (s, v) = post(
        &h,
        &h.admin,
        "/engine/embed",
        json!({"inputs": "x", "task": "query", "expect_fingerprint": "embeddinggemma-300m-4bit@mlx-python"}),
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["dims"], 768);

    // Native generate with a raw prompt and with a schema.
    let (s, v) = post(&h, &h.admin, "/engine/generate", json!({"model": "gemma4-e2b", "prompt": "raw"})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["text"], "gen:raw");
    let (s, v) = post(
        &h,
        &h.admin,
        "/engine/generate",
        json!({"messages": [{"role": "user", "content": "json please"}], "format": {"type": "json_object"}}),
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["json"]["title"], "T");
    assert_eq!(v["repaired"], true);
    assert!(last_logged(&h)["system"][0].as_str().unwrap_or_default().contains("one JSON object"), "{}", last_logged(&h));
    // Plain text gets no such system message.
    post(&h, &h.admin, "/engine/generate", json!({"messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(last_logged(&h)["system"], json!([]));

    // Defaults: bind check enforced through the API.
    let (s, v) = get(&h, Some(&h.admin), "/engine/defaults").await;
    assert_eq!(s, 200);
    assert_eq!(v["text"]["family"], "gemma4-e4b");
    // Roles set over the API persist to config.json (survive a restart).
    let mut roles = v.clone();
    roles["writer"] = json!({"family": "gemma4-e2b"});
    let r = reqwest::Client::new().put(format!("{}/engine/defaults", h.base)).bearer_auth(&h.admin).json(&roles).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(h._dir.join("config.json")).unwrap()).unwrap();
    assert_eq!(saved["roles"]["writer"]["family"], "gemma4-e2b");
    let r = reqwest::Client::new().put(format!("{}/engine/defaults", h.base)).bearer_auth(&h.embed_only).json(&roles).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 403, "roles need admin");

    // The test client page is open; the runtime install it drives needs admin.
    let r = reqwest::Client::new().get(format!("{}/client", h.base)).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert!(r.text().await.unwrap().contains("/engine/pair"));
    let (s, _) = post(&h, &h.embed_only, "/engine/runtime/install", json!({})).await;
    assert_eq!(s, 403);

    let (s, _) = post(&h, &h.admin, "/engine/models/pull", json!({"id": "no-such-model"})).await;
    assert_eq!(s, 404);
    let (s, v) = get(&h, Some(&h.admin), "/engine/stats").await;
    assert_eq!(s, 200);
    assert!(v["loaded"].as_array().unwrap().len() >= 2, "{v}");

    // Pairing: open request, pending until the operator approves (through the
    // shared store), then the token is handed over exactly once and works.
    let (s, v) = post(&h, "", "/engine/pair", json!({"name": "phone", "scopes": ["generate"]})).await;
    assert_eq!(s, 202, "{v}");
    let id = v["id"].as_str().unwrap().to_string();
    let (s, v) = get(&h, None, &format!("/engine/pair/{id}")).await;
    assert_eq!(s, 200);
    assert_eq!(v["status"], "pending");
    let store = estia_server::pairing::PairingStore::new(&h._dir);
    let tokens = TokenStore::open(h._dir.join("tokens.json")).unwrap();
    store.approve(&id, &tokens).unwrap();
    let (s, v) = get(&h, None, &format!("/engine/pair/{id}")).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["status"], "approved");
    let paired = v["token"].as_str().expect("token once").to_string();
    let (_, v) = get(&h, None, &format!("/engine/pair/{id}")).await;
    assert!(v["token"].is_null(), "second poll gets no token");
    // The approval was written by another TokenStore instance (as `estia pair
    // approve` does); the daemon must honour it without a restart.
    let (s, v) =
        post(&h, &paired, "/v1/chat/completions", json!({"model": "fast", "messages": [{"role": "user", "content": "paired hello"}]}))
            .await;
    assert_eq!(s, 200, "paired token generates: {v}");
    let (s, _) = post(&h, &paired, "/v1/embeddings", json!({"model": "embeddinggemma-300m-4bit", "input": "x"})).await;
    assert_eq!(s, 403, "paired token lacks embed scope");
}

/// DNS rebinding: a page on `attacker.example` that re-points its name at the
/// engine is same-origin with it, but its requests carry `Host:
/// attacker.example`. Those are refused on every route, open ones included,
/// even with a valid token; IP literals, localhost, `.local` and allowed names
/// pass. A state-changing request whose `Origin` is not the origin it was sent
/// to is refused; same-origin (the /client page) passes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn host_guard_blocks_rebinding_and_cross_origin_writes() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    let port = h.addr.port();
    let get = reqwest::Method::GET;
    let post = reqwest::Method::POST;

    // Rebinding names are refused, on open and authenticated routes alike.
    let evil = format!("rebind.attacker.example:{port}");
    let (s, v) = send(&h, get.clone(), "/engine/health", None, &[("host", &evil)], None).await;
    assert_eq!(s, 403, "{v}");
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(msg.contains("rebind.attacker.example") && msg.contains("--allow-host"), "says how to allow the name: {msg}");
    let (s, v) = send(&h, get.clone(), "/v1/models", Some(&h.admin), &[("host", &evil)], None).await;
    assert_eq!(s, 403, "a valid token does not make a foreign Host acceptable: {v}");
    let (s, _) = send(&h, post.clone(), "/engine/pair", None, &[("host", &evil)], Some(json!({"name": "x", "scopes": ["admin"]}))).await;
    assert_eq!(s, 403, "a rebinding page cannot file a pairing request");
    let (s, _) = send(&h, get.clone(), "/client", None, &[("host", "localhost.attacker.example")], None).await;
    assert_eq!(s, 403);

    // IP literals (any port), localhost, *.localhost and *.local are ours.
    for host in [
        format!("127.0.0.1:{port}"),
        "127.0.0.1".to_string(),
        format!("[::1]:{port}"),
        "192.168.1.20:8080".to_string(),
        format!("localhost:{port}"),
        "LocalHost".to_string(),
        "estia.localhost".to_string(),
        format!("studio.local:{port}"),
        "studio.local.".to_string(),
    ] {
        let (s, v) = send(&h, get.clone(), "/engine/health", None, &[("host", &host)], None).await;
        assert_eq!(s, 200, "Host `{host}` should be accepted: {v}");
    }

    // Operator-allowed names (the CLI's --allow-host / ESTIA_ALLOWED_HOSTS).
    let (s, _) = send(&h, get.clone(), "/engine/health", None, &[("host", "studio.lan")], None).await;
    assert_eq!(s, 403);
    h.state.allow_hosts(["studio.lan:27200, *.tail.example"]);
    for host in ["studio.lan", "Studio.LAN.:27200", "mac.tail.example"] {
        let (s, v) = send(&h, get.clone(), "/engine/health", None, &[("host", host)], None).await;
        assert_eq!(s, 200, "allowed Host `{host}`: {v}");
    }
    let (s, _) = send(&h, get.clone(), "/engine/health", None, &[("host", "tail.example")], None).await;
    assert_eq!(s, 403, "`*.tail.example` allows subdomains, not the bare name");

    // Cross-origin writes are refused: pairing, and admin routes even with a token.
    let pair = json!({"name": "phone", "scopes": ["generate"]});
    for origin in ["http://evil.example", "null", &format!("http://127.0.0.1:{}", port.wrapping_add(1)), "http://127.0.0.1"] {
        let (s, v) = send(&h, post.clone(), "/engine/pair", None, &[("origin", origin)], Some(pair.clone())).await;
        assert_eq!(s, 403, "Origin `{origin}` is cross-origin: {v}");
    }
    let (s, _) =
        send(&h, reqwest::Method::PUT, "/engine/defaults", Some(&h.admin), &[("origin", "http://evil.example")], Some(json!({}))).await;
    assert_eq!(s, 403);
    assert!(!h._dir.join("config.json").exists(), "the refused PUT wrote nothing");
    let (s, _) = send(&h, post.clone(), "/engine/runtime/install", Some(&h.admin), &[("origin", "http://evil.example")], None).await;
    assert_eq!(s, 403);

    // Same-origin writes pass (browsers send Origin on same-origin POSTs), as
    // do reads with any Origin (CORS already keeps those unreadable).
    let (s, v) = send(&h, post.clone(), "/engine/pair", None, &[("origin", &format!("http://127.0.0.1:{port}"))], Some(pair.clone())).await;
    assert_eq!(s, 202, "same-origin pairing request: {v}");
    let (s, v) = send(
        &h,
        post.clone(),
        "/engine/pair",
        None,
        &[("host", &format!("studio.local:{port}")), ("origin", &format!("http://Studio.local:{port}"))],
        Some(pair.clone()),
    )
    .await;
    assert_eq!(s, 202, "same origin by name: {v}");
    let (s, v) = send(&h, post.clone(), "/engine/pair", None, &[], Some(pair.clone())).await;
    assert_eq!(s, 202, "no Origin (not a browser): {v}");
    let (s, _) = send(&h, get.clone(), "/engine/health", None, &[("origin", "http://evil.example")], None).await;
    assert_eq!(s, 200);
}

/// Unknown paths are a JSON 404, not a 401 asking for a token; every real
/// route still requires one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_paths_are_404_and_real_routes_stay_closed() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    let (s, v) = send(&h, reqwest::Method::POST, "/v1/chat/completion", None, &[], Some(json!({}))).await;
    assert_eq!(s, 404, "{v}");
    assert_eq!(v["error"]["type"], "not_found_error", "OpenAI error shape: {v}");
    let (s, v) = get(&h, Some(&h.admin), "/engine/nope").await;
    assert_eq!(s, 404, "{v}");
    assert_eq!(v["error"]["code"], 404);
    // `/client/` redirects to the page.
    let r = reqwest::Client::new().get(format!("{}/client/", h.base)).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert!(r.url().path() == "/client");
    // Real routes are still closed without a token, whatever the method.
    for (m, p) in [
        (reqwest::Method::GET, "/v1/models"),
        (reqwest::Method::POST, "/v1/chat/completions"),
        (reqwest::Method::GET, "/v1/chat/completions"),
        (reqwest::Method::GET, "/engine/defaults"),
        (reqwest::Method::GET, "/engine/pairings"),
        (reqwest::Method::GET, "/engine/jobs/some-id"),
        (reqwest::Method::DELETE, "/engine/models/gemma4-e2b-it-4bit-mlx"),
    ] {
        let (s, v) = send(&h, m.clone(), p, None, &[], None).await;
        assert_eq!(s, 401, "{m} {p} needs a token: {v}");
    }
}

/// Two tokens never share a prompt-cache entry: the runner's cache reports
/// reused prefix tokens, which would let one token probe another's
/// conversation. Same token, same conversation: same key, so reuse still works.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prompt_cache_keys_are_scoped_per_token() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    let alice = h.state.tokens.mint("alice", &[SCOPE_GENERATE]).unwrap();
    let bob = h.state.tokens.mint("bob", &[SCOPE_GENERATE]).unwrap();
    let chat = |user: Option<&str>| {
        let mut b =
            json!({"model": "fast", "messages": [{"role": "system", "content": "terse"}, {"role": "user", "content": "my PIN is 4719"}]});
        if let Some(u) = user {
            b["user"] = json!(u);
        }
        b
    };

    // Same client-supplied key.
    let (s, v) = post(&h, &alice, "/v1/chat/completions", chat(Some("conv-1"))).await;
    assert_eq!(s, 200, "{v}");
    let (a1, _) = last_chat(&h);
    post(&h, &alice, "/v1/chat/completions", chat(Some("conv-1"))).await;
    let (a2, _) = last_chat(&h);
    post(&h, &bob, "/v1/chat/completions", chat(Some("conv-1"))).await;
    let (b1, _) = last_chat(&h);
    let a1 = a1.expect("a key reached the runner");
    assert_eq!(Some(&a1), a2.as_ref(), "one token, one conversation: a stable key");
    assert_ne!(Some(&a1), b1.as_ref(), "another token with the same `user` gets its own entry");
    assert_ne!(a1, "conv-1", "the raw client key never reaches the runner");

    // Derived key (no `user`): same opening messages, different tokens.
    post(&h, &alice, "/v1/chat/completions", chat(None)).await;
    let (ad, _) = last_chat(&h);
    post(&h, &bob, "/v1/chat/completions", chat(None)).await;
    let (bd, _) = last_chat(&h);
    assert!(ad.is_some() && bd.is_some());
    assert_ne!(ad, bd, "derived keys are per token too");

    // Native generate scopes the same way.
    let gen = |key: &str| json!({"model": "fast", "messages": [{"role": "user", "content": "hi"}], "cache_key": key});
    let (s, v) = post(&h, &alice, "/engine/generate", gen("conv-1")).await;
    assert_eq!(s, 200, "{v}");
    let (ag, _) = last_chat(&h);
    post(&h, &bob, "/engine/generate", gen("conv-1")).await;
    let (bg, _) = last_chat(&h);
    assert_ne!(ag, bg, "/engine/generate cache_key is per token");
    assert_eq!(ag.as_ref(), Some(&a1), "a token's key names the same entry on both surfaces");
}

/// `/v1` requests cannot ask for unbounded work.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v1_work_is_capped() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start().await;
    let msgs = json!([{"role": "user", "content": "hi"}]);
    let (s, v) = post(&h, &h.admin, "/v1/chat/completions", json!({"model": "fast", "messages": msgs, "max_tokens": u32::MAX})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(last_chat(&h).1, Some(MAX_TOKENS_CEILING as u64), "max_tokens clamped");
    let (s, _) =
        post(&h, &h.admin, "/v1/chat/completions", json!({"model": "fast", "messages": msgs, "max_completion_tokens": 4_000_000_000u64}))
            .await;
    assert_eq!(s, 200);
    assert_eq!(last_chat(&h).1, Some(MAX_TOKENS_CEILING as u64), "max_completion_tokens clamped");
    let (s, _) = post(&h, &h.admin, "/v1/chat/completions", json!({"model": "fast", "messages": msgs, "max_tokens": 100})).await;
    assert_eq!(s, 200);
    assert_eq!(last_chat(&h).1, Some(100), "a sane value passes through");

    let too_many: Vec<String> = (0..=MAX_EMBED_INPUTS).map(|i| format!("x{i}")).collect();
    let (s, v) = post(&h, &h.embed_only, "/v1/embeddings", json!({"model": "embeddinggemma-300m-4bit", "input": too_many})).await;
    assert_eq!(s, 400, "{v}");
    let ok: Vec<String> = (0..MAX_EMBED_INPUTS).map(|i| format!("x{i}")).collect();
    let (s, v) = post(&h, &h.embed_only, "/v1/embeddings", json!({"model": "embeddinggemma-300m-4bit", "input": ok})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["data"].as_array().unwrap().len(), MAX_EMBED_INPUTS);
}

/// Slowloris: connections that never finish their request head are closed
/// after the header timeout, one peer cannot hold more than its share, and
/// the engine keeps answering everyone else meanwhile.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn idle_and_half_open_connections_are_bounded() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let limits =
        ConnLimits { header_read_timeout: std::time::Duration::from_millis(800), per_peer: 4, exempt_loopback: false, total: Some(64) };
    let h = start_with(limits).await;
    async fn half_open(addr: std::net::SocketAddr) -> tokio::net::TcpStream {
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        s.write_all(b"GET /engine/health HTTP/1.1\r\n").await.unwrap();
        s
    }
    /// Bytes until the server closes (bounded wait).
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

    // A peer's share: four held, the fifth is closed at once.
    let mut held = Vec::new();
    for _ in 0..4 {
        held.push(half_open(h.addr).await);
    }
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let mut fifth = half_open(h.addr).await;
    assert!(closed_within(&mut fifth, 300).await, "over the per-peer cap: closed at once");

    // The header timeout closes the held ones, freeing the share.
    for s in held.iter_mut() {
        assert!(closed_within(s, 2_000).await, "a request head that never ends is cut off");
    }
    let (s, _) = get(&h, None, "/engine/health").await;
    assert_eq!(s, 200, "the peer's share is back");

    // Idle keep-alive connections are closed the same way.
    let mut idle = tokio::net::TcpStream::connect(h.addr).await.unwrap();
    idle.write_all(format!("GET /engine/health HTTP/1.1\r\nHost: {}\r\n\r\n", h.addr).as_bytes()).await.unwrap();
    let mut buf = vec![0u8; 4096];
    let n = idle.read(&mut buf).await.unwrap();
    assert!(String::from_utf8_lossy(&buf[..n]).starts_with("HTTP/1.1 200"));
    assert!(closed_within(&mut idle, 2_000).await, "idle keep-alive closed after the header timeout");
}

/// A llama.cpp-shaped fake: it declares `structured` output, parses tool calls
/// itself (returned in `meta.tool_calls`, never in the text) and names its
/// backend. Every chat is logged beside the script with its `format`, its
/// `model_path` and the adapter command line it was started with.
const FAKE_LLAMA: &str = r#"import sys, json, os
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_log.jsonl")
ARGS = sys.argv[1:]
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
        out({"ok": True, "runner": "fake-llama", "version": "0", "protocol": 2,
             "capabilities": {"generate": True, "stream": True, "embed": True, "cancel": True, "load": True, "chat": True,
                              "tools": True, "prompt_cache": True, "count_tokens": True, "structured": ["json", "json_schema"],
                              "parses_tool_calls": True, "backend": "llama-cpp"}})
    elif t == "load":
        out({"ok": True, "loaded": True, "ms": 1})
    elif t in ("chat", "chat_stream"):
        with open(LOG, "a") as f:
            f.write(json.dumps({"type": t, "cache_key": req.get("cache_key"), "max_tokens": req.get("max_tokens"),
                                "format": req.get("format"), "model_path": req.get("model_path"), "args": ARGS,
                                "system": [m["content"] for m in req["messages"] if m["role"] == "system"]}) + "\n")
        last = req["messages"][-1]["content"]
        calls = None
        if req.get("tools") and "raw" in last:
            text = '<|tool_call>call:wrong{}<tool_call|>'
        elif req.get("tools"):
            text = "Let me check."
            calls = [{"id": "call_w1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Athens\"}"}}]
        elif req.get("format"):
            text = '{"title": "T", "facts": ["a"]}'
        else:
            text = "llama:" + last
        meta = {"prompt_tokens": 7, "cached_tokens": 0, "generation_tokens": 3, "template": "native", "generation_tps": 42.5}
        if calls:
            meta["tool_calls"] = calls
        if t == "chat":
            out({"text": text, "meta": meta})
        else:
            for piece in [text[:4], text[4:]]:
                if piece:
                    out({"type": "token", "text": piece})
            out(dict(type="meta", **meta))
            out({"done": True})
    elif t == "embed_batch":
        out({"embeddings": [[0.6, 0.8, 0.0] for _ in req["inputs"]]})
    elif t == "generate":
        out({"text": "gen:" + req["prompt"]})
    else:
        out({"error": "unknown type %r" % (t,)})
"#;

/// An import as `estia import` leaves it: `model.gguf` and the manifest.
fn write_import(models: &std::path::Path, id: &str, kind: &str, family: &str, dims: Option<usize>) {
    let dir = models.join(id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.gguf"), b"GGUF").unwrap();
    let manifest = json!({
        "id": id, "kind": kind, "format": "gguf", "family": family, "label": format!("{id} (test)"),
        "architecture": "gemma3", "context_length": 4096, "embedding_dims": dims, "pooling": dims.map(|_| "mean"),
        "has_chat_template": kind == "generation", "tools": false,
        "source_path": "/nowhere/model.gguf", "sha256": "00", "bytes": 4, "mode": "copy", "imported_at": 0,
    });
    std::fs::write(dir.join("estia-model.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
}

/// A llama.cpp engine over the fake: the built-in GGUF artifacts for `fast`,
/// `text` and `embed` present, one imported chat model (family
/// `wire-test-fam`) and one imported embedding model.
async fn start_llama() -> Harness {
    let dir = scratch_dir();
    let models = dir.join("models");
    for id in ["gemma4-e2b-it-qat-q4_0-gguf", "gemma4-e4b-it-qat-q4_0-gguf", "embeddinggemma-300m-q8_0-gguf"] {
        std::fs::create_dir_all(models.join(id)).unwrap();
        std::fs::write(models.join(id).join("model.gguf"), b"GGUF").unwrap();
    }
    write_import(&models, "wire-test-chat", "generation", "wire-test-fam", None);
    write_import(&models, "wire-test-embed", "embedding", "wire-test-embed", Some(3));
    let runner = dir.join("fake_llama.py");
    std::fs::write(&runner, FAKE_LLAMA).unwrap();
    // `python3 fake_llama.py --server … --run-dir … --ctx …`: the adapter's
    // command line, with the script standing in for the adapter binary.
    let launch = LlamaLaunch::new("python3", dir.join("run")).prefix_arg(&runner).with_server(LlamaServer::Path(runner.clone()));
    let cfg = EngineConfig::new(ModelStore::new(models), PythonRuntime::new(dir.join("runtime")), dir.join("no-mlx-runner.py"))
        .with_llama(launch);
    serve_engine(dir, cfg, ConnLimits::default()).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn llama_backend_resolves_forwards_format_and_uses_runner_tool_calls() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start_llama().await;

    // Health names the active backend first; the other is listed, inactive.
    let (s, v) = get(&h, None, "/engine/health").await;
    assert_eq!(s, 200);
    assert_eq!(v["backend"], "llama-cpp");
    assert_eq!(v["backends"][0]["id"], "llama-cpp", "{v}");
    assert_eq!(v["backends"][0]["active"], true);
    assert_eq!(v["backends"][0]["server"], "custom", "ESTIA_LLAMA_SERVER-style path: {v}");
    assert!(v["backends"][0]["build"].as_str().unwrap().starts_with('b'));
    assert_eq!(v["backends"][1]["id"], "mlx-python");
    assert_eq!(v["backends"][1]["active"], false);

    // Roles resolve to the GGUF artifacts; the runner gets the .gguf path and
    // the adapter command line carries the artifact's context.
    let (s, v) =
        post(&h, &h.admin, "/v1/chat/completions", json!({"model": "fast", "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["model"], "gemma4-e2b-it-qat-q4_0-gguf");
    assert_eq!(v["choices"][0]["message"]["content"], "llama:hi");
    assert_eq!(v["x_estia"]["backend"], "llama-cpp");
    assert_eq!(v["x_estia"]["generation_tps"], 42.5);
    let logged = last_logged(&h);
    let model_path = logged["model_path"].as_str().unwrap();
    assert!(model_path.ends_with("gemma4-e2b-it-qat-q4_0-gguf/model.gguf") && std::path::Path::new(model_path).is_absolute(), "{logged}");
    let args: Vec<&str> = logged["args"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(args.windows(2).any(|w| w == ["--ctx", "32768"]) && args.contains(&"--run-dir"), "{args:?}");
    // An MLX artifact id on a llama.cpp engine: refused, with the way out.
    let (s, v) = post(
        &h,
        &h.admin,
        "/v1/chat/completions",
        json!({"model": "gemma4-e2b-it-4bit-mlx", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(s, 400, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("ask by family or role"), "{v}");

    // Tools: the runner's own calls, with its prose as content.
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]);
    let ask = |content: &str, stream: bool| json!({"model": "text", "messages": [{"role": "user", "content": content}], "tools": tools, "stream": stream});
    let (s, v) = post(&h, &h.admin, "/v1/chat/completions", ask("weather in Athens?", false)).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["content"], "Let me check.");
    assert_eq!(msg["tool_calls"].as_array().unwrap().len(), 1);
    assert_eq!(msg["tool_calls"][0]["id"], "call_w1");
    assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(
        serde_json::from_str::<Value>(msg["tool_calls"][0]["function"]["arguments"].as_str().unwrap()).unwrap(),
        json!({"city": "Athens"})
    );
    // Call syntax in a parsing runner's text is content, not a call.
    let (s, v) = post(&h, &h.admin, "/v1/chat/completions", ask("raw please", false)).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(v["choices"][0]["message"]["tool_calls"].is_null(), "{v}");
    // Streamed: the prose streams live, the calls come in the last chunk.
    let r = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", h.base))
        .bearer_auth(&h.admin)
        .json(&ask("weather in Athens?", true))
        .send()
        .await
        .unwrap();
    let body = r.text().await.unwrap();
    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let text: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(text, "Let me check.", "{body}");
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls", "{body}");
    let call = &last["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(call["index"], 0);
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(last["x_estia"]["backend"], "llama-cpp");

    // Structured output: the format reaches the runner, and the engine still
    // validates what comes back (clean here, so nothing is repaired).
    let schema = json!({"type": "object", "required": ["title", "facts"]});
    let (s, v) = post(
        &h,
        &h.admin,
        "/v1/chat/completions",
        json!({"model": "text", "messages": [{"role": "user", "content": "facts"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "d", "schema": schema}}}),
    )
    .await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(last_logged(&h)["format"], json!({"type": "json_schema", "schema": schema}));
    assert_eq!(last_logged(&h)["system"], json!([]), "a runner that constrains decoding needs no schema in the prompt");
    assert_eq!(v["x_estia"]["repaired"], false);
    assert_eq!(serde_json::from_str::<Value>(v["choices"][0]["message"]["content"].as_str().unwrap()).unwrap()["title"], "T");
    // Not together with tools (llama-server refuses a grammar with tools).
    post(
        &h,
        &h.admin,
        "/v1/chat/completions",
        json!({"model": "text", "messages": [{"role": "user", "content": "x"}], "tools": tools,
        "response_format": {"type": "json_object"}}),
    )
    .await;
    assert!(last_logged(&h)["format"].is_null());
    // Native generate: a raw prompt with a format goes through chat.
    let (s, v) = post(&h, &h.admin, "/engine/generate", json!({"prompt": "facts please", "format": {"type": "json_object"}})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(last_logged(&h)["type"], "chat");
    assert_eq!(last_logged(&h)["format"], json!({"type": "json"}));
    assert_eq!((v["attempts"].as_u64(), v["repaired"].as_bool()), (Some(1), Some(false)), "{v}");
    assert_eq!(v["backend"], "llama-cpp");
    assert_eq!(v["meta"]["generation_tps"], 42.5);
    let (s, v) =
        post(&h, &h.admin, "/engine/generate", json!({"messages": [{"role": "user", "content": "weather?"}], "tools": tools})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["tool_calls"][0]["function"]["name"], "get_weather");
    let (s, v) = post(&h, &h.admin, "/engine/generate", json!({"prompt": "plain"})).await;
    assert_eq!((s, v["text"].as_str()), (200, Some("gen:plain")), "no format: the raw generate path");

    // Embeddings: this backend's artifact and fingerprint; MLX vectors are
    // another space.
    let (s, v) = post(&h, &h.embed_only, "/v1/embeddings", json!({"model": "embed", "input": ["a", "b"]})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["x_estia"]["fingerprint"], "embeddinggemma-300m-q8_0-gguf@llama-cpp");
    let (s, v) =
        post(&h, &h.admin, "/engine/embed", json!({"inputs": "x", "expect_fingerprint": "embeddinggemma-300m-4bit@mlx-python"})).await;
    assert_eq!(s, 422, "{v}");
    let (s, v) = post(&h, &h.admin, "/engine/embed", json!({"inputs": "x"})).await;
    assert_eq!((s, v["backend"].as_str()), (200, Some("llama-cpp")), "{v}");
    let (s, v) = post(&h, &h.embed_only, "/v1/embeddings", json!({"model": "multilingual-e5-small-mlx", "input": "x"})).await;
    assert_eq!(s, 404, "no GGUF artifact for this model: {v}");
    let (_, v) = get(&h, Some(&h.admin), "/engine/stats").await;
    let loaded: Vec<&str> = v["loaded"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
    assert!(loaded.contains(&"embeddinggemma-300m-q8_0-gguf") && loaded.contains(&"gemma4-e2b-it-qat-q4_0-gguf"), "{loaded:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn imported_models_are_listed_resolved_and_bindable() {
    if !python3_available() {
        eprintln!("skip: python3 not available");
        return;
    }
    let h = start_llama().await;

    // Listed on both surfaces, marked imported; the other backend's artifacts
    // are listed but not runnable.
    let (s, v) = get(&h, Some(&h.admin), "/v1/models").await;
    assert_eq!(s, 200);
    let find = |id: &str| v["data"].as_array().unwrap().iter().find(|m| m["id"] == id).cloned().unwrap_or(Value::Null);
    let chat = find("wire-test-chat");
    assert_eq!(chat["x_estia"]["imported"], true, "{v}");
    assert_eq!(chat["x_estia"]["format"], "gguf");
    assert_eq!(chat["x_estia"]["runnable"], true);
    assert_eq!(chat["x_estia"]["installed"], true);
    assert_eq!(find("gemma4-e2b-it-4bit-mlx")["x_estia"]["runnable"], false);
    assert_eq!(find("wire-test-embed")["x_estia"]["fingerprint"], "wire-test-embed@llama-cpp");
    let (s, v) = get(&h, Some(&h.admin), "/engine/models").await;
    assert_eq!(s, 200);
    assert_eq!(v["backend"], "llama-cpp");
    let gen = v["generation"].as_array().unwrap().iter().find(|m| m["id"] == "wire-test-chat").expect("import listed").clone();
    assert_eq!(
        (gen["imported"].as_bool(), gen["family"].as_str(), gen["backend"].as_str()),
        (Some(true), Some("wire-test-fam"), Some("llama-cpp"))
    );
    let emb = v["embedding"].as_array().unwrap().iter().find(|m| m["id"] == "embeddinggemma-300m-4bit").unwrap().clone();
    assert_eq!(emb["artifact"], "embeddinggemma-300m-q8_0-gguf");
    assert_eq!(emb["installed"], true);
    assert_eq!(emb["fingerprint"], "embeddinggemma-300m-q8_0-gguf@llama-cpp");

    // Resolvable by id and by family.
    for name in ["wire-test-chat", "wire-test-fam"] {
        let (s, v) =
            post(&h, &h.admin, "/v1/chat/completions", json!({"model": name, "messages": [{"role": "user", "content": "hi"}]})).await;
        assert_eq!(s, 200, "{name}: {v}");
        assert_eq!(v["model"], "wire-test-chat");
        assert_eq!(v["x_estia"]["family"], "wire-test-fam");
    }
    assert!(last_logged(&h)["args"].as_array().unwrap().windows(2).any(|w| w == [json!("--ctx"), json!("4096")]), "the import's context");

    // Roles bind to imported families (generation) and imported embedding models.
    let (_, mut roles) = get(&h, Some(&h.admin), "/engine/defaults").await;
    roles["fast"] = json!({"family": "wire-test-fam"});
    roles["embed"] = json!({"family": "wire-test-embed"});
    let r = reqwest::Client::new().put(format!("{}/engine/defaults", h.base)).bearer_auth(&h.admin).json(&roles).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200, "{}", r.text().await.unwrap());
    let (s, v) =
        post(&h, &h.admin, "/v1/chat/completions", json!({"model": "fast", "messages": [{"role": "user", "content": "hi"}]})).await;
    assert_eq!((s, v["model"].as_str()), (200, Some("wire-test-chat")), "{v}");
    let (s, v) = post(&h, &h.embed_only, "/v1/embeddings", json!({"model": "embed", "input": "x"})).await;
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["x_estia"]["fingerprint"], "wire-test-embed@llama-cpp");
    // Capabilities are checked by kind: an embedding import cannot serve chat.
    roles["text"] = json!({"family": "wire-test-embed"});
    let r = reqwest::Client::new().put(format!("{}/engine/defaults", h.base)).bearer_auth(&h.admin).json(&roles).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 400);

    // The runtime install route takes a backend; a bad one starts nothing.
    let (s, v) = post(&h, &h.admin, "/engine/runtime/install", json!({"backend": "onnx"})).await;
    assert_eq!(s, 400, "{v}");
    let (_, v) = get(&h, Some(&h.admin), "/engine/jobs").await;
    assert!(v["jobs"].as_array().unwrap().is_empty(), "{v}");

    // Imports are not downloads; removing one is allowed.
    let (s, v) = post(&h, &h.admin, "/engine/models/pull", json!({"id": "wire-test-chat"})).await;
    assert_eq!(s, 400, "{v}");
    let (s, _) = post(&h, &h.admin, "/engine/models/pull", json!({"id": "no-such-model"})).await;
    assert_eq!(s, 404);
    let r = reqwest::Client::new().delete(format!("{}/engine/models/wire-test-chat", h.base)).bearer_auth(&h.admin).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert!(!h._dir.join("models/wire-test-chat").exists());
    let (s, _) =
        post(&h, &h.admin, "/v1/chat/completions", json!({"model": "wire-test-chat", "messages": [{"role": "user", "content": "hi"}]}))
            .await;
    assert_eq!(s, 404, "a removed import no longer resolves");
}
