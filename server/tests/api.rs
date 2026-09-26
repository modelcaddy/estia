//! The HTTP surface against a stdlib-only fake runner: auth, models, chat
//! completions (tool calls, response_format), embeddings, fingerprint checks,
//! native generate. Skipped without python3.

use estia_engine::models::ModelStore;
use estia_engine::runtime::PythonRuntime;
use estia_engine::{Engine, EngineConfig};
use estia_server::tokens::{TokenStore, SCOPE_ADMIN, SCOPE_EMBED};
use estia_server::{router, AppState};
use serde_json::{json, Value};
use std::io::Write as _;
use std::sync::Arc;

fn python3_available() -> bool {
    std::process::Command::new("python3").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// A v2 fake: chat echoes the roles and the last user turn, or answers a
/// Gemma-style tool call when tools are declared; embed returns 2-dim vectors.
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
             "capabilities": {"generate": True, "stream": True, "embed": True, "cancel": True, "load": True, "chat": True, "tools": True, "prompt_cache": True, "count_tokens": True}})
    elif t == "load":
        out({"ok": True, "loaded": True, "ms": 1})
    elif t in ("chat", "chat_stream"):
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
    admin: String,
    embed_only: String,
    _dir: std::path::PathBuf,
}

async fn start() -> Harness {
    let dir = std::env::temp_dir().join(format!(
        "estia-server-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(dir.join("models/gemma4-e2b-it-4bit-mlx")).unwrap();
    std::fs::create_dir_all(dir.join("models/gemma4-e4b-it-4bit-mlx")).unwrap();
    std::fs::create_dir_all(dir.join("models/embeddinggemma-300m-4bit")).unwrap();
    let runner = dir.join("fake_runner.py");
    std::fs::File::create(&runner).unwrap().write_all(FAKE.as_bytes()).unwrap();
    let cfg =
        EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), &runner).with_python("python3");
    let engine = Arc::new(Engine::new(cfg));
    let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
    let admin = tokens.mint("admin", &[SCOPE_ADMIN]).unwrap();
    let embed_only = tokens.mint("embedder", &[SCOPE_EMBED]).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(AppState::new(engine, tokens, true, addr));
    tokio::spawn(async move {
        axum::serve(listener, router(state).into_make_service_with_connect_info::<std::net::SocketAddr>()).await.unwrap();
    });
    Harness { base: format!("http://{addr}"), admin, embed_only, _dir: dir }
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
