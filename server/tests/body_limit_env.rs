//! `ESTIA_MAX_BODY_BYTES` sets the request-body limit, and a value that is
//! not a positive byte count leaves the default. Its own test binary with a
//! single test, so changing the environment races nothing.

use estia_engine::models::ModelStore;
use estia_engine::runtime::PythonRuntime;
use estia_engine::{Engine, EngineConfig};
use estia_server::tokens::{TokenStore, SCOPE_ADMIN};
use estia_server::{router, serve_router, AppState, ConnLimits, DEFAULT_MAX_BODY_BYTES, MAX_BODY_ENV};
use serde_json::{json, Value};
use std::sync::Arc;

fn state(dir: &std::path::Path) -> (Arc<AppState>, String) {
    let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), dir.join("no-runner.py"));
    let tokens = TokenStore::open(dir.join("tokens.json")).unwrap();
    let admin = tokens.mint("admin", &[SCOPE_ADMIN]).unwrap();
    (Arc::new(AppState::new(Arc::new(Engine::new(cfg)), tokens, true, "127.0.0.1:0".parse().unwrap())), admin)
}

#[test]
fn the_environment_sets_the_body_limit() {
    let dir = std::env::temp_dir().join(format!("estia-body-env-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    std::env::set_var(MAX_BODY_ENV, "2048");
    let (st, admin) = state(&dir);
    assert_eq!(st.max_body_bytes(), 2048);
    for bad in ["lots", "0", "-5", ""] {
        std::env::set_var(MAX_BODY_ENV, bad);
        assert_eq!(state(&dir.join(format!("bad{}", bad.len()))).0.max_body_bytes(), DEFAULT_MAX_BODY_BYTES, "{bad:?}");
    }
    std::env::remove_var(MAX_BODY_ENV);
    assert_eq!(state(&dir.join("unset")).0.max_body_bytes(), DEFAULT_MAX_BODY_BYTES);

    // The limit from the environment is the one the router enforces.
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = router(Arc::clone(&st));
        tokio::spawn(async move { serve_router(listener, app, ConnLimits::default(), std::future::pending()).await.unwrap() });
        let r = reqwest::Client::new()
            .post(format!("{base}/v1/embeddings"))
            .bearer_auth(&admin)
            .json(&json!({"model": "embed", "input": "z".repeat(4096)}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status().as_u16(), 413);
        let v: Value = r.json().await.unwrap();
        assert!(v["error"]["message"].as_str().unwrap().contains("2048 bytes"), "{v}");
    });
    let _ = std::fs::remove_dir_all(&dir);
}
