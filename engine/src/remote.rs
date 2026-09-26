//! A daemon somewhere else, used through the same shapes as a local session.
//!
//! Blocking HTTP on purpose: hosts call sessions from blocking contexts
//! (`spawn_blocking`, worker threads), and a `Session` is blocking too. The
//! native `/engine/*` routes are used, not `/v1/*`, so the caller gets the
//! engine's own facts back: fingerprint, family, cached tokens, repairs.

use crate::error::SessionError;
use crate::proto::{GenerationMeta, Message};
use crate::session::{CancelToken, Priority};
use serde::Deserialize;
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
    client: reqwest::blocking::Client,
}

fn prio_str(p: Priority) -> &'static str {
    match p {
        Priority::Interactive => "interactive",
        Priority::Background => "background",
    }
}

impl RemoteEngine {
    /// `base_url` like `http://192.168.1.20:27200`.
    pub fn new(base_url: impl Into<String>, token: Option<String>) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            // Generations run for minutes; this is the whole-request ceiling.
            .timeout(Duration::from_secs(600))
            .build()
            .map_err(|e| err(format!("http client: {e}")))?;
        Ok(Self { base_url: base_url.into().trim_end_matches('/').to_string(), token, client })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn post(&self, path: &str, body: &Value) -> reqwest::blocking::RequestBuilder {
        let mut r = self.client.post(format!("{}{path}", self.base_url)).json(body);
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
        let resp = self
            .client
            .get(format!("{}/engine/health", self.base_url))
            .timeout(Duration::from_secs(5))
            .send()
            .map_err(|e| err(format!("remote engine unreachable: {e}")))?;
        Self::check(resp)?.json().map_err(|e| err(format!("health: {e}")))
    }

    /// Non-streaming generation of a raw prompt.
    pub fn generate(&self, model: &str, prompt: &str, max_tokens: Option<u32>, temperature: Option<f32>, prio: Priority) -> Result<String> {
        let body =
            json!({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "priority": prio_str(prio)});
        let resp = Self::check(self.post("/engine/generate", &body).send().map_err(|e| err(format!("remote generate: {e}")))?)?;
        let v: Value = resp.json().map_err(|e| err(format!("remote generate: {e}")))?;
        v["text"].as_str().map(str::to_string).ok_or_else(|| err("remote generate: no text in response"))
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
        let resp = Self::check(self.post("/engine/generate", &body).send().map_err(|e| err(format!("remote stream: {e}")))?)?;
        let reader = BufReader::new(resp);
        let mut full = String::new();
        for line in reader.lines() {
            if cancel.map(CancelToken::is_cancelled).unwrap_or(false) {
                // Dropping the reader closes the connection; the daemon sees
                // the client go and cancels the runner.
                return Err(SessionError::Cancelled { partial: full });
            }
            let line = line.map_err(|e| err(format!("remote stream read: {e}")))?;
            let Some(data) = line.strip_prefix("data: ") else { continue };
            let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
            if let Some(e) = v.get("error") {
                return Err(err(format!("remote engine: {}", e.as_str().unwrap_or(&e.to_string()))));
            }
            if let Some(t) = v.get("token").and_then(Value::as_str) {
                full.push_str(t);
                on_token(t);
                continue;
            }
            if v.get("done").and_then(Value::as_bool) == Some(true) {
                let meta = v.get("meta").cloned().and_then(|m| serde_json::from_value(m).ok());
                // Prefer the host's assembled text (identical bytes, no drift).
                if let Some(t) = v.get("text").and_then(Value::as_str) {
                    return Ok((t.to_string(), meta));
                }
                return Ok((full, meta));
            }
        }
        Err(err("remote stream ended without a done line"))
    }

    /// Ask a daemon for a token: the first half of pairing. Needs no token.
    /// Returns the pairing id to poll with [`RemoteEngine::pair_poll`].
    pub fn pair_request(&self, name: &str, scopes: &[&str]) -> Result<String> {
        let resp = self
            .client
            .post(format!("{}/engine/pair", self.base_url))
            .json(&json!({"name": name, "scopes": scopes}))
            .send()
            .map_err(|e| err(format!("pair request: {e}")))?;
        let v: Value = Self::check(resp)?.json().map_err(|e| err(format!("pair request: {e}")))?;
        v["id"].as_str().map(str::to_string).ok_or_else(|| err("pair request: no id"))
    }

    /// Poll a pairing: `pending`, `approved` (with the token, once) or `denied`.
    pub fn pair_poll(&self, id: &str) -> Result<(String, Option<String>)> {
        let resp = self.client.get(format!("{}/engine/pair/{id}", self.base_url)).send().map_err(|e| err(format!("pair poll: {e}")))?;
        let v: Value = Self::check(resp)?.json().map_err(|e| err(format!("pair poll: {e}")))?;
        Ok((v["status"].as_str().unwrap_or("pending").to_string(), v["token"].as_str().map(str::to_string)))
    }

    /// Admin: pending and recent pairings on the daemon.
    pub fn pairings(&self) -> Result<Value> {
        let mut r = self.client.get(format!("{}/engine/pairings", self.base_url));
        if let Some(t) = &self.token {
            r = r.bearer_auth(t);
        }
        Self::check(r.send().map_err(|e| err(format!("pairings: {e}")))?)?.json().map_err(|e| err(format!("pairings: {e}")))
    }

    /// Admin: approve or deny a pairing from wherever this client is.
    pub fn decide_pairing(&self, id: &str, approve: bool) -> Result<Value> {
        let verb = if approve { "approve" } else { "deny" };
        let resp =
            self.post(&format!("/engine/pairings/{id}/{verb}"), &json!({})).send().map_err(|e| err(format!("pairing {verb}: {e}")))?;
        Self::check(resp)?.json().map_err(|e| err(format!("pairing {verb}: {e}")))
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
        let body =
            json!({"model": model, "inputs": inputs, "task": "none", "expect_fingerprint": expect_fingerprint, "priority": prio_str(prio)});
        let resp = Self::check(self.post("/engine/embed", &body).send().map_err(|e| err(format!("remote embed: {e}")))?)?;
        #[derive(Deserialize)]
        struct EmbedResp {
            vectors: Vec<Vec<f32>>,
            fingerprint: String,
        }
        let r: EmbedResp = resp.json().map_err(|e| err(format!("remote embed: {e}")))?;
        Ok((r.vectors, r.fingerprint))
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
