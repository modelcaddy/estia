//! Resident model sessions: a [`Session`] plus the model it was spawned for.
//!
//! [`EmbedSession`] and [`GenSession`] are the two shapes a host keeps warm.
//! They know the request types and the model path; everything about keeping
//! the child alive safely is [`Session`]'s.

use crate::error::SessionError;
use crate::proto::{
    Capabilities, ChatResp, CountTokensResp, EmbedBatchResp, EmbedResp, GenerateResp, GenerationMeta, HelloResp, LoadResp, Message,
    PingResp, Request, UnloadResp,
};
use crate::session::{CancelToken, Launch, Priority, Session, SessionConfig, SessionObserver};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

type Result<T> = std::result::Result<T, SessionError>;

/// What both session types share: the handshake result and preloading.
fn handshake(session: &Session) -> Result<Option<HelloResp>> {
    let hello = session.hello()?;
    let model = session.launch().label().unwrap_or("");
    match &hello {
        Some(h) => tracing::info!(
            model = %model,
            pid = session.pid(),
            runner = %h.runner,
            runner_version = %h.version,
            protocol = h.protocol,
            capabilities = %capability_list(&h.capabilities),
            "runner handshake"
        ),
        None => tracing::info!(model = %model, pid = session.pid(), protocol = 1, "runner handshake: no hello, protocol v1"),
    }
    Ok(hello)
}

/// The capabilities a runner declared, as `chat,stream,…`, for a log line.
pub(crate) fn capability_list(c: &Capabilities) -> String {
    let flags = [
        ("generate", c.generate),
        ("stream", c.stream),
        ("embed", c.embed),
        ("cancel", c.cancel),
        ("load", c.load),
        ("chat", c.chat),
        ("tools", c.tools),
        ("prompt_cache", c.prompt_cache),
        ("count_tokens", c.count_tokens),
    ];
    let mut out: Vec<String> = flags.iter().filter(|(_, on)| *on).map(|(n, _)| n.to_string()).collect();
    out.extend(c.structured.iter().map(|s| format!("structured:{s}")));
    out.join(",")
}

fn load(session: &Session, model_path: &str, kind: &str) -> Result<LoadResp> {
    let model = session.launch().label().unwrap_or("");
    match session.call_typed_with::<_, LoadResp>(&Request::Load { model_path, kind }, crate::session::Priority::Interactive) {
        Ok(r) => {
            tracing::info!(model = %model, kind = %kind, load_ms = r.ms, "model loaded");
            Ok(r)
        }
        Err(e) => {
            tracing::warn!(model = %model, kind = %kind, error = %e, "model load failed");
            Err(e)
        }
    }
}

fn unload(session: &Session, model_path: &str) -> Result<bool> {
    let r: UnloadResp = session.call_typed_with(&Request::Unload { model_path }, crate::session::Priority::Interactive)?;
    tracing::info!(model = %session.launch().label().unwrap_or(""), unloaded = r.unloaded, "model unloaded");
    Ok(r.unloaded)
}

fn ping(session: &Session, what: &str) -> Result<()> {
    let v = session.call_unobserved(&Request::Ping)?;
    let parsed: PingResp = serde_json::from_value(v).map_err(SessionError::Parse)?;
    if parsed.ok {
        Ok(())
    } else {
        Err(SessionError::Runner(format!("{what} ping returned ok=false")))
    }
}

/// A long-lived embedding runner: model loaded once, kept resident across
/// calls. Inputs are embedded exactly as given — prefixing is the caller's
/// policy (see `models::embed`).
pub struct EmbedSession {
    session: Session,
    model_path: String,
    /// Identity every vector from this session carries: model **and** backend.
    fingerprint: String,
    /// The runner's `hello`, when it speaks protocol v2.
    hello: Option<HelloResp>,
}

impl EmbedSession {
    /// Spawn the runner. The model loads lazily on the first embed, not here.
    pub fn spawn(
        launch: Launch,
        cfg: SessionConfig,
        observer: Arc<dyn SessionObserver>,
        model_path: impl Into<String>,
        fingerprint: impl Into<String>,
    ) -> Result<Self> {
        let model_path = model_path.into();
        let launch = if launch.label().is_some() { launch } else { launch.with_label(model_label(&model_path)) };
        let session = Session::spawn(launch, cfg, observer)?;
        let hello = handshake(&session)?;
        Ok(Self { session, model_path, fingerprint: fingerprint.into(), hello })
    }

    pub fn set_call_timeout(&mut self, timeout: Duration) {
        self.session.set_call_timeout(timeout);
    }

    /// Health check, not counted as a model call.
    pub fn ping(&self) -> Result<()> {
        ping(&self.session, "embed runner")
    }

    /// The runner's declared capabilities; empty for a v1 runner.
    pub fn capabilities(&self) -> Capabilities {
        self.hello.as_ref().map(|h| h.capabilities.clone()).unwrap_or_default()
    }

    pub fn hello(&self) -> Option<&HelloResp> {
        self.hello.as_ref()
    }

    /// Load the model now (protocol v2) and report how long it took. On a v1
    /// runner this is a no-op returning `None`; the model loads on first use.
    pub fn load(&self) -> Result<Option<Duration>> {
        if !self.capabilities().load {
            return Ok(None);
        }
        let r = load(&self.session, &self.model_path, "embedding")?;
        Ok(Some(Duration::from_millis(r.ms)))
    }

    /// Drop the model from the runner (protocol v2), freeing its memory while
    /// the process stays alive. `Ok(false)` when nothing was loaded or on v1.
    pub fn unload(&self) -> Result<bool> {
        if !self.capabilities().load {
            return Ok(false);
        }
        unload(&self.session, &self.model_path)
    }

    /// Embed `inputs` in one batched round-trip; one unit vector per input.
    /// Background priority — a bulk ingest.
    pub fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with(inputs, Priority::Background)
    }

    /// [`EmbedSession::embed_batch`] at an explicit priority: a search query
    /// someone is waiting on is `Interactive`.
    pub fn embed_batch_with(&self, inputs: &[String], prio: Priority) -> Result<Vec<Vec<f32>>> {
        let resp: EmbedBatchResp = self.session.call_typed_with(&Request::EmbedBatch { model_path: &self.model_path, inputs }, prio)?;
        Ok(resp.embeddings)
    }

    pub fn embed(&self, input: &str) -> Result<Vec<f32>> {
        let resp: EmbedResp = self.session.call_typed(&Request::Embed { model_path: &self.model_path, input })?;
        Ok(resp.embedding)
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn in_flight(&self) -> usize {
        self.session.in_flight()
    }

    /// Callers queued at `prio` on this session (not counting the one running).
    pub fn waiting(&self, prio: Priority) -> usize {
        self.session.waiting(prio)
    }

    pub fn is_idle(&self, timeout: Duration) -> bool {
        self.session.is_idle(timeout)
    }

    pub fn maybe_shutdown(&self, timeout: Duration) -> bool {
        self.session.maybe_shutdown(timeout)
    }
}

/// A long-lived generation runner: model loaded once, kept warm across calls.
pub struct GenSession {
    session: Session,
    model_path: String,
    /// The artifact id this session generates with.
    model_id: String,
    hello: Option<HelloResp>,
}

impl GenSession {
    pub fn spawn(
        launch: Launch,
        cfg: SessionConfig,
        observer: Arc<dyn SessionObserver>,
        model_path: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Result<Self> {
        let model_id = model_id.into();
        let launch = if launch.label().is_some() { launch } else { launch.with_label(model_id.clone()) };
        let session = Session::spawn(launch, cfg, observer)?;
        let hello = handshake(&session)?;
        Ok(Self { session, model_path: model_path.into(), model_id, hello })
    }

    pub fn set_call_timeout(&mut self, timeout: Duration) {
        self.session.set_call_timeout(timeout);
    }

    pub fn ping(&self) -> Result<()> {
        ping(&self.session, "gen runner")
    }

    /// The runner's declared capabilities; empty for a v1 runner.
    pub fn capabilities(&self) -> Capabilities {
        self.hello.as_ref().map(|h| h.capabilities.clone()).unwrap_or_default()
    }

    pub fn hello(&self) -> Option<&HelloResp> {
        self.hello.as_ref()
    }

    /// Load the model now (protocol v2); `None` on a v1 runner.
    pub fn load(&self) -> Result<Option<Duration>> {
        if !self.capabilities().load {
            return Ok(None);
        }
        let r = load(&self.session, &self.model_path, "generation")?;
        Ok(Some(Duration::from_millis(r.ms)))
    }

    /// Drop the model from the runner (protocol v2); `Ok(false)` on v1.
    pub fn unload(&self) -> Result<bool> {
        if !self.capabilities().load {
            return Ok(false);
        }
        unload(&self.session, &self.model_path)
    }

    /// Non-streaming generation at background priority.
    pub fn generate(&self, prompt: &str, max_tokens: Option<u32>, temperature: Option<f32>) -> Result<String> {
        self.generate_with(prompt, max_tokens, temperature, Priority::Background)
    }

    /// [`GenSession::generate`] at an explicit priority. Not cancellable: the
    /// runner's one-shot decode has no stopping point; use the streaming call
    /// when a cancel matters.
    pub fn generate_with(&self, prompt: &str, max_tokens: Option<u32>, temperature: Option<f32>, prio: Priority) -> Result<String> {
        let resp: GenerateResp = self
            .session
            .call_typed_with(&Request::Generate { model_path: &self.model_path, prompt, max_tokens, temperature, json: None }, prio)?;
        Ok(resp.text)
    }

    /// Token lines reach `on_token` as they arrive; the assembled text is
    /// returned. See [`Session::stream`] for the retry and deadline rules.
    pub fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        on_token: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_stream_with(prompt, max_tokens, temperature, Priority::Background, None, on_token)
    }

    /// [`GenSession::generate_stream`] at an explicit priority with an
    /// optional [`CancelToken`]; see [`Session::stream_with`].
    pub fn generate_stream_with(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        prio: Priority,
        cancel: Option<&CancelToken>,
        on_token: impl FnMut(&str),
    ) -> Result<String> {
        self.session.stream_with(
            &Request::GenerateStream { model_path: &self.model_path, prompt, max_tokens, temperature },
            prio,
            cancel,
            on_token,
        )
    }

    /// Protocol v2 chat, streamed: `messages` rendered by the model's chat
    /// template, optional `tools` declared natively, optional `cache_key` so
    /// the runner reuses this conversation's KV cache. Requires the `chat`
    /// capability; errors otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn chat_stream_with(
        &self,
        messages: &[Message],
        tools: Option<&[Value]>,
        cache_key: Option<&str>,
        format: Option<&Value>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        prio: Priority,
        cancel: Option<&CancelToken>,
        on_token: impl FnMut(&str),
    ) -> Result<ChatOutcome> {
        if !self.capabilities().chat {
            return Err(SessionError::Runner("this runner does not support chat (protocol v2 `chat_stream`)".into()));
        }
        let outcome = self.session.stream_full(
            &Request::ChatStream { model_path: &self.model_path, messages, tools, cache_key, format, max_tokens, temperature },
            prio,
            cancel,
            on_token,
        )?;
        let meta = outcome.meta.and_then(|v| serde_json::from_value::<GenerationMeta>(v).ok()).unwrap_or_default();
        Ok(ChatOutcome { text: outcome.text, meta })
    }

    /// Non-streaming [`GenSession::chat_stream_with`].
    #[allow(clippy::too_many_arguments)]
    pub fn chat_with(
        &self,
        messages: &[Message],
        tools: Option<&[Value]>,
        cache_key: Option<&str>,
        format: Option<&Value>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        prio: Priority,
    ) -> Result<ChatOutcome> {
        if !self.capabilities().chat {
            return Err(SessionError::Runner("this runner does not support chat (protocol v2 `chat`)".into()));
        }
        let r: ChatResp = self.session.call_typed_with(
            &Request::Chat { model_path: &self.model_path, messages, tools, cache_key, format, max_tokens, temperature },
            prio,
        )?;
        Ok(ChatOutcome { text: r.text, meta: r.meta.unwrap_or_default() })
    }

    /// Tokens in `text` under this model's tokenizer (protocol v2); `None` on a
    /// runner without `count_tokens`.
    pub fn count_tokens(&self, text: &str) -> Result<Option<u64>> {
        if !self.capabilities().count_tokens {
            return Ok(None);
        }
        let r: CountTokensResp =
            self.session.call_typed_with(&Request::CountTokens { model_path: &self.model_path, text }, Priority::Interactive)?;
        Ok(Some(r.tokens))
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn in_flight(&self) -> usize {
        self.session.in_flight()
    }

    /// Callers queued at `prio` on this session (not counting the one running).
    pub fn waiting(&self, prio: Priority) -> usize {
        self.session.waiting(prio)
    }

    pub fn is_idle(&self, timeout: Duration) -> bool {
        self.session.is_idle(timeout)
    }

    pub fn maybe_shutdown(&self, timeout: Duration) -> bool {
        self.session.maybe_shutdown(timeout)
    }
}

/// A model directory's last component (the artifact id in a model store), for log lines.
fn model_label(model_path: &str) -> String {
    std::path::Path::new(model_path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| model_path.to_string())
}

/// Text plus the runner's token accounting for one chat generation.
#[derive(Debug, Clone, Default)]
pub struct ChatOutcome {
    pub text: String,
    pub meta: GenerationMeta,
}
