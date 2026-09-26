//! One shape for a model wherever it runs. A host holds `GenHandle` /
//! `EmbedHandle` and never branches on local versus remote itself.

use crate::error::SessionError;
use crate::proto::Capabilities;
use crate::remote::{RemoteEmbed, RemoteGen};
use crate::resident::{EmbedSession, GenSession};
use crate::session::{CancelToken, Priority};
use std::time::Duration;

type Result<T> = std::result::Result<T, SessionError>;

// Few handles exist and they live long; boxing the local session would
// change the public variant type for every caller.
#[allow(clippy::large_enum_variant)]
pub enum GenHandle {
    Local(GenSession),
    Remote(RemoteGen),
}

impl GenHandle {
    pub fn model_id(&self) -> &str {
        match self {
            GenHandle::Local(s) => s.model_id(),
            GenHandle::Remote(r) => r.model_id(),
        }
    }

    pub fn is_remote(&self) -> bool {
        matches!(self, GenHandle::Remote(_))
    }

    pub fn capabilities(&self) -> Capabilities {
        match self {
            GenHandle::Local(s) => s.capabilities(),
            // A daemon speaks the native API; it always chats and counts.
            GenHandle::Remote(_) => Capabilities { generate: true, stream: true, cancel: true, chat: true, ..Default::default() },
        }
    }

    /// Health: a local ping, or the daemon's `/engine/health`.
    pub fn ping(&self) -> Result<()> {
        match self {
            GenHandle::Local(s) => s.ping(),
            GenHandle::Remote(r) => r.engine().health().map(|_| ()),
        }
    }

    /// Per-call deadline of a local session; no effect on a remote handle.
    pub fn set_call_timeout(&mut self, timeout: Duration) {
        if let GenHandle::Local(s) = self {
            s.set_call_timeout(timeout);
        }
    }

    pub fn generate(&self, prompt: &str, max_tokens: Option<u32>, temperature: Option<f32>) -> Result<String> {
        self.generate_with(prompt, max_tokens, temperature, Priority::Background)
    }

    pub fn generate_with(&self, prompt: &str, max_tokens: Option<u32>, temperature: Option<f32>, prio: Priority) -> Result<String> {
        match self {
            GenHandle::Local(s) => s.generate_with(prompt, max_tokens, temperature, prio),
            GenHandle::Remote(r) => r.generate_with(prompt, max_tokens, temperature, prio),
        }
    }

    pub fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        on_token: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_stream_with(prompt, max_tokens, temperature, Priority::Background, None, on_token)
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
        match self {
            GenHandle::Local(s) => s.generate_stream_with(prompt, max_tokens, temperature, prio, cancel, on_token),
            GenHandle::Remote(r) => r.generate_stream_with(prompt, max_tokens, temperature, prio, cancel, on_token),
        }
    }

    /// Calls in flight on a local session; a remote host keeps its own count.
    pub fn in_flight(&self) -> usize {
        match self {
            GenHandle::Local(s) => s.in_flight(),
            GenHandle::Remote(_) => 0,
        }
    }

    /// A remote handle is never idle-reaped: dropping it frees nothing here.
    pub fn is_idle(&self, timeout: Duration) -> bool {
        match self {
            GenHandle::Local(s) => s.is_idle(timeout),
            GenHandle::Remote(_) => false,
        }
    }

    pub fn maybe_shutdown(&self, timeout: Duration) -> bool {
        match self {
            GenHandle::Local(s) => s.maybe_shutdown(timeout),
            GenHandle::Remote(_) => false,
        }
    }

    pub fn waiting(&self, prio: Priority) -> usize {
        match self {
            GenHandle::Local(s) => s.waiting(prio),
            GenHandle::Remote(_) => 0,
        }
    }
}

// Few handles exist and they live long; boxing the local session would
// change the public variant type for every caller.
#[allow(clippy::large_enum_variant)]
pub enum EmbedHandle {
    Local(EmbedSession),
    Remote(RemoteEmbed),
}

impl EmbedHandle {
    /// Model **and** backend identity every vector carries. For a remote host
    /// this is what the caller expects; the host refuses to serve anything else.
    pub fn fingerprint(&self) -> &str {
        match self {
            EmbedHandle::Local(s) => s.fingerprint(),
            EmbedHandle::Remote(r) => r.fingerprint(),
        }
    }

    pub fn is_remote(&self) -> bool {
        matches!(self, EmbedHandle::Remote(_))
    }

    pub fn ping(&self) -> Result<()> {
        match self {
            EmbedHandle::Local(s) => s.ping(),
            EmbedHandle::Remote(r) => r.engine().health().map(|_| ()),
        }
    }

    pub fn set_call_timeout(&mut self, timeout: Duration) {
        if let EmbedHandle::Local(s) = self {
            s.set_call_timeout(timeout);
        }
    }

    /// One input, one vector.
    pub fn embed(&self, input: &str) -> Result<Vec<f32>> {
        match self {
            EmbedHandle::Local(s) => s.embed(input),
            EmbedHandle::Remote(r) => r
                .embed_batch_with(&[input.to_string()], Priority::Interactive)?
                .into_iter()
                .next()
                .ok_or_else(|| SessionError::Runner("remote embed returned no vector".into())),
        }
    }

    pub fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_with(inputs, Priority::Background)
    }

    pub fn embed_batch_with(&self, inputs: &[String], prio: Priority) -> Result<Vec<Vec<f32>>> {
        match self {
            EmbedHandle::Local(s) => s.embed_batch_with(inputs, prio),
            EmbedHandle::Remote(r) => r.embed_batch_with(inputs, prio),
        }
    }

    pub fn in_flight(&self) -> usize {
        match self {
            EmbedHandle::Local(s) => s.in_flight(),
            EmbedHandle::Remote(_) => 0,
        }
    }

    pub fn is_idle(&self, timeout: Duration) -> bool {
        match self {
            EmbedHandle::Local(s) => s.is_idle(timeout),
            EmbedHandle::Remote(_) => false,
        }
    }

    pub fn maybe_shutdown(&self, timeout: Duration) -> bool {
        match self {
            EmbedHandle::Local(s) => s.maybe_shutdown(timeout),
            EmbedHandle::Remote(_) => false,
        }
    }
}
