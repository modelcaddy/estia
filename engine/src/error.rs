use estia_proto::ProtoError;

/// Everything a [`crate::Session`] call can fail with.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("spawn runner `{program}`: {source}")]
    Spawn {
        program: String,
        #[source]
        source: std::io::Error,
    },
    #[error("runner stdin/stdout not piped")]
    NotPiped,
    #[error("write to runner: {0}")]
    Write(#[source] std::io::Error),
    /// The child accepted the request but never answered. It has been killed;
    /// the next call respawns it.
    #[error("runner timed out after {secs}s and was terminated")]
    Timeout { secs: u64 },
    /// The child closed stdout (it exited or was killed).
    #[error("runner closed stdout (EOF)")]
    Eof,
    /// The runner answered with an error envelope.
    #[error("runner error: {0}")]
    Runner(String),
    #[error("serialize request: {0}")]
    Serialize(#[source] serde_json::Error),
    #[error("parse runner response: {0}")]
    Parse(#[source] serde_json::Error),
    /// The first attempt failed, the child was respawned, and the retry
    /// failed too.
    #[error("runner failed after respawn: {0}")]
    AfterRespawn(#[source] Box<SessionError>),
    /// The child died and could not be respawned.
    #[error("respawn runner after `{cause}`: {source}")]
    Respawn {
        cause: String,
        #[source]
        source: Box<SessionError>,
    },
    /// A streaming call saw no output line for the whole deadline. The child
    /// has been killed; tokens already delivered stand.
    #[error("runner stream produced no output for {secs}s and was terminated")]
    StreamSilence { secs: u64 },
    #[error("runner closed stdout mid-stream (EOF)")]
    StreamEof,
    /// The stream was cancelled through its [`crate::CancelToken`]. `partial`
    /// is everything delivered before the stop.
    #[error("generation cancelled after {} chars", partial.len())]
    Cancelled { partial: String },
    /// Waiting on the child failed at the OS level.
    #[error("wait for runner: {0}")]
    Wait(#[source] std::io::Error),
    /// A one-shot runner exited with a failure status. `stderr` is whatever it
    /// wrote there, trimmed — usually the only clue.
    #[error("runner exited with {status}: {stderr}")]
    Exited { status: String, stderr: String },
    /// A one-shot runner exited cleanly but streamed nothing usable.
    #[error("runner produced no output: {stderr}")]
    NoOutput { stderr: String },
}

impl From<ProtoError> for SessionError {
    fn from(e: ProtoError) -> Self {
        match e {
            ProtoError::Runner(msg) => SessionError::Runner(msg),
            ProtoError::Invalid(err) => SessionError::Parse(err),
        }
    }
}
