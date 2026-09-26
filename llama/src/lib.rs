//! Estia's llama.cpp backend.
//!
//! The adapter speaks Estia's runner protocol v2 on stdin/stdout (see
//! `docs/protocol.md`) and serves each request by calling one upstream
//! `llama-server` process over a private UNIX socket (or loopback TCP on
//! Windows) with a random API key. The engine starts it like any resident
//! runner, so `Session`'s deadlines, priority gate, cancel and respawn apply
//! unchanged. Design: `docs/design/llama-backend.md`.
//!
//! The `estia` binary runs it as the hidden subcommand `estia runner llama`,
//! so a release stays one binary plus the downloaded llama.cpp build.

use std::path::PathBuf;

/// How the adapter is started.
#[derive(Debug, Clone)]
pub struct AdapterOptions {
    /// The `llama-server` executable to run.
    pub server_bin: PathBuf,
    /// A private directory for the socket, the API-key file and the pid
    /// record (created 0700 if missing).
    pub run_dir: PathBuf,
    /// Context length for generation models (`-c`); `None` = the registry
    /// did not say, use 8192.
    pub context_length: Option<u32>,
    /// Extra `llama-server` arguments from the operator (e.g. `-ngl 0`).
    pub extra_args: Vec<String>,
}

/// Run the adapter on this process's stdin/stdout until stdin closes.
pub fn run_stdio(opts: AdapterOptions) -> anyhow::Result<()> {
    let _ = opts;
    anyhow::bail!("estia-llama: not implemented yet")
}
