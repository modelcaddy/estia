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
//! so a release stays one binary plus the downloaded llama.cpp build:
//!
//! ```no_run
//! // after `estia runner llama`:
//! let opts = estia_llama::parse_args(std::env::args_os().skip(3))?;
//! estia_llama::run_stdio(opts)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! How it ends, and what happens to `llama-server`:
//!
//! - stdin closes: requests already read are answered, then the server is
//!   stopped (SIGTERM, then SIGKILL after 3 s) and [`run_stdio`] returns.
//! - SIGTERM, SIGINT or SIGHUP: the server is stopped and the process exits
//!   with 128 + the signal number.
//! - the parent dies (the adapter is re-parented): the server is stopped and
//!   the process exits.
//! - the adapter is killed outright: on Linux the server gets SIGKILL through
//!   `PR_SET_PDEATHSIG`; on macOS a small `/bin/sh` guard holding a pipe from
//!   the adapter kills it. The pid record in the run directory names it too.
//! - `llama-server` dies under a loaded model: the call in flight gets an
//!   error line and [`run_stdio`] returns an error, so the engine respawns
//!   the adapter.
//!
//! `run_stdio` starts servers only from the thread that calls it, and on
//! Linux that thread must live as long as the adapter: `PR_SET_PDEATHSIG`
//! fires when the thread that started the child exits.
//!
//! Where llama-server could behave differently from the MLX Python runner
//! (`runners/mlx-python/estia-runner.py`), the adapter follows the runner:
//!
//! - Sampling: every value is sent. Null `max_tokens` and `temperature` mean
//!   256 and 0.0; top-k, top-p, min-p and the repeat penalty are neutral.
//!   Thinking is off (`chat_template_kwargs.enable_thinking = false`).
//! - `generate` and `generate_stream`: the prompt is sent as one user turn
//!   through the model's chat template, as the Python runner renders it, not
//!   as raw text to `/completion`. Callers (`estia chat`, `/engine/generate`)
//!   send instructions and expect an answer, not a continuation.
//! - Embeddings: L2-normalised (`embd_normalize: 2`), as mlx-embeddings'
//!   `text_embeds` are, pooled the way the GGUF file says, and cut to 512
//!   tokens (the Python runner's `max_length=512`), keeping the special
//!   tokens at either end. Non-finite values become 0.0.
//! - Prompt cache: one slot. It reuses its prefix only for the `cache_key`
//!   whose last request filled it and succeeded; a new key, no key, or any
//!   request after one that failed or was cancelled is sent with
//!   `cache_prompt: false`, so `cached_tokens` never reveals another
//!   conversation.

mod adapter;
mod args;
mod http;
mod lifecycle;
mod server;

pub use adapter::{capabilities, RUNNER};
pub use args::{parse_args, USAGE};

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

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

impl AdapterOptions {
    /// Options with no `--ctx` and no extra arguments.
    pub fn new(server_bin: impl Into<PathBuf>, run_dir: impl Into<PathBuf>) -> AdapterOptions {
        AdapterOptions { server_bin: server_bin.into(), run_dir: run_dir.into(), context_length: None, extra_args: Vec::new() }
    }
}

/// Run the adapter on this process's stdin/stdout until stdin closes.
///
/// Installs handlers for SIGTERM, SIGINT and SIGHUP and starts a watchdog
/// thread, so call it once per process, from a process that does nothing
/// else. Nothing but protocol lines is written to stdout; logs, including
/// `llama-server`'s output (prefixed `[llama-server <pid>]`), go to stderr.
pub fn run_stdio(opts: AdapterOptions) -> anyhow::Result<()> {
    // Which llama.cpp build this adapter drives, for the hello handshake
    // (#28): read once here, so a hanging or missing binary cannot stall a
    // request later; `None` simply leaves the field out of the reply.
    let _ = server::LLAMA_BUILD.set(server::probe_llama_build(&opts.server_bin));
    let shared = Arc::new(server::Shared::default());
    lifecycle::install_signal_handlers();
    lifecycle::spawn_watchdog(Arc::clone(&shared));
    let rx = lifecycle::spawn_stdin_reader(Arc::clone(&shared), std::io::stdin());
    let out = adapter::Out::new(Box::new(std::io::stdout()));
    let result = serve(opts, Arc::clone(&shared), rx, &out);
    shared.close(server::STOP_GRACE);
    result
}

/// The request loop, one request at a time, in order.
fn serve(opts: AdapterOptions, shared: Arc<server::Shared>, rx: Receiver<lifecycle::Incoming>, out: &adapter::Out) -> anyhow::Result<()> {
    let mut a = adapter::Adapter::new(opts, shared);
    while let Ok(incoming) = rx.recv() {
        let flow = match incoming {
            lifecycle::Incoming::Bad { error } => {
                out.emit(&serde_json::json!({ "error": error }))?;
                adapter::Flow::Continue
            }
            lifecycle::Incoming::Request { seq, req } => match catch_unwind(AssertUnwindSafe(|| a.handle(seq, &req, out))) {
                Ok(flow) => flow.map_err(|e| anyhow::anyhow!("stdout closed: {e}"))?,
                Err(panic) => {
                    let what = panic
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "panic".into());
                    out.emit(&serde_json::json!({ "error": format!("internal error: {what}") }))?;
                    adapter::Flow::Continue
                }
            },
        };
        if let adapter::Flow::Exit(why) = flow {
            a.stop();
            anyhow::bail!("{why}");
        }
    }
    a.stop();
    Ok(())
}

/// One line on stderr, prefixed. Never panics, unlike `eprintln!`.
pub(crate) fn log(args: std::fmt::Arguments) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr().lock(), "[estia-llama] {args}");
}
