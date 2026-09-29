//! `estia` — the terminal front for the Estia engine: set a machine up,
//! install the service, run the HTTP daemon, pair LAN clients, list, pull and
//! import models, install a backend's runtime, generate (with an optional JSON
//! schema), chat, embed, and bench.
//!
//! One backend per data directory: `mlx-python` (Apple Silicon) or
//! `llama-cpp` (everywhere), from `--backend`, `ESTIA_BACKEND`, `config.json`
//! or the machine's default. On llama.cpp this binary is also the runner: the
//! engine starts `estia runner llama …`, the llama.cpp adapter.

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use estia_engine::models::custom::{ImportMode, ImportOptions};
use estia_engine::models::embed::embed_models;
use estia_engine::models::registry::family_known;
use estia_engine::models::{find_artifact, generation_artifacts, Artifact, DownloadSpec, ModelKind, ModelStore};
use estia_engine::proto::Message;
use estia_engine::runtime::{LlamaRuntime, PythonRuntime};
use estia_engine::structured::{self, OutputFormat, Structured, StructuredError};
use estia_engine::{Backend, CancelToken, Engine, EngineConfig, LlamaLaunch, LlamaServer, Priority, RoleBinding, Roles};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing_subscriber::filter::{Directive, EnvFilter, LevelFilter};

/// What `estia --version` prints after the name: `0.4.0 (<commit>, <date>)`.
/// build.rs sets the commit and date; see docs/versioning.md.
const VERSION_LINE: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("ESTIA_BUILD_COMMIT"), ", ", env!("ESTIA_BUILD_DATE"), ")");

/// The `estia-engine` features this binary is built with, as cli/Cargo.toml
/// lists them (a test keeps the two in step).
const ENGINE_FEATURES: &[&str] = &["python-mlx", "llama-runtime"];

#[derive(Parser)]
#[command(
    name = "estia",
    version = VERSION_LINE,
    about = "Estia: a local LLM inference engine. Serve an OpenAI-compatible API, manage models and roles, pair devices on the LAN."
)]
struct Cli {
    /// Engine data directory (holds models/, runtime/, config.json).
    #[arg(long, global = true, env = "ESTIA_DATA_DIR")]
    data_dir: Option<PathBuf>,
    /// The resident runner script (the Python MLX runner).
    #[arg(long, global = true, env = "ESTIA_RUNNER")]
    runner: Option<PathBuf>,
    /// Python interpreter for the runner. Default: the installed runtime, else `python3`.
    #[arg(long, global = true, env = "ESTIA_PYTHON")]
    python: Option<PathBuf>,
    /// Which backend runs models: `mlx-python` (Apple Silicon) or `llama-cpp`
    /// (aliases `mlx`, `llama`). Default: `backend` in config.json, else MLX
    /// on Apple Silicon and llama.cpp everywhere else. On llama.cpp,
    /// `ESTIA_LLAMA_SERVER` names your own llama-server and `ESTIA_LLAMA_ARGS`
    /// adds llama-server arguments (`-ngl 0`).
    #[arg(long, global = true, env = "ESTIA_BACKEND", value_name = "BACKEND")]
    backend: Option<Backend>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// One command to a working engine: runtime, default models, first token, roles.
    Setup {
        /// Roles to make ready (their default families are pulled).
        #[arg(long, value_delimiter = ',', default_value = "text,fast,embed")]
        roles: Vec<String>,
        /// Skip model pulls (runtime + token + config only).
        #[arg(long)]
        no_models: bool,
        /// llama.cpp only: the build to install (`cpu`, `metal`, `vulkan`,
        /// `cuda-12`, `cuda-13`, `rocm`, …). Default: probe this machine.
        #[arg(long)]
        variant: Option<String>,
    },
    /// Run the engine at login (launchd on macOS, systemd --user on Linux).
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },
    /// Live terminal view of a running engine: health, load, models, pairings.
    Dashboard {
        /// Engine URL; default: the daemon recorded in this data dir.
        #[arg(long)]
        engine: Option<String>,
        /// Token for the engine (health works without one; stats need `models:read`, pairings need `admin`).
        #[arg(long)]
        token: Option<String>,
        #[arg(long, default_value_t = 2)]
        interval: u64,
        /// Print once and exit.
        #[arg(long)]
        once: bool,
    },
    /// List known and imported models, their format, and whether they are installed.
    Models,
    /// Download a model: an artifact id, or a family, role or embedding model
    /// id (this backend's artifact of it).
    Pull { id: String },
    /// Register a local GGUF file as a model (llama-cpp backend). Its metadata
    /// fills in the kind, context and width; bind it with `estia roles set`.
    Import {
        /// The `.gguf` file.
        file: PathBuf,
        /// Model id (lowercase letters, digits, `.`, `_`, `-`). Default: from
        /// the file's name.
        #[arg(long)]
        id: Option<String>,
        /// `generation` or `embedding`. Default: from the file's metadata.
        #[arg(long, value_parser = ["generation", "embedding"])]
        kind: Option<String>,
        /// The family roles bind to. Default: the id.
        #[arg(long)]
        family: Option<String>,
        /// Display name, shown in `/engine/models` and the test client.
        /// Default: the file's `general.name`, else the id.
        #[arg(long)]
        label: Option<String>,
        /// Context length to run it with. Default: the file's, capped at 32768.
        #[arg(long = "ctx", value_name = "TOKENS")]
        context_length: Option<u32>,
        /// Embedding models: vector width, when a projection head changes it.
        #[arg(long)]
        dims: Option<usize>,
        /// Embedding models: text put before each query (`query: `). Default: none.
        #[arg(long)]
        query_prefix: Option<String>,
        /// Embedding models: text put before each document (`passage: `). Default: none.
        #[arg(long)]
        doc_prefix: Option<String>,
        /// Link to the file where it is instead of copying it.
        #[arg(long)]
        link: bool,
        /// Replace an earlier import with the same id.
        #[arg(long)]
        replace: bool,
        /// Generation models: the image projector (`mmproj-….gguf`) that lets
        /// the model read images. Stored beside it as `mmproj.gguf`.
        #[arg(long, value_name = "FILE")]
        mmproj: Option<PathBuf>,
    },
    /// Remove an installed model and any partial download.
    Rm { id: String },
    /// Data dir, runtime, runner, installed models.
    Status,
    /// What this machine can hold: its memory tier, the budget for resident
    /// models, and which models to bind to `text`, `fast` and `vision`.
    Recommend {
        /// Bind the recommended models to their roles (config.json).
        #[arg(long)]
        apply: bool,
        /// Print the recommendation as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show or change role → family bindings (persisted in config.json).
    Roles {
        #[command(subcommand)]
        action: Option<RolesAction>,
    },
    /// Generate from the prompt on stdin.
    Run {
        /// Role name, family, or artifact id.
        #[arg(long, default_value = "text")]
        model: String,
        #[arg(long, default_value_t = 512)]
        max_tokens: u32,
        #[arg(long, default_value_t = 0.2)]
        temperature: f32,
        /// Require output that validates against this JSON Schema file (retries once).
        #[arg(long)]
        schema: Option<PathBuf>,
        /// Require any valid JSON.
        #[arg(long)]
        json: bool,
        /// Collect the whole answer before printing.
        #[arg(long)]
        no_stream: bool,
        /// Queue behind interactive work instead of ahead of background work.
        #[arg(long)]
        background: bool,
        /// Testing aid: cancel the generation this many milliseconds after it starts.
        #[arg(long)]
        cancel_after_ms: Option<u64>,
    },
    /// Embed each line of stdin; prints a JSON array of vectors.
    Embed {
        #[arg(long)]
        model: Option<String>,
    },
    /// Time a generation and an embed batch on this machine.
    Bench {
        #[arg(long, default_value = "text")]
        model: String,
        #[arg(long)]
        embed_model: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
    },
    /// The backend's runtime: Python + MLX (`mlx-python`), or the pinned
    /// llama.cpp build (`llama-cpp`). Pick one with `--backend`.
    Runtime {
        #[command(subcommand)]
        action: RuntimeAction,
    },
    /// Handshake with the runner: protocol version and capabilities. Loads no model.
    RunnerCheck,
    /// Run a backend's runner on stdin/stdout (what the engine starts):
    /// `estia runner llama --server <llama-server> --run-dir <dir> [--ctx <n>] [-- <args>]`.
    #[command(hide = true)]
    Runner {
        kind: String,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<OsString>,
    },
    /// Chat (protocol v2): stdin is a JSON array of {role, content} messages, or plain
    /// text for a single user turn. Rendered by the model's own chat template.
    Chat {
        #[arg(long, default_value = "text")]
        model: String,
        /// System message prepended when stdin is plain text.
        #[arg(long)]
        system: Option<String>,
        /// Keep the KV cache for this conversation across invocations of the same runner.
        #[arg(long)]
        cache_key: Option<String>,
        /// JSON file with an array of OpenAI tool schemas, declared to the model.
        #[arg(long)]
        tools: Option<PathBuf>,
        #[arg(long, default_value_t = 512)]
        max_tokens: u32,
        #[arg(long, default_value_t = 0.2)]
        temperature: f32,
        /// Run two turns in one runner process (the second appends the first answer
        /// and a follow-up) to show the prompt cache at work.
        #[arg(long)]
        two_turns: bool,
        /// Attach an image (PNG, JPEG, WebP or GIF) to the user message; repeat
        /// for several. Needs a model that reads images, such as the `vision` role.
        #[arg(long = "image", value_name = "FILE")]
        images: Vec<PathBuf>,
    },
    /// Count tokens of stdin under a model's tokenizer.
    Tokens {
        #[arg(long, default_value = "text")]
        model: String,
    },
    /// Run the HTTP daemon: OpenAI-compatible /v1/* plus native /engine/*.
    Serve {
        #[arg(long, default_value_t = 27200)]
        port: u16,
        /// Address to bind (default 127.0.0.1, or 0.0.0.0 with --lan). Anything
        /// but loopback needs --lan.
        #[arg(long)]
        bind: Option<String>,
        /// Serve the LAN: binds 0.0.0.0 unless --bind says otherwise, requires
        /// auth (clients pair for a token), advertises over Bonjour.
        #[arg(long)]
        lan: bool,
        /// Do not advertise over Bonjour even on the LAN.
        #[arg(long)]
        no_advertise: bool,
        /// Bonjour instance name (default: this machine's hostname).
        #[arg(long)]
        name: Option<String>,
        /// Testing aid: accept requests without a bearer token (loopback only).
        #[arg(long)]
        no_auth: bool,
        /// Release a resident model after this many idle minutes (0 = never).
        /// Default: 3 on a constrained machine (16 GB or less, or no fan),
        /// else 15. See `estia recommend`.
        #[arg(long, value_name = "MINUTES")]
        idle_unload_minutes: Option<u64>,
        /// Most memory resident models may hold together, such as `8GB`, or
        /// `off`. Before a model loads, idle models are unloaded, least
        /// recently used first, until it fits; one larger than the whole
        /// budget is refused with a 503. Default: half of RAM on a
        /// constrained machine, 60 % on a standard one, 70 % on a capable
        /// one (`ESTIA_MEMORY_BUDGET` overrides).
        #[arg(long, value_name = "SIZE", value_parser = parse_memory_budget)]
        memory_budget: Option<MemoryBudget>,
        /// Also accept requests whose Host header names this host (repeatable,
        /// or comma-separated): a reverse proxy's name, or a LAN name such as
        /// `mac.lan`. IP literals, `localhost` and `<name>.local` are always
        /// accepted.
        #[arg(long = "allow-host", value_name = "NAME", value_delimiter = ',')]
        allow_host: Vec<String>,
        /// Largest request body to accept, in bytes; a larger one gets a JSON
        /// 413. Default: `ESTIA_MAX_BODY_BYTES`, else 33554432 (32 MiB, a
        /// full embed batch of 256 inputs of ~100 KB).
        #[arg(long, value_name = "BYTES", value_parser = clap::value_parser!(u64).range(1..))]
        max_body_bytes: Option<u64>,
        /// What to log: a level (`debug`) or comma-separated filter directives
        /// (`estia_server=debug,mdns_sd=info`), on top of the default `info`
        /// for Estia and `warn` for libraries. See docs/logging.md.
        #[arg(long, env = "ESTIA_LOG", value_name = "FILTER")]
        log_level: Option<String>,
        /// Log format on stderr: `text` for people, `json` for log shippers
        /// (one object per line).
        #[arg(long, env = "ESTIA_LOG_FORMAT", value_enum, default_value_t = LogFormat::Text)]
        log_format: LogFormat,
    },
    /// Pairing: approve LAN clients (operator side) or request a token (client side).
    Pair {
        #[command(subcommand)]
        action: PairAction,
    },
    /// Find engines advertising on the LAN.
    Discover {
        #[arg(long, default_value_t = 3)]
        seconds: u64,
    },
    /// Talk to a daemon as a client: health, a generation and an embed through
    /// the same remote path a host uses.
    RemoteCheck {
        #[arg(long)]
        engine: String,
        #[arg(long)]
        token: Option<String>,
        #[arg(long, default_value = "fast")]
        model: String,
    },
    /// Bearer tokens for the daemon (stored hashed in tokens.json).
    Token {
        #[command(subcommand)]
        action: TokenAction,
    },
    /// This build: version, commit, date, target, API and protocol versions,
    /// backends, features and the pinned llama.cpp build.
    Version {
        /// Print one JSON object, for scripts.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum ServiceAction {
    /// Write the unit and start it now (runs `estia serve --lan` at login).
    Install {
        #[arg(long, default_value_t = 27200)]
        port: u16,
        /// Loopback only (no LAN, no pairing needed for local clients with a token).
        #[arg(long)]
        local: bool,
        /// Passed to `serve --allow-host`: extra Host names the service
        /// answers to (repeatable, or comma-separated).
        #[arg(long = "allow-host", value_name = "NAME", value_delimiter = ',')]
        allow_host: Vec<String>,
        /// Passed to `serve --log-level` (see `estia serve --help`). Only this
        /// flag is passed on; `ESTIA_LOG` in your shell is not.
        #[arg(long, value_name = "FILTER")]
        log_level: Option<String>,
        /// Passed to `serve --log-format`: `text` (default) or `json`.
        #[arg(long, value_enum)]
        log_format: Option<LogFormat>,
        /// Passed to `serve --max-body-bytes`: the largest request body the
        /// service accepts, in bytes (default 32 MiB).
        #[arg(long, value_name = "BYTES", value_parser = clap::value_parser!(u64).range(1..))]
        max_body_bytes: Option<u64>,
        /// Passed to `serve --memory-budget`: the most memory resident models
        /// may hold (`8GB`, or `off`). Default: from this machine's tier.
        #[arg(long, value_name = "SIZE", value_parser = parse_memory_budget_arg)]
        memory_budget: Option<String>,
        /// Passed to `serve --idle-unload-minutes`. Default: from this
        /// machine's tier (3 when constrained, else 15).
        #[arg(long, value_name = "MINUTES")]
        idle_unload_minutes: Option<u64>,
        // `--backend` (global) is passed to `serve` too when given.
    },
    Uninstall,
    Start,
    Stop,
    Restart,
    Status,
    /// Print the end of the service log (macOS: the log files; Linux: the journal).
    Logs {
        #[arg(long, short = 'n', default_value_t = 40)]
        lines: usize,
        /// Keep printing new lines as they are written, until Ctrl-C.
        #[arg(long, short = 'f')]
        follow: bool,
    },
}

#[derive(Subcommand)]
enum PairAction {
    /// Pending and recent pairing requests (locally, or on a remote engine with --engine/--token).
    List {
        /// Ask a running engine (`http://host:port`) instead of reading this
        /// machine's data directory. Needs --token.
        #[arg(long, value_name = "URL")]
        engine: Option<String>,
        /// An `admin` token for the engine given with --engine.
        #[arg(long)]
        token: Option<String>,
    },
    /// Approve a request: mints a token with the requested scopes for the client to collect.
    Approve {
        /// The pairing id, as the client and `estia pair list` show it.
        id: String,
        /// Decide on a running engine (`http://host:port`) instead of in this
        /// machine's data directory. Needs --token.
        #[arg(long, value_name = "URL")]
        engine: Option<String>,
        /// An `admin` token for the engine given with --engine.
        #[arg(long)]
        token: Option<String>,
        /// Approve a request that asks for the `admin` scope (full control of
        /// the engine). Without it such a request is refused.
        #[arg(long)]
        allow_admin: bool,
    },
    /// Deny a pending request, or take back an approved one: its token is
    /// revoked, whether or not the client has collected it yet.
    Deny {
        /// The pairing id, as the client and `estia pair list` show it.
        id: String,
        /// Decide on a running engine (`http://host:port`) instead of in this
        /// machine's data directory. Needs --token.
        #[arg(long, value_name = "URL")]
        engine: Option<String>,
        /// An `admin` token for the engine given with --engine.
        #[arg(long)]
        token: Option<String>,
    },
    /// Client side: ask an engine for a token and wait for the operator's approval.
    Request {
        /// `http://host:port` of the engine.
        #[arg(long)]
        engine: String,
        /// How the operator sees this device: up to 64 letters, digits, single
        /// spaces and `. _ - ' ( )`.
        #[arg(long, default_value = "estia-cli")]
        name: String,
        /// Scopes to ask for, comma-separated. Ask for the least you need.
        #[arg(long, value_delimiter = ',', default_value = "generate,embed,models:read")]
        scopes: Vec<String>,
        /// How long to wait for the operator. The engine drops an undecided
        /// request 300 s after it was made, so waiting longer gains nothing;
        /// the default stops just before that.
        #[arg(long, default_value_t = PAIR_WAIT_DEFAULT_S)]
        wait_seconds: u64,
    },
}

/// How long an engine keeps an undecided pairing request.
const PAIR_EXPIRY_S: u64 = estia_server::pairing::PAIRING_TTL_SECS;
/// `estia pair request` stops waiting a little before the engine drops the
/// request, so it ends with its own message rather than a 404.
const PAIR_WAIT_DEFAULT_S: u64 = PAIR_EXPIRY_S - 10;

#[derive(Subcommand)]
enum TokenAction {
    /// Mint a token; the plaintext is printed once. Names are unique.
    New {
        name: String,
        /// Scopes: generate, embed, models:read, models:write, admin. Default admin
        /// (with --replace: the replaced token's scopes).
        #[arg(long, value_delimiter = ',')]
        scopes: Option<Vec<String>>,
        /// Rotate an existing token of this name: the previous one stops working now.
        #[arg(long)]
        replace: bool,
    },
    List,
    Revoke {
        name: String,
    },
}

#[derive(Subcommand)]
enum RolesAction {
    /// Bind a role to a family (checked against the family's capabilities).
    Set { role: String, family: String },
    /// Remove a binding; the role falls back along its chain.
    Rm { role: String },
}

#[derive(Subcommand)]
enum RuntimeAction {
    Status,
    Install {
        /// llama.cpp only: `cpu`, `metal`, `vulkan`, `cuda-12`, `cuda-13`,
        /// `rocm`, … Default: probe this machine (`ESTIA_LLAMA_VARIANT`).
        #[arg(long)]
        variant: Option<String>,
    },
    Remove,
}

/// `config.json`. Other keys in the file are kept when it is written.
#[derive(Debug, Default, Serialize, Deserialize)]
struct FileConfig {
    #[serde(default)]
    roles: Option<Roles>,
    /// Chosen by `estia setup`; `--backend` / `ESTIA_BACKEND` override it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    backend: Option<Backend>,
}

/// The backend for this invocation and where the choice came from: the
/// flag or `ESTIA_BACKEND`, then `config.json`, then the machine's default
/// (MLX on Apple Silicon, llama.cpp elsewhere).
const BACKEND_SOURCE_DEFAULT: &str = "default for this machine";

fn choose_backend(flag: Option<Backend>, file: Option<Backend>) -> (Backend, &'static str) {
    match (flag, file) {
        (Some(b), _) => (b, "--backend / ESTIA_BACKEND"),
        (None, Some(b)) => (b, "config.json"),
        (None, None) => (Backend::platform_default(), BACKEND_SOURCE_DEFAULT),
    }
}

/// Extra llama-server arguments from `ESTIA_LLAMA_ARGS`, split on whitespace.
fn llama_extra_args() -> Vec<String> {
    std::env::var("ESTIA_LLAMA_ARGS").map(|v| v.split_whitespace().map(str::to_string).collect()).unwrap_or_default()
}

struct Ctx {
    data_dir: PathBuf,
    store: ModelStore,
    runtime: PythonRuntime,
    llama_runtime: LlamaRuntime,
    backend: Backend,
    /// Where `backend` came from, for `estia status`.
    backend_source: &'static str,
    /// `--backend` / `ESTIA_BACKEND` when given: what `service install` passes on.
    backend_flag: Option<Backend>,
    /// `--runner` / `ESTIA_RUNNER`, if given.
    runner_arg: Option<PathBuf>,
    /// Resolved on first use: resolving may write the compiled-in runner into
    /// the data dir, which commands such as `discover` must not do.
    runner_found: std::sync::OnceLock<Option<PathBuf>>,
    python: Option<PathBuf>,
    roles: Roles,
}

fn default_data_dir() -> PathBuf {
    estia_engine::default_data_dir()
}

/// The resident runner, compiled in so an installed binary (`cargo install`,
/// a release tarball without its `runners/` folder) still has one.
/// `cli/estia-runner.py` is a symlink to `runners/mlx-python/estia-runner.py`,
/// so the script is inside this crate and ships in its package.
const EMBEDDED_RUNNER: &str = include_str!("../estia-runner.py");

/// The runner script, by the first rule that finds one:
///
/// 1. `--runner` / `ESTIA_RUNNER`.
/// 2. `runners/mlx-python/estia-runner.py` beside the binary: a release
///    tarball, or what `service install` stages.
/// 3. The same path two levels up, only when the binary sits in a cargo
///    `target/<profile>/` directory: `target/debug/estia` in a checkout.
/// 4. The compiled-in copy, written to `<data_dir>/engine/runners/`.
///
/// Never the current directory: running `estia` inside a downloaded or cloned
/// folder must not execute a script that folder happens to contain.
fn find_runner(explicit: Option<PathBuf>, data_dir: &Path) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p);
    }
    const REL: &str = "runners/mlx-python/estia-runner.py";
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(REL));
            if dir.parent().and_then(Path::file_name).is_some_and(|n| n == "target") {
                candidates.push(dir.join("../..").join(REL));
            }
        }
    }
    if let Some(found) = candidates.into_iter().find(|p| p.exists()).and_then(|p| p.canonicalize().ok()) {
        return Some(found);
    }
    // Nothing on disk: write the compiled-in copy into the data dir, where
    // `service install` would stage it anyway, and keep it in step with this
    // binary.
    let path = data_dir.join("engine").join(REL);
    if std::fs::read_to_string(&path).ok().as_deref() != Some(EMBEDDED_RUNNER) {
        std::fs::create_dir_all(path.parent()?).ok()?;
        let tmp = path.with_extension("staging");
        std::fs::write(&tmp, EMBEDDED_RUNNER).ok()?;
        std::fs::rename(&tmp, &path).ok()?;
    }
    Some(path)
}

impl Ctx {
    fn new(cli: &Cli) -> Result<Self> {
        let data_dir = cli.data_dir.clone().unwrap_or_else(default_data_dir);
        let agent = format!("estia/{}", env!("CARGO_PKG_VERSION"));
        let store = ModelStore::new(data_dir.join("models")).with_user_agent(agent.clone());
        let runtime = PythonRuntime::new(data_dir.join("runtime")).with_user_agent(agent.clone());
        let llama_runtime = LlamaRuntime::new(data_dir.join("runtime")).with_user_agent(agent);
        let file = match std::fs::read_to_string(data_dir.join("config.json")) {
            Ok(text) => serde_json::from_str::<FileConfig>(&text).context("parse config.json")?,
            Err(_) => FileConfig::default(),
        };
        let (backend, backend_source) = choose_backend(cli.backend, file.backend);
        // Imported models resolve (and roles bind to them) like built-ins.
        store.load_imported();
        Ok(Self {
            data_dir,
            store,
            runtime,
            llama_runtime,
            backend,
            backend_source,
            backend_flag: cli.backend,
            runner_arg: cli.runner.clone(),
            runner_found: std::sync::OnceLock::new(),
            python: cli.python.clone(),
            roles: file.roles.unwrap_or_else(Roles::defaults),
        })
    }

    fn runner(&self) -> Option<PathBuf> {
        self.runner_found.get_or_init(|| find_runner(self.runner_arg.clone(), &self.data_dir)).clone()
    }

    /// Write `config.json`: `roles`, and `backend` when given, over whatever
    /// else the file holds.
    fn save_config(&self, backend: Option<Backend>) -> Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        let path = self.data_dir.join("config.json");
        let mut cfg: serde_json::Value =
            std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_else(|| serde_json::json!({}));
        if !cfg.is_object() {
            cfg = serde_json::json!({});
        }
        cfg["roles"] = serde_json::to_value(&self.roles)?;
        if let Some(b) = backend {
            cfg["backend"] = serde_json::to_value(b)?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(&cfg)?)?;
        Ok(())
    }

    fn save_roles(&self) -> Result<()> {
        self.save_config(None)
    }

    fn python(&self) -> PathBuf {
        self.python.clone().or_else(|| self.runtime.python_path()).unwrap_or_else(|| PathBuf::from("python3"))
    }

    /// `<data dir>/run`: the llama adapter's sockets, key files and pid records.
    fn run_dir(&self) -> PathBuf {
        self.data_dir.join("run")
    }

    /// An engine that starts runners on this invocation's backend.
    fn engine(&self) -> Result<Engine> {
        let cfg = match self.backend {
            Backend::MlxPython => {
                if !Backend::MlxPython.supported_here() {
                    return Err(anyhow!("the mlx-python backend needs Apple Silicon; use --backend llama-cpp"));
                }
                let runner = self
                    .runner()
                    .ok_or_else(|| anyhow!("no runner script found — pass --runner or set ESTIA_RUNNER to estia-runner.py"))?;
                EngineConfig::new(self.store.clone(), self.runtime.clone(), runner).with_python(self.python())
            }
            Backend::LlamaCpp => {
                // This binary is the adapter: `estia runner llama …`.
                let exe = std::env::current_exe().context("locate the estia binary, which runs the llama.cpp adapter")?;
                let mut launch =
                    LlamaLaunch::new(exe, self.run_dir()).prefix_arg("runner").prefix_arg("llama").with_server(LlamaServer::from_env());
                for a in llama_extra_args() {
                    launch = launch.server_arg(a);
                }
                EngineConfig::new(self.store.clone(), self.runtime.clone(), PathBuf::new()).with_llama(launch)
            }
        };
        Ok(Engine::new(cfg.with_roles(self.roles.clone()).with_llama_runtime(self.llama_runtime.clone())))
    }

    /// An engine for resolving names and listing models only: it never
    /// starts a runner, so it needs no runtime or runner script.
    fn resolver(&self) -> Engine {
        let cfg = EngineConfig::new(self.store.clone(), self.runtime.clone(), PathBuf::new())
            .with_backend(self.backend)
            .with_roles(self.roles.clone())
            .with_llama_runtime(self.llama_runtime.clone());
        Engine::new(cfg)
    }

    /// Role name, family or artifact id → this backend's artifact.
    fn resolve_generation(&self, engine: &Engine, name: &str) -> Result<&'static Artifact> {
        let artifact = engine.resolve_generation(name).map_err(|e| anyhow!("{e}"))?;
        if find_artifact(name).is_none() && !family_known(name) {
            if let Ok(res) = self.roles.resolve(name) {
                if res.served_by != res.asked {
                    eprintln!("role `{}` is unbound; served by `{}` ({})", res.asked, res.served_by, artifact.id);
                }
            }
        }
        Ok(artifact)
    }
}

// ── Printing text this process did not write ──────────────────────────────────

/// True for characters that can steer a terminal or hide what it shows: C0
/// and C1 controls (ESC opens every escape sequence, CR rewrites the line),
/// bidi embeddings, overrides and isolates, zero-width and other invisible
/// format characters, the Unicode line and paragraph separators, and tag
/// characters.
fn is_unsafe_char(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{00AD}'
                | '\u{061C}'
                | '\u{180E}'
                | '\u{200B}'..='\u{200F}'
                | '\u{2028}'..='\u{202E}'
                | '\u{2060}'..='\u{206F}'
                | '\u{FEFF}'
                | '\u{FFF9}'..='\u{FFFB}'
                | '\u{E0000}'..='\u{E007F}'
        )
}

/// `s` with each unsafe character (see [`is_unsafe_char`]) written as a
/// visible escape — `\u{1b}`, `\r`, `\u{202e}` — and everything else as is,
/// apostrophes and non-Latin letters included.
///
/// Every string that reached this process from outside goes through this
/// before it is printed: pairing and token names (chosen by whoever sent the
/// pairing request), anything a remote engine or an mDNS answer said, error
/// text built from those. A pairing name ending in `ESC[8m` once hid the
/// `admin` scope printed after it.
fn safe(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if is_unsafe_char(c) {
            out.extend(c.escape_default());
        } else {
            out.push(c);
        }
    }
    out
}

/// [`safe`], line by line: line breaks are kept (errors quote runner output
/// and tracebacks), every other unsafe character is escaped.
fn safe_lines(s: &str) -> String {
    s.split('\n').map(safe).collect::<Vec<_>>().join("\n")
}

/// A JSON value from an engine as printable text: a string escaped by
/// [`safe`], `?` for a missing value, anything else in its JSON form.
fn jtext(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => safe(s),
        serde_json::Value::Null => "?".into(),
        other => safe(&other.to_string()),
    }
}

/// A JSON array of strings from an engine, each escaped.
fn jlist(v: &serde_json::Value) -> Vec<String> {
    v.as_array().into_iter().flatten().filter_map(|x| x.as_str()).map(safe).collect()
}

fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
}

fn print_progress(p: estia_engine::models::DownloadProgress) {
    let pct = match p.total_bytes {
        Some(t) if t > 0 => format!("{:>3}%", p.bytes_downloaded * 100 / t),
        _ => "    ".to_string(),
    };
    eprint!("\r{:<12} {pct} {} {}                    ", p.phase, gb(p.bytes_downloaded), safe(p.file_name.as_deref().unwrap_or("")));
    if p.phase == "complete" {
        eprintln!();
    }
}

fn read_stdin() -> Result<String> {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;
    Ok(s.trim().to_string())
}

/// Google's terms for the Gemma models, and the use policy they incorporate.
const GEMMA_TERMS_URL: &str = "https://ai.google.dev/gemma/terms";
const GEMMA_PROHIBITED_USE_URL: &str = "https://ai.google.dev/gemma/prohibited_use_policy";

/// Where to read a model licence that is not an SPDX id: its terms and its
/// prohibited-use policy. The test client (server/client/index.html) links
/// the same pages.
fn license_links(license: &str) -> Option<(&'static str, &'static str)> {
    (license == "Gemma Terms of Use").then_some((GEMMA_TERMS_URL, GEMMA_PROHIBITED_USE_URL))
}

/// Whether `license` is a single SPDX licence id (`Apache-2.0`, `MIT`)
/// rather than a vendor's licence name (`Gemma Terms of Use`).
fn is_spdx_id(license: &str) -> bool {
    !license.is_empty() && license.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '+'))
}

/// A licence name, followed by its terms and prohibited-use pages when it
/// has them.
fn license_detail(license: &str) -> String {
    match license_links(license) {
        Some((terms, prohibited)) => format!("{} · terms: {terms} · prohibited use: {prohibited}", safe(license)),
        None => safe(license),
    }
}

/// An artifact's licence as printed before it is downloaded: an SPDX id on
/// its own; any other licence with the pages to read, or, when Estia knows
/// none, the model card.
fn license_text(a: &Artifact) -> String {
    if is_spdx_id(a.license) || license_links(a.license).is_some() || a.repo_id.is_empty() {
        license_detail(a.license)
    } else {
        format!("{} · read the model card: https://huggingface.co/{}", safe(a.license), safe(a.repo_id))
    }
}

async fn pull(ctx: &Ctx, id: &str) -> Result<()> {
    let artifact = estia_server::catalog::pull_artifact(&ctx.resolver(), id).map_err(|e| anyhow!("{e} (see `estia models`)"))?;
    let spec = DownloadSpec::from(artifact);
    eprintln!("pulling {} from {}@{} ({} required)", spec.id, spec.repo_id, spec.revision, gb(spec.required_disk_bytes));
    eprintln!("licence: {}", license_text(artifact));
    let summary = ctx.store.download(&spec, print_progress).await?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

/// One row of `estia models`.
fn model_row(ctx: &Ctx, a: &Artifact, family: &str, kind: &str, extra: &str) -> String {
    let imported = estia_server::catalog::is_imported(&ctx.store, a.id);
    let (state, size) = if ctx.store.is_installed(a.id) && imported {
        // The model's own size: a linked import takes no space here.
        ("installed", gb(a.required_disk_bytes))
    } else if ctx.store.is_installed(a.id) {
        ("installed", ctx.store.bytes_on_disk(a.id).map(gb).unwrap_or_default())
    } else if let Some(b) = ctx.store.partial_bytes_on_disk(a.id) {
        ("partial", format!("{} of {}", gb(b), gb(a.required_disk_bytes)))
    } else {
        ("missing", gb(a.required_disk_bytes))
    };
    let runs = if a.format == ctx.backend.format() { "*" } else { " " };
    let source = if imported { "imported" } else { "built-in" };
    format!(
        "{runs} {:<36} {:<24} {:<5} {:<6} {:<8} {:<9} {:<18} {size}{extra}",
        safe(a.id),
        safe(family),
        kind,
        a.format.id(),
        source,
        state,
        safe(a.license)
    )
}

/// The lines under `estia models` that say where to read each listed
/// licence that has its own terms, once per licence.
fn license_notes<'a>(licenses: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut seen: Vec<&str> = Vec::new();
    for l in licenses {
        if license_links(l).is_some() && !seen.contains(&l) {
            seen.push(l);
        }
    }
    seen.into_iter().map(license_detail).collect()
}

fn models(ctx: &Ctx) -> Result<()> {
    println!("backend {} ({}) · * = runs on it", ctx.backend, ctx.backend_source);
    println!("  {:<36} {:<24} {:<5} {:<6} {:<8} {:<9} {:<18} size", "id", "family / model", "kind", "format", "source", "state", "licence");
    let mut licenses = Vec::new();
    for a in generation_artifacts() {
        println!("{}", model_row(ctx, a, a.family, "gen", ""));
        licenses.push(a.license);
    }
    for e in embed_models() {
        for a in e.artifacts {
            let fp = if a.format == ctx.backend.format() {
                format!(" · {}", safe(&format!("{}@{}", a.id, ctx.backend)))
            } else {
                String::new()
            };
            let extra = format!(" ({}-dim{fp})", e.dims);
            println!("{}", model_row(ctx, a, e.id, "embed", &extra));
            licenses.push(a.license);
        }
    }
    let notes = license_notes(licenses);
    if !notes.is_empty() {
        println!();
        for n in notes {
            println!("{n}");
        }
    }
    Ok(())
}

/// The llama.cpp runtime in one line: a user's own `llama-server`, the
/// installed pinned build, or how to install it.
fn llama_runtime_line(ctx: &Ctx) -> String {
    if let LlamaServer::Path(p) = LlamaServer::from_env() {
        return format!("ESTIA_LLAMA_SERVER={}{}", safe(&p.display().to_string()), if p.is_file() { "" } else { " (missing!)" });
    }
    let st = ctx.llama_runtime.status();
    match (&st.variant, &st.server_path) {
        (Some(v), Some(path)) => format!("llama.cpp {} ({v}) · {}", st.build, safe(path)),
        _ => format!("not installed  (estia runtime install --backend llama-cpp: {})", st.build),
    }
}

fn status(ctx: &Ctx) -> Result<()> {
    println!("data dir : {}", ctx.data_dir.display());
    println!("models   : {}", ctx.store.models_dir().display());
    println!("backend  : {} ({})", ctx.backend, ctx.backend_source);
    println!("llama.cpp: {}", llama_runtime_line(ctx));
    let [machine, memory] = machine_lines();
    println!("machine  : {machine}");
    println!("memory   : {memory}");
    // The MLX lines only where MLX can run.
    if Backend::MlxPython.supported_here() {
        let rt = ctx.runtime.status();
        println!(
            "mlx      : {:?}{}",
            rt.state,
            rt.python_version
                .as_deref()
                .map(|v| format!(" (python {v}, mlx-lm {})", rt.mlx_lm_version.as_deref().unwrap_or("?")))
                .unwrap_or_default()
        );
        if ctx.backend == Backend::MlxPython {
            println!("python   : {}", ctx.python().display());
            println!("runner   : {}", ctx.runner().as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(none found)".into()));
        }
    }
    let mut installed: Vec<String> = generation_artifacts()
        .into_iter()
        .chain(embed_models().into_iter().flat_map(|e| e.artifacts.iter()))
        .filter(|a| ctx.store.is_installed(a.id))
        .map(|a| if a.format == ctx.backend.format() { safe(a.id) } else { format!("{} ({})", safe(a.id), a.format.id()) })
        .collect();
    installed.dedup();
    println!("installed: {}", if installed.is_empty() { "(none)".to_string() } else { installed.join(", ") });
    println!("roles    :");
    for (role, b) in ctx.roles.iter() {
        println!("  {:<8} → {}{}", safe(role), safe(&b.family), if b.pin { " (pinned)" } else { "" });
    }
    match estia_server::another_engine_running(&ctx.data_dir) {
        Some(rec) => {
            let health = reqwest_blocking()
                .ok()
                .and_then(|c| c.get(format!("{}/engine/health", local_url(&rec.bind, rec.port))).send().ok())
                .and_then(|r| r.json::<serde_json::Value>().ok());
            let up = health
                .as_ref()
                .and_then(|h| h["uptime_s"].as_u64())
                .map(|s| format!("{}m{}s", s / 60, s % 60))
                .unwrap_or_else(|| "?".into());
            let loaded = health
                .as_ref()
                .and_then(|h| h["loaded"].as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str()).map(safe).collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            let backend = health.as_ref().map(|h| jtext(&h["backend"])).unwrap_or_else(|| "?".into());
            println!(
                "daemon   : running · pid {} · {} · backend {backend} · up {up} · loaded [{loaded}]",
                rec.pid,
                host_port(&rec.bind, rec.port)
            );
            if rec.bind.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_unspecified()) {
                for ip in lan_ips() {
                    let url = http_url(&ip, rec.port);
                    println!("  reach it: {url}   (clients: estia pair request --engine {url})");
                }
            }
        }
        None => println!("daemon   : not running  (estia serve: this machine only · estia serve --lan: let other devices pair)"),
    }
    let pending: Vec<_> = estia_server::pairing::PairingStore::new(&ctx.data_dir)
        .list()
        .into_iter()
        .filter(|p| p.status == estia_server::pairing::PairingStatus::Pending)
        .collect();
    if !pending.is_empty() {
        println!("pairings : {} pending — estia pair list / estia pair approve <id>", pending.len());
    }
    println!("service  : {}", service_status_line());
    Ok(())
}

// One parameter per `estia import` flag.
#[allow(clippy::too_many_arguments)]
fn import(
    ctx: &Ctx,
    file: &Path,
    id: Option<String>,
    kind: Option<String>,
    family: Option<String>,
    label: Option<String>,
    context_length: Option<u32>,
    dims: Option<usize>,
    query_prefix: Option<String>,
    doc_prefix: Option<String>,
    link: bool,
    replace: bool,
    mmproj: Option<PathBuf>,
) -> Result<()> {
    let kind = match kind.as_deref() {
        Some("generation") => Some(ModelKind::Generation),
        Some("embedding") => Some(ModelKind::Embedding),
        Some(other) => return Err(anyhow!("--kind is generation or embedding, not `{}`", safe(other))),
        None => None,
    };
    let opts = ImportOptions {
        id,
        kind,
        family,
        label,
        context_length,
        embedding_dims: dims,
        query_prefix,
        doc_prefix,
        mode: if link { ImportMode::Symlink } else { ImportMode::Copy },
        replace,
        projector: mmproj,
    };
    if !link {
        eprintln!("copying and hashing {} …", file.display());
    }
    let m = ctx.store.import_gguf(file, opts)?;
    let kind = match m.kind {
        ModelKind::Generation => "generation",
        ModelKind::Embedding => "embedding",
    };
    println!("imported {} ({kind}, {})", safe(&m.id), safe(m.architecture.as_deref().unwrap_or("unknown architecture")));
    println!("  file     : {}", ctx.store.load_path(&m.id).display());
    match m.mode {
        ImportMode::Symlink => println!("  source   : {} (linked: moving it breaks the import)", safe(&m.source_path)),
        ImportMode::Copy => println!("  source   : {} (copied)", safe(&m.source_path)),
    }
    println!("  size     : {} · sha256 {}", gb(m.bytes), &m.sha256[..m.sha256.len().min(16)]);
    match m.kind {
        ModelKind::Generation => {
            println!("  family   : {}", safe(&m.family));
            println!(
                "  context  : {}{}",
                m.context_length.map(|c| c.to_string()).unwrap_or_else(|| "?".into()),
                m.declared_context_length.map(|c| format!(" (the file declares {c})")).unwrap_or_default()
            );
            println!(
                "  template : {}{}",
                if m.has_chat_template { "yes" } else { "none (chat will fail)" },
                if m.tools { ", tools" } else { "" }
            );
            println!("  use it   : estia chat --model {} · estia roles set fast {}", safe(&m.id), safe(&m.family));
        }
        ModelKind::Embedding => {
            println!(
                "  vectors  : {}-dim, {} pooling · fingerprint {}",
                m.embedding_dims.map(|d| d.to_string()).unwrap_or_else(|| "?".into()),
                m.pooling.map(|p| p.llama_arg()).unwrap_or("mean"),
                safe(&m.fingerprint().unwrap_or_default())
            );
            println!("  use it   : estia embed --model {} · estia roles set embed {}", safe(&m.id), safe(&m.id));
        }
    }
    if ctx.backend != Backend::LlamaCpp {
        println!(
            "  note     : imported GGUF models run on the llama-cpp backend (--backend llama-cpp, or `estia setup --backend llama-cpp`)"
        );
    }
    Ok(())
}

/// Non-loopback IPv4 addresses of this machine.
fn lan_ips() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(o) = std::process::Command::new("ifconfig").output() {
        let text = String::from_utf8_lossy(&o.stdout);
        for line in text.lines() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("inet ") {
                let ip = rest.split_whitespace().next().unwrap_or("");
                if !ip.starts_with("127.") && !ip.is_empty() && !ip.starts_with("169.254.") {
                    out.push(ip.to_string());
                }
            }
        }
    }
    out
}

/// `--memory-budget`: a size, or `off` (`None`).
#[derive(Debug, Clone, Copy)]
struct MemoryBudget(Option<u64>);

fn parse_memory_budget(raw: &str) -> std::result::Result<MemoryBudget, String> {
    estia_engine::machine::parse_budget(raw).map(MemoryBudget)
}

/// `service install --memory-budget`: checked here, passed on as written.
fn parse_memory_budget_arg(raw: &str) -> std::result::Result<String, String> {
    estia_engine::machine::parse_budget(raw).map(|_| raw.to_string())
}

/// This machine and its memory policy, for `status` and `recommend`.
fn machine_lines() -> [String; 2] {
    use estia_engine::machine::{gb, profile, MemoryPolicy};
    let p = profile();
    let m = MemoryPolicy::detect();
    let overrides = if p.overrides.is_empty() { String::new() } else { format!(" · overridden: {}", p.overrides.join(" ")) };
    [
        format!(
            "{} ({}) · {} RAM · {} cores{} · tier {}{overrides}",
            safe(&p.chip),
            safe(&p.model_id),
            gb(p.ram_bytes),
            p.logical_cores,
            if p.fanless { " · no fan" } else { "" },
            p.tier
        ),
        format!(
            "budget {} · idle unload {} · {}",
            m.budget_bytes.map(gb).unwrap_or_else(|| "off".into()),
            m.idle_unload.map(|d| format!("{} min", d.as_secs() / 60)).unwrap_or_else(|| "never".into()),
            match m.max_generation_models {
                Some(1) => "one generation model at a time".to_string(),
                Some(n) => format!("{n} generation models at a time"),
                None => "generation models as the budget allows".to_string(),
            }
        ),
    ]
}

fn recommend(ctx: &mut Ctx, apply: bool, json: bool) -> Result<()> {
    use estia_engine::machine::{self, gb};
    let policy = machine::MemoryPolicy::detect();
    let backend = ctx.backend;
    let embed_reserve = ctx
        .engine()
        .ok()
        .and_then(|e| e.resolve_embedding(None).ok().map(|(_, a)| machine::estimate_resident_bytes(a.required_disk_bytes, a.format)))
        .unwrap_or(0);
    let r = machine::recommend(&policy, backend, embed_reserve);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"machine": machine::profile(), "memory": policy, "recommendation": r}))?
        );
    } else {
        let [m, p] = machine_lines();
        println!("machine : {m}");
        println!("memory  : {p}");
        println!("backend : {backend}");
        println!();
        println!("  {:<18} {:>9}  fits", "family", "memory");
        for f in &r.families {
            println!(
                "  {:<18} {:>9}  {}{}",
                safe(f.family),
                gb(f.estimate_bytes),
                if f.fits { "yes" } else { "no" },
                if f.vision { "  (reads images)" } else { "" }
            );
        }
        if embed_reserve > 0 {
            println!("  (each beside the embedding model, about {})", gb(embed_reserve));
        }
        println!();
    }
    let wanted: Vec<(&str, &str)> =
        [("text", r.text), ("fast", r.fast), ("vision", r.vision)].into_iter().filter_map(|(role, f)| f.map(|f| (role, f))).collect();
    let mut changes = Vec::new();
    for (role, family) in &wanted {
        let current = ctx.roles.get(role).map(|b| b.family.clone());
        if current.as_deref() != Some(*family) {
            changes.push((*role, *family, current));
        }
    }
    if !json {
        for (role, family) in &wanted {
            println!("{role:<7} → {}", safe(family));
        }
        if r.families.iter().all(|f| !f.fits) {
            println!("\nno model fits the budget beside the embedding model; raise it with `estia serve --memory-budget`");
        }
        if changes.is_empty() {
            println!("\nthe roles already match");
        } else if !apply {
            println!();
            for (role, family, current) in &changes {
                println!(
                    "  estia roles set {role} {}   # now {}",
                    safe(family),
                    current.as_deref().map(safe).unwrap_or_else(|| "unset".into())
                );
            }
            println!("\nor run `estia recommend --apply`; then `estia pull` any family not yet installed");
        }
    }
    if apply && !changes.is_empty() {
        for (role, family, _) in &changes {
            ctx.roles.bind(role, RoleBinding::family(*family)).map_err(|e| anyhow!("{e}"))?;
        }
        ctx.save_roles()?;
        if !json {
            println!("\nroles updated; a running engine picks them up on restart (estia service restart)");
        }
    }
    Ok(())
}

fn roles(ctx: &mut Ctx, action: Option<RolesAction>) -> Result<()> {
    match action {
        None => {
            for (role, b) in ctx.roles.iter() {
                println!("{:<8} {}{}", role, b.family, if b.pin { " (pinned)" } else { "" });
            }
        }
        Some(RolesAction::Set { role, family }) => {
            ctx.roles.bind(&role, RoleBinding::family(family.clone())).map_err(|e| anyhow!("{e}"))?;
            ctx.save_roles()?;
            println!("{role} → {family}");
        }
        Some(RolesAction::Rm { role }) => match ctx.roles.unbind(&role) {
            Some(_) => {
                ctx.save_roles()?;
                println!("{role} unbound");
            }
            None => return Err(anyhow!("role `{role}` was not bound; nothing removed")),
        },
    }
    Ok(())
}

// One parameter per `estia run` flag.
#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &Ctx,
    model: &str,
    max_tokens: u32,
    temperature: f32,
    schema: Option<PathBuf>,
    json: bool,
    no_stream: bool,
    background: bool,
    cancel_after_ms: Option<u64>,
) -> Result<()> {
    let engine = ctx.engine()?;
    let artifact = ctx.resolve_generation(&engine, model)?;
    let prompt = read_stdin()?;
    if prompt.is_empty() {
        return Err(anyhow!("empty prompt on stdin"));
    }
    let format = match schema {
        Some(path) => {
            let text = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
            OutputFormat::JsonSchema { schema: serde_json::from_str(&text).context("schema is not JSON")? }
        }
        None if json => OutputFormat::Json,
        None => OutputFormat::Text,
    };
    let prio = if background { Priority::Background } else { Priority::Interactive };
    let started = Instant::now();
    let session = engine.spawn_gen_session(artifact.id)?;
    eprintln!("model {} · backend {} · spawned in {} ms", artifact.id, engine.backend(), started.elapsed().as_millis());
    // A runner that constrains decoding (llama.cpp) does so in `chat`: the
    // prompt goes there as one user turn. The output is validated either way.
    let caps = session.capabilities();
    let constrain = estia_server::openai::runner_format(&caps, &format, false).filter(|_| caps.chat);
    // Any other runner (MLX) is shown the schema after the prompt instead.
    let prompt = if constrain.is_none() { structured::prompt_with_hint(&prompt, &format) } else { prompt };

    let stream_out = !no_stream && !format.wants_json();
    let mut out = std::io::stdout();
    let mut generate = |prompt: &str| -> Result<String> {
        let cancel = CancelToken::new();
        if let Some(ms) = cancel_after_ms {
            let flip = cancel.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(ms));
                flip.cancel();
            });
        }
        let t0 = Instant::now();
        let mut first: Option<u128> = None;
        let on_token = |tok: &str| {
            if first.is_none() {
                first = Some(t0.elapsed().as_millis());
            }
            if stream_out {
                let _ = out.write_all(tok.as_bytes());
                let _ = out.flush();
            }
        };
        let result = match &constrain {
            Some(f) => session
                .chat_stream_with(
                    &[Message::new("user", prompt)],
                    None,
                    None,
                    Some(f),
                    Some(max_tokens),
                    Some(temperature),
                    prio,
                    Some(&cancel),
                    on_token,
                )
                .map(|o| o.text),
            None => session.generate_stream_with(prompt, Some(max_tokens), Some(temperature), prio, Some(&cancel), on_token),
        };
        let text = match result {
            Ok(text) => text,
            Err(estia_engine::SessionError::Cancelled { partial }) => {
                eprintln!("\ncancelled after {} ms with {} chars delivered", t0.elapsed().as_millis(), partial.chars().count());
                partial
            }
            Err(e) => return Err(e.into()),
        };
        eprintln!(
            "\n{} chars in {} ms (first token {} ms)",
            text.chars().count(),
            t0.elapsed().as_millis(),
            first.map(|ms| ms.to_string()).unwrap_or_else(|| "-".into())
        );
        Ok(text)
    };

    let text = generate(&prompt)?;
    if !format.wants_json() {
        if !stream_out {
            println!("{text}");
        } else {
            println!();
        }
        return Ok(());
    }
    let result: std::result::Result<Structured, StructuredError> = match structured::enforce(&text, &format) {
        Ok(s) => Ok(s),
        Err(first_err) => {
            eprintln!("first attempt rejected: {first_err}; retrying once");
            let retry_prompt = format!("{prompt}{}", Structured::retry_hint(&first_err));
            let second = generate(&retry_prompt)?;
            structured::enforce(&second, &format)
        }
    };
    match result {
        Ok(s) => {
            if s.repaired {
                eprintln!("repaired: {}", s.repairs.join(", "));
            }
            println!("{}", serde_json::to_string_pretty(&s.value)?);
            Ok(())
        }
        Err(e) => Err(anyhow!("structured output failed after retry: {e}")),
    }
}

fn embed(ctx: &Ctx, model: Option<&str>) -> Result<()> {
    let engine = ctx.engine()?;
    let (spec, _) = engine.resolve_embedding(model).map_err(|e| anyhow!("{e}"))?;
    let session = engine.spawn_embed_model(spec)?;
    let fingerprint = session.fingerprint().to_string();
    let inputs: Vec<String> = std::io::stdin().lock().lines().map_while(Result::ok).filter(|l| !l.trim().is_empty()).collect();
    if inputs.is_empty() {
        return Err(anyhow!("no input lines on stdin"));
    }
    let t0 = Instant::now();
    let vectors = session.embed_batch_with(&inputs, Priority::Interactive)?;
    eprintln!(
        "{} vectors × {} dims in {} ms · fingerprint {}",
        vectors.len(),
        vectors.first().map(|v| v.len()).unwrap_or(0),
        t0.elapsed().as_millis(),
        fingerprint
    );
    println!("{}", serde_json::to_string(&vectors)?);
    Ok(())
}

fn bench(ctx: &Ctx, model: &str, embed_model: Option<&str>, max_tokens: u32) -> Result<()> {
    let engine = ctx.engine()?;
    let artifact = ctx.resolve_generation(&engine, model)?;
    println!("machine   : {} · {}", std::env::consts::ARCH, std::env::consts::OS);
    println!("backend   : {}", engine.backend());

    // Generation: an explicit load (protocol v2) so the first generation is
    // measured warm-ish, then a second one fully warm. On a v1 runner the
    // load is a no-op and "first" includes the model load.
    let session = engine.spawn_gen_session(artifact.id)?;
    match session.load()? {
        Some(d) => println!("load          : {} in {} ms (protocol v2 load)", artifact.id, d.as_millis()),
        None => println!("load          : (v1 runner: model loads on first generation)"),
    }
    let prompt = "Write three plain sentences about the sea, then stop.";
    for label in ["first", "warm"] {
        let t0 = Instant::now();
        let mut first: Option<u128> = None;
        let mut pieces = 0usize;
        let on_token = |_: &str| {
            pieces += 1;
            if first.is_none() {
                first = Some(t0.elapsed().as_millis());
            }
        };
        // Chat when the runner has it: its meta line carries token counts
        // and the decode rate the runner measured.
        let (text, meta) = if session.capabilities().chat {
            let o = session.chat_stream_with(
                &[Message::new("user", prompt)],
                None,
                None,
                None,
                Some(max_tokens),
                Some(0.0),
                Priority::Interactive,
                None,
                on_token,
            )?;
            (o.text, Some(o.meta))
        } else {
            (session.generate_stream_with(prompt, Some(max_tokens), Some(0.0), Priority::Interactive, None, on_token)?, None)
        };
        let total = t0.elapsed().as_millis().max(1);
        let chars = text.chars().count();
        let decode_ms = total.saturating_sub(first.unwrap_or(0)).max(1);
        let tokens = match meta.as_ref().and_then(|m| m.generation_tokens) {
            Some(n) => {
                let tps = meta.as_ref().and_then(|m| m.generation_tps).unwrap_or(n as f64 * 1000.0 / decode_ms as f64);
                format!(" · {n} tokens · ≈{tps:.1} tokens/s")
            }
            None => String::new(),
        };
        println!(
            "generate {:<4}: {} · first token {} ms · total {} ms · {} chars · ≈{:.1} chars/s decode ({} pieces){tokens}",
            label,
            artifact.id,
            first.map(|m| m.to_string()).unwrap_or_else(|| "-".into()),
            total,
            chars,
            chars as f64 * 1000.0 / decode_ms as f64,
            pieces
        );
    }

    // Embedding: one warm batch of 32 short units.
    let (spec, eartifact) = engine.resolve_embedding(embed_model).map_err(|e| anyhow!("{e}"))?;
    let esession = engine.spawn_embed_model(spec)?;
    let units: Vec<String> =
        (0..32).map(|i| format!("title: none | text: sample sentence number {i} about the sea and the shore")).collect();
    if esession.load()?.is_none() {
        let _ = esession.embed_batch(&units[..1])?; // v1: load by embedding once
    }
    let t0 = Instant::now();
    let v = esession.embed_batch(&units)?;
    let ms = t0.elapsed().as_millis().max(1);
    println!(
        "embed warm    : {} · {} units × {} dims in {} ms · ≈{:.1} units/s",
        eartifact.id,
        v.len(),
        v.first().map(|x| x.len()).unwrap_or(0),
        ms,
        v.len() as f64 * 1000.0 / ms as f64
    );
    Ok(())
}

fn runner_check(ctx: &Ctx) -> Result<()> {
    let engine = ctx.engine()?;
    // No model is named, so the llama adapter gets no `--ctx`; nothing loads.
    let launch = engine.launch_for("")?;
    let command: Vec<String> = std::iter::once(launch.program().display().to_string())
        .chain(launch.args().iter().map(|a| a.to_string_lossy().into_owned()))
        .collect();
    let t0 = Instant::now();
    let session =
        estia_engine::Session::spawn(launch, estia_engine::SessionConfig::default(), std::sync::Arc::new(estia_engine::NoopObserver))?;
    let spawned = t0.elapsed().as_millis();
    let hello = session.hello()?;
    let ping_ok = session.call_unobserved(&estia_engine::proto::Request::Ping).is_ok();
    println!("backend  : {} ({})", engine.backend(), ctx.backend_source);
    println!("command  : {}", safe(&command.join(" ")));
    if engine.backend() == Backend::LlamaCpp {
        let server = engine.llama_server_path()?;
        let version = estia_engine::runtime::llama::server_version(&server, estia_engine::runtime::llama::FIRST_RUN_TIMEOUT)
            .map(|v| safe(&v))
            .unwrap_or_else(|e| format!("? ({})", safe(&format!("{e:#}"))));
        println!("server   : {} · {version}", server.display());
    }
    println!("spawn    : {spawned} ms");
    println!("ping     : {}", if ping_ok { "ok" } else { "FAILED" });
    match hello {
        Some(h) => {
            println!("protocol : v{} ({} {})", h.protocol, safe(&h.runner), safe(&h.version));
            println!("{}", serde_json::to_string_pretty(&h.capabilities)?);
            if h.protocol > estia_engine::proto::PROTOCOL_VERSION {
                eprintln!(
                    "warning: runner speaks protocol v{} but this engine knows v{}",
                    h.protocol,
                    estia_engine::proto::PROTOCOL_VERSION
                );
            }
        }
        None => println!("protocol : v1 (no hello; capabilities unknown, treated as none)"),
    }
    Ok(())
}

// One parameter per `estia chat` flag.
#[allow(clippy::too_many_arguments)]
fn chat(
    ctx: &Ctx,
    model: &str,
    system: Option<&str>,
    cache_key: Option<&str>,
    tools: Option<PathBuf>,
    max_tokens: u32,
    temperature: f32,
    two_turns: bool,
    images: &[PathBuf],
) -> Result<()> {
    let engine = ctx.engine()?;
    let artifact = ctx.resolve_generation(&engine, model)?;
    if !images.is_empty() && !artifact.has(estia_engine::models::Capability::Vision) {
        return Err(anyhow!("{} does not read images; try --model vision", artifact.id));
    }
    let images: Vec<estia_engine::proto::ImageData> = images
        .iter()
        .map(|p| {
            let bytes = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
            estia_server::openai::image_from_bytes(&bytes).map_err(|e| anyhow!("{}: {e}", p.display()))
        })
        .collect::<Result<_>>()?;
    let input = read_stdin()?;
    let mut messages: Vec<Message> = if input.trim_start().starts_with('[') {
        serde_json::from_str(&input).context("stdin is not a JSON array of messages")?
    } else {
        let mut m = Vec::new();
        if let Some(sys) = system {
            m.push(Message::new("system", sys));
        }
        m.push(Message::new("user", input).with_images(images.clone()));
        m
    };
    if !images.is_empty() && !Message::any_images(&messages) {
        // A JSON conversation on stdin: the images go with its last user turn.
        if let Some(last) = messages.iter_mut().rev().find(|m| m.role == "user") {
            last.images = Some(images);
        }
    }
    let tools: Option<Vec<serde_json::Value>> = match tools {
        Some(path) => Some(
            serde_json::from_str(&std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?)
                .context("tools file is not JSON")?,
        ),
        None => None,
    };
    let session = engine.spawn_gen_session(artifact.id)?;
    if !session.capabilities().chat {
        return Err(anyhow!("runner has no chat capability (protocol v1)"));
    }
    if let Some(d) = session.load()? {
        eprintln!("loaded {} in {} ms", artifact.id, d.as_millis());
    }
    let turns = if two_turns { 2 } else { 1 };
    let mut out = std::io::stdout();
    for turn in 0..turns {
        let t0 = Instant::now();
        let mut first: Option<u128> = None;
        let outcome = session.chat_stream_with(
            &messages,
            tools.as_deref(),
            cache_key,
            None,
            Some(max_tokens),
            Some(temperature),
            Priority::Interactive,
            None,
            |tok| {
                if first.is_none() {
                    first = Some(t0.elapsed().as_millis());
                }
                let _ = out.write_all(tok.as_bytes());
                let _ = out.flush();
            },
        )?;
        println!();
        let m = &outcome.meta;
        eprintln!(
            "turn {}: first token {} ms · total {} ms · prompt {} tokens ({} cached) · generated {}{} · template {}",
            turn + 1,
            first.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            t0.elapsed().as_millis(),
            m.prompt_tokens.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            m.cached_tokens.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            m.generation_tokens.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            m.generation_tps.map(|t| format!(" (≈{t:.1} tokens/s)")).unwrap_or_default(),
            m.template.as_deref().unwrap_or("?")
        );
        // A runner that parses tool calls itself (llama.cpp) reports them here.
        for c in m.tool_calls.iter().flatten() {
            eprintln!("tool call: {}", safe(&c.to_string()));
        }
        if turn + 1 < turns {
            messages.push(Message::new("assistant", outcome.text.clone()));
            messages.push(Message::new("user", "Now say the same thing in exactly five words."));
        }
    }
    Ok(())
}

fn tokens(ctx: &Ctx, model: &str) -> Result<()> {
    let engine = ctx.engine()?;
    let artifact = ctx.resolve_generation(&engine, model)?;
    let text = read_stdin()?;
    let session = engine.spawn_gen_session(artifact.id)?;
    match session.count_tokens(&text)? {
        Some(n) => println!("{n}"),
        None => return Err(anyhow!("runner has no count_tokens capability (protocol v1)")),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn serve(
    ctx: &Ctx,
    port: u16,
    bind: Option<&str>,
    lan: bool,
    no_advertise: bool,
    name: Option<String>,
    no_auth: bool,
    idle_unload_minutes: Option<u64>,
    memory_budget: Option<MemoryBudget>,
    allow_host: Vec<String>,
    max_body_bytes: Option<u64>,
) -> Result<()> {
    use estia_server::{tokens::TokenStore, AppState, ServeOptions};
    // An explicit --bind always wins; only the default follows --lan.
    let bind = bind.unwrap_or(if lan { "0.0.0.0" } else { "127.0.0.1" });
    let addr = parse_bind(bind, port)?;
    let engine = std::sync::Arc::new(ctx.engine()?);
    let tokens = TokenStore::open(ctx.data_dir.join("tokens.json"))?;
    if !no_auth && tokens.is_empty() {
        let t = tokens.mint("local", &[estia_server::tokens::SCOPE_ADMIN])?;
        // Printed for a person at a terminal only: a service's stderr is a
        // log file, and tokens never go into logs.
        if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            eprintln!(
                "minted the first token (name `local`, scope admin). Shown once — keep it:\n\n  {t}\n\n  Authorization: Bearer {t}\n"
            );
        } else {
            tracing::warn!(
                name = "local",
                "minted the first admin token; it is not printed because stderr is not a terminal. \
                 Get a usable one with `estia token new local --replace`"
            );
        }
    }
    let state = std::sync::Arc::new(AppState::new(engine, tokens, !no_auth, addr));
    if no_auth {
        tracing::warn!("--no-auth: every process on this machine can use this engine without a token");
    }
    if lan {
        tracing::info!(
            "LAN mode: devices pair with `estia pair request --engine http://<this machine>:{port}`; approve with `estia pair approve <id>`"
        );
    }
    if let Some(MemoryBudget(b)) = memory_budget {
        state.set_memory_policy(state.memory_policy().with_budget(b));
    }
    // Some(ZERO) is "never"; None keeps the memory policy's window.
    let idle_unload = idle_unload_minutes.map(|m| std::time::Duration::from_secs(m * 60));
    estia_server::serve(
        state,
        ctx.data_dir.clone(),
        ServeOptions {
            lan,
            advertise: lan && !no_advertise,
            name,
            idle_unload,
            allowed_hosts: allow_host,
            max_body_bytes: max_body_bytes.map(|n| usize::try_from(n).unwrap_or(usize::MAX)),
        },
    )
    .await
}

/// `--bind` as an address: an IP literal (IPv6 with or without brackets),
/// `localhost`, or a host name that resolves (IPv4 preferred).
fn parse_bind(bind: &str, port: u16) -> Result<std::net::SocketAddr> {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr, ToSocketAddrs};
    let bare = bind.strip_prefix('[').and_then(|b| b.strip_suffix(']')).unwrap_or(bind);
    if let Ok(ip) = bare.parse::<IpAddr>() {
        return Ok(SocketAddr::new(ip, port));
    }
    if bind.eq_ignore_ascii_case("localhost") {
        return Ok(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port));
    }
    let addrs: Vec<SocketAddr> = (bind, port)
        .to_socket_addrs()
        .with_context(|| format!("--bind `{bind}` is neither an IP address nor a host name that resolves"))?
        .collect();
    addrs.iter().find(|a| a.is_ipv4()).or(addrs.first()).copied().ok_or_else(|| anyhow!("--bind `{bind}` resolves to no address"))
}

/// `host:port`, with an IPv6 literal in brackets.
fn host_port(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        // A zone (`fe80::1%en0`) is written `%25` inside a URL.
        format!("[{}]:{port}", host.replace('%', "%25"))
    } else {
        format!("{host}:{port}")
    }
}

fn http_url(host: &str, port: u16) -> String {
    format!("http://{}", host_port(host, port))
}

/// The URL that reaches, from this machine, a daemon bound to `bind`:
/// the wildcard addresses are reached through loopback.
fn local_url(bind: &str, port: u16) -> String {
    match bind.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) if ip.is_unspecified() => http_url("127.0.0.1", port),
        Ok(std::net::IpAddr::V6(ip)) if ip.is_unspecified() => http_url("::1", port),
        _ => http_url(bind, port),
    }
}

/// Progress of a runtime install on one line of stderr.
fn print_setup_progress(p: &estia_engine::runtime::SetupProgress) {
    let pct = match (p.bytes_done, p.bytes_total) {
        (Some(d), Some(t)) if t > 0 => format!("{:>3}%", d.min(t) * 100 / t),
        _ => "    ".to_string(),
    };
    eprint!("\r  {:<24} {pct} {:<60}", p.phase, safe(&p.message.chars().take(60).collect::<String>()));
}

/// Make the backend's runtime ready: the Python + MLX packages, or the pinned
/// `llama-server` (unless `ESTIA_LLAMA_SERVER` names one). Prints one line.
async fn ensure_runtime(ctx: &Ctx, variant: Option<&str>) -> Result<()> {
    match ctx.backend {
        Backend::MlxPython => {
            if ctx.runtime.is_installed() {
                let st = ctx.runtime.status();
                println!(
                    "✓ runtime  : installed (python {}, mlx-lm {})",
                    st.python_version.unwrap_or_else(|| "?".into()),
                    st.mlx_lm_version.unwrap_or_else(|| "?".into())
                );
            } else {
                println!("… runtime  : installing Python + MLX (~700 MB, several minutes)");
                ctx.runtime.preflight(estia_engine::runtime::RUNTIME_APPROX_BYTES)?;
                let summary = ctx.runtime.install(|p| print_setup_progress(&p)).await?;
                eprintln!();
                println!("✓ runtime  : python {} / mlx-lm {}", summary.python_version, summary.mlx_lm_version);
            }
        }
        Backend::LlamaCpp => {
            if let LlamaServer::Path(server) = LlamaServer::from_env() {
                let info = tokio::task::block_in_place(|| {
                    estia_engine::runtime::inspect_server(&server, estia_engine::runtime::llama::FIRST_RUN_TIMEOUT)
                })
                .with_context(|| format!("ESTIA_LLAMA_SERVER={}", server.display()))?;
                println!(
                    "✓ runtime  : your llama-server {} ({}){}",
                    server.display(),
                    safe(&info.version),
                    if info.pinned { "" } else { " — not the build Estia is tested with" }
                );
            } else if ctx.llama_runtime.is_installed() && variant.is_none() {
                let st = ctx.llama_runtime.status();
                println!("✓ runtime  : llama.cpp {} ({})", st.build, st.variant.unwrap_or_default());
            } else {
                let probe = LlamaRuntime::probe();
                let size = probe.download_bytes.map(|b| format!(", {} MB", b / 1_000_000)).unwrap_or_default();
                println!("… runtime  : installing llama.cpp {}{size}", variant.or(probe.variant.as_deref()).unwrap_or("?"));
                let summary = ctx.llama_runtime.install(variant, |p| print_setup_progress(&p)).await?;
                eprintln!();
                println!("✓ runtime  : llama.cpp {} ({}) · {}", summary.build, summary.variant, safe(&summary.reason));
            }
        }
    }
    Ok(())
}

async fn setup(ctx: &Ctx, roles: &[String], no_models: bool, variant: Option<&str>) -> Result<()> {
    use estia_server::tokens::{TokenStore, SCOPE_ADMIN};
    println!("estia setup — data dir {}", ctx.data_dir.display());
    if !ctx.backend.supported_here() {
        return Err(anyhow!("the {} backend cannot run on this machine; use --backend llama-cpp", ctx.backend));
    }
    if variant.is_some() && ctx.backend != Backend::LlamaCpp {
        return Err(anyhow!("--variant is for the llama-cpp backend"));
    }
    std::fs::create_dir_all(&ctx.data_dir)?;
    println!("✓ backend  : {} ({})", ctx.backend, ctx.backend_source);

    // 1. Runtime.
    ensure_runtime(ctx, variant).await?;

    // 2. Models for the requested roles, in this backend's format.
    if !no_models {
        let engine = ctx.resolver();
        let mut wanted: Vec<&'static Artifact> = Vec::new();
        for role in roles {
            let found = if role == estia_engine::roles::EMBED {
                engine.resolve_embedding(None).map(|(_, a)| a)
            } else {
                engine.resolve_generation(role)
            };
            match found {
                Ok(a) => wanted.push(a),
                Err(e) => println!("! role `{}`: {} — skipped", safe(role), safe(&e.to_string())),
            }
        }
        wanted.dedup_by(|a, b| a.id == b.id);
        for a in wanted {
            if ctx.store.is_installed(a.id) {
                // An import's own size: a linked one takes no space here.
                let (bytes, how) = if a.repo_id.is_empty() {
                    (a.required_disk_bytes, ", imported")
                } else {
                    (ctx.store.bytes_on_disk(a.id).unwrap_or(0), "")
                };
                println!("✓ model    : {} ({}{how})", a.id, gb(bytes));
            } else if a.repo_id.is_empty() {
                println!("! model    : {} is imported but its file is missing (moved?) — `estia import` it again", safe(a.id));
            } else {
                let spec = DownloadSpec::from(a);
                println!("… model    : pulling {} ({} required)", spec.id, gb(spec.required_disk_bytes));
                println!("  licence  : {}", license_text(a));
                ctx.store.download(&spec, print_progress).await?;
                println!("✓ model    : {}", spec.id);
            }
        }
    }

    // 3. config.json (roles and the backend) and the first token.
    let had_config = ctx.data_dir.join("config.json").exists();
    // Pin the backend only when it was chosen (flag, env or an existing pin);
    // a machine default stays a default, so re-running setup changes nothing
    // and a later default change still applies.
    let pin = if ctx.backend_source == BACKEND_SOURCE_DEFAULT { None } else { Some(ctx.backend) };
    ctx.save_config(pin)?;
    println!("✓ config   : {} config.json (backend {})", if had_config { "updated" } else { "wrote" }, ctx.backend);
    let tokens = TokenStore::open(ctx.data_dir.join("tokens.json"))?;
    if tokens.is_empty() {
        let t = tokens.mint("local", &[SCOPE_ADMIN])?;
        println!("✓ token    : admin token `local` minted — shown once, keep it:\n\n    {t}\n");
    } else {
        println!("✓ token    : {} token(s) in tokens.json (estia token new <name> for another)", tokens.list().len());
    }
    if ctx.backend == Backend::MlxPython && ctx.runner().is_none() {
        println!("! runner   : estia-runner.py not found — pass --runner or set ESTIA_RUNNER");
    }
    println!(
        "\nnext:
  estia serve                    run it now, for this machine only (loopback)
  estia service install --local  or run it at login, loopback only
  estia status · estia dashboard

  to let other devices on your network use it (plain HTTP; each device pairs for a token):
  estia serve --lan              or: estia service install
  from the other device: estia discover · estia pair request --engine http://<this machine>:27200"
    );
    Ok(())
}

// ── Service (launchd / systemd --user) ────────────────────────────────────────

const LAUNCHD_LABEL: &str = "com.modelcaddy.estia";

fn launchd_plist_path() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join("Library/LaunchAgents").join(format!("{LAUNCHD_LABEL}.plist"))
}

fn systemd_unit_path() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".config/systemd/user/estia.service")
}

fn uid() -> String {
    std::process::Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "501".into())
}

fn sh(cmd: &str, args: &[&str]) -> Result<String> {
    let o = std::process::Command::new(cmd).args(args).output().with_context(|| format!("run {cmd}"))?;
    let out = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    if !o.status.success() {
        return Err(anyhow!("{cmd} {} failed: {}", args.join(" "), out.trim()));
    }
    Ok(out)
}

fn service_status_line() -> String {
    if cfg!(target_os = "macos") {
        if !launchd_plist_path().exists() {
            return "not installed  (estia service install)".into();
        }
        match std::process::Command::new("launchctl").args(["print", &format!("gui/{}/{LAUNCHD_LABEL}", uid())]).output() {
            Ok(o) if o.status.success() => {
                let text = String::from_utf8_lossy(&o.stdout);
                let state = text.lines().find_map(|l| l.trim().strip_prefix("state = ")).unwrap_or("?");
                let pid = text.lines().find_map(|l| l.trim().strip_prefix("pid = ")).unwrap_or("-");
                format!("installed · {state} · pid {pid}  ({})", launchd_plist_path().display())
            }
            _ => format!("installed but not loaded  ({})", launchd_plist_path().display()),
        }
    } else if systemd_unit_path().exists() {
        let active = std::process::Command::new("systemctl")
            .args(["--user", "is-active", "estia"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "?".into());
        format!("installed · {active}  ({})", systemd_unit_path().display())
    } else {
        "not installed  (estia service install)".into()
    }
}

/// Copy the running binary and its runner scripts to `<data_dir>/engine/`, so
/// the service never points into a build tree or a checkout that can move or
/// vanish (a deleted worktree left a running service with no binary on disk).
/// Same layout as a release: `estia` with `runners/` beside it. Each file goes
/// through a temp name and a rename, so a running copy is never overwritten in
/// place.
fn stage_install(data_dir: &Path, runner: &Path) -> Result<(PathBuf, PathBuf)> {
    let root = data_dir.join("engine");
    let exe_src = std::env::current_exe()?.canonicalize()?;
    let exe_dst = root.join(if cfg!(windows) { "estia.exe" } else { "estia" });
    let runner_dir_src = runner.parent().ok_or_else(|| anyhow!("runner has no directory"))?;
    let runner_dir_name = runner_dir_src.file_name().ok_or_else(|| anyhow!("runner directory has no name"))?;
    let runner_dir_dst = root.join("runners").join(runner_dir_name);
    std::fs::create_dir_all(&runner_dir_dst)?;
    let copy = |from: &Path, to: &Path| -> Result<()> {
        if from == to {
            return Ok(());
        }
        let tmp = to.with_extension("staging");
        std::fs::copy(from, &tmp).with_context(|| format!("copy {} → {}", from.display(), tmp.display()))?;
        std::fs::rename(&tmp, to)?;
        Ok(())
    };
    copy(&exe_src, &exe_dst)?;
    for entry in std::fs::read_dir(runner_dir_src)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("py") {
            copy(&path, &runner_dir_dst.join(path.file_name().unwrap()))?;
        }
    }
    let runner_dst = runner_dir_dst.join(runner.file_name().ok_or_else(|| anyhow!("runner has no file name"))?);
    Ok((exe_dst, runner_dst))
}

fn service(ctx: &Ctx, action: ServiceAction) -> Result<()> {
    let logs = ctx.data_dir.join("logs");
    match action {
        ServiceAction::Install { port, local, allow_host, log_level, log_format, max_body_bytes, memory_budget, idle_unload_minutes } => {
            // Checked before anything is staged: every path below lives under it.
            service_path(&ctx.data_dir)?;
            let mut args = service_args(port, local, &allow_host, log_level.as_deref(), log_format, ctx.backend_flag)?;
            if let Some(n) = max_body_bytes {
                args.extend(["--max-body-bytes".to_string(), n.to_string()]);
            }
            if let Some(b) = memory_budget {
                args.extend(["--memory-budget".to_string(), b.trim().to_string()]);
            }
            if let Some(m) = idle_unload_minutes {
                args.extend(["--idle-unload-minutes".to_string(), m.to_string()]);
            }
            std::fs::create_dir_all(&logs)?;
            let runner = ctx
                .runner()
                .ok_or_else(|| anyhow!("no runner script found — pass --runner or set ESTIA_RUNNER before installing the service"))?;
            let (exe, runner) = stage_install(&ctx.data_dir, &runner)?;
            let (exe_s, runner_s, data_s) = (service_path(&exe)?, service_path(&runner)?, service_path(&ctx.data_dir)?);
            println!("staged {} and {}", exe.display(), runner.display());
            if cfg!(target_os = "macos") {
                let plist = launchd_plist_path();
                std::fs::create_dir_all(plist.parent().unwrap())?;
                let arg_xml: String = std::iter::once(exe_s.to_string())
                    .chain(args.iter().cloned())
                    .map(|a| format!("      <string>{}</string>\n", xml_escape(&a)))
                    .collect();
                let body = format!(
                    r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{arg_xml}  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>ESTIA_DATA_DIR</key><string>{data}</string>
    <key>ESTIA_RUNNER</key><string>{runner}</string>
    <key>PATH</key><string>/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin</string>
  </dict>
  <key>WorkingDirectory</key><string>{data}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>{out}</string>
  <key>StandardErrorPath</key><string>{err}</string>
</dict>
</plist>
"#,
                    data = xml_escape(data_s),
                    runner = xml_escape(runner_s),
                    out = xml_escape(service_path(&logs.join("estia.out.log"))?),
                    err = xml_escape(service_path(&logs.join("estia.err.log"))?)
                );
                std::fs::write(&plist, body)?;
                let target = format!("gui/{}", uid());
                let _ = std::process::Command::new("launchctl").args(["bootout", &format!("{target}/{LAUNCHD_LABEL}")]).output();
                // `bootout` returns before the old job is gone; bootstrapping
                // into that window fails with "5: Input/output error".
                let loaded = || {
                    std::process::Command::new("launchctl")
                        .args(["print", &format!("{target}/{LAUNCHD_LABEL}")])
                        .output()
                        .map(|o| o.status.success())
                        .unwrap_or(false)
                };
                for _ in 0..50 {
                    if !loaded() {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                if sh("launchctl", &["bootstrap", &target, plist.to_str().unwrap()]).is_err() {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    sh("launchctl", &["bootstrap", &target, plist.to_str().unwrap()])?;
                }
                println!("installed {} and started it: {}", LAUNCHD_LABEL, service_status_line());
                println!("logs: {}", logs.join("estia.err.log").display());
            } else {
                let unit = systemd_unit_path();
                std::fs::create_dir_all(unit.parent().unwrap())?;
                std::fs::write(&unit, systemd_unit(exe_s, &args, data_s, runner_s)?)?;
                sh("systemctl", &["--user", "daemon-reload"])?;
                sh("systemctl", &["--user", "enable", "--now", "estia"])?;
                println!("installed estia.service and started it");
            }
        }
        ServiceAction::Uninstall => {
            if cfg!(target_os = "macos") {
                let _ = std::process::Command::new("launchctl").args(["bootout", &format!("gui/{}/{LAUNCHD_LABEL}", uid())]).output();
                let plist = launchd_plist_path();
                if plist.exists() {
                    std::fs::remove_file(&plist)?;
                }
                println!("uninstalled {LAUNCHD_LABEL}");
            } else {
                let _ = std::process::Command::new("systemctl").args(["--user", "disable", "--now", "estia"]).output();
                let unit = systemd_unit_path();
                if unit.exists() {
                    std::fs::remove_file(&unit)?;
                }
                let _ = sh("systemctl", &["--user", "daemon-reload"]);
                println!("uninstalled estia.service");
            }
        }
        ServiceAction::Start | ServiceAction::Restart => {
            if cfg!(target_os = "macos") {
                let target = format!("gui/{}", uid());
                let loaded = std::process::Command::new("launchctl")
                    .args(["print", &format!("{target}/{LAUNCHD_LABEL}")])
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false);
                if loaded {
                    sh("launchctl", &["kickstart", "-k", &format!("{target}/{LAUNCHD_LABEL}")])?;
                } else {
                    // `stop` unloads the job (KeepAlive would revive it otherwise).
                    let plist = launchd_plist_path();
                    if !plist.exists() {
                        return Err(anyhow!("service not installed — run `estia service install`"));
                    }
                    sh("launchctl", &["bootstrap", &target, plist.to_str().unwrap()])?;
                }
            } else {
                sh("systemctl", &["--user", "restart", "estia"])?;
            }
            println!("{}", service_status_line());
        }
        ServiceAction::Stop => {
            if cfg!(target_os = "macos") {
                // KeepAlive would restart it; bootout unloads until the next login or `start`.
                let _ = std::process::Command::new("launchctl").args(["bootout", &format!("gui/{}/{LAUNCHD_LABEL}", uid())]).output();
                println!("stopped (unloaded until login or `estia service start`, which re-bootstraps)");
            } else {
                sh("systemctl", &["--user", "stop", "estia"])?;
                println!("stopped");
            }
        }
        ServiceAction::Status => println!("{}", service_status_line()),
        ServiceAction::Logs { lines, follow } => {
            if !cfg!(target_os = "macos") {
                // systemd sends the service's stderr to the journal.
                let mut cmd = std::process::Command::new("journalctl");
                cmd.args(["--user", "-u", "estia", "--no-pager", "-n", &lines.to_string()]);
                if follow {
                    cmd.arg("-f");
                }
                return run_in_place(cmd);
            }
            // Log lines go to stderr (estia.err.log); stdout is kept for anything else.
            let files: Vec<PathBuf> = ["estia.err.log", "estia.out.log"].iter().map(|n| logs.join(n)).filter(|p| p.exists()).collect();
            if files.is_empty() {
                return Err(anyhow!("no service logs in {} (is the service installed? `estia service install`)", logs.display()));
            }
            if follow {
                // `tail -F` follows by name, so it keeps going if a file is
                // truncated or replaced; Ctrl-C ends it.
                let mut cmd = std::process::Command::new("tail");
                cmd.args(["-n", &lines.to_string(), "-F"]).args(&files);
                return run_in_place(cmd);
            }
            for p in files {
                if let Ok(text) = std::fs::read_to_string(&p) {
                    println!("── {} ──", p.display());
                    let all: Vec<&str> = text.lines().collect();
                    for l in all.iter().rev().take(lines).rev() {
                        println!("{l}");
                    }
                }
            }
        }
    }
    Ok(())
}

/// Run a log viewer as this process (on Unix, `exec`), so Ctrl-C or a
/// signal to `estia` reaches it and nothing is left behind.
fn run_in_place(mut cmd: std::process::Command) -> Result<()> {
    let program = cmd.get_program().to_string_lossy().into_owned();
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let e = cmd.exec();
        Err(anyhow!("run {program}: {e}"))
    }
    #[cfg(not(unix))]
    {
        let status = cmd.status().with_context(|| format!("run {program}"))?;
        if !status.success() {
            return Err(anyhow!("{program} exited with {status}"));
        }
        Ok(())
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// A path as it goes into a service definition: UTF-8 (a lossy rendering
/// would name a different file) and free of control characters (a newline
/// starts a new directive in a unit file; XML 1.0 cannot carry most of them).
/// The `estia` arguments a service runs with. Host names and the log filter
/// go into a plist or a systemd command line, so they are checked here rather
/// than left for `serve` to trip over in a restart loop.
fn service_args(
    port: u16,
    local: bool,
    allow_host: &[String],
    log_level: Option<&str>,
    log_format: Option<LogFormat>,
    backend: Option<Backend>,
) -> Result<Vec<String>> {
    let mut args = vec!["serve".to_string(), "--port".into(), port.to_string()];
    // Only an explicit choice; otherwise the service reads config.json.
    if let Some(b) = backend {
        args.push("--backend".into());
        args.push(b.id().into());
    }
    if !local {
        args.push("--lan".into());
    }
    for name in allow_host.iter().map(|n| n.trim()).filter(|n| !n.is_empty()) {
        if name.chars().any(|c| c.is_control() || c.is_whitespace() || c == '"' || c == '\\' || c == '%' || c == '$') {
            return Err(anyhow!("`{}` is not a host name", safe(name)));
        }
        args.push("--allow-host".into());
        args.push(name.to_string());
    }
    if let Some(filter) = log_level.map(str::trim).filter(|f| !f.is_empty()) {
        if filter.chars().any(|c| c.is_control() || c.is_whitespace() || c == '\\' || c == '%' || c == '$') {
            return Err(anyhow!("`{}` is not a log filter", safe(filter)));
        }
        log_filter(SERVE_LOG_DEFAULT, None, Some(filter)).map_err(|e| anyhow!("--log-level: {e}"))?;
        args.push("--log-level".into());
        args.push(filter.to_string());
    }
    if let Some(format) = log_format {
        args.push("--log-format".into());
        args.push(format.as_str().into());
    }
    Ok(args)
}

fn service_path(p: &Path) -> Result<&str> {
    let s = p.to_str().ok_or_else(|| anyhow!("{} is not valid UTF-8; a service definition cannot name it", p.display()))?;
    if s.chars().any(char::is_control) {
        return Err(anyhow!("{} contains a control character; a service definition cannot name it", safe(s)));
    }
    Ok(s)
}

/// One double-quoted word for a systemd unit (systemd.syntax(7), "Quoting"):
/// `\` and `"` are backslash-escaped and `%`, which starts a specifier, is
/// doubled. On a command line (`exec`), `$` starts a variable and is doubled
/// too; in `Environment=` it has no special meaning.
fn systemd_quote(s: &str, exec: bool) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '%' => out.push_str("%%"),
            '$' if exec => out.push_str("$$"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The `systemd --user` unit. Callers pass paths through [`service_path`].
fn systemd_unit(exe: &str, args: &[String], data_dir: &str, runner: &str) -> Result<String> {
    // WorkingDirectory= is not unquoted: the rest of the line, trimmed, is the
    // path, with only specifiers expanded. A trailing backslash would continue
    // the line; surrounding whitespace would be trimmed away.
    if data_dir.ends_with('\\') || data_dir.trim() != data_dir {
        return Err(anyhow!("data dir `{}` ends in a backslash or whitespace; a systemd unit cannot name it", safe(data_dir)));
    }
    let exec: Vec<String> = std::iter::once(exe).chain(args.iter().map(String::as_str)).map(|w| systemd_quote(w, true)).collect();
    Ok(format!(
        "[Unit]\nDescription=estia\nAfter=network.target\n\n[Service]\nExecStart={}\nEnvironment={}\nEnvironment={}\nRestart=always\nRestartSec=3\nWorkingDirectory={}\n\n[Install]\nWantedBy=default.target\n",
        exec.join(" "),
        systemd_quote(&format!("ESTIA_DATA_DIR={data_dir}"), false),
        systemd_quote(&format!("ESTIA_RUNNER={runner}"), false),
        data_dir.replace('%', "%%")
    ))
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

fn dashboard(ctx: &Ctx, engine: Option<String>, token: Option<String>, interval: u64, once: bool) -> Result<()> {
    let base = match engine {
        Some(e) => e.trim_end_matches('/').to_string(),
        None => match estia_server::another_engine_running(&ctx.data_dir) {
            Some(rec) => local_url(&rec.bind, rec.port),
            None => {
                return Err(anyhow!("no running engine for {} — start one with `estia serve` or pass --engine", ctx.data_dir.display()))
            }
        },
    };
    let client = reqwest_blocking()?;
    let get = |path: &str, auth: bool| -> Option<serde_json::Value> {
        let mut r = client.get(format!("{base}{path}"));
        if auth {
            if let Some(t) = &token {
                r = r.bearer_auth(t);
            }
        }
        r.send().ok().filter(|r| r.status().is_success()).and_then(|r| r.json().ok())
    };
    loop {
        let health = get("/engine/health", false);
        let stats = get("/engine/stats", true);
        let models = get("/engine/models", true);
        let pairings = get("/engine/pairings", true);
        if !once {
            print!("\x1b[2J\x1b[H");
        }
        println!("estia dashboard · {} · {}", safe(&base), clock_now());
        match &health {
            Some(h) => println!(
                "engine   : v{} api v{} · backend {} · up {}s · bind {} · auth {}",
                jtext(&h["version"]),
                jtext(&h["api_version"]),
                jtext(&h["backend"]),
                jtext(&h["uptime_s"]),
                jtext(&h["bind"]),
                if h["auth_required"].as_bool().unwrap_or(true) { "required" } else { "OFF" }
            ),
            None => println!("engine   : unreachable"),
        }
        match &stats {
            Some(s) => println!(
                "loaded   : {} · queue interactive {} / background {} · jobs {}",
                Some(jlist(&s["loaded"]).join(", ")).filter(|l| !l.is_empty()).unwrap_or_else(|| "none".into()),
                jtext(&s["queue"]["interactive"]),
                jtext(&s["queue"]["background"]),
                jtext(&s["jobs"])
            ),
            None => println!("loaded   : (stats need a token: --token)"),
        }
        if let Some(m) = &models {
            println!("models   :");
            for a in m["generation"].as_array().into_iter().flatten().chain(m["embedding"].as_array().into_iter().flatten()) {
                let state = if a["installed"].as_bool().unwrap_or(false) {
                    "installed"
                } else if a["pulling"].is_string() {
                    "pulling…"
                } else {
                    "missing"
                };
                println!("  {:<36} {:<10} {}", jtext(&a["id"]), state, a["bytes_on_disk"].as_u64().map(gb).unwrap_or_default());
            }
        }
        match &pairings {
            Some(p) => {
                let list = p["pairings"].as_array().cloned().unwrap_or_default();
                let pending: Vec<_> = list.iter().filter(|x| x["status"] == "pending").collect();
                println!(
                    "pairings : {} pending{}",
                    pending.len(),
                    if pending.is_empty() { "".to_string() } else { "  — approve with: estia pair approve <id>".to_string() }
                );
                for x in pending {
                    let scopes = jlist(&x["scopes"]);
                    println!("  {:<18} {:<28} from {:<16} {}", jtext(&x["id"]), scopes.join(","), jtext(&x["from"]), jtext(&x["name"]));
                    if let Some(w) = scope_warning(&scopes) {
                        println!("  {:<18} ! {w}", "");
                    }
                }
            }
            None => println!("pairings : (admin token needed)"),
        }
        if once {
            break;
        }
        println!("\n(refresh {interval}s · Ctrl-C to quit)");
        std::thread::sleep(std::time::Duration::from_secs(interval.max(1)));
    }
    Ok(())
}

/// Wall-clock time for the dashboard header: `HH:MM:SS` in local time with
/// the zone's abbreviation, or `HH:MM:SS UTC` where local time is unavailable.
fn clock_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        let t = secs as libc::time_t;
        // SAFETY: `localtime_r` writes only into `tm`, which outlives the
        // call; `tm_zone`, when set, points at static zone data.
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        if !unsafe { libc::localtime_r(&t, &mut tm) }.is_null() {
            let zone = if tm.tm_zone.is_null() {
                String::new()
            } else {
                format!(" {}", unsafe { std::ffi::CStr::from_ptr(tm.tm_zone) }.to_string_lossy())
            };
            return format!("{:02}:{:02}:{:02}{zone}", tm.tm_hour, tm.tm_min, tm.tm_sec);
        }
    }
    utc_clock(secs)
}

fn utc_clock(unix_secs: u64) -> String {
    let s = unix_secs % 86_400;
    format!("{:02}:{:02}:{:02} UTC", s / 3600, (s / 60) % 60, s % 60)
}

/// One pairing as `pair list` prints it, every field already escaped.
struct PairRow {
    id: String,
    status: String,
    pending: bool,
    scopes: Vec<String>,
    from: String,
    name: String,
}

impl PairRow {
    fn local(p: &estia_server::pairing::Pairing) -> Self {
        PairRow {
            id: safe(&p.id),
            status: format!(
                "{:?}{}",
                p.status,
                if p.revoked {
                    " (token revoked)"
                } else if p.claimed {
                    " (collected)"
                } else {
                    ""
                }
            ),
            pending: p.status == estia_server::pairing::PairingStatus::Pending,
            scopes: p.scopes.iter().map(|s| safe(s)).collect(),
            from: safe(p.from.as_deref().unwrap_or("?")),
            name: safe(&p.name),
        }
    }

    fn remote(p: &serde_json::Value) -> Self {
        PairRow {
            id: jtext(&p["id"]),
            status: format!(
                "{}{}",
                jtext(&p["status"]),
                if p["revoked"].as_bool() == Some(true) {
                    " (token revoked)"
                } else if p["claimed"].as_bool() == Some(true) {
                    " (collected)"
                } else {
                    ""
                }
            ),
            pending: p["status"] == "pending",
            scopes: jlist(&p["scopes"]),
            from: jtext(&p["from"]),
            name: jtext(&p["name"]),
        }
    }
}

/// `pair list` output: the trusted columns first and the name — chosen by
/// whoever sent the request — last, so nothing the engine vouches for is
/// printed after it. Pending requests for more than inference get a warning.
fn pairing_lines(rows: &[PairRow]) -> Vec<String> {
    if rows.is_empty() {
        return vec!["no pairing requests".into()];
    }
    let mut out = vec![format!("{:<18} {:<22} {:<28} {:<16} NAME", "ID", "STATUS", "SCOPES", "FROM")];
    for r in rows {
        out.push(format!("{:<18} {:<22} {:<28} {:<16} {}", r.id, r.status, r.scopes.join(","), r.from, r.name));
        if r.pending {
            if let Some(w) = scope_warning(&r.scopes) {
                out.push(format!("{:<18} ! {w}", ""));
            }
        }
    }
    out
}

/// What a pending request's scopes allow beyond inference, if anything.
fn scope_warning(scopes: &[String]) -> Option<&'static str> {
    use estia_server::tokens::{SCOPE_ADMIN, SCOPE_MODELS_WRITE};
    if scopes.iter().any(|s| s == SCOPE_ADMIN) {
        Some("asks for admin: full control of this engine (models, roles, runtime, tokens, pairings); approving needs --allow-admin")
    } else if scopes.iter().any(|s| s == SCOPE_MODELS_WRITE) {
        Some("asks for models:write: can pull and delete models")
    } else {
        None
    }
}

/// Refuse to approve an `admin` request unless the operator said so; warn on
/// `models:write`. Arguments are already escaped.
fn check_approval(id: &str, scopes: &[String], from: &str, name: &str, allow_admin: bool) -> Result<()> {
    use estia_server::tokens::{SCOPE_ADMIN, SCOPE_MODELS_WRITE};
    if scopes.iter().any(|s| s == SCOPE_ADMIN) && !allow_admin {
        return Err(anyhow!(
            "not approved: pairing {id} asks for scopes [{}] from {from}, under the name `{name}`.\n\
             admin is full control of this engine: models, roles, runtime, tokens and further pairings.\n\
             If you started this request yourself, run the same command again with --allow-admin.",
            scopes.join(",")
        ));
    }
    if scopes.iter().any(|s| s == SCOPE_MODELS_WRITE) {
        eprintln!("warning: pairing {id} gets models:write: it can pull and delete models on this engine");
    }
    Ok(())
}

fn pair(ctx: &Ctx, action: PairAction) -> Result<()> {
    use estia_server::pairing::PairingStore;
    use estia_server::tokens::TokenStore;
    let store = PairingStore::new(&ctx.data_dir);
    let remote = |engine: &Option<String>, token: &Option<String>| -> Result<Option<estia_engine::RemoteEngine>> {
        match engine {
            Some(e) => Ok(Some(estia_engine::RemoteEngine::new(e.clone(), token.clone())?)),
            None => Ok(None),
        }
    };
    match action {
        PairAction::List { engine, token } => {
            let rows: Vec<PairRow> = match remote(&engine, &token)? {
                Some(r) => r.pairings()?["pairings"].as_array().into_iter().flatten().map(PairRow::remote).collect(),
                None => store.list().iter().map(PairRow::local).collect(),
            };
            for line in pairing_lines(&rows) {
                println!("{line}");
            }
        }
        PairAction::Approve { id, engine, token, allow_admin } => {
            if let Some(r) = remote(&engine, &token)? {
                // Read the request first: the approve route takes no scopes, so
                // this is the only place to see what is being granted.
                let list = r.pairings()?;
                let p = list["pairings"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .find(|p| p["id"].as_str() == Some(id.as_str()))
                    .ok_or_else(|| anyhow!("no pairing `{}` on {} (expired?)", safe(&id), safe(r.base_url())))?;
                let row = PairRow::remote(p);
                // Only a pending request can be approved; the engine says why otherwise.
                if row.pending {
                    check_approval(&row.id, &row.scopes, &row.from, &row.name, allow_admin)?;
                }
                let v = r.decide_pairing(&id, true)?;
                println!(
                    "approved {} with scopes [{}] from {}, name `{}`, on {} — the client collects its token on its next poll",
                    jtext(&v["id"]),
                    jlist(&v["scopes"]).join(","),
                    row.from,
                    jtext(&v["name"]),
                    safe(r.base_url())
                );
                return Ok(());
            }
            let pending = store.list().into_iter().find(|p| p.id == id).ok_or_else(|| anyhow!("no pairing `{}` (expired?)", safe(&id)))?;
            let row = PairRow::local(&pending);
            if row.pending {
                check_approval(&row.id, &row.scopes, &row.from, &row.name, allow_admin)?;
            }
            let tokens = TokenStore::open(ctx.data_dir.join("tokens.json"))?;
            let p = PairRow::local(&store.approve(&id, &tokens)?);
            println!(
                "approved {} with scopes [{}] from {}, name `{}` — the client collects its token on its next poll",
                p.id,
                p.scopes.join(","),
                p.from,
                p.name
            );
        }
        PairAction::Deny { id, engine, token } => {
            if let Some(r) = remote(&engine, &token)? {
                let v = r.decide_pairing(&id, false)?;
                let revoked = if v["revoked"].as_bool() == Some(true) {
                    format!("; its token {} was revoked", jtext(&v["token_name"]))
                } else {
                    String::new()
                };
                println!("denied {} (name `{}`) on {}{revoked}", jtext(&v["id"]), jtext(&v["name"]), safe(r.base_url()));
                return Ok(());
            }
            let p = store.deny(&id)?;
            let revoked = match (&p.revoked, &p.token_name) {
                (true, Some(t)) => format!("; its token {} was revoked", safe(t)),
                (true, None) => "; its token was revoked".to_string(),
                _ => String::new(),
            };
            let row = PairRow::local(&p);
            println!("denied {} (name `{}`){revoked}", row.id, row.name);
        }
        PairAction::Request { engine, name, scopes, wait_seconds } => {
            let client = reqwest_blocking()?;
            let base = engine.trim_end_matches('/');
            let r: serde_json::Value = client
                .post(format!("{base}/engine/pair"))
                .json(&serde_json::json!({"name": name, "scopes": scopes}))
                .send()
                .context("reach the engine")?
                .error_for_status()
                .context("pairing request refused")?
                .json()?;
            let id = r["id"].as_str().ok_or_else(|| anyhow!("no pairing id in response"))?.to_string();
            // The id goes into a URL path below and in front of the operator.
            if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                return Err(anyhow!("the engine returned a malformed pairing id `{}`", safe(&id)));
            }
            eprintln!(
                "pairing request `{id}` sent as `{}` — waiting up to {wait_seconds}s for the operator to run: estia pair approve {id}",
                safe(&name)
            );
            let deadline = Instant::now() + std::time::Duration::from_secs(wait_seconds);
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                // A 5xx (the engine could not read or write pairings.json) or a
                // dropped connection is worth another poll: the token is only
                // handed out once its claim is saved, so nothing is lost. A 4xx
                // (unknown or expired id) is final.
                let v: Option<serde_json::Value> = match client.get(format!("{base}/engine/pair/{id}")).send() {
                    Ok(r) if r.status().is_server_error() => {
                        eprintln!("the engine answered {} while polling; retrying", r.status());
                        None
                    }
                    Ok(r) if r.status() == reqwest::StatusCode::NOT_FOUND => {
                        return Err(anyhow!(
                            "pairing request `{id}` has expired: the engine drops a request {PAIR_EXPIRY_S} s after it was made; \
                             run `estia pair request` again"
                        ));
                    }
                    Ok(r) => Some(r.error_for_status()?.json()?),
                    Err(e) if e.is_connect() || e.is_timeout() => {
                        eprintln!("could not reach the engine while polling ({}); retrying", safe(&e.to_string()));
                        None
                    }
                    Err(e) => return Err(e.into()),
                };
                let v = v.unwrap_or(serde_json::Value::Null);
                match v["status"].as_str() {
                    Some("approved") => {
                        let token = v["token"].as_str().ok_or_else(|| anyhow!("approved but the token was already collected"))?;
                        println!("{}", safe(token));
                        return Ok(());
                    }
                    Some("denied") => return Err(anyhow!("pairing denied by the operator")),
                    _ => {}
                }
                if Instant::now() >= deadline {
                    return Err(anyhow!(
                        "no decision on pairing request `{id}` within {wait_seconds} s; the engine drops a request \
                         {PAIR_EXPIRY_S} s after it was made, so run `estia pair request` again"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn reqwest_blocking() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(20)).build()?)
}

fn discover(seconds: u64) -> Result<()> {
    let found = estia_server::discover(std::time::Duration::from_secs(seconds))?;
    if found.is_empty() {
        println!("no engines advertising {} within {seconds}s", estia_server::MDNS_SERVICE_TYPE);
        return Ok(());
    }
    println!("{:<20} {:<28} {:<30} {:<8} ENGINE", "NAME", "HOST", "URL", "API");
    for d in found {
        // The list is ordered by what a client can dial, so the head is the
        // address to hand to another device; the rest are loopback and IPv6
        // spellings of the same engine and only add noise here.
        // Everything here came off the network from whoever answered.
        let url = safe(&http_url(d.addresses.first().unwrap_or(&d.host), d.port));
        println!(
            "{:<20} {:<28} {:<30} v{:<7} {}",
            safe(&d.name),
            safe(&d.host),
            url,
            safe(d.api_version.as_deref().unwrap_or("?")),
            safe(d.engine_version.as_deref().unwrap_or("?"))
        );
    }
    Ok(())
}

fn remote_check(engine: &str, token: Option<String>, model: &str) -> Result<()> {
    use estia_engine::{Priority, RemoteEmbed, RemoteEngine, RemoteGen};
    let remote = std::sync::Arc::new(RemoteEngine::new(engine, token)?);
    let health = remote.health()?;
    println!("health   : api v{} engine {} bind {}", jtext(&health["api_version"]), jtext(&health["version"]), jtext(&health["bind"]));
    let gen = RemoteGen::new(std::sync::Arc::clone(&remote), model);
    let t0 = Instant::now();
    let mut pieces = 0;
    let text = gen.generate_stream_with("Name one sea in two words.", Some(16), Some(0.0), Priority::Interactive, None, |_| pieces += 1)?;
    println!("generate : \"{}\" in {} ms ({pieces} pieces, streamed)", safe(text.trim()), t0.elapsed().as_millis());
    let spec = estia_engine::models::embed::EMBEDDING_GEMMA_300M_4BIT;
    let emb = RemoteEmbed::new(remote, spec.id, spec.fingerprint_for("mlx-python"));
    let t0 = Instant::now();
    let v = emb.embed_batch_with(&["the sea at dawn".into()], Priority::Interactive)?;
    println!("embed    : {} × {} dims in {} ms, fingerprint {}", v.len(), v[0].len(), t0.elapsed().as_millis(), safe(emb.fingerprint()));
    Ok(())
}

fn token(ctx: &Ctx, action: TokenAction) -> Result<()> {
    use estia_server::tokens::{TokenExists, TokenStore, ALL_SCOPES, SCOPE_ADMIN};
    let store = TokenStore::open(ctx.data_dir.join("tokens.json"))?;
    match action {
        TokenAction::New { name, scopes, replace } => {
            if name.trim().is_empty() || name.chars().any(is_unsafe_char) {
                return Err(anyhow!("token name `{}` is empty or holds control or invisible characters", safe(&name)));
            }
            // Rotating keeps the replaced token's scopes unless told otherwise.
            let scopes: Vec<String> = match scopes {
                Some(s) => s,
                None if replace => store.get(&name).map(|old| old.scopes).unwrap_or_else(|| vec![SCOPE_ADMIN.to_string()]),
                None => vec![SCOPE_ADMIN.to_string()],
            };
            for s in &scopes {
                if !ALL_SCOPES.contains(&s.as_str()) {
                    return Err(anyhow!("unknown scope `{}` (one of {})", safe(s), ALL_SCOPES.join(", ")));
                }
            }
            let refs: Vec<&str> = scopes.iter().map(String::as_str).collect();
            // Names are how tokens are listed and revoked, so they are unique:
            // minting over one would cut off whoever holds the old token.
            let t = if replace {
                let (t, old) = store.replace(&name, &refs)?;
                if old.is_some() {
                    eprintln!("replaced `{}` (now {}): the previous token stops working now", safe(&name), scopes.join(","));
                }
                t
            } else {
                store.mint(&name, &refs).map_err(|e| match e.downcast_ref::<TokenExists>() {
                    Some(x) => anyhow!(
                        "token `{}` already exists (scopes {}, created {}); pass --replace to rotate it (the old token stops working), or `estia token revoke` it first",
                        safe(&x.name),
                        x.existing.scopes.join(","),
                        x.existing.created_unix
                    ),
                    None => e,
                })?
            };
            println!("{t}");
        }
        TokenAction::List => {
            let list = store.list();
            if list.is_empty() {
                println!("no tokens");
                return Ok(());
            }
            // The name last: a paired device's token is named after the
            // device, which chose its own name.
            println!("{:<40} {:<12} NAME", "SCOPES", "CREATED");
            for r in list {
                println!("{:<40} {:<12} {}", safe(&r.scopes.join(",")), r.created_unix, safe(&r.name));
            }
        }
        TokenAction::Revoke { name } => {
            // A name as `token list` printed it also works: a device that
            // named itself with control characters is listed escaped, and
            // the escaped form is what the operator can type.
            let target = match store.list().into_iter().find(|r| r.name == name) {
                Some(r) => r.name,
                None => {
                    let shown: Vec<String> = store.list().into_iter().map(|r| r.name).filter(|n| safe(n) == name).collect();
                    match shown.as_slice() {
                        [one] => one.clone(),
                        [] => return Err(anyhow!("no token named `{}` (see `estia token list`); nothing revoked", safe(&name))),
                        _ => return Err(anyhow!("`{}` matches {} tokens as printed; nothing revoked", safe(&name), shown.len())),
                    }
                }
            };
            if !store.revoke(&target)? {
                return Err(anyhow!("no token named `{}`; nothing revoked", safe(&name)));
            }
            println!("revoked `{}`", safe(&target));
        }
    }
    Ok(())
}

async fn runtime(ctx: &Ctx, action: RuntimeAction) -> Result<()> {
    match (ctx.backend, action) {
        (Backend::MlxPython, RuntimeAction::Status) => {
            let s = ctx.runtime.status();
            println!("{}", serde_json::to_string_pretty(&s)?);
        }
        (Backend::MlxPython, RuntimeAction::Install { variant }) => {
            if variant.is_some() {
                return Err(anyhow!("--variant is for the llama-cpp backend (--backend llama-cpp)"));
            }
            if !Backend::MlxPython.supported_here() {
                return Err(anyhow!("the mlx-python backend needs Apple Silicon; use --backend llama-cpp"));
            }
            ctx.runtime.preflight(estia_engine::runtime::RUNTIME_APPROX_BYTES)?;
            let summary = ctx.runtime.install(|p| print_setup_progress(&p)).await?;
            eprintln!();
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        (Backend::MlxPython, RuntimeAction::Remove) => {
            println!("{}", if ctx.runtime.remove().await? { "removed" } else { "nothing to remove" });
        }
        (Backend::LlamaCpp, RuntimeAction::Status) => {
            let mut v = serde_json::to_value(ctx.llama_runtime.status())?;
            let probe = LlamaRuntime::probe();
            v["probe"] = serde_json::json!({
                "variant": probe.variant, "asset": probe.asset, "download_bytes": probe.download_bytes, "reason": probe.reason,
            });
            if let LlamaServer::Path(p) = LlamaServer::from_env() {
                v["env_server"] = serde_json::json!(p);
            }
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
        (Backend::LlamaCpp, RuntimeAction::Install { variant }) => {
            let summary = ctx.llama_runtime.install(variant.as_deref(), |p| print_setup_progress(&p)).await?;
            eprintln!();
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        (Backend::LlamaCpp, RuntimeAction::Remove) => {
            println!("{}", if ctx.llama_runtime.remove().await? { "removed" } else { "nothing to remove" });
        }
    }
    Ok(())
}

// ── Logging ───────────────────────────────────────────────────────────────────

/// How log events are written to stderr.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
enum LogFormat {
    /// One readable line per event, coloured on a terminal.
    #[default]
    Text,
    /// One JSON object per line, for log shippers.
    Json,
}

impl LogFormat {
    fn as_str(self) -> &'static str {
        match self {
            LogFormat::Text => "text",
            LogFormat::Json => "json",
        }
    }
}

/// `serve`: Estia's own events from `info`, libraries from `warn`.
const SERVE_LOG_DEFAULT: &str = "warn,estia=info";
/// Every other command prints its own output; it logs Estia's warnings and
/// the runner's stderr (Python tracebacks among them), and no library output.
const CLI_LOG_DEFAULT: &str = "estia=warn,estia_engine::runner=info";
/// The same on llama.cpp, without the runner's stderr: llama-server logs
/// several lines per request. Its failures come back as protocol errors, and
/// `ESTIA_LOG=estia_engine::runner=info` shows the log.
const CLI_LOG_DEFAULT_LLAMA: &str = "estia=warn,estia_engine::runner=warn";

/// The filter: `defaults`, then `RUST_LOG`, then `ESTIA_LOG` / `--log-level`,
/// each a comma-separated list of `tracing` filter directives; a later
/// directive for the same target replaces an earlier one. A bare level
/// (`debug`) sets every target, Estia's included, rather than only the
/// libraries. Bad directives in `ESTIA_LOG` are an error; bad ones in
/// `RUST_LOG`, which other programs read too, are skipped and returned.
fn log_filter(defaults: &str, rust_log: Option<&str>, estia_log: Option<&str>) -> Result<(EnvFilter, Vec<String>), String> {
    let targeted: Vec<&str> = defaults.split(',').filter_map(|d| d.split_once('=').map(|(t, _)| t.trim())).collect();
    let mut filter = EnvFilter::default();
    let mut skipped = Vec::new();
    for (source, spec, strict) in [("defaults", Some(defaults), true), ("RUST_LOG", rust_log, false), ("ESTIA_LOG", estia_log, true)] {
        let Some(spec) = spec else { continue };
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let expanded: Vec<String> = match part.parse::<LevelFilter>() {
                Ok(level) => std::iter::once(part.to_string()).chain(targeted.iter().map(|t| format!("{t}={level}"))).collect(),
                Err(_) => vec![part.to_string()],
            };
            for d in expanded {
                match d.parse::<Directive>() {
                    Ok(d) => filter = filter.add_directive(d),
                    Err(e) if strict => return Err(format!("`{}` is not a log filter directive ({e})", safe(part))),
                    Err(_) => skipped.push(format!("{source}: {}", safe(part))),
                }
            }
        }
    }
    Ok((filter, skipped))
}

/// Install the `tracing` subscriber: events to stderr, as text or JSON lines.
/// Records from crates that use `log` (mdns-sd) come through it too.
fn init_logging(filter: EnvFilter, format: LogFormat) {
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;
    let registry = tracing_subscriber::registry().with(filter);
    let _ = match format {
        LogFormat::Text => registry
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(std::io::stderr)
                    .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr())),
            )
            .try_init(),
        LogFormat::Json => registry
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .flatten_event(true)
                    .with_current_span(true)
                    .with_span_list(false)
                    .with_writer(std::io::stderr),
            )
            .try_init(),
    };
}

/// Logging for this invocation: `serve` takes `--log-level` / `--log-format`
/// (or `ESTIA_LOG` / `ESTIA_LOG_FORMAT`); other commands read the variables.
fn setup_logging(cmd: &Cmd, backend: Backend) -> Result<()> {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    let (defaults, level, format) = match cmd {
        Cmd::Serve { log_level, log_format, .. } => (SERVE_LOG_DEFAULT, log_level.clone(), *log_format),
        _ => {
            let format = match env("ESTIA_LOG_FORMAT").as_deref().map(str::to_ascii_lowercase).as_deref() {
                Some("json") => LogFormat::Json,
                _ => LogFormat::Text,
            };
            let defaults = if backend == Backend::LlamaCpp { CLI_LOG_DEFAULT_LLAMA } else { CLI_LOG_DEFAULT };
            (defaults, env("ESTIA_LOG"), format)
        }
    };
    let (filter, skipped) =
        log_filter(defaults, env("RUST_LOG").as_deref(), level.as_deref()).map_err(|e| anyhow!("ESTIA_LOG / --log-level: {e}"))?;
    init_logging(filter, format);
    for s in skipped {
        tracing::warn!("ignored a log filter directive it could not parse: {s}");
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    // `estia runner llama …` is the llama.cpp adapter the engine starts. It
    // runs before clap, logging and the async runtime: it owns stdin and
    // stdout, and it must run on the main thread (on Linux, llama-server's
    // parent-death signal is tied to the thread that started it).
    let argv: Vec<OsString> = std::env::args_os().collect();
    if let Some(rest) = llama_runner_args(&argv) {
        return std::process::ExitCode::from(llama_runner(rest.to_vec()));
    }
    let rt = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Error: could not start the async runtime: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    rt.block_on(cli_main())
}

/// The adapter's arguments when this process was started as
/// `estia runner llama …`.
fn llama_runner_args(argv: &[OsString]) -> Option<&[OsString]> {
    match argv {
        [_, runner, llama, rest @ ..] if runner == "runner" && llama == "llama" => Some(rest),
        _ => None,
    }
}

/// Run the llama.cpp adapter on stdin/stdout, as the standalone `estia-llama`
/// binary does. Exit codes: 0 when stdin closes, 1 when llama-server died, 2
/// on bad arguments (a signal exits with 128 + its number from inside).
fn llama_runner(args: Vec<OsString>) -> u8 {
    let usage = estia_llama::USAGE.replacen("usage: estia-llama", "usage: estia runner llama", 1);
    // Only our own flags: everything after `--` belongs to llama-server.
    for a in args.iter().take_while(|a| *a != "--") {
        if a == "-h" || a == "--help" {
            println!("{usage}");
            return 0;
        }
        if a == "-V" || a == "--version" {
            println!("{} {} (estia runner llama)", estia_llama::RUNNER, env!("CARGO_PKG_VERSION"));
            return 0;
        }
    }
    let opts = match estia_llama::parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("estia runner llama: {e:#}\n\n{usage}");
            return 2;
        }
    };
    match estia_llama::run_stdio(opts) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("estia runner llama: {e:#}");
            1
        }
    }
}

async fn cli_main() -> std::process::ExitCode {
    match run_cli().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            // As `Result` from `main` would print it, but escaped: errors
            // carry text from remote engines and pairing requests.
            eprintln!("Error: {}", safe_lines(&e.to_string()));
            let causes: Vec<String> = e.chain().skip(1).map(|c| safe_lines(&c.to_string())).collect();
            match causes.as_slice() {
                [] => {}
                [one] => eprintln!("\nCaused by:\n    {one}"),
                many => {
                    eprintln!("\nCaused by:");
                    for (i, c) in many.iter().enumerate() {
                        eprintln!("    {i}: {c}");
                    }
                }
            }
            std::process::ExitCode::FAILURE
        }
    }
}

async fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    // Reached only with global flags before `runner` (`estia --data-dir x
    // runner llama …`); this is still the main thread (`block_on`).
    if let Cmd::Runner { kind, args } = &cli.cmd {
        if kind != "llama" {
            return Err(anyhow!("unknown runner `{}` (llama)", safe(kind)));
        }
        std::process::exit(llama_runner(args.clone()) as i32);
    }
    // Reads no data directory, config or environment, so it works anywhere.
    if let Cmd::Version { json } = &cli.cmd {
        return print_version(*json);
    }
    // The backend decides how much runner output the terminal gets, so it is
    // read before logging starts; Ctx reads config.json properly below.
    let data_dir = cli.data_dir.clone().unwrap_or_else(default_data_dir);
    let configured = std::fs::read_to_string(data_dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<FileConfig>(&t).ok())
        .and_then(|c| c.backend);
    setup_logging(&cli.cmd, choose_backend(cli.backend, configured).0)?;
    let mut ctx = Ctx::new(&cli)?;
    match cli.cmd {
        Cmd::Models => models(&ctx),
        Cmd::Pull { id } => pull(&ctx, &id).await,
        Cmd::Import { file, id, kind, family, label, context_length, dims, query_prefix, doc_prefix, link, replace, mmproj } => {
            tokio::task::block_in_place(|| {
                import(&ctx, &file, id, kind, family, label, context_length, dims, query_prefix, doc_prefix, link, replace, mmproj)
            })
        }
        Cmd::Runner { .. } => unreachable!("handled above"),
        Cmd::Rm { id } => {
            let removed = ctx.store.remove(&id).await?;
            println!("{}", if removed { "removed" } else { "nothing to remove" });
            Ok(())
        }
        // Blocking HTTP inside: must leave the async context.
        Cmd::Status => tokio::task::block_in_place(|| status(&ctx)),
        Cmd::Roles { action } => roles(&mut ctx, action),
        Cmd::Recommend { apply, json } => recommend(&mut ctx, apply, json),
        Cmd::Run { model, max_tokens, temperature, schema, json, no_stream, background, cancel_after_ms } => {
            tokio::task::block_in_place(|| run(&ctx, &model, max_tokens, temperature, schema, json, no_stream, background, cancel_after_ms))
        }
        Cmd::Embed { model } => tokio::task::block_in_place(|| embed(&ctx, model.as_deref())),
        Cmd::Bench { model, embed_model, max_tokens } => {
            tokio::task::block_in_place(|| bench(&ctx, &model, embed_model.as_deref(), max_tokens))
        }
        Cmd::Runtime { action } => runtime(&ctx, action).await,
        Cmd::RunnerCheck => tokio::task::block_in_place(|| runner_check(&ctx)),
        Cmd::Chat { model, system, cache_key, tools, max_tokens, temperature, two_turns, images } => tokio::task::block_in_place(|| {
            chat(&ctx, &model, system.as_deref(), cache_key.as_deref(), tools, max_tokens, temperature, two_turns, &images)
        }),
        Cmd::Tokens { model } => tokio::task::block_in_place(|| tokens(&ctx, &model)),
        Cmd::Setup { roles, no_models, variant } => setup(&ctx, &roles, no_models, variant.as_deref()).await,
        Cmd::Service { action } => service(&ctx, action),
        Cmd::Dashboard { engine, token, interval, once } => tokio::task::block_in_place(|| dashboard(&ctx, engine, token, interval, once)),
        Cmd::Serve {
            port,
            bind,
            lan,
            no_advertise,
            name,
            no_auth,
            idle_unload_minutes,
            memory_budget,
            allow_host,
            max_body_bytes,
            log_level: _,
            log_format: _,
        } => {
            serve(
                &ctx,
                port,
                bind.as_deref(),
                lan,
                no_advertise,
                name,
                no_auth,
                idle_unload_minutes,
                memory_budget,
                allow_host,
                max_body_bytes,
            )
            .await
        }
        Cmd::Token { action } => token(&ctx, action),
        Cmd::Pair { action } => tokio::task::block_in_place(|| pair(&ctx, action)),
        Cmd::Discover { seconds } => tokio::task::block_in_place(|| discover(seconds)),
        Cmd::RemoteCheck { engine, token, model } => tokio::task::block_in_place(|| remote_check(&engine, token, &model)),
        Cmd::Version { .. } => unreachable!("handled above"),
    }
}

/// `RUNNER_VERSION` of the MLX runner compiled into this binary.
fn embedded_runner_version() -> Option<&'static str> {
    EMBEDDED_RUNNER.lines().find_map(|l| l.strip_prefix("RUNNER_VERSION = \"")?.strip_suffix('"'))
}

/// What `estia version --json` prints. `version` and `build.commit` /
/// `build.date` match the fields of the same names in `/engine/health`.
fn version_info() -> serde_json::Value {
    let default = Backend::platform_default();
    let backends: Vec<serde_json::Value> =
        Backend::ALL.iter().map(|b| serde_json::json!({"id": b.id(), "supported": b.supported_here(), "default": *b == default})).collect();
    serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "build": {
            "commit": env!("ESTIA_BUILD_COMMIT"),
            "date": env!("ESTIA_BUILD_DATE"),
            "target": env!("ESTIA_BUILD_TARGET"),
            "profile": env!("ESTIA_BUILD_PROFILE"),
            "rustc": env!("ESTIA_BUILD_RUSTC"),
        },
        "api_version": estia_server::API_VERSION,
        "protocol_version": estia_engine::proto::PROTOCOL_VERSION,
        "backends": backends,
        "features": ENGINE_FEATURES,
        "llama_cpp_build": estia_engine::runtime::llama_pins::LLAMA_BUILD,
        "mlx_runner_version": embedded_runner_version(),
    })
}

/// `estia version` without `--json`: one `key : value` line per field.
fn version_text() -> String {
    let default = Backend::platform_default();
    let backends: Vec<String> = Backend::ALL
        .iter()
        .map(|b| match (*b == default, b.supported_here()) {
            (true, _) => format!("{b} (default on this machine)"),
            (false, true) => b.to_string(),
            (false, false) => format!("{b} (not on this machine)"),
        })
        .collect();
    let rows = [
        ("version", env!("CARGO_PKG_VERSION").to_string()),
        ("commit", env!("ESTIA_BUILD_COMMIT").to_string()),
        ("date", env!("ESTIA_BUILD_DATE").to_string()),
        ("target", format!("{} ({} build)", env!("ESTIA_BUILD_TARGET"), env!("ESTIA_BUILD_PROFILE"))),
        ("rustc", env!("ESTIA_BUILD_RUSTC").to_string()),
        ("api", format!("v{} (HTTP /engine/*)", estia_server::API_VERSION)),
        ("protocol", format!("v{} (runners)", estia_engine::proto::PROTOCOL_VERSION)),
        ("backends", backends.join(", ")),
        ("features", ENGINE_FEATURES.join(", ")),
        ("llama.cpp", format!("{} (pinned for `estia runtime install --backend llama`)", estia_engine::runtime::llama_pins::LLAMA_BUILD)),
        ("runner", format!("mlx-python {} (compiled in)", embedded_runner_version().unwrap_or("unknown"))),
    ];
    rows.iter().map(|(k, v)| format!("{k:<9}: {v}\n")).collect()
}

/// `estia version`: which build this is, for bug reports and for checking a
/// release download. A closed pipe (`estia version | head -1`) is not an
/// error.
fn print_version(json: bool) -> Result<()> {
    let out = if json { format!("{}\n", serde_json::to_string_pretty(&version_info())?) } else { version_text() };
    match std::io::stdout().lock().write_all(out.as_bytes()) {
        Err(e) if e.kind() != std::io::ErrorKind::BrokenPipe => Err(e.into()),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("estia-cli-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn ctx_for(dir: &Path) -> Ctx {
        let cli = Cli::try_parse_from(["estia", "--data-dir", dir.to_str().unwrap(), "status"]).unwrap();
        Ctx::new(&cli).unwrap()
    }

    #[test]
    fn safe_escapes_terminal_controls_and_keeps_names() {
        assert_eq!(safe("ipad\x1b[8m"), "ipad\\u{1b}[8m");
        assert_eq!(safe("a\rb\nc\td"), "a\\rb\\nc\\td");
        assert_eq!(safe("\u{202e}dapi\u{2066}"), "\\u{202e}dapi\\u{2066}");
        assert_eq!(safe("x\u{9b}31m\u{200b}\u{feff}"), "x\\u{9b}31m\\u{200b}\\u{feff}");
        for ordinary in ["George\u{2019}s iPad", "Γιώργος", "it's \"mine\" \\ ok", "laptop-2.lan"] {
            assert_eq!(safe(ordinary), ordinary);
        }
        // The stealthy payload from the review: cursor moves and autowrap off.
        let payload = "ipad\x1b[17Cgenerate\x1b[21CPending  from 192.168.1.7\x1b[?7l\x1b[999C";
        let shown = safe(payload);
        assert!(!shown.chars().any(is_unsafe_char), "{shown}");
        assert_eq!(safe(&shown), shown, "escaping is idempotent");
        assert_eq!(safe_lines("line one\x1b[2K\nline two"), "line one\\u{1b}[2K\nline two");
        assert_eq!(jtext(&serde_json::json!("a\u{1b}b")), "a\\u{1b}b");
        assert_eq!(jtext(&serde_json::Value::Null), "?");
        assert_eq!(jtext(&serde_json::json!(3)), "3");
        assert_eq!(jlist(&serde_json::json!(["admin", "x\u{7}"])), vec!["admin".to_string(), "x\\u{7}".to_string()]);
    }

    #[test]
    fn pair_list_puts_scopes_before_the_name_and_flags_admin() {
        let row = PairRow::remote(&serde_json::json!({
            "id": "6de47a30928f4f15", "status": "pending", "from": "127.0.0.1",
            "name": "ipad                 generate   Pending\u{1b}[8m", "scopes": ["admin"],
        }));
        let lines = pairing_lines(&[row]);
        assert_eq!(lines.len(), 3, "{lines:#?}");
        let line = &lines[1];
        assert!(!line.contains('\x1b'), "{line}");
        let (scopes_at, name_at) = (line.find("admin").unwrap(), line.find("ipad").unwrap());
        assert!(scopes_at < name_at, "scopes must come before the untrusted name: {line}");
        assert!(line.ends_with("Pending\\u{1b}[8m"), "{line}");
        assert!(lines[2].contains("admin") && lines[2].contains("--allow-admin"), "{}", lines[2]);
        // Plain inference scopes get no warning.
        let plain = PairRow::remote(&serde_json::json!({"id": "a", "status": "pending", "scopes": ["generate"], "name": "phone"}));
        assert_eq!(pairing_lines(&[plain]).len(), 2);
    }

    #[test]
    fn approve_refuses_admin_without_the_flag() {
        use estia_server::pairing::{PairingStatus, PairingStore};
        let dir = scratch("approve");
        let ctx = ctx_for(&dir);
        let store = PairingStore::new(&dir);
        let admin = store.request("ipad", &["admin".into()], Some("10.0.0.9".into())).unwrap();
        // The server now refuses such names; records written before that
        // still hold them, and are what the CLI must print safely.
        let file = dir.join("pairings.json");
        let text = std::fs::read_to_string(&file).unwrap().replace("\"ipad\"", "\"ipad\\u001b[8m\"");
        std::fs::write(&file, text).unwrap();
        assert_eq!(store.list()[0].name, "ipad\x1b[8m");
        let err = pair(&ctx, PairAction::Approve { id: admin.id.clone(), engine: None, token: None, allow_admin: false }).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("[admin]") && msg.contains("10.0.0.9") && msg.contains("--allow-admin"), "{msg}");
        assert!(!msg.contains('\x1b'), "{msg}");
        let still = store.list().into_iter().find(|p| p.id == admin.id).unwrap();
        assert_eq!(still.status, PairingStatus::Pending, "refused means nothing was minted");
        assert!(estia_server::tokens::TokenStore::open(dir.join("tokens.json")).unwrap().is_empty());
        pair(&ctx, PairAction::Approve { id: admin.id.clone(), engine: None, token: None, allow_admin: true }).unwrap();
        assert_eq!(store.list().into_iter().find(|p| p.id == admin.id).unwrap().status, PairingStatus::Approved);
        let tokens = || estia_server::tokens::TokenStore::open(dir.join("tokens.json")).unwrap().list();
        assert_eq!(tokens().len(), 1);
        pair(&ctx, PairAction::Deny { id: admin.id.clone(), engine: None, token: None }).unwrap();
        assert!(tokens().is_empty(), "deny after approve revokes the minted token");
        // models:write is approved with a warning; plain scopes need nothing.
        assert!(check_approval("x", &["models:write".into()], "?", "n", false).is_ok());
        assert!(check_approval("x", &["generate".into(), "embed".into()], "?", "n", false).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn token_names_are_unique_and_revoke_fails_loudly() {
        use estia_server::tokens::TokenStore;
        let dir = scratch("tokens");
        let ctx = ctx_for(&dir);
        let new = |name: &str, scopes: Option<Vec<&str>>, replace: bool| {
            token(&ctx, TokenAction::New { name: name.into(), scopes: scopes.map(|v| v.into_iter().map(String::from).collect()), replace })
        };
        new("editor", Some(vec!["generate", "embed"]), false).unwrap();
        let first = TokenStore::open(dir.join("tokens.json")).unwrap().list();
        let err = new("editor", Some(vec!["embed"]), false).unwrap_err().to_string();
        assert!(err.contains("already exists") && err.contains("--replace"), "{err}");
        assert_eq!(TokenStore::open(dir.join("tokens.json")).unwrap().list()[0].sha256, first[0].sha256, "refused mint left it alone");
        // --replace rotates and keeps the scopes when none are given.
        new("editor", None, true).unwrap();
        let after = TokenStore::open(dir.join("tokens.json")).unwrap().list();
        assert_eq!(after.len(), 1);
        assert_ne!(after[0].sha256, first[0].sha256);
        assert_eq!(after[0].scopes, vec!["generate".to_string(), "embed".to_string()]);
        assert!(new("bad\x1bname", None, false).is_err());

        assert!(token(&ctx, TokenAction::Revoke { name: "nosuch".into() }).is_err());
        // A device-chosen name with escapes is revocable as `token list` shows it.
        let raw = "pair:ipad\x1b[8m:6de47a30928f4f15";
        TokenStore::open(dir.join("tokens.json")).unwrap().mint(raw, &["admin"]).unwrap();
        token(&ctx, TokenAction::Revoke { name: safe(raw) }).unwrap();
        assert!(TokenStore::open(dir.join("tokens.json")).unwrap().list().iter().all(|r| r.name != raw));
        token(&ctx, TokenAction::Revoke { name: "editor".into() }).unwrap();
        assert!(token(&ctx, TokenAction::Revoke { name: "editor".into() }).is_err(), "second revoke finds nothing");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roles_rm_of_an_unbound_role_fails() {
        let dir = scratch("roles");
        let mut ctx = ctx_for(&dir);
        assert!(roles(&mut ctx, Some(RolesAction::Rm { role: "nosuch".into() })).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn systemd_unit_quotes_paths() {
        let data = r#"/home/u/My Models/100%/$HOME/"q"\b"#;
        let unit = systemd_unit(&format!("{data}/engine/estia"), &["serve".into(), "--lan".into()], data, &format!("{data}/r.py")).unwrap();
        let line = |key: &str| unit.lines().find(|l| l.starts_with(key)).unwrap().to_string();
        assert_eq!(line("ExecStart="), r#"ExecStart="/home/u/My Models/100%%/$$HOME/\"q\"\\b/engine/estia" "serve" "--lan""#);
        let env: Vec<&str> = unit.lines().filter(|l| l.starts_with("Environment=")).collect();
        assert_eq!(env[0], r#"Environment="ESTIA_DATA_DIR=/home/u/My Models/100%%/$HOME/\"q\"\\b""#);
        assert_eq!(env[1], r#"Environment="ESTIA_RUNNER=/home/u/My Models/100%%/$HOME/\"q\"\\b/r.py""#);
        assert_eq!(line("WorkingDirectory="), r#"WorkingDirectory=/home/u/My Models/100%%/$HOME/"q"\b"#);
        assert_eq!(unit.lines().count(), 14, "no path added a line:\n{unit}");
        assert!(systemd_unit("/x/estia", &[], "/data\\", "/r.py").is_err(), "a trailing backslash would continue the line");
        assert!(service_path(Path::new("/tmp/a\nExecStartPre=/bin/sh")).is_err());
        assert!(service_path(Path::new("/tmp/My Models")).is_ok());
        assert_eq!(xml_escape(r#"/a&b/<c>/"d""#), "/a&amp;b/&lt;c&gt;/&quot;d&quot;");
    }

    /// `service install --allow-host` reaches `serve`, and a name that could
    /// break out of a plist string or a systemd word is refused up front.
    #[test]
    fn service_args_carry_allowed_hosts() {
        assert_eq!(service_args(27200, true, &[], None, None, None).unwrap(), ["serve", "--port", "27200"]);
        assert_eq!(
            service_args(1, false, &["studio.lan".into(), " *.home.arpa ".into(), "".into()], None, None, None).unwrap(),
            ["serve", "--port", "1", "--lan", "--allow-host", "studio.lan", "--allow-host", "*.home.arpa"]
        );
        for bad in ["a b", "a\nExecStartPre=/bin/sh", "x\"y", "100%", "$HOME"] {
            assert!(service_args(1, false, &[bad.into()], None, None, None).is_err(), "{bad:?} accepted");
        }
    }

    /// `service install --log-level/--log-format` reach `serve`; a filter that
    /// would not parse, or could break out of a service file, is refused.
    #[test]
    fn service_args_carry_log_settings() {
        assert_eq!(
            service_args(1, true, &[], Some(" estia_server=debug,mdns_sd=info "), Some(LogFormat::Json), None).unwrap(),
            ["serve", "--port", "1", "--log-level", "estia_server=debug,mdns_sd=info", "--log-format", "json"]
        );
        assert_eq!(service_args(1, true, &[], Some(""), None, None).unwrap(), ["serve", "--port", "1"]);
        for bad in ["estia=loud", "debug\nExecStartPre=/bin/sh", "a b", "$HOME", "100%"] {
            assert!(service_args(1, true, &[], Some(bad), None, None).is_err(), "{bad:?} accepted");
        }
    }

    /// The events a filter lets through, out of ERROR to DEBUG from a few targets.
    fn passes(f: EnvFilter) -> Vec<String> {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::layer::SubscriberExt as _;
        #[derive(Clone, Default)]
        struct Seen(Arc<Mutex<Vec<String>>>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Seen {
            fn on_event(&self, e: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
                self.0.lock().unwrap().push(format!("{} {}", e.metadata().target(), e.metadata().level()));
            }
        }
        let seen = Seen::default();
        tracing::subscriber::with_default(tracing_subscriber::registry().with(f).with(seen.clone()), || {
            macro_rules! probe {
                ($($t:literal),*) => {$(
                    tracing::event!(target: $t, tracing::Level::ERROR, "p");
                    tracing::event!(target: $t, tracing::Level::WARN, "p");
                    tracing::event!(target: $t, tracing::Level::INFO, "p");
                    tracing::event!(target: $t, tracing::Level::DEBUG, "p");
                )*};
            }
            probe!("estia_server::access", "estia_server", "estia_engine::session", "estia_engine::runner", "mdns_sd::x", "hyper::proto");
        });
        let v = seen.0.lock().unwrap().clone();
        v
    }

    #[test]
    fn log_filters_layer_and_bare_levels_cover_estia() {
        let on = |v: &[String], t: &str, l: &str| v.iter().any(|e| e == &format!("{t} {l}"));
        let f = |rust: Option<&str>, estia: Option<&str>| passes(log_filter(SERVE_LOG_DEFAULT, rust, estia).unwrap().0);

        let v = f(None, None);
        assert!(on(&v, "estia_server::access", "INFO") && on(&v, "estia_engine::session", "INFO"));
        assert!(!on(&v, "estia_server::access", "DEBUG"));
        assert!(!on(&v, "mdns_sd::x", "INFO") && on(&v, "mdns_sd::x", "WARN"), "libraries at warn");

        let v = f(None, Some("debug"));
        assert!(on(&v, "estia_server::access", "DEBUG"), "a bare level covers Estia too");
        assert!(on(&v, "hyper::proto", "DEBUG"));

        let v = f(None, Some("estia_server=debug"));
        assert!(on(&v, "estia_server::access", "DEBUG"));
        assert!(!on(&v, "estia_engine::session", "DEBUG"), "only the named crate");
        assert!(on(&v, "estia_engine::session", "INFO"), "the rest keeps its default");

        let v = f(Some("mdns_sd=debug"), Some("warn"));
        assert!(on(&v, "mdns_sd::x", "DEBUG"), "RUST_LOG still honoured");
        assert!(!on(&v, "estia_server::access", "INFO") && on(&v, "estia_server::access", "WARN"), "ESTIA_LOG comes last");

        let v = f(None, Some("estia_server::access=off"));
        assert!(!on(&v, "estia_server::access", "ERROR"));
        assert!(on(&v, "estia_server", "INFO"));

        let v = passes(log_filter(CLI_LOG_DEFAULT_LLAMA, None, None).unwrap().0);
        assert!(!on(&v, "estia_engine::runner", "INFO") && on(&v, "estia_engine::runner", "WARN"), "llama-server's chatter stays out");
        let v = passes(log_filter(CLI_LOG_DEFAULT, None, None).unwrap().0);
        assert!(on(&v, "estia_engine::runner", "INFO"), "other commands still show runner stderr");
        assert!(!on(&v, "estia_server::access", "INFO") && on(&v, "estia_server::access", "WARN"));
        assert!(!on(&v, "mdns_sd::x", "ERROR"));

        let (_, skipped) = log_filter(SERVE_LOG_DEFAULT, Some("mdns_sd=loud,info"), None).unwrap();
        assert_eq!(skipped, ["RUST_LOG: mdns_sd=loud"]);
        assert!(log_filter(SERVE_LOG_DEFAULT, None, Some("estia=loud")).is_err());
    }

    #[test]
    fn bind_accepts_localhost_and_ipv6() {
        use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
        assert_eq!(parse_bind("127.0.0.1", 1).unwrap().ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(parse_bind("localhost", 1).unwrap().ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(parse_bind("::1", 7).unwrap(), "[::1]:7".parse().unwrap());
        assert_eq!(parse_bind("[::1]", 7).unwrap().ip(), IpAddr::V6(Ipv6Addr::LOCALHOST));
        assert!(parse_bind("0.0.0.0", 1).unwrap().ip().is_unspecified());
        assert!(parse_bind("no such host.invalid", 1).is_err());
        assert_eq!(http_url("::1", 27200), "http://[::1]:27200");
        assert_eq!(http_url("192.168.1.7", 27200), "http://192.168.1.7:27200");
        assert_eq!(http_url("fe80::1%en0", 1), "http://[fe80::1%25en0]:1");
        assert_eq!(local_url("0.0.0.0", 5), "http://127.0.0.1:5");
        assert_eq!(local_url("::", 5), "http://[::1]:5");
        assert_eq!(local_url("::1", 5), "http://[::1]:5");
        assert_eq!(local_url("127.0.0.1", 5), "http://127.0.0.1:5");
    }

    #[test]
    fn dashboard_clock_is_a_time_of_day() {
        assert_eq!(utc_clock(3603), "01:00:03 UTC");
        assert_eq!(utc_clock(86_399), "23:59:59 UTC");
        let now = clock_now();
        let hms = now.split(' ').next().unwrap();
        assert_eq!(hms.len(), 8, "{now}");
        assert!(hms.chars().enumerate().all(|(i, c)| if i == 2 || i == 5 { c == ':' } else { c.is_ascii_digit() }), "{now}");
    }

    #[test]
    fn runner_is_never_taken_from_the_current_directory() {
        let dir = scratch("runner");
        let planted = dir.join("cwd/runners/mlx-python");
        std::fs::create_dir_all(&planted).unwrap();
        std::fs::write(planted.join("estia-runner.py"), "raise SystemExit('planted')\n").unwrap();
        let data = dir.join("data");
        let before = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir.join("cwd")).unwrap();
        let found = find_runner(None, &data);
        std::env::set_current_dir(before).unwrap();
        let found = found.expect("the compiled-in runner");
        assert_eq!(found, data.join("engine/runners/mlx-python/estia-runner.py"));
        assert_eq!(std::fs::read_to_string(&found).unwrap(), EMBEDDED_RUNNER);
        assert_eq!(find_runner(Some(PathBuf::from("/x/r.py")), &data), Some(PathBuf::from("/x/r.py")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn backend_comes_from_the_flag_then_config_then_the_machine() {
        let cli = Cli::try_parse_from(["estia", "--backend", "llama", "models"]).unwrap();
        assert_eq!(cli.backend, Some(Backend::LlamaCpp));
        // Global: accepted after the subcommand too.
        let cli = Cli::try_parse_from(["estia", "runtime", "install", "--backend", "llama-cpp", "--variant", "cpu"]).unwrap();
        assert_eq!(cli.backend, Some(Backend::LlamaCpp));
        assert!(matches!(cli.cmd, Cmd::Runtime { action: RuntimeAction::Install { variant: Some(ref v) } } if v == "cpu"));
        assert!(Cli::try_parse_from(["estia", "--backend", "onnx", "models"]).is_err());
        assert_eq!(choose_backend(Some(Backend::LlamaCpp), Some(Backend::MlxPython)).0, Backend::LlamaCpp);
        assert_eq!(choose_backend(None, Some(Backend::LlamaCpp)), (Backend::LlamaCpp, "config.json"));
        assert_eq!(choose_backend(None, None).0, Backend::platform_default());
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert_eq!(choose_backend(None, None).0, Backend::MlxPython, "Apple Silicon keeps MLX");
        }
    }

    #[test]
    fn pair_request_stops_waiting_before_the_engine_drops_the_request() {
        let cli = Cli::try_parse_from(["estia", "pair", "request", "--engine", "http://127.0.0.1:1"]).unwrap();
        let Cmd::Pair { action: PairAction::Request { wait_seconds, .. } } = cli.cmd else { panic!("pair request") };
        assert_eq!(wait_seconds, 290);
        assert!(wait_seconds < PAIR_EXPIRY_S);
        assert_eq!(PAIR_EXPIRY_S, 300);
    }

    #[test]
    fn setup_and_service_carry_the_backend() {
        let cli = Cli::try_parse_from(["estia", "--backend", "llama", "setup", "--roles", "fast,embed", "--variant", "cpu"]).unwrap();
        assert!(matches!(cli.cmd, Cmd::Setup { ref roles, variant: Some(ref v), .. } if roles == &["fast", "embed"] && v == "cpu"));
        let cli = Cli::try_parse_from(["estia", "service", "install", "--local", "--backend", "llama-cpp"]).unwrap();
        assert_eq!(cli.backend, Some(Backend::LlamaCpp));
        assert_eq!(
            service_args(1, true, &[], None, None, Some(Backend::LlamaCpp)).unwrap(),
            ["serve", "--port", "1", "--backend", "llama-cpp"]
        );
        let cli = Cli::try_parse_from(["estia", "serve", "--backend", "mlx", "--port", "27351"]).unwrap();
        assert_eq!(cli.backend, Some(Backend::MlxPython));
    }

    #[test]
    fn import_takes_its_options() {
        let cli = Cli::try_parse_from([
            "estia",
            "import",
            "/m/tiny.gguf",
            "--id",
            "tiny",
            "--kind",
            "embedding",
            "--family",
            "tinies",
            "--ctx",
            "4096",
            "--dims",
            "384",
            "--link",
        ])
        .unwrap();
        match cli.cmd {
            Cmd::Import { file, id, kind, family, context_length, dims, link, replace, .. } => {
                assert_eq!(file, PathBuf::from("/m/tiny.gguf"));
                assert_eq!((id.as_deref(), kind.as_deref(), family.as_deref()), (Some("tiny"), Some("embedding"), Some("tinies")));
                assert_eq!((context_length, dims, link, replace), (Some(4096), Some(384), true, false));
            }
            _ => panic!("not an import"),
        }
        assert!(Cli::try_parse_from(["estia", "import", "x.gguf", "--kind", "chat"]).is_err());
    }

    #[test]
    fn every_import_flag_has_its_own_help() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        let import = cmd.find_subcommand("import").unwrap();
        for arg in import.get_arguments().filter(|a| !a.is_global_set() && a.get_id() != "help") {
            let help = arg.get_help().map(|h| h.to_string()).unwrap_or_default();
            assert!(!help.is_empty(), "`estia import --{}` has no help text", arg.get_id());
        }
        let help = |id: &str| import.get_arguments().find(|a| a.get_id() == id).unwrap().get_help().unwrap().to_string();
        assert!(help("query_prefix").contains("query") && !help("query_prefix").contains("document"), "{}", help("query_prefix"));
        assert!(help("doc_prefix").contains("document"), "{}", help("doc_prefix"));
    }

    #[test]
    fn licences_print_with_where_to_read_them() {
        use estia_engine::models::embed::EMBEDDING_GEMMA_300M_4BIT;
        let gemma = &EMBEDDING_GEMMA_300M_4BIT.artifacts[0];
        assert_eq!(
            license_text(gemma),
            "Gemma Terms of Use · terms: https://ai.google.dev/gemma/terms \
             · prohibited use: https://ai.google.dev/gemma/prohibited_use_policy"
        );
        let e4b = find_artifact("gemma4-e4b-it-4bit-mlx").unwrap();
        assert_eq!(license_text(e4b), "Apache-2.0");
        assert!(is_spdx_id("MIT") && is_spdx_id("Apache-2.0") && is_spdx_id("GPL-2.0+"));
        assert!(!is_spdx_id("Gemma Terms of Use") && !is_spdx_id("unknown (imported)") && !is_spdx_id(""));
        // A vendor licence Estia has no pages for points at the model card.
        let other = Artifact { license: "Acme Model Licence", ..e4b.clone() };
        assert_eq!(
            license_text(&other),
            "Acme Model Licence · read the model card: https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit"
        );
        // `estia models` names each licence's pages once, under the table.
        assert_eq!(license_notes(["Apache-2.0", "Gemma Terms of Use", "MIT", "Gemma Terms of Use"]), vec![license_text(gemma)]);
        assert!(license_notes(["Apache-2.0", "MIT"]).is_empty());
    }

    /// Every built-in model's licence is an SPDX id or one whose pages
    /// `estia pull` can print; a new vendor licence needs its links added to
    /// `license_links` (and the test client's `LICENSE_LINKS`).
    #[test]
    fn every_built_in_licence_is_spdx_or_has_its_pages() {
        // Imports have no repository and the licence `unknown (imported)`.
        let embed = embed_models().into_iter().filter(|e| !e.repo_id.is_empty()).collect::<Vec<_>>();
        for e in &embed {
            assert!(is_spdx_id(e.license) || license_links(e.license).is_some(), "{}: licence `{}`", e.id, e.license);
        }
        let artifacts = generation_artifacts().into_iter().chain(embed.iter().flat_map(|e| e.artifacts.iter()));
        for a in artifacts.filter(|a| !a.repo_id.is_empty()) {
            assert!(is_spdx_id(a.license) || license_links(a.license).is_some(), "{}: licence `{}`", a.id, a.license);
        }
    }

    /// The engine starts `<estia> runner llama --server … --run-dir … [-- …]`;
    /// the adapter's arguments are everything after `runner llama`, `--`
    /// included, whichever way they reach it.
    #[test]
    fn runner_llama_arguments_reach_the_adapter_whole() {
        let argv: Vec<OsString> =
            ["estia", "runner", "llama", "--server", "/s", "--run-dir", "/r", "--", "-ngl", "0"].map(OsString::from).to_vec();
        let rest = llama_runner_args(&argv).unwrap();
        let opts = estia_llama::parse_args(rest.to_vec()).unwrap();
        assert_eq!((opts.server_bin.as_path(), opts.run_dir.as_path()), (Path::new("/s"), Path::new("/r")));
        assert_eq!(opts.extra_args, ["-ngl", "0"]);
        assert!(llama_runner_args(&["estia", "runner"].map(OsString::from)).is_none());
        assert!(llama_runner_args(&["estia", "models", "llama"].map(OsString::from)).is_none());
        // With global flags first, clap routes it to the hidden subcommand.
        let cli =
            Cli::try_parse_from(["estia", "--data-dir", "/d", "runner", "llama", "--server", "/s", "--run-dir", "/r", "--", "-ngl", "0"])
                .unwrap();
        match cli.cmd {
            Cmd::Runner { kind, args } => {
                assert_eq!(kind, "llama");
                let opts = estia_llama::parse_args(args).unwrap();
                assert_eq!(opts.extra_args, ["-ngl", "0"], "`--` must survive clap");
            }
            _ => panic!("not the runner"),
        }
        assert_eq!(llama_runner(vec!["--bogus".into()]), 2, "bad arguments exit 2");
    }

    /// `config.json` keeps keys it does not know; `backend` is written by
    /// setup and read back as the default.
    #[test]
    fn config_writes_keep_other_keys_and_record_the_backend() {
        let dir = scratch("config");
        std::fs::write(dir.join("config.json"), r#"{"host_app": {"x": 1}, "roles": {"text": {"family": "gemma4-e4b"}}}"#).unwrap();
        let ctx = ctx_for(&dir);
        ctx.save_config(Some(Backend::LlamaCpp)).unwrap();
        let saved: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        assert_eq!(saved["host_app"]["x"], 1, "{saved}");
        assert_eq!(saved["backend"], "llama-cpp");
        assert_eq!(saved["roles"]["text"]["family"], "gemma4-e4b");
        ctx.save_roles().unwrap();
        let saved: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        assert_eq!(saved["backend"], "llama-cpp", "saving roles keeps the backend");
        if std::env::var_os("ESTIA_BACKEND").is_none() {
            let ctx = ctx_for(&dir);
            assert_eq!((ctx.backend, ctx.backend_source), (Backend::LlamaCpp, "config.json"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// On llama.cpp the engine runs this binary as the adapter, names the
    /// llama-server (the installed build, or ESTIA_LLAMA_SERVER) and the run
    /// directory, and resolves roles to GGUF artifacts.
    #[test]
    fn llama_engine_runs_this_binary_as_the_adapter() {
        let dir = scratch("llama-engine");
        let installed = dir.join(format!("runtime/llama/{}-cpu", estia_engine::runtime::llama_pins::LLAMA_BUILD));
        std::fs::create_dir_all(&installed).unwrap();
        std::fs::write(installed.join(estia_engine::runtime::llama::server_file_name()), b"").unwrap();
        let cli = Cli::try_parse_from(["estia", "--data-dir", dir.to_str().unwrap(), "--backend", "llama", "status"]).unwrap();
        let ctx = Ctx::new(&cli).unwrap();
        let engine = ctx.engine().unwrap();
        assert_eq!(engine.backend(), Backend::LlamaCpp);
        let launch = engine.launch_for("gemma4-e2b-it-qat-q4_0-gguf").unwrap();
        assert_eq!(launch.program(), std::env::current_exe().unwrap());
        let args: Vec<String> = launch.args().iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(&args[..3], ["runner", "llama", "--server"]);
        let server = match LlamaServer::from_env() {
            LlamaServer::Path(p) => p,
            LlamaServer::Installed => installed.join(estia_engine::runtime::llama::server_file_name()),
        };
        assert_eq!(args[3], server.display().to_string());
        assert_eq!(&args[4..6], ["--run-dir", &dir.join("run").display().to_string()]);
        assert!(args.windows(2).any(|w| w == ["--ctx", "32768"]), "{args:?}");
        assert_eq!(ctx.resolve_generation(&engine, "fast").unwrap().id, "gemma4-e2b-it-qat-q4_0-gguf");
        assert_eq!(engine.resolve_embedding(None).unwrap().1.id, "embeddinggemma-300m-q8_0-gguf");
        // Pulls follow the backend too.
        assert_eq!(estia_server::catalog::pull_spec(&ctx.resolver(), "embed").unwrap().id, "embeddinggemma-300m-q8_0-gguf");
        assert_eq!(
            estia_server::catalog::pull_spec(&ctx.resolver(), "embeddinggemma-300m-4bit").unwrap().id,
            "embeddinggemma-300m-q8_0-gguf"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `cli/estia-runner.py` is a symlink; a checkout without symlink support
    /// would turn it into a one-line text file holding the target path.
    #[test]
    fn embedded_runner_is_the_real_script() {
        assert!(EMBEDDED_RUNNER.starts_with("#!/usr/bin/env python3"), "{}", &EMBEDDED_RUNNER[..EMBEDDED_RUNNER.len().min(80)]);
        if let Ok(canonical) = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../runners/mlx-python/estia-runner.py")) {
            assert_eq!(EMBEDDED_RUNNER, canonical);
        }
    }

    /// `estia --version` is one line: `estia <version> (<commit>, <date>)`.
    #[test]
    fn version_flag_is_one_line_with_commit_and_date() {
        use clap::CommandFactory;
        let line = Cli::command().render_version();
        let line = line.trim_end();
        assert!(!line.contains('\n'), "{line:?}");
        let rest = line.strip_prefix(concat!("estia ", env!("CARGO_PKG_VERSION"), " (")).unwrap_or_else(|| panic!("{line:?}"));
        let inner = rest.strip_suffix(')').unwrap_or_else(|| panic!("{line:?}"));
        let (commit, date) = inner.split_once(", ").unwrap_or_else(|| panic!("{line:?}"));
        assert_eq!(commit, env!("ESTIA_BUILD_COMMIT"));
        assert_eq!(date, env!("ESTIA_BUILD_DATE"));
        assert!(!commit.is_empty() && !commit.contains(char::is_whitespace), "{commit:?}");
        let b = date.as_bytes();
        assert!(date == "unknown" || (b.len() == 10 && b[4] == b'-' && b[7] == b'-'), "{date:?}");
        let cli = Cli::try_parse_from(["estia", "version", "--json"]).unwrap();
        assert!(matches!(cli.cmd, Cmd::Version { json: true }));
    }

    /// `estia version --json` is one JSON object with every field scripts
    /// read, and the build fields match what `/engine/health` reports.
    #[test]
    fn version_json_has_the_fields() {
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string_pretty(&version_info()).unwrap()).unwrap();
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(v["build"]["commit"], estia_server::BUILD_COMMIT);
        assert_eq!(v["build"]["date"], env!("ESTIA_BUILD_DATE"));
        for k in ["target", "profile", "rustc"] {
            assert!(v["build"][k].as_str().is_some_and(|s| !s.is_empty()), "build.{k}: {v}");
        }
        assert_eq!(v["build"]["target"], env!("ESTIA_BUILD_TARGET"));
        assert_eq!(v["api_version"], estia_server::API_VERSION);
        assert_eq!(v["protocol_version"], estia_engine::proto::PROTOCOL_VERSION);
        let backends = v["backends"].as_array().unwrap();
        let ids: Vec<&str> = backends.iter().map(|b| b["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["mlx-python", "llama-cpp"]);
        assert_eq!(backends.iter().filter(|b| b["default"] == true).count(), 1);
        assert!(backends.iter().all(|b| b["supported"].is_boolean()));
        assert_eq!(v["features"], serde_json::json!(ENGINE_FEATURES));
        assert_eq!(v["llama_cpp_build"], estia_engine::runtime::llama_pins::LLAMA_BUILD);
        assert!(v["mlx_runner_version"].as_str().is_some_and(|s| s.split('.').count() == 3), "{v}");
    }

    /// `estia version` lines up its keys and leads with the version and commit.
    #[test]
    fn version_text_lists_the_build() {
        let text = version_text();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], concat!("version  : ", env!("CARGO_PKG_VERSION")));
        assert_eq!(lines[1], concat!("commit   : ", env!("ESTIA_BUILD_COMMIT")));
        assert!(lines.iter().all(|l| l.as_bytes().get(9) == Some(&b':')), "{text}");
        for key in ["date", "target", "rustc", "api", "protocol", "backends", "features", "llama.cpp", "runner"] {
            assert!(lines.iter().any(|l| l.starts_with(key)), "no {key}: {text}");
        }
    }

    /// `ENGINE_FEATURES` is what cli/Cargo.toml turns on for `estia-engine`.
    #[test]
    fn version_features_match_the_manifest() {
        let manifest = include_str!("../Cargo.toml");
        let line = manifest.lines().find(|l| l.starts_with("estia-engine = ")).expect("estia-engine dependency line");
        let list = line.split("features = [").nth(1).and_then(|s| s.split(']').next()).expect("features list");
        let declared: Vec<&str> = list.split(',').map(|s| s.trim().trim_matches('"')).filter(|s| !s.is_empty()).collect();
        assert_eq!(declared, ENGINE_FEATURES);
    }
}
