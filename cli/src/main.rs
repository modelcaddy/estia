//! `estia` — the terminal front for the Estia engine: set a machine up,
//! install the service, run the HTTP daemon, pair LAN clients, list and pull
//! models, install the runtime, generate (with an optional JSON schema), chat,
//! embed, and bench.

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use estia_engine::models::embed::{find_embed_model, EmbedModel, BACKEND_MLX_PYTHON, EMBEDDING_MODELS};
use estia_engine::models::{find_artifact, find_family_default, Artifact, DownloadSpec, Format, ModelStore, GENERATION_MODELS};
use estia_engine::runtime::PythonRuntime;
use estia_engine::structured::{self, OutputFormat, Structured, StructuredError};
use estia_engine::{CancelToken, Engine, EngineConfig, Priority, RoleBinding, Roles};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing_subscriber::filter::{Directive, EnvFilter, LevelFilter};

#[derive(Parser)]
#[command(
    name = "estia",
    version,
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
    /// List known models and whether they are installed.
    Models,
    /// Download a model by id (generation or embedding).
    Pull { id: String },
    /// Remove an installed model and any partial download.
    Rm { id: String },
    /// Data dir, runtime, runner, installed models.
    Status,
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
    /// The Python runtime for the MLX backend.
    Runtime {
        #[command(subcommand)]
        action: RuntimeAction,
    },
    /// Handshake with the runner: protocol version and capabilities. Loads no model.
    RunnerCheck,
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
        #[arg(long, default_value_t = 15)]
        idle_unload_minutes: u64,
        /// Also accept requests whose Host header names this host (repeatable,
        /// or comma-separated): a reverse proxy's name, or a LAN name such as
        /// `mac.lan`. IP literals, `localhost` and `<name>.local` are always
        /// accepted.
        #[arg(long = "allow-host", value_name = "NAME", value_delimiter = ',')]
        allow_host: Vec<String>,
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
        #[arg(long)]
        engine: Option<String>,
        #[arg(long)]
        token: Option<String>,
    },
    /// Approve a request: mints a token with the requested scopes for the client to collect.
    Approve {
        id: String,
        #[arg(long)]
        engine: Option<String>,
        #[arg(long)]
        token: Option<String>,
        /// Approve a request that asks for the `admin` scope (full control of
        /// the engine). Without it such a request is refused.
        #[arg(long)]
        allow_admin: bool,
    },
    Deny {
        id: String,
        #[arg(long)]
        engine: Option<String>,
        #[arg(long)]
        token: Option<String>,
    },
    /// Client side: ask an engine for a token and wait for the operator's approval.
    Request {
        /// `http://host:port` of the engine.
        #[arg(long)]
        engine: String,
        #[arg(long, default_value = "estia-cli")]
        name: String,
        #[arg(long, value_delimiter = ',', default_value = "generate,embed,models:read")]
        scopes: Vec<String>,
        #[arg(long, default_value_t = 300)]
        wait_seconds: u64,
    },
}

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
    Install,
    Remove,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct FileConfig {
    #[serde(default)]
    roles: Option<Roles>,
}

struct Ctx {
    data_dir: PathBuf,
    store: ModelStore,
    runtime: PythonRuntime,
    /// `--runner` / `ESTIA_RUNNER`, if given.
    runner_arg: Option<PathBuf>,
    /// Resolved on first use: resolving may write the compiled-in runner into
    /// the data dir, which commands such as `discover` must not do.
    runner_found: std::sync::OnceLock<Option<PathBuf>>,
    python: Option<PathBuf>,
    roles: Roles,
}

fn default_data_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    if cfg!(target_os = "macos") {
        home.join("Library/Application Support/estia")
    } else {
        home.join(".local/share/estia")
    }
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
        let store = ModelStore::new(data_dir.join("models")).with_user_agent(format!("estia/{}", env!("CARGO_PKG_VERSION")));
        let runtime = PythonRuntime::new(data_dir.join("runtime")).with_user_agent(format!("estia/{}", env!("CARGO_PKG_VERSION")));
        let roles = match std::fs::read_to_string(data_dir.join("config.json")) {
            Ok(text) => serde_json::from_str::<FileConfig>(&text).context("parse config.json")?.roles.unwrap_or_else(Roles::defaults),
            Err(_) => Roles::defaults(),
        };
        Ok(Self {
            data_dir,
            store,
            runtime,
            runner_arg: cli.runner.clone(),
            runner_found: std::sync::OnceLock::new(),
            python: cli.python.clone(),
            roles,
        })
    }

    fn runner(&self) -> Option<PathBuf> {
        self.runner_found.get_or_init(|| find_runner(self.runner_arg.clone(), &self.data_dir)).clone()
    }

    fn save_roles(&self) -> Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        let cfg = FileConfig { roles: Some(self.roles.clone()) };
        std::fs::write(self.data_dir.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
        Ok(())
    }

    fn python(&self) -> PathBuf {
        self.python.clone().or_else(|| self.runtime.python_path()).unwrap_or_else(|| PathBuf::from("python3"))
    }

    fn engine(&self) -> Result<Engine> {
        let runner =
            self.runner().ok_or_else(|| anyhow!("no runner script found — pass --runner or set ESTIA_RUNNER to estia-runner.py"))?;
        let cfg =
            EngineConfig::new(self.store.clone(), self.runtime.clone(), runner).with_python(self.python()).with_roles(self.roles.clone());
        Ok(Engine::new(cfg))
    }

    /// Role name, family, or artifact id → the artifact to load.
    fn resolve_generation(&self, name: &str) -> Result<&'static Artifact> {
        if let Some(a) = find_artifact(name) {
            return Ok(a);
        }
        if let Some(a) = find_family_default(name, Format::Mlx) {
            return Ok(a);
        }
        let (res, artifact) = self.roles.resolve_artifact(name, Format::Mlx).map_err(|e| anyhow!("{name}: {e}"))?;
        if res.served_by != res.asked {
            eprintln!("role `{}` is unbound; served by `{}` ({})", res.asked, res.served_by, artifact.id);
        }
        Ok(artifact)
    }

    fn resolve_embedding(&self, name: Option<&str>) -> Result<&'static EmbedModel> {
        let id = name.unwrap_or("embeddinggemma-300m-4bit");
        find_embed_model(id).ok_or_else(|| anyhow!("unknown embedding model `{id}`"))
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

async fn pull(ctx: &Ctx, id: &str) -> Result<()> {
    let spec = if let Some(a) = find_artifact(id) {
        DownloadSpec::from(a)
    } else if let Some(e) = find_embed_model(id) {
        DownloadSpec {
            id: e.id.to_string(),
            repo_id: e.repo_id.to_string(),
            revision: e.revision.to_string(),
            required_disk_bytes: e.required_disk_bytes,
        }
    } else {
        return Err(anyhow!("unknown model id `{id}` (see `estia models`)"));
    };
    eprintln!("pulling {} from {}@{} ({} required)", spec.id, spec.repo_id, spec.revision, gb(spec.required_disk_bytes));
    let summary = ctx.store.download(&spec, print_progress).await?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn models(ctx: &Ctx) -> Result<()> {
    println!("{:<36} {:<16} {:<6} {:<10} size", "id", "family", "format", "state");
    for a in GENERATION_MODELS {
        let (state, size) = if ctx.store.is_installed(a.id) {
            ("installed", ctx.store.bytes_on_disk(a.id).map(gb).unwrap_or_default())
        } else if let Some(b) = ctx.store.partial_bytes_on_disk(a.id) {
            ("partial", format!("{} of {}", gb(b), gb(a.required_disk_bytes)))
        } else {
            ("missing", gb(a.required_disk_bytes))
        };
        println!("{:<36} {:<16} {:<6} {:<10} {}", a.id, a.family, format!("{:?}", a.format).to_lowercase(), state, size);
    }
    for e in EMBEDDING_MODELS {
        let (state, size) = if ctx.store.is_installed(e.id) {
            ("installed", ctx.store.bytes_on_disk(e.id).map(gb).unwrap_or_default())
        } else {
            ("missing", gb(e.required_disk_bytes))
        };
        println!("{:<36} {:<16} {:<6} {:<10} {} ({}-dim, {})", e.id, "embed", "mlx", state, size, e.dims, e.arch.model_type());
    }
    Ok(())
}

fn status(ctx: &Ctx) -> Result<()> {
    println!("data dir : {}", ctx.data_dir.display());
    println!("models   : {}", ctx.store.models_dir().display());
    let rt = ctx.runtime.status();
    println!(
        "runtime  : {:?}{}",
        rt.state,
        rt.python_version
            .as_deref()
            .map(|v| format!(" (python {v}, mlx-lm {})", rt.mlx_lm_version.as_deref().unwrap_or("?")))
            .unwrap_or_default()
    );
    println!("python   : {}", ctx.python().display());
    println!("runner   : {}", ctx.runner().as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(none found)".into()));
    let installed: Vec<&str> = GENERATION_MODELS
        .iter()
        .map(|a| a.id)
        .chain(EMBEDDING_MODELS.iter().map(|e| e.id))
        .filter(|id| ctx.store.is_installed(id))
        .collect();
    println!("installed: {}", if installed.is_empty() { "(none)".to_string() } else { installed.join(", ") });
    println!("roles    :");
    for (role, b) in ctx.roles.iter() {
        println!("  {:<8} → {}{}", role, b.family, if b.pin { " (pinned)" } else { "" });
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
            println!("daemon   : running · pid {} · {} · up {up} · loaded [{loaded}]", rec.pid, host_port(&rec.bind, rec.port));
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
    let artifact = ctx.resolve_generation(model)?;
    let engine = ctx.engine()?;
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
    eprintln!("model {} · backend {} · spawned in {} ms", artifact.id, BACKEND_MLX_PYTHON, started.elapsed().as_millis());

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
        let text = match session.generate_stream_with(prompt, Some(max_tokens), Some(temperature), prio, Some(&cancel), |tok| {
            if first.is_none() {
                first = Some(t0.elapsed().as_millis());
            }
            if stream_out {
                let _ = out.write_all(tok.as_bytes());
                let _ = out.flush();
            }
        }) {
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
    let spec = ctx.resolve_embedding(model)?;
    let engine = ctx.engine()?;
    let fingerprint = spec.fingerprint_for(BACKEND_MLX_PYTHON);
    let session = engine.spawn_embed_session(spec.id, &fingerprint)?;
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
    let artifact = ctx.resolve_generation(model)?;
    let engine = ctx.engine()?;
    println!("machine   : {} · {}", std::env::consts::ARCH, std::env::consts::OS);
    println!("backend   : {}", BACKEND_MLX_PYTHON);

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
        let text = session.generate_stream_with(prompt, Some(max_tokens), Some(0.0), Priority::Interactive, None, |_| {
            pieces += 1;
            if first.is_none() {
                first = Some(t0.elapsed().as_millis());
            }
        })?;
        let total = t0.elapsed().as_millis().max(1);
        let chars = text.chars().count();
        let decode_ms = total.saturating_sub(first.unwrap_or(0)).max(1);
        println!(
            "generate {:<4}: {} · first token {} ms · total {} ms · {} chars · ≈{:.1} chars/s decode ({} pieces)",
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
    let spec = ctx.resolve_embedding(embed_model)?;
    let esession = engine.spawn_embed_session(spec.id, &spec.fingerprint_for(BACKEND_MLX_PYTHON))?;
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
        spec.id,
        v.len(),
        v.first().map(|x| x.len()).unwrap_or(0),
        ms,
        v.len() as f64 * 1000.0 / ms as f64
    );
    Ok(())
}

fn runner_check(ctx: &Ctx) -> Result<()> {
    let engine = ctx.engine()?;
    let runner = engine.resident_runner().to_path_buf();
    let t0 = Instant::now();
    let session = estia_engine::Session::spawn(
        estia_engine::Launch::new(ctx.python()).arg(runner.clone()),
        estia_engine::SessionConfig::default(),
        std::sync::Arc::new(estia_engine::NoopObserver),
    )?;
    let spawned = t0.elapsed().as_millis();
    let hello = session.hello()?;
    let ping_ok = session.call_unobserved(&estia_engine::proto::Request::Ping).is_ok();
    println!("runner   : {}", runner.display());
    println!("python   : {}", ctx.python().display());
    println!("spawn    : {spawned} ms");
    println!("ping     : {}", if ping_ok { "ok" } else { "FAILED" });
    match hello {
        Some(h) => {
            println!("protocol : v{} ({} {})", h.protocol, h.runner, h.version);
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
) -> Result<()> {
    use estia_engine::proto::Message;
    let artifact = ctx.resolve_generation(model)?;
    let engine = ctx.engine()?;
    let input = read_stdin()?;
    let mut messages: Vec<Message> = if input.trim_start().starts_with('[') {
        serde_json::from_str(&input).context("stdin is not a JSON array of messages")?
    } else {
        let mut m = Vec::new();
        if let Some(sys) = system {
            m.push(Message::new("system", sys));
        }
        m.push(Message::new("user", input));
        m
    };
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
            "turn {}: first token {} ms · total {} ms · prompt {} tokens ({} cached) · generated {} · template {}",
            turn + 1,
            first.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            t0.elapsed().as_millis(),
            m.prompt_tokens.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            m.cached_tokens.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            m.generation_tokens.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            m.template.as_deref().unwrap_or("?")
        );
        if turn + 1 < turns {
            messages.push(Message::new("assistant", outcome.text.clone()));
            messages.push(Message::new("user", "Now say the same thing in exactly five words."));
        }
    }
    Ok(())
}

fn tokens(ctx: &Ctx, model: &str) -> Result<()> {
    let artifact = ctx.resolve_generation(model)?;
    let engine = ctx.engine()?;
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
    idle_unload_minutes: u64,
    allow_host: Vec<String>,
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
    let idle_unload = if idle_unload_minutes == 0 { None } else { Some(std::time::Duration::from_secs(idle_unload_minutes * 60)) };
    estia_server::serve(
        state,
        ctx.data_dir.clone(),
        ServeOptions { lan, advertise: lan && !no_advertise, name, idle_unload, allowed_hosts: allow_host },
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

async fn setup(ctx: &Ctx, roles: &[String], no_models: bool) -> Result<()> {
    use estia_server::tokens::{TokenStore, SCOPE_ADMIN};
    println!("estia setup — data dir {}", ctx.data_dir.display());
    std::fs::create_dir_all(&ctx.data_dir)?;

    // 1. Runtime.
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
        let summary =
            ctx.runtime.install(|p| eprint!("\r  {:<32} {:<60}", p.phase, safe(&p.message.chars().take(60).collect::<String>()))).await?;
        eprintln!();
        println!("✓ runtime  : python {} / mlx-lm {}", summary.python_version, summary.mlx_lm_version);
    }

    // 2. Models for the requested roles.
    if !no_models {
        let mut wanted: Vec<DownloadSpec> = Vec::new();
        for role in roles {
            if role == "embed" {
                let e = ctx.resolve_embedding(None)?;
                wanted.push(DownloadSpec {
                    id: e.id.into(),
                    repo_id: e.repo_id.into(),
                    revision: e.revision.into(),
                    required_disk_bytes: e.required_disk_bytes,
                });
            } else {
                match ctx.roles.resolve_artifact(role, Format::Mlx) {
                    Ok((_, a)) => wanted.push(DownloadSpec::from(a)),
                    Err(e) => println!("! role `{role}`: {e} — skipped"),
                }
            }
        }
        wanted.dedup_by(|a, b| a.id == b.id);
        for spec in wanted {
            if ctx.store.is_installed(&spec.id) {
                println!("✓ model    : {} ({})", spec.id, gb(ctx.store.bytes_on_disk(&spec.id).unwrap_or(0)));
            } else {
                println!("… model    : pulling {} ({} required)", spec.id, gb(spec.required_disk_bytes));
                ctx.store.download(&spec, print_progress).await?;
                println!("✓ model    : {}", spec.id);
            }
        }
    }

    // 3. Roles file and first token.
    if !ctx.data_dir.join("config.json").exists() {
        ctx.save_roles()?;
        println!("✓ roles    : defaults written to config.json");
    } else {
        println!("✓ roles    : config.json present");
    }
    let tokens = TokenStore::open(ctx.data_dir.join("tokens.json"))?;
    if tokens.is_empty() {
        let t = tokens.mint("local", &[SCOPE_ADMIN])?;
        println!("✓ token    : admin token `local` minted — shown once, keep it:\n\n    {t}\n");
    } else {
        println!("✓ token    : {} token(s) in tokens.json (estia token new <name> for another)", tokens.list().len());
    }
    if ctx.runner().is_none() {
        println!("! runner   : estia-runner.py not found — pass --runner or set ESTIA_RUNNER");
    }
    println!(
        "\nnext:
  estia serve                    run it now, for this machine only (loopback)
  estia service install --local  or run it at login, loopback only
  estia status · estia dashboard

  to let other devices on your network use it (plain HTTP; each device pairs for a token):
  estia serve --lan              or: estia service install
  from the other device: estia discover · estia pair request --engine http://<this-mac>:27200"
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
        ServiceAction::Install { port, local, allow_host, log_level, log_format } => {
            // Checked before anything is staged: every path below lives under it.
            service_path(&ctx.data_dir)?;
            let args = service_args(port, local, &allow_host, log_level.as_deref(), log_format)?;
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
) -> Result<Vec<String>> {
    let mut args = vec!["serve".to_string(), "--port".into(), port.to_string()];
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
                "engine   : v{} api v{} · up {}s · bind {} · auth {}",
                jtext(&h["version"]),
                jtext(&h["api_version"]),
                jtext(&h["uptime_s"]),
                jtext(&h["bind"]),
                if h["auth_required"].as_bool().unwrap_or(true) { "required" } else { "OFF" }
            ),
            None => println!("engine   : unreachable"),
        }
        match &stats {
            Some(s) => println!(
                "loaded   : {} · queue interactive {} / background {} · jobs {}",
                jlist(&s["loaded"]).join(", "),
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
                    return Err(anyhow!("timed out waiting for approval"));
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
    match action {
        RuntimeAction::Status => {
            let s = ctx.runtime.status();
            println!("{}", serde_json::to_string_pretty(&s)?);
        }
        RuntimeAction::Install => {
            ctx.runtime.preflight(estia_engine::runtime::RUNTIME_APPROX_BYTES)?;
            let summary = ctx
                .runtime
                .install(|p| {
                    eprint!("\r{:<32} {}                    ", p.phase, safe(&p.message.chars().take(60).collect::<String>()));
                })
                .await?;
            eprintln!();
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        RuntimeAction::Remove => {
            println!("{}", if ctx.runtime.remove().await? { "removed" } else { "nothing to remove" });
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
fn setup_logging(cmd: &Cmd) -> Result<()> {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    let (defaults, level, format) = match cmd {
        Cmd::Serve { log_level, log_format, .. } => (SERVE_LOG_DEFAULT, log_level.clone(), *log_format),
        _ => {
            let format = match env("ESTIA_LOG_FORMAT").as_deref().map(str::to_ascii_lowercase).as_deref() {
                Some("json") => LogFormat::Json,
                _ => LogFormat::Text,
            };
            (CLI_LOG_DEFAULT, env("ESTIA_LOG"), format)
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

#[tokio::main]
async fn main() -> std::process::ExitCode {
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
    setup_logging(&cli.cmd)?;
    let mut ctx = Ctx::new(&cli)?;
    match cli.cmd {
        Cmd::Models => models(&ctx),
        Cmd::Pull { id } => pull(&ctx, &id).await,
        Cmd::Rm { id } => {
            let removed = ctx.store.remove(&id).await?;
            println!("{}", if removed { "removed" } else { "nothing to remove" });
            Ok(())
        }
        // Blocking HTTP inside: must leave the async context.
        Cmd::Status => tokio::task::block_in_place(|| status(&ctx)),
        Cmd::Roles { action } => roles(&mut ctx, action),
        Cmd::Run { model, max_tokens, temperature, schema, json, no_stream, background, cancel_after_ms } => {
            tokio::task::block_in_place(|| run(&ctx, &model, max_tokens, temperature, schema, json, no_stream, background, cancel_after_ms))
        }
        Cmd::Embed { model } => tokio::task::block_in_place(|| embed(&ctx, model.as_deref())),
        Cmd::Bench { model, embed_model, max_tokens } => {
            tokio::task::block_in_place(|| bench(&ctx, &model, embed_model.as_deref(), max_tokens))
        }
        Cmd::Runtime { action } => runtime(&ctx, action).await,
        Cmd::RunnerCheck => tokio::task::block_in_place(|| runner_check(&ctx)),
        Cmd::Chat { model, system, cache_key, tools, max_tokens, temperature, two_turns } => tokio::task::block_in_place(|| {
            chat(&ctx, &model, system.as_deref(), cache_key.as_deref(), tools, max_tokens, temperature, two_turns)
        }),
        Cmd::Tokens { model } => tokio::task::block_in_place(|| tokens(&ctx, &model)),
        Cmd::Setup { roles, no_models } => setup(&ctx, &roles, no_models).await,
        Cmd::Service { action } => service(&ctx, action),
        Cmd::Dashboard { engine, token, interval, once } => tokio::task::block_in_place(|| dashboard(&ctx, engine, token, interval, once)),
        Cmd::Serve { port, bind, lan, no_advertise, name, no_auth, idle_unload_minutes, allow_host, log_level: _, log_format: _ } => {
            serve(&ctx, port, bind.as_deref(), lan, no_advertise, name, no_auth, idle_unload_minutes, allow_host).await
        }
        Cmd::Token { action } => token(&ctx, action),
        Cmd::Pair { action } => tokio::task::block_in_place(|| pair(&ctx, action)),
        Cmd::Discover { seconds } => tokio::task::block_in_place(|| discover(seconds)),
        Cmd::RemoteCheck { engine, token, model } => tokio::task::block_in_place(|| remote_check(&engine, token, &model)),
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
        assert_eq!(service_args(27200, true, &[], None, None).unwrap(), ["serve", "--port", "27200"]);
        assert_eq!(
            service_args(1, false, &["studio.lan".into(), " *.home.arpa ".into(), "".into()], None, None).unwrap(),
            ["serve", "--port", "1", "--lan", "--allow-host", "studio.lan", "--allow-host", "*.home.arpa"]
        );
        for bad in ["a b", "a\nExecStartPre=/bin/sh", "x\"y", "100%", "$HOME"] {
            assert!(service_args(1, false, &[bad.into()], None, None).is_err(), "{bad:?} accepted");
        }
    }

    /// `service install --log-level/--log-format` reach `serve`; a filter that
    /// would not parse, or could break out of a service file, is refused.
    #[test]
    fn service_args_carry_log_settings() {
        assert_eq!(
            service_args(1, true, &[], Some(" estia_server=debug,mdns_sd=info "), Some(LogFormat::Json)).unwrap(),
            ["serve", "--port", "1", "--log-level", "estia_server=debug,mdns_sd=info", "--log-format", "json"]
        );
        assert_eq!(service_args(1, true, &[], Some(""), None).unwrap(), ["serve", "--port", "1"]);
        for bad in ["estia=loud", "debug\nExecStartPre=/bin/sh", "a b", "$HOME", "100%"] {
            assert!(service_args(1, true, &[], Some(bad), None).is_err(), "{bad:?} accepted");
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

    /// `cli/estia-runner.py` is a symlink; a checkout without symlink support
    /// would turn it into a one-line text file holding the target path.
    #[test]
    fn embedded_runner_is_the_real_script() {
        assert!(EMBEDDED_RUNNER.starts_with("#!/usr/bin/env python3"), "{}", &EMBEDDED_RUNNER[..EMBEDDED_RUNNER.len().min(80)]);
        if let Ok(canonical) = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../runners/mlx-python/estia-runner.py")) {
            assert_eq!(EMBEDDED_RUNNER, canonical);
        }
    }
}
