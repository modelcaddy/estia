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
        /// Address to bind. Anything but loopback needs --lan.
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
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
    },
    Uninstall,
    Start,
    Stop,
    Restart,
    Status,
    /// Tail the service logs.
    Logs {
        #[arg(long, default_value_t = 40)]
        lines: usize,
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
    /// Mint a token; the plaintext is printed once.
    New {
        name: String,
        /// Scopes: generate, embed, models:read, models:write, admin. Default admin.
        #[arg(long, value_delimiter = ',')]
        scopes: Option<Vec<String>>,
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
const EMBEDDED_RUNNER: &str = include_str!("../../runners/mlx-python/estia-runner.py");

fn find_runner(explicit: Option<PathBuf>, data_dir: &Path) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p);
    }
    const REL: &str = "runners/mlx-python/estia-runner.py";
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            // Installed layout: runners beside the binary (a release tarball,
            // or what `service install` stages). Dev layout: target/debug/ is
            // two levels below the repo root.
            candidates.push(dir.join(REL));
            candidates.push(dir.join("../..").join(REL));
        }
    }
    candidates.push(PathBuf::from(REL));
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

fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
}

fn print_progress(p: estia_engine::models::DownloadProgress) {
    let pct = match p.total_bytes {
        Some(t) if t > 0 => format!("{:>3}%", p.bytes_downloaded * 100 / t),
        _ => "    ".to_string(),
    };
    eprint!("\r{:<12} {pct} {} {}                    ", p.phase, gb(p.bytes_downloaded), p.file_name.as_deref().unwrap_or(""));
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
                .and_then(|c| c.get(format!("http://127.0.0.1:{}/engine/health", rec.port)).send().ok())
                .and_then(|r| r.json::<serde_json::Value>().ok());
            let up = health
                .as_ref()
                .and_then(|h| h["uptime_s"].as_u64())
                .map(|s| format!("{}m{}s", s / 60, s % 60))
                .unwrap_or_else(|| "?".into());
            let loaded = health
                .as_ref()
                .and_then(|h| h["loaded"].as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            println!("daemon   : running · pid {} · {}:{} · up {up} · loaded [{loaded}]", rec.pid, rec.bind, rec.port);
            if rec.bind == "0.0.0.0" {
                for ip in lan_ips() {
                    println!("  reach it: http://{ip}:{}   (clients: estia pair request --engine http://{ip}:{})", rec.port, rec.port);
                }
            }
        }
        None => println!("daemon   : not running  (estia serve --lan, or estia service install)"),
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
            None => println!("{role} was not bound"),
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
    bind: &str,
    lan: bool,
    no_advertise: bool,
    name: Option<String>,
    no_auth: bool,
    idle_unload_minutes: u64,
) -> Result<()> {
    use estia_server::{tokens::TokenStore, AppState, ServeOptions};
    let engine = std::sync::Arc::new(ctx.engine()?);
    let tokens = TokenStore::open(ctx.data_dir.join("tokens.json"))?;
    if !no_auth && tokens.is_empty() {
        let t = tokens.mint("local", &[estia_server::tokens::SCOPE_ADMIN])?;
        eprintln!("minted the first token (name `local`, scope admin). Shown once — keep it:\n\n  {t}\n\n  Authorization: Bearer {t}\n");
    }
    let bind = if lan && bind == "127.0.0.1" { "0.0.0.0" } else { bind };
    let addr: std::net::SocketAddr = format!("{bind}:{port}").parse().context("bad --bind/--port")?;
    let state = std::sync::Arc::new(AppState::new(engine, tokens, !no_auth, addr));
    if no_auth {
        eprintln!("warning: --no-auth — every loopback process can use this engine");
    }
    if lan {
        eprintln!(
            "LAN mode: clients pair with `estia pair request --engine http://<this-host>:{port}`; approve with `estia pair approve <id>`"
        );
    }
    let idle_unload = if idle_unload_minutes == 0 { None } else { Some(std::time::Duration::from_secs(idle_unload_minutes * 60)) };
    estia_server::serve(state, ctx.data_dir.clone(), ServeOptions { lan, advertise: lan && !no_advertise, name, idle_unload }).await
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
            ctx.runtime.install(|p| eprint!("\r  {:<32} {:<60}", p.phase, p.message.chars().take(60).collect::<String>())).await?;
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
    println!("\nnext:\n  estia service install          run at login, on the LAN, with pairing\n  estia serve --lan              or run it by hand\n  estia status · estia dashboard\n  from another device: estia discover · estia pair request --engine http://<this-mac>:27200");
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
        ServiceAction::Install { port, local } => {
            std::fs::create_dir_all(&logs)?;
            let runner = ctx
                .runner()
                .ok_or_else(|| anyhow!("no runner script found — pass --runner or set ESTIA_RUNNER before installing the service"))?;
            let (exe, runner) = stage_install(&ctx.data_dir, &runner)?;
            println!("staged {} and {}", exe.display(), runner.display());
            let mut args = vec!["serve".to_string(), "--port".into(), port.to_string()];
            if !local {
                args.push("--lan".into());
            }
            if cfg!(target_os = "macos") {
                let plist = launchd_plist_path();
                std::fs::create_dir_all(plist.parent().unwrap())?;
                let arg_xml: String = std::iter::once(exe.display().to_string())
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
                    data = xml_escape(&ctx.data_dir.display().to_string()),
                    runner = xml_escape(&runner.display().to_string()),
                    out = xml_escape(&logs.join("estia.out.log").display().to_string()),
                    err = xml_escape(&logs.join("estia.err.log").display().to_string())
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
                let body = format!("[Unit]\nDescription=estia\nAfter=network.target\n\n[Service]\nExecStart={} {}\nEnvironment=ESTIA_DATA_DIR={}\nEnvironment=ESTIA_RUNNER={}\nRestart=always\nRestartSec=3\nWorkingDirectory={}\n\n[Install]\nWantedBy=default.target\n", exe.display(), args.join(" "), ctx.data_dir.display(), runner.display(), ctx.data_dir.display());
                std::fs::write(&unit, body)?;
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
        ServiceAction::Logs { lines } => {
            for name in ["estia.err.log", "estia.out.log"] {
                let p = logs.join(name);
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

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

fn dashboard(ctx: &Ctx, engine: Option<String>, token: Option<String>, interval: u64, once: bool) -> Result<()> {
    let base = match engine {
        Some(e) => e.trim_end_matches('/').to_string(),
        None => match estia_server::another_engine_running(&ctx.data_dir) {
            Some(rec) => format!("http://127.0.0.1:{}", rec.port),
            None => {
                return Err(anyhow!(
                    "no running engine for {} — start one with `estia serve --lan` or pass --engine",
                    ctx.data_dir.display()
                ))
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
        println!("estia dashboard · {base} · {}", chrono_now());
        match &health {
            Some(h) => println!(
                "engine   : v{} api v{} · up {}s · bind {} · auth {}",
                h["version"].as_str().unwrap_or("?"),
                h["api_version"],
                h["uptime_s"],
                h["bind"].as_str().unwrap_or("?"),
                if h["auth_required"].as_bool().unwrap_or(true) { "required" } else { "OFF" }
            ),
            None => println!("engine   : unreachable"),
        }
        match &stats {
            Some(s) => println!(
                "loaded   : {} · queue interactive {} / background {} · jobs {}",
                s["loaded"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default(),
                s["queue"]["interactive"],
                s["queue"]["background"],
                s["jobs"]
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
                println!(
                    "  {:<36} {:<10} {}",
                    a["id"].as_str().unwrap_or("?"),
                    state,
                    a["bytes_on_disk"].as_u64().map(gb).unwrap_or_default()
                );
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
                    println!(
                        "  {:<18} {:<20} {:<28} from {}",
                        x["id"].as_str().unwrap_or("?"),
                        x["name"].as_str().unwrap_or("?"),
                        x["scopes"]
                            .as_array()
                            .map(|a| a.iter().filter_map(|s| s.as_str()).collect::<Vec<_>>().join(","))
                            .unwrap_or_default(),
                        x["from"].as_str().unwrap_or("?")
                    );
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

fn chrono_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{}h{:02}m{:02}s UTC", (secs / 3600) % 24, (secs / 60) % 60, secs % 60)
}

fn pair(ctx: &Ctx, action: PairAction) -> Result<()> {
    use estia_server::pairing::{PairingStatus, PairingStore};
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
            if let Some(r) = remote(&engine, &token)? {
                let v = r.pairings()?;
                for p in v["pairings"].as_array().cloned().unwrap_or_default() {
                    println!(
                        "{:<18} {:<20} {:<28} {:<9} from {}",
                        p["id"].as_str().unwrap_or("?"),
                        p["name"].as_str().unwrap_or("?"),
                        p["scopes"]
                            .as_array()
                            .map(|a| a.iter().filter_map(|s| s.as_str()).collect::<Vec<_>>().join(","))
                            .unwrap_or_default(),
                        p["status"].as_str().unwrap_or("?"),
                        p["from"].as_str().unwrap_or("?")
                    );
                }
                return Ok(());
            }
            let list = store.list();
            if list.is_empty() {
                println!("no pairing requests");
            }
            for p in list {
                println!(
                    "{:<18} {:<20} {:<28} {:?}{}  from {}",
                    p.id,
                    p.name,
                    p.scopes.join(","),
                    p.status,
                    if p.claimed { " (collected)" } else { "" },
                    p.from.as_deref().unwrap_or("?")
                );
            }
        }
        PairAction::Approve { id, engine, token } => {
            if let Some(r) = remote(&engine, &token)? {
                let v = r.decide_pairing(&id, true)?;
                println!(
                    "approved `{}` ({}) on {}",
                    v["name"].as_str().unwrap_or("?"),
                    v["scopes"].as_array().map(|a| a.iter().filter_map(|s| s.as_str()).collect::<Vec<_>>().join(",")).unwrap_or_default(),
                    r.base_url()
                );
                return Ok(());
            }
            let tokens = TokenStore::open(ctx.data_dir.join("tokens.json"))?;
            let p = store.approve(&id, &tokens)?;
            println!("approved `{}` ({}) — the client collects its token on its next poll", p.name, p.scopes.join(","));
        }
        PairAction::Deny { id, engine, token } => {
            if let Some(r) = remote(&engine, &token)? {
                let v = r.decide_pairing(&id, false)?;
                println!("denied `{}` on {}", v["name"].as_str().unwrap_or("?"), r.base_url());
                return Ok(());
            }
            let p = store.deny(&id)?;
            println!("denied `{}`", p.name);
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
            eprintln!(
                "pairing request `{id}` sent as `{name}` — waiting up to {wait_seconds}s for the operator to run: estia pair approve {id}"
            );
            let deadline = Instant::now() + std::time::Duration::from_secs(wait_seconds);
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let v: serde_json::Value = client.get(format!("{base}/engine/pair/{id}")).send()?.error_for_status()?.json()?;
                match v["status"].as_str() {
                    Some("approved") => {
                        let token = v["token"].as_str().ok_or_else(|| anyhow!("approved but the token was already collected"))?;
                        println!("{token}");
                        return Ok(());
                    }
                    Some("denied") => return Err(anyhow!("pairing denied by the operator")),
                    _ => {}
                }
                if Instant::now() >= deadline {
                    return Err(anyhow!("timed out waiting for approval"));
                }
                let _ = PairingStatus::Pending;
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
        let url = match d.addresses.first() {
            Some(a) => format!("http://{a}:{}", d.port),
            None => format!("http://{}:{}", d.host, d.port),
        };
        println!(
            "{:<20} {:<28} {:<30} v{:<7} {}",
            d.name,
            d.host,
            url,
            d.api_version.unwrap_or_else(|| "?".into()),
            d.engine_version.unwrap_or_else(|| "?".into())
        );
    }
    Ok(())
}

fn remote_check(engine: &str, token: Option<String>, model: &str) -> Result<()> {
    use estia_engine::{Priority, RemoteEmbed, RemoteEngine, RemoteGen};
    let remote = std::sync::Arc::new(RemoteEngine::new(engine, token)?);
    let health = remote.health()?;
    println!("health   : api v{} engine {} bind {}", health["api_version"], health["version"], health["bind"]);
    let gen = RemoteGen::new(std::sync::Arc::clone(&remote), model);
    let t0 = Instant::now();
    let mut pieces = 0;
    let text = gen.generate_stream_with("Name one sea in two words.", Some(16), Some(0.0), Priority::Interactive, None, |_| pieces += 1)?;
    println!("generate : {:?} in {} ms ({pieces} pieces, streamed)", text.trim(), t0.elapsed().as_millis());
    let spec = estia_engine::models::embed::EMBEDDING_GEMMA_300M_4BIT;
    let emb = RemoteEmbed::new(remote, spec.id, spec.fingerprint_for("mlx-python"));
    let t0 = Instant::now();
    let v = emb.embed_batch_with(&["the sea at dawn".into()], Priority::Interactive)?;
    println!("embed    : {} × {} dims in {} ms, fingerprint {}", v.len(), v[0].len(), t0.elapsed().as_millis(), emb.fingerprint());
    Ok(())
}

fn token(ctx: &Ctx, action: TokenAction) -> Result<()> {
    use estia_server::tokens::{TokenStore, ALL_SCOPES, SCOPE_ADMIN};
    let store = TokenStore::open(ctx.data_dir.join("tokens.json"))?;
    match action {
        TokenAction::New { name, scopes } => {
            let scopes: Vec<String> = scopes.unwrap_or_else(|| vec![SCOPE_ADMIN.to_string()]);
            for s in &scopes {
                if !ALL_SCOPES.contains(&s.as_str()) {
                    return Err(anyhow!("unknown scope `{s}` (one of {})", ALL_SCOPES.join(", ")));
                }
            }
            let refs: Vec<&str> = scopes.iter().map(String::as_str).collect();
            let t = store.mint(&name, &refs)?;
            println!("{t}");
        }
        TokenAction::List => {
            for r in store.list() {
                println!("{:<12} {:<40} created {}", r.name, r.scopes.join(","), r.created_unix);
            }
        }
        TokenAction::Revoke { name } => {
            println!("{}", if store.revoke(&name)? { "revoked" } else { "no such token" });
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
                    eprint!("\r{:<32} {}                    ", p.phase, p.message.chars().take(60).collect::<String>());
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

/// Stderr logger for the libraries the daemon leans on, off unless `RUST_LOG`
/// is set (`RUST_LOG=mdns_sd=debug estia serve …`).
///
/// mdns-sd reports the things that make an engine undiscoverable — a socket it
/// could not bind, an interface it skipped — through the `log` crate and
/// nowhere else, so without a logger installed those failures are invisible and
/// the registration still looks healthy. The filter is a comma-separated list
/// of `target=level` (or a bare level for everything).
struct StderrLogger {
    filters: Vec<(String, log::LevelFilter)>,
    default: log::LevelFilter,
}

impl log::Log for StderrLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        let level = self.filters.iter().find(|(t, _)| m.target().starts_with(t.as_str())).map(|(_, l)| *l).unwrap_or(self.default);
        m.level() <= level
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            eprintln!("[{} {}] {}", r.level(), r.target(), r.args());
        }
    }
    fn flush(&self) {}
}

fn init_logging() {
    let Ok(spec) = std::env::var("RUST_LOG") else { return };
    let mut filters = Vec::new();
    let mut default = log::LevelFilter::Off;
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match part.split_once('=') {
            Some((target, level)) => {
                if let Ok(l) = level.parse() {
                    filters.push((target.replace('-', "_"), l));
                }
            }
            None => {
                if let Ok(l) = part.parse() {
                    default = l;
                }
            }
        }
    }
    let max = filters.iter().map(|(_, l)| *l).chain(std::iter::once(default)).max().unwrap_or(log::LevelFilter::Off);
    let logger = Box::leak(Box::new(StderrLogger { filters, default }));
    if log::set_logger(logger).is_ok() {
        log::set_max_level(max);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();
    let cli = Cli::parse();
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
        Cmd::Serve { port, bind, lan, no_advertise, name, no_auth, idle_unload_minutes } => {
            serve(&ctx, port, &bind, lan, no_advertise, name, no_auth, idle_unload_minutes).await
        }
        Cmd::Token { action } => token(&ctx, action),
        Cmd::Pair { action } => tokio::task::block_in_place(|| pair(&ctx, action)),
        Cmd::Discover { seconds } => tokio::task::block_in_place(|| discover(seconds)),
        Cmd::RemoteCheck { engine, token, model } => tokio::task::block_in_place(|| remote_check(&engine, token, &model)),
    }
}
