//! Run models inside your own program with `Engine`: no server, no HTTP.
//!
//! What it shows:
//!   - building an `Engine` from a model directory, the Python runtime and
//!     the runner script
//!   - resolving a role (`fast`) to the model artifact the engine loads
//!   - a chat, then a streamed follow-up that reuses the prompt cache
//!   - structured output with `structured::enforce` and the retry hint
//!   - embeddings with the model's task prefixes and its fingerprint
//!
//! It needs what `estia setup` installs: the Python runtime and the models.
//!
//! ```text
//! cargo run -p estia-engine --example in_process
//! ESTIA_DATA_DIR=/path/to/estia-data ESTIA_MODEL=text cargo run -p estia-engine --example in_process
//! ```
//!
//! Every call here blocks. In an async program, run them in `spawn_blocking`
//! or on a worker thread. This process starts one runner per model and stops
//! them when the sessions are dropped.
//!
//! The engine reports runner starts, model loads, cancels and restarts, and
//! the runner's own stderr, as `tracing` events. This example installs no
//! subscriber, so they are discarded; your program can install one to log them.

use estia_engine::models::embed::{EmbedTask, BACKEND_MLX_PYTHON, EMBEDDING_GEMMA_300M_4BIT};
use estia_engine::models::{find_artifact, find_family_default, Artifact, Format, ModelStore};
use estia_engine::proto::{GenerationMeta, Message};
use estia_engine::runtime::PythonRuntime;
use estia_engine::structured::{self, OutputFormat, Structured};
use estia_engine::{Engine, EngineConfig, Priority};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

type Error = Box<dyn std::error::Error>;

/// The same default the `estia` CLI uses.
fn data_dir() -> PathBuf {
    if let Ok(d) = std::env::var("ESTIA_DATA_DIR") {
        return PathBuf::from(d);
    }
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
    if cfg!(target_os = "macos") {
        home.join("Library/Application Support/estia")
    } else {
        home.join(".local/share/estia")
    }
}

/// The resident runner script. A shipped app bundles its own copy.
fn runner() -> PathBuf {
    std::env::var("ESTIA_RUNNER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../runners/mlx-python/estia-runner.py")))
}

/// Artifact id, family or role, in that order, as the server resolves them.
fn resolve(engine: &Engine, name: &str) -> Result<&'static Artifact, Error> {
    if let Some(a) = find_artifact(name).or_else(|| find_family_default(name, Format::Mlx)) {
        return Ok(a);
    }
    Ok(engine.roles().resolve_artifact(name, Format::Mlx)?.1)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    let dir = data_dir();
    let mut cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), runner());
    if let Ok(python) = std::env::var("ESTIA_PYTHON") {
        cfg = cfg.with_python(python);
    }
    // Roles default to text → gemma4-e4b, fast → gemma4-e2b. The engine does
    // not read the CLI's config.json; pass your own table with `with_roles`.
    let engine = Engine::new(cfg);
    if !engine.runtime().is_installed() && std::env::var("ESTIA_PYTHON").is_err() {
        eprintln!("no Python runtime under {}. Run `estia setup` (or set ESTIA_DATA_DIR).", dir.join("runtime").display());
        std::process::exit(2);
    }

    let name = std::env::var("ESTIA_MODEL").unwrap_or_else(|_| "fast".to_string());
    let artifact = resolve(&engine, &name)?;
    if !engine.store().is_installed(artifact.id) {
        eprintln!("model {} is not downloaded. Run: estia pull {}", artifact.id, artifact.id);
        std::process::exit(2);
    }

    // 1. Start a runner for the model and load it.
    let t0 = Instant::now();
    let gen = engine.spawn_gen_session(artifact.id)?;
    let load = gen.load()?;
    let load_ms = load.map(|d| d.as_millis().to_string()).unwrap_or_else(|| "?".into());
    println!(
        "`{name}` → {} ({}): runner up and model loaded in {} ms (load {load_ms} ms)",
        artifact.id,
        artifact.family,
        t0.elapsed().as_millis()
    );

    // 2. A chat, then a streamed follow-up under the same cache key.
    let mut messages = vec![
        Message::new("system", "You are a concise assistant. Answer in one sentence."),
        Message::new("user", "Name one sea near Greece."),
    ];
    let first = gen.chat_with(&messages, None, Some("demo-1"), None, Some(60), Some(0.2), Priority::Interactive)?;
    println!("assistant> {}", first.text.trim());
    println!("  [{}]", describe(&first.meta));
    messages.push(Message::new("assistant", first.text));
    messages.push(Message::new("user", "And one near Italy?"));
    print!("assistant> ");
    let second =
        gen.chat_stream_with(&messages, None, Some("demo-1"), None, Some(60), Some(0.2), Priority::Interactive, None, |piece| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
        })?;
    println!("\n  [{}]", describe(&second.meta));

    // 3. Structured output. The MLX runner cannot constrain decoding, so show
    //    the model the schema (`with_prompt_hint` puts it in the system
    //    prompt) and validate afterwards. On failure, send the retry hint as
    //    the next user turn and try once more.
    let schema = json!({
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"]
    });
    let format = OutputFormat::JsonSchema { schema };
    let ask = structured::with_prompt_hint(&[Message::new("user", "Where is the Acropolis?")], &format);
    let out = gen.chat_with(&ask, None, None, None, Some(80), Some(0.0), Priority::Interactive)?;
    match structured::enforce(&out.text, &format) {
        Ok(s) => println!("\nJSON: {} (repaired: {}, repairs: {:?})", s.value, s.repaired, s.repairs),
        Err(e) => println!("\ninvalid: {e}\nretry with: {}", Structured::retry_hint(&e).trim()),
    }

    // 4. Embeddings. The session embeds inputs exactly as given: add the
    //    model's own prefix per task. Store the fingerprint with the vectors.
    let spec = EMBEDDING_GEMMA_300M_4BIT;
    let fingerprint = spec.fingerprint_for(BACKEND_MLX_PYTHON);
    let emb = engine.spawn_embed_session(spec.id, &fingerprint)?;
    let passages =
        ["The spare key is in the blue tin in the garage.", "Water the lemon tree twice a week.", "The router is in the hallway cupboard."];
    let docs: Vec<String> = passages.iter().map(|p| format!("{}{p}", spec.prefix(EmbedTask::Document))).collect();
    let vectors = emb.embed_batch_with(&docs, Priority::Background)?;
    let question = "where did I put the key?";
    let q = emb.embed_batch_with(&[format!("{}{question}", spec.prefix(EmbedTask::Query))], Priority::Interactive)?.remove(0);
    println!("\n{} vectors of {} dims, fingerprint {}", vectors.len(), q.len(), emb.fingerprint());
    let mut ranked: Vec<(f32, &str)> = vectors.iter().zip(passages).map(|(v, p)| (cosine(&q, v), p)).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("{question}");
    for (score, p) in ranked {
        println!("  {score:.3}  {p}");
    }
    Ok(())
}

/// The runner's token accounting for one generation.
fn describe(m: &GenerationMeta) -> String {
    format!(
        "{} prompt tokens, {} from cache, {} generated",
        m.prompt_tokens.unwrap_or(0),
        m.cached_tokens.unwrap_or(0),
        m.generation_tokens.unwrap_or(0)
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b)).max(f32::EPSILON)
}
