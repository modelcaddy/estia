//! Use a running Estia server from Rust, with `RemoteEngine`.
//!
//! What it shows:
//!   - checking an engine with `health()` before you have a token
//!   - getting a token by pairing (`--pair <name>`), or using `ESTIA_TOKEN`
//!   - pinning an engine on the LAN: it serves https with its own
//!     certificate, and `RemoteEngine::connect` trusts exactly the one whose
//!     fingerprint it is given (`ESTIA_PIN`, or what pairing returns).
//!     Pairing shows a 6-digit code here, and `estia pair approve` shows the
//!     engine's own on its machine. The operator approves only when the two
//!     are the same, which proves to the engine that the device saw its real
//!     certificate; once approved, this program asks whether the engine shows
//!     the same code and pins only on a yes, which proves to the device that
//!     the engine saw the same exchange
//!   - a streamed chat with a cache key: the second turn reuses the cache
//!   - cancelling a stream with a `CancelToken`
//!   - `GenHandle` / `EmbedHandle`: code that holds a handle works the same
//!     whether the model runs in this process or on another machine
//!   - embeddings by role (`embed`), on either backend: the first response
//!     reports the fingerprint (`<artifact>@<backend>`), which names the
//!     model and so its task prefixes, and every later call asks the server
//!     to refuse a different fingerprint. `RemoteEngine` sends inputs as given
//!     (`task: none`), so the client adds the prefix.
//!
//! Scopes: generate, embed
//!
//! ```text
//! ESTIA_TOKEN=estia_... cargo run -p estia-engine --example remote_client
//! ESTIA_URL=https://192.168.1.20:27200 cargo run -p estia-engine --features remote-tls --example remote_client -- --pair "my laptop"
//! ESTIA_URL=https://192.168.1.20:27200 ESTIA_TOKEN=estia_... ESTIA_PIN=3f2a… cargo run -p estia-engine --features remote-tls --example remote_client
//! ```
//!
//! An https engine needs the `remote-tls` feature of estia-engine (the pinned
//! client); without it the https calls say so.
//!
//! `RemoteEngine` uses the native `/engine/*` routes and blocking HTTP. Call it
//! from a thread or `spawn_blocking`, not from inside an async task.

use estia_engine::models::embed::{find_embed_model, EmbedTask};
use estia_engine::proto::Message;
use estia_engine::{
    CancelToken, EmbedHandle, Fingerprint, GenHandle, PairingOutcome, Priority, RemoteEmbed, RemoteEngine, RemoteGen, SessionError,
};
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, SessionError>;

fn main() {
    let url = std::env::var("ESTIA_URL").unwrap_or_else(|_| "http://127.0.0.1:27200".to_string());
    if let Err(e) = run(&url) {
        // RemoteEngine reports HTTP failures as `SessionError::Runner` with
        // text such as "remote engine 401 Unauthorized: unknown token".
        let message = match &e {
            SessionError::Runner(m) => m.clone(),
            other => other.to_string(),
        };
        eprintln!("error: {message}");
        if matches!(e, SessionError::CertificateMismatch { .. }) {
            eprintln!("The engine's certificate is not the one this app pinned (ESTIA_PIN). Pair again if its owner rotated it.");
        } else if message.contains("https_required") || message.contains("over HTTPS only") {
            eprintln!("This engine serves other devices over https: set ESTIA_URL=https://… and pair again (--pair).");
        } else if message.contains(" 401 ") {
            eprintln!("Check ESTIA_TOKEN, or mint a token: estia token new rust-app --scopes generate,embed");
        } else if message.contains("unreachable") {
            eprintln!("Is `estia serve` running? Set ESTIA_URL to the engine's address.");
        }
        std::process::exit(1);
    }
}

fn run(url: &str) -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // ESTIA_PIN is for an https engine; one exported for a LAN engine is
    // ignored for the engine on this machine (http://127.0.0.1).
    let pin = match std::env::var("ESTIA_PIN") {
        Ok(p) if !p.trim().is_empty() && url.starts_with("https://") => {
            Some(Fingerprint::parse(&p).map_err(|e| SessionError::Runner(format!("ESTIA_PIN: {e}")))?)
        }
        _ => None,
    };
    let (token, pin) = if args.first().map(String::as_str) == Some("--pair") {
        // Pairing checks the certificate itself and returns the pin with the token.
        pair(url, args.get(1).map(String::as_str).unwrap_or("rust example"))?
    } else {
        if url.starts_with("https://") && pin.is_none() {
            eprintln!(
                "ESTIA_PIN is not set: an https engine needs its certificate fingerprint. Pair from here (--pair \"my laptop\"),\n\
                 or run `estia tls fingerprint` on the engine's machine and export ESTIA_PIN=<what it prints>."
            );
            std::process::exit(2);
        }
        match std::env::var("ESTIA_TOKEN") {
            Ok(t) if !t.is_empty() => (t, pin),
            _ => {
                eprintln!(
                    "ESTIA_TOKEN is not set. Mint one on the engine's machine:\n  estia token new rust-app --scopes generate,embed\n\
                     or pair from here: cargo run -p estia-engine --features remote-tls --example remote_client -- --pair \"my laptop\""
                );
                std::process::exit(2);
            }
        }
    };
    // Health needs no token: use it to check the address and the API version.
    // `connect` pins the certificate when there is a pin (https), and uses
    // plain HTTP otherwise (the engine on this machine).
    let health = RemoteEngine::connect(url, None, pin)?.health()?;
    println!("engine {} (api v{}) at {url}", health["version"].as_str().unwrap_or("?"), health["api_version"]);
    if let Some(p) = pin {
        println!("certificate {} (pinned)", p.short());
    }
    let engine = Arc::new(RemoteEngine::connect(url, Some(token), pin)?);
    // A role, not a model id: the server decides which model serves it.
    let model = std::env::var("ESTIA_MODEL").unwrap_or_else(|_| "fast".to_string());

    // 1. A streamed chat, two turns under one cache key.
    let cache_key = format!("rust-demo-{}", std::process::id());
    let mut messages = vec![
        Message::new("system", "You are a concise assistant. Answer in one sentence."),
        Message::new("user", "Name one sea near Greece."),
    ];
    for follow_up in ["And one near Italy?", ""] {
        let t0 = Instant::now();
        let mut first = true;
        let (text, meta) =
            engine.chat_stream(&model, &messages, None, Some(&cache_key), Some(80), Some(0.2), Priority::Interactive, None, |piece| {
                if std::mem::take(&mut first) {
                    print!("assistant> ");
                }
                print!("{piece}");
                let _ = std::io::stdout().flush();
            })?;
        let meta = meta.unwrap_or_default();
        println!(
            "\n  [{} prompt tokens, {} from cache, {} ms]",
            meta.prompt_tokens.unwrap_or(0),
            meta.cached_tokens.unwrap_or(0),
            t0.elapsed().as_millis()
        );
        messages.push(Message::new("assistant", text));
        if follow_up.is_empty() {
            break;
        }
        messages.push(Message::new("user", follow_up));
    }

    // 2. Cancel a stream. Flipping the token drops the connection, and the
    //    server cancels the generation in its runner.
    let cancel = CancelToken::new();
    let mut pieces = 0;
    let long = [Message::new("user", "Write the numbers from one to two hundred in words, separated by commas.")];
    let result = engine.chat_stream(&model, &long, None, None, Some(600), Some(0.2), Priority::Interactive, Some(&cancel), |_| {
        pieces += 1;
        if pieces == 10 {
            cancel.cancel();
        }
    });
    match result {
        Err(SessionError::Cancelled { partial }) => {
            println!("\ncancelled after {pieces} pieces: {:?}…", partial.chars().take(40).collect::<String>())
        }
        Ok((text, _)) => println!("\nfinished before the cancel: {} chars", text.len()),
        Err(e) => return Err(e),
    }

    // 3. The handle shape. A GenHandle is Local or Remote; the calls are the same.
    let gen = GenHandle::Remote(RemoteGen::new(Arc::clone(&engine), model.clone()));
    let text = gen.generate_with("Complete in three words: The capital of Greece is", Some(12), Some(0.0), Priority::Interactive)?;
    println!("\nraw prompt → {:?}", text.trim());

    // 4. Embeddings, by role like the chat: the operator decides which model
    //    answers `embed`, and the engine's backend which artifact of it runs.
    //    The fingerprint, `<artifact id>@<backend>`, names the vector space:
    //    `embeddinggemma-300m-4bit@mlx-python` on MLX,
    //    `embeddinggemma-300m-q8_0-gguf@llama-cpp` on llama.cpp. Vectors from
    //    two fingerprints do not compare.
    //
    //    A new index learns the fingerprint from the engine's first response,
    //    as here. An app stores it with the vectors and passes the stored
    //    value from then on, so a server that changed its model or backend
    //    answers 422 instead of returning vectors that do not compare.
    let (_, fingerprint) = engine.embed_batch("embed", &["which model answers embed?".to_string()], None, Priority::Background)?;
    // RemoteEngine sends `task: none`, so the prefix is the caller's job, and
    // it depends on the model. The fingerprint names the model. A model
    // imported on the engine is not known here: ask its operator for its
    // prefixes, if it has any.
    let spec = find_embed_model(&fingerprint);
    if spec.is_none() {
        println!("\n{fingerprint} is not a built-in model; embedding without task prefixes");
    }
    let prefix = |task: EmbedTask| spec.map(|s| s.prefix(task)).unwrap_or("");
    let emb = EmbedHandle::Remote(RemoteEmbed::new(Arc::clone(&engine), "embed", fingerprint));
    let passages =
        ["The spare key is in the blue tin in the garage.", "Water the lemon tree twice a week.", "The router is in the hallway cupboard."];
    let docs: Vec<String> = passages.iter().map(|p| format!("{}{p}", prefix(EmbedTask::Document))).collect();
    let vectors = emb.embed_batch_with(&docs, Priority::Background)?;
    let question = "where did I put the key?";
    let q = emb.embed(&format!("{}{question}", prefix(EmbedTask::Query)))?;
    println!("\n{} vectors of {} dims, fingerprint {}", vectors.len(), q.len(), emb.fingerprint());
    let mut ranked: Vec<(f32, &str)> = vectors.iter().zip(passages).map(|(v, p)| (cosine(&q, v), p)).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("{question}");
    for (score, p) in ranked {
        println!("  {score:.3}  {p}");
    }
    Ok(())
}

/// Ask for a token and wait for the operator. The token comes back once;
/// over https the certificate to pin comes once the user says the engine
/// shows the same code as this program.
fn pair(url: &str, name: &str) -> Result<(String, Option<Fingerprint>)> {
    // Over https: connects without trusting the engine yet, commits to a
    // nonce, and computes the code from the certificate it saw.
    let pending = estia_engine::begin_pairing(url, name, &["generate", "embed"])?;
    if let Some(code) = pending.code() {
        println!("\n    code: {code}\n");
        println!("`estia pair approve` shows the engine's code on its machine: the operator approves only if it is");
        println!("{code} too. If the engine shows another code, stop: something between this device and the engine");
        println!("answered with its own certificate.");
    }
    println!("Approve it on the engine's machine:\n  {}", pending.approve_command());
    // The engine drops an undecided request 300 s after it arrived. Stop
    // polling a little before that, so the last answer is not "expired".
    let deadline = Instant::now() + Duration::from_secs(290);
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_secs(2));
        match pending.poll() {
            Ok(PairingOutcome::Approved { token }) => {
                // Over https "approved" alone proves nothing: the user checks
                // that the engine shows the code this program shows.
                let pin = confirm(pending)?;
                println!("approved. Store the token somewhere only this app can read; it is not shown again.");
                println!("To run again without pairing:\n  export ESTIA_URL={url} ESTIA_TOKEN={token}");
                if let Some(p) = pin {
                    println!("  export ESTIA_PIN={p}");
                }
                return Ok((token, pin));
            }
            Ok(PairingOutcome::AlreadyCollected) => {
                return Err(SessionError::Runner("approved, but another poll collected the token".into()))
            }
            Ok(PairingOutcome::Denied) => return Err(SessionError::Runner("the operator denied the pairing".into())),
            Ok(PairingOutcome::Expired) => return Err(SessionError::Runner("the request expired. Pair again.".into())),
            Ok(PairingOutcome::Pending) => {}
            // A certificate swapped mid-pairing is final; anything else is
            // worth another poll (the engine restarting, Wi-Fi dropping).
            Err(e @ SessionError::CertificateMismatch { .. }) => return Err(e),
            Err(e) => eprintln!("poll failed ({e}); retrying"),
        }
    }
    Err(SessionError::Runner("no decision in time: the request expires 5 minutes after it was made. Pair again.".into()))
}

/// Ask whether the engine shows this program's code: the pin on a yes
/// (`None` over http://127.0.0.1, which has no code), an error otherwise.
/// A yes counts only from a terminal: piped in, nobody compared anything.
/// Without a terminal, the answer must be the six digits the engine shows
/// (`estia pair list` on its machine), and they must equal this code.
fn confirm(pending: estia_engine::PendingPairing) -> Result<Option<Fingerprint>> {
    use std::io::IsTerminal;
    let Some(code) = pending.code().copied() else { return Ok(pending.confirm_match()) };
    let at_terminal = std::io::stdin().is_terminal();
    if at_terminal {
        print!("approved. Does the engine show {code} for this request? [y/N] ");
    } else {
        print!("approved. The six digits the engine shows for this request: ");
    }
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
    let answer = line.trim().to_ascii_lowercase();
    let matched = match estia_engine::PairingCode::parse(&answer) {
        Some(shown) => shown == code,
        None => at_terminal && matches!(answer.as_str(), "y" | "yes"),
    };
    if matched {
        return Ok(pending.confirm_match());
    }
    Err(SessionError::Runner(format!(
        "not confirmed, so the token is not used. If the engine shows another code than {code}, something between this \
         device and the engine answered in its place: deny the request there (estia pair deny {})",
        pending.id()
    )))
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b)).max(f32::EPSILON)
}
