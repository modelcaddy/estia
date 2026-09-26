//! One `llama-server` child: flags, start, readiness, stop, and the files it
//! leaves in the run directory.
//!
//! Files, all named after the adapter's pid so two adapters sharing a run
//! directory never collide:
//!
//! - `llama-<pid>.sock`: the UNIX socket llama-server listens on. llama-server
//!   does not remove it on exit, so a stale one is removed before each start.
//! - `llama-<pid>.key`: the random API key, mode 0600, read by llama-server at
//!   start (`--api-key-file` keeps it out of `ps`). Removed once the server is
//!   up.
//! - `llama-<pid>.json`: `{adapter_pid, server_pid, socket, addr, model, kind}`,
//!   so a host can find and stop a server whose adapter was killed outright
//!   (on macOS nothing else would; see [`Guard`]).
//!
//! The process itself is held in [`Shared`], so the watchdog thread can stop
//! it while the main thread is busy in a request.

use crate::http::{self, Endpoint, Stream};
use crate::{log, AdapterOptions};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// `-c` for generation models when the command line gives no `--ctx`.
pub(crate) const DEFAULT_CTX: u32 = 8192;
/// Embedding inputs are cut to this many tokens, as the MLX runner does
/// (`max_length=512` in estia-runner.py). It also bounds each slot's context.
pub(crate) const EMBED_MAX_TOKENS: usize = 512;
/// Embedding slots: inputs of one batch are decoded this many at a time.
/// Measured on all-MiniLM-L6-v2 (M1 Pro): 256 short inputs take 0.9 s with
/// one slot and 0.25 to 0.35 s with eight.
const EMBED_SLOTS: usize = 4;
/// How long a model may take to become ready. A first run of an unsigned
/// build on macOS can stall for tens of seconds before loading starts.
const LOAD_TIMEOUT: Duration = Duration::from_secs(600);
/// SIGTERM, then this long, then SIGKILL.
pub(crate) const STOP_GRACE: Duration = Duration::from_secs(3);
/// llama-server output lines kept for error messages.
const LOG_TAIL: usize = 60;

/// What a server was started for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Generation,
    Embedding,
}

impl Kind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Kind::Generation => "generation",
            Kind::Embedding => "embedding",
        }
    }
}

/// The model file a `model_path` names: the path itself when it is a file,
/// `<dir>/model.gguf` when it is a directory.
pub(crate) fn resolve_model(model_path: &str) -> Result<PathBuf, String> {
    if model_path.trim().is_empty() {
        return Err("model_path is empty".into());
    }
    let p = PathBuf::from(model_path);
    if p.is_dir() {
        let f = p.join("model.gguf");
        if f.is_file() {
            return Ok(f);
        }
        return Err(format!("no model.gguf in {model_path}"));
    }
    if p.is_file() {
        return Ok(p);
    }
    Err(format!("model not found: {model_path}"))
}

/// The llama-server command line (after the program name).
pub(crate) fn server_args(
    model: &Path,
    kind: Kind,
    endpoint: &Endpoint,
    key_file: &Path,
    ctx: Option<u32>,
    extra: &[String],
) -> Vec<OsString> {
    let mut a: Vec<OsString> = vec!["-m".into(), model.into()];
    match endpoint {
        #[cfg(unix)]
        Endpoint::Unix(sock) => a.extend(["--host".into(), sock.into()]),
        Endpoint::Tcp(addr) => a.extend(["--host".into(), addr.ip().to_string().into(), "--port".into(), addr.port().to_string().into()]),
    }
    a.extend(["--api-key-file".into(), key_file.into()]);
    for f in ["--no-ui", "--no-slots", "--offline", "--cache-ram", "0"] {
        a.push(f.into());
    }
    match kind {
        Kind::Generation => {
            let c = ctx.unwrap_or(DEFAULT_CTX).to_string();
            a.extend(["-np".into(), "1".into(), "-c".into(), c.into()]);
        }
        Kind::Embedding => {
            // No --pooling: the model's own pooling type from its GGUF
            // metadata. Each slot gets EMBED_MAX_TOKENS of context, and a
            // physical batch holds several whole inputs (a non-causal model
            // needs a whole input in one ubatch).
            let ctx = (EMBED_SLOTS * EMBED_MAX_TOKENS).to_string();
            let batch = (EMBED_SLOTS * EMBED_MAX_TOKENS).max(2048).to_string();
            a.extend([
                "--embedding".into(),
                "-np".into(),
                EMBED_SLOTS.to_string().into(),
                "-c".into(),
                ctx.into(),
                "-b".into(),
                batch.clone().into(),
                "-ub".into(),
                batch.into(),
            ]);
        }
    }
    // Last, so an operator's value wins over ours (llama-server keeps the
    // last occurrence of a flag).
    a.extend(extra.iter().map(OsString::from));
    a
}

/// Environment variables llama-server would read as configuration.
fn env_is_dropped(name: &str) -> bool {
    name.starts_with("LLAMA_ARG_") || name == "LLAMA_API_KEY" || name == "HF_TOKEN"
}

/// sun_path holds 104 bytes on macOS and the BSDs, 108 on Linux, NUL included.
#[cfg(unix)]
const MAX_SOCKET_PATH: usize = if cfg!(target_os = "linux") { 107 } else { 103 };

/// Where the server listens: a socket in the run directory; if that path is
/// too long for a socket, one in a private directory under the temp dir
/// (returned so it is removed later); if that is too long too, loopback TCP.
pub(crate) fn choose_endpoint(run_dir: &Path, pid: u32) -> io::Result<(Endpoint, Option<PathBuf>)> {
    #[cfg(unix)]
    {
        let sock = run_dir.join(format!("llama-{pid}.sock"));
        if sock.as_os_str().len() <= MAX_SOCKET_PATH {
            return Ok((Endpoint::Unix(sock), None));
        }
        let dir = std::env::temp_dir().join(format!("estia-llama-{pid}-{}", &random_hex(4)?));
        let sock = dir.join(format!("llama-{pid}.sock"));
        if sock.as_os_str().len() <= MAX_SOCKET_PATH {
            make_private_dir(&dir)?;
            return Ok((Endpoint::Unix(sock), Some(dir)));
        }
    }
    let _ = (run_dir, pid);
    Ok((Endpoint::Tcp(free_loopback_port()?), None))
}

fn free_loopback_port() -> io::Result<std::net::SocketAddr> {
    // Bind port 0, read the port, release it. llama-server binds it a moment
    // later; the API key protects it in the meantime and after.
    let l = std::net::TcpListener::bind("127.0.0.1:0")?;
    l.local_addr()
}

/// Create `dir` (and parents) with mode 0700 where it is created.
pub(crate) fn make_private_dir(dir: &Path) -> io::Result<()> {
    let mut b = fs::DirBuilder::new();
    b.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        b.mode(0o700);
    }
    b.create(dir)
}

pub(crate) fn random_hex(bytes: usize) -> io::Result<String> {
    let mut buf = vec![0u8; bytes];
    getrandom::getrandom(&mut buf).map_err(|e| io::Error::other(format!("no randomness for the API key: {e}")))?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

fn write_private_file(path: &Path, contents: &[u8]) -> io::Result<()> {
    let _ = fs::remove_file(path);
    let mut o = fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    let mut f = o.open(path)?;
    f.write_all(contents)?;
    f.sync_all()
}

fn write_record(path: &Path, v: &Value) -> io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    write_private_file(&tmp, format!("{v}\n").as_bytes())?;
    fs::rename(&tmp, path)
}

/// The last lines llama-server printed, for error messages.
#[derive(Clone, Default)]
pub(crate) struct LogTail(Arc<Mutex<VecDeque<String>>>);

impl LogTail {
    fn push(&self, line: String) {
        let mut q = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if q.len() == LOG_TAIL {
            q.pop_front();
        }
        q.push_back(line);
    }

    /// Error lines if there are any, otherwise the last few lines that are
    /// not part of a crash backtrace; joined for a one-line error message.
    pub(crate) fn summary(&self) -> String {
        let q = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let mut picked: Vec<&str> = q.iter().map(String::as_str).filter(|l| is_error_line(l)).collect();
        if picked.is_empty() {
            picked = q.iter().map(String::as_str).filter(|l| !is_backtrace_line(l)).collect();
        }
        picked[picked.len().saturating_sub(5)..].join(" | ")
    }
}

/// llama.cpp logs errors at level ` E `; a failed assertion or an uncaught
/// exception is printed raw before the process aborts.
fn is_error_line(l: &str) -> bool {
    l.split_whitespace().nth(1) == Some("E")
        || l.contains("GGML_ASSERT")
        || l.contains("GGML_ABORT")
        || l.contains("terminate called")
        || l.contains("terminating due to uncaught exception")
}

/// What an abort prints after the message: `3   libllama.dylib   0x… symbol`
/// frames (macOS), `#3 0x…` frames (Linux), and the notes about them.
fn is_backtrace_line(l: &str) -> bool {
    let mut w = l.split_whitespace();
    let first = w.next().unwrap_or("");
    let frame = first.trim_start_matches('#').parse::<u32>().is_ok() && l.contains("0x");
    frame || l.contains("GGML_BACKTRACE_LLDB") || l.contains("native backtrace") || l.starts_with("See: https://github.com/ggml-org")
}

/// Copy one of llama-server's output pipes to our stderr, line by line.
fn forward_output(pipe: impl Read + Send + 'static, pid: u32, tail: LogTail) {
    let _ = std::thread::Builder::new().name("llama-server-log".into()).spawn(move || {
        let mut r = BufReader::new(pipe);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match r.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let line = String::from_utf8_lossy(&buf).trim_end().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    let _ = writeln!(io::stderr().lock(), "[llama-server {pid}] {line}");
                    tail.push(line);
                }
            }
        }
    });
}

/// Kills llama-server if the adapter dies without stopping it, on systems
/// without `PR_SET_PDEATHSIG` (macOS, the BSDs).
///
/// A `/bin/sh` child that reads its stdin, a pipe only the adapter holds.
/// When the adapter exits by any means, even SIGKILL, the pipe closes and the
/// shell kills the server. Before stopping the server itself, the adapter
/// writes `stop`, and the shell exits without killing anything, so it never
/// signals a pid that has been reaped and reused.
#[cfg_attr(not(all(unix, not(target_os = "linux"))), allow(dead_code))]
struct Guard(Child);

#[cfg(all(unix, not(target_os = "linux")))]
const GUARD_SCRIPT: &str = r#"while IFS= read -r line; do [ "$line" = stop ] && exit 0; done; kill -KILL "$1" 2>/dev/null; exit 0"#;

impl Guard {
    #[cfg(all(unix, not(target_os = "linux")))]
    fn spawn(server_pid: u32) -> Option<Guard> {
        match Command::new("/bin/sh")
            .arg("-c")
            .arg(GUARD_SCRIPT)
            .arg("estia-llama-guard")
            .arg(server_pid.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => Some(Guard(c)),
            Err(e) => {
                log(format_args!("could not start the orphan guard for llama-server {server_pid}: {e}"));
                None
            }
        }
    }

    #[cfg(not(all(unix, not(target_os = "linux"))))]
    fn spawn(_server_pid: u32) -> Option<Guard> {
        None
    }

    fn stand_down(mut self) {
        if let Some(mut stdin) = self.0.stdin.take() {
            let _ = stdin.write_all(b"stop\n");
        }
        wait_or_kill(&mut self.0, Duration::from_secs(1));
    }
}

fn wait_or_kill(child: &mut Child, grace: Duration) {
    let deadline = Instant::now() + grace;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return;
            }
        }
    }
}

/// A started llama-server process and what to clean up after it.
struct Running {
    child: Child,
    guard: Option<Guard>,
    files: Vec<PathBuf>,
    tmp_dir: Option<PathBuf>,
}

impl Running {
    fn cleanup_files(&mut self) {
        for f in self.files.drain(..) {
            let _ = fs::remove_file(f);
        }
        if let Some(d) = self.tmp_dir.take() {
            let _ = fs::remove_dir_all(d);
        }
    }

    fn terminate(mut self, grace: Duration) {
        // The guard first: once the server is reaped its pid may be reused.
        if let Some(g) = self.guard.take() {
            g.stand_down();
        }
        if matches!(self.child.try_wait(), Ok(None)) {
            // SIGTERM lets llama-server shut down cleanly; Windows has no
            // such thing, so it is killed there.
            #[cfg(unix)]
            // SAFETY: plain kill(2) on our own unreaped child.
            unsafe {
                libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM);
            }
            #[cfg(not(unix))]
            let _ = self.child.kill();
        }
        wait_or_kill(&mut self.child, grace);
        self.cleanup_files();
    }
}

/// How a server that was running is doing.
pub(crate) enum Liveness {
    Running,
    /// It exited; the text says how. Its files are cleaned up.
    Exited(String),
    /// Nothing is running (stopped, or never started).
    Gone,
}

/// State the main thread shares with the stdin reader and the watchdog.
#[derive(Default)]
pub(crate) struct Shared {
    running: Mutex<Option<Running>>,
    /// Set when the process is exiting: nothing new may start.
    closing: AtomicBool,
    /// Sequence number of the last request line read from stdin.
    last_read_seq: AtomicU64,
    /// Sequence number of the request a cancel line was meant for.
    cancelled_seq: AtomicU64,
    /// The connection serving request `seq`, so a cancel can shut it down.
    active: Mutex<Option<(u64, Stream)>>,
}

impl Shared {
    fn lock_running(&self) -> MutexGuard<'_, Option<Running>> {
        self.running.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Stop the server, if any, and remove its files.
    pub(crate) fn stop_server(&self, grace: Duration) {
        let r = self.lock_running().take();
        if let Some(r) = r {
            r.terminate(grace);
        }
    }

    /// Stop everything for good: used on exit, from any thread.
    pub(crate) fn close(&self, grace: Duration) {
        self.closing.store(true, Ordering::SeqCst);
        if let Some((_, s)) = self.active.lock().unwrap_or_else(PoisonError::into_inner).take() {
            s.shutdown();
        }
        self.stop_server(grace);
    }

    pub(crate) fn liveness(&self) -> Liveness {
        let mut guard = self.lock_running();
        let Some(r) = guard.as_mut() else { return Liveness::Gone };
        match r.child.try_wait() {
            Ok(None) => Liveness::Running,
            Ok(Some(status)) => {
                let mut r = guard.take().expect("checked above");
                if let Some(g) = r.guard.take() {
                    g.stand_down();
                }
                r.cleanup_files();
                Liveness::Exited(format!("{status}"))
            }
            Err(e) => Liveness::Exited(format!("unknown ({e})")),
        }
    }

    /// The stdin reader numbers each request line.
    pub(crate) fn next_seq(&self) -> u64 {
        self.last_read_seq.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// A cancel line arrived: it is for the request read last. If that
    /// request has a connection open, shut it down now; llama-server notices
    /// the closed connection and cancels its task.
    pub(crate) fn cancel(&self) {
        let seq = self.last_read_seq.load(Ordering::SeqCst);
        self.cancelled_seq.store(seq, Ordering::SeqCst);
        if let Some((s, stream)) = self.active.lock().unwrap_or_else(PoisonError::into_inner).as_ref() {
            if *s == seq {
                stream.shutdown();
            }
        }
    }

    pub(crate) fn is_cancelled(&self, seq: u64) -> bool {
        seq != 0 && self.cancelled_seq.load(Ordering::SeqCst) == seq
    }

    /// Register (or with `None`, forget) the connection serving `seq`.
    pub(crate) fn set_active(&self, seq: u64, stream: Option<Stream>) {
        *self.active.lock().unwrap_or_else(PoisonError::into_inner) = stream.map(|s| (seq, s));
    }
}

/// Why waiting for a server to become ready ended early.
pub(crate) enum WaitError {
    /// The process exited while loading; the text is its last output.
    Exited(String),
    TimedOut,
    /// The tick callback asked to stop (a cancel, or stdout closed). The
    /// server keeps loading.
    Aborted(io::Error),
}

/// The main thread's handle on the running server.
pub(crate) struct Server {
    pub model: PathBuf,
    pub kind: Kind,
    pub endpoint: Endpoint,
    pub key: String,
    pub pid: u32,
    pub ready: bool,
    pub log: LogTail,
    /// Embedding models: the special tokens `add_special` adds to empty text,
    /// used to cut long inputs without losing them.
    pub specials: Option<Vec<i64>>,
    spawned: Instant,
    key_file: PathBuf,
}

impl Server {
    /// Start llama-server for `model`. It is loading when this returns; call
    /// [`Server::wait_ready`].
    pub(crate) fn start(opts: &AdapterOptions, shared: &Shared, model: &Path, kind: Kind) -> Result<Server, String> {
        let pid = std::process::id();
        let run_dir = std::path::absolute(&opts.run_dir).unwrap_or_else(|_| opts.run_dir.clone());
        make_private_dir(&run_dir).map_err(|e| format!("cannot create run dir {}: {e}", run_dir.display()))?;
        let (endpoint, tmp_dir) = choose_endpoint(&run_dir, pid).map_err(|e| format!("cannot pick a socket: {e}"))?;
        #[cfg(unix)]
        if let Endpoint::Unix(sock) = &endpoint {
            // llama-server leaves its socket behind on exit, and would fail
            // to bind over it.
            let _ = fs::remove_file(sock);
        }
        let key = random_hex(32).map_err(|e| e.to_string())?;
        let key_file = run_dir.join(format!("llama-{pid}.key"));
        write_private_file(&key_file, format!("{key}\n").as_bytes()).map_err(|e| format!("cannot write {}: {e}", key_file.display()))?;
        let record = run_dir.join(format!("llama-{pid}.json"));

        let mut cmd = Command::new(&opts.server_bin);
        cmd.args(server_args(model, kind, &endpoint, &key_file, opts.context_length, &opts.extra_args))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // llama-server reads LLAMA_ARG_* variables as flags, and LLAMA_API_KEY
        // as one more accepted key. The flags above are the whole
        // configuration; operators use `-- <args>`. HF_TOKEN is only for
        // downloads, which --offline turns off: no reason to hand it over.
        for (k, _) in std::env::vars_os() {
            if env_is_dropped(&k.to_string_lossy()) {
                cmd.env_remove(k);
            }
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            // SAFETY: only async-signal-safe calls between fork and exec.
            let parent = unsafe { libc::getpid() };
            unsafe {
                cmd.pre_exec(move || {
                    // Killed when the thread that started it exits. The
                    // adapter starts servers only from its request thread,
                    // which lives as long as the adapter.
                    if libc::prctl(
                        libc::PR_SET_PDEATHSIG,
                        libc::SIGKILL as libc::c_ulong,
                        0 as libc::c_ulong,
                        0 as libc::c_ulong,
                        0 as libc::c_ulong,
                    ) != 0
                    {
                        return Err(io::Error::last_os_error());
                    }
                    if libc::getppid() != parent {
                        return Err(io::Error::from_raw_os_error(libc::ESRCH));
                    }
                    Ok(())
                });
            }
        }

        let cleanup_early = |tmp: &Option<PathBuf>| {
            let _ = fs::remove_file(&key_file);
            if let Some(d) = tmp {
                let _ = fs::remove_dir_all(d);
            }
        };
        // Spawn under the lock, so the watchdog either sees the child or has
        // already closed and the child is never started.
        let mut running = shared.lock_running();
        if shared.closing.load(Ordering::SeqCst) {
            cleanup_early(&tmp_dir);
            return Err("the adapter is shutting down".into());
        }
        let spawned = Instant::now();
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                cleanup_early(&tmp_dir);
                return Err(format!("cannot start {}: {e}", opts.server_bin.display()));
            }
        };
        let server_pid = child.id();
        let guard = Guard::spawn(server_pid);
        let log_tail = LogTail::default();
        if let Some(out) = child.stdout.take() {
            forward_output(out, server_pid, log_tail.clone());
        }
        if let Some(err) = child.stderr.take() {
            forward_output(err, server_pid, log_tail.clone());
        }
        let (socket, addr) = match &endpoint {
            #[cfg(unix)]
            Endpoint::Unix(p) => (json!(p.to_string_lossy()), Value::Null),
            Endpoint::Tcp(a) => (Value::Null, json!(a.to_string())),
        };
        let rec = json!({
            "adapter_pid": pid,
            "server_pid": server_pid,
            "socket": socket,
            "addr": addr,
            "model": model.to_string_lossy(),
            "kind": kind.as_str(),
        });
        if let Err(e) = write_record(&record, &rec) {
            log(format_args!("cannot write {}: {e}", record.display()));
        }
        let files = match &endpoint {
            #[cfg(unix)]
            Endpoint::Unix(sock) => vec![key_file.clone(), record, sock.clone()],
            Endpoint::Tcp(_) => vec![key_file.clone(), record],
        };
        *running = Some(Running { child, guard, files, tmp_dir });
        drop(running);
        log(format_args!("started llama-server {server_pid} for {} ({}) on {}", model.display(), kind.as_str(), endpoint.describe()));
        Ok(Server {
            model: model.to_path_buf(),
            kind,
            endpoint,
            key,
            pid: server_pid,
            ready: false,
            log: log_tail,
            specials: None,
            spawned,
            key_file,
        })
    }

    /// Poll `/health` until the model is loaded. `tick` runs between polls
    /// and may abort the wait. Returns the milliseconds from spawn to ready.
    pub(crate) fn wait_ready(&mut self, shared: &Shared, tick: &mut dyn FnMut() -> io::Result<()>) -> Result<u64, WaitError> {
        let mut pause = Duration::from_millis(10);
        loop {
            match shared.liveness() {
                Liveness::Running => {}
                Liveness::Exited(how) => {
                    // Give the log threads a moment to catch the last lines.
                    std::thread::sleep(Duration::from_millis(100));
                    let tail = self.log.summary();
                    return Err(WaitError::Exited(if tail.is_empty() { how } else { format!("{how}: {tail}") }));
                }
                Liveness::Gone => return Err(WaitError::Exited("stopped".into())),
            }
            if health(&self.endpoint) == Some(200) {
                self.ready = true;
                // It has read the key; the file is no longer needed.
                let _ = fs::remove_file(&self.key_file);
                return Ok(self.spawned.elapsed().as_millis() as u64);
            }
            if self.spawned.elapsed() > LOAD_TIMEOUT {
                return Err(WaitError::TimedOut);
            }
            tick().map_err(WaitError::Aborted)?;
            std::thread::sleep(pause);
            pause = (pause * 2).min(Duration::from_millis(100));
        }
    }
}

/// `GET /health`: the status code, or `None` when nothing answers.
/// llama-server answers 503 while loading and 200 when ready; it needs no key.
pub(crate) fn health(endpoint: &Endpoint) -> Option<u16> {
    let stream = endpoint.connect().ok()?;
    let conn = http::send(stream, &http::Request { method: "GET", path: "/health", key: None, body: None }).ok()?;
    // A healthy server answers at once; give up after about five seconds.
    let mut ticks = 0;
    let mut idle = || {
        ticks += 1;
        if ticks > 25 {
            Err(io::Error::new(io::ErrorKind::TimedOut, "no answer"))
        } else {
            Ok(())
        }
    };
    let mut resp = conn.read_head(&mut idle).ok()?;
    let _ = resp.read_all(&mut idle, 64 * 1024);
    Some(resp.head.status)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strs(v: &[OsString]) -> Vec<String> {
        v.iter().map(|s| s.to_string_lossy().into_owned()).collect()
    }

    #[cfg(unix)]
    #[test]
    fn generation_flags() {
        let a = server_args(
            Path::new("/m/model.gguf"),
            Kind::Generation,
            &Endpoint::Unix("/r/llama-1.sock".into()),
            Path::new("/r/llama-1.key"),
            None,
            &["-ngl".into(), "0".into()],
        );
        assert_eq!(
            strs(&a),
            [
                "-m",
                "/m/model.gguf",
                "--host",
                "/r/llama-1.sock",
                "--api-key-file",
                "/r/llama-1.key",
                "--no-ui",
                "--no-slots",
                "--offline",
                "--cache-ram",
                "0",
                "-np",
                "1",
                "-c",
                "8192",
                "-ngl",
                "0"
            ]
        );
        let a = server_args(Path::new("/m.gguf"), Kind::Generation, &Endpoint::Unix("/s".into()), Path::new("/k"), Some(2048), &[]);
        let s = strs(&a);
        assert_eq!(&s[s.len() - 2..], ["-c", "2048"]);
        assert!(!s.contains(&"--embedding".to_string()));
    }

    #[test]
    fn embedding_flags_and_tcp() {
        let a = server_args(
            Path::new("/e.gguf"),
            Kind::Embedding,
            &Endpoint::Tcp("127.0.0.1:4567".parse().unwrap()),
            Path::new("/k"),
            Some(99),
            &[],
        );
        let s = strs(&a).join(" ");
        assert!(s.contains("--host 127.0.0.1 --port 4567"), "{s}");
        assert!(s.contains("--embedding -np 4 -c 2048 -b 2048 -ub 2048"), "{s}");
        // --ctx is for generation models only; no pooling override.
        assert!(!s.contains("-c 99") && !s.contains("--pooling"), "{s}");
    }

    #[cfg(unix)]
    #[test]
    fn socket_path_falls_back_when_too_long() {
        let (ep, tmp) = choose_endpoint(Path::new("/tmp/r"), 42).unwrap();
        assert_eq!(ep, Endpoint::Unix("/tmp/r/llama-42.sock".into()));
        assert!(tmp.is_none());
        let deep = PathBuf::from(format!("/tmp/{}", "d".repeat(120)));
        let (ep, tmp) = choose_endpoint(&deep, 42).unwrap();
        let tmp = tmp.expect("a private temp dir");
        match &ep {
            Endpoint::Unix(p) => assert!(p.starts_with(&tmp) && p.as_os_str().len() <= MAX_SOCKET_PATH),
            other => panic!("{other:?}"),
        }
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(fs::metadata(&tmp).unwrap().permissions().mode() & 0o777, 0o700);
        fs::remove_dir_all(tmp).unwrap();
    }

    #[test]
    fn model_paths_resolve() {
        let dir = std::env::temp_dir().join(format!("estia-llama-test-{}", random_hex(6).unwrap()));
        fs::create_dir_all(&dir).unwrap();
        assert!(resolve_model(dir.to_str().unwrap()).unwrap_err().contains("no model.gguf"));
        let f = dir.join("model.gguf");
        fs::write(&f, b"GGUF").unwrap();
        assert_eq!(resolve_model(dir.to_str().unwrap()).unwrap(), f);
        assert_eq!(resolve_model(f.to_str().unwrap()).unwrap(), f);
        assert!(resolve_model("/nonexistent/x.gguf").unwrap_err().contains("not found"));
        assert!(resolve_model(" ").is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn key_file_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let p = std::env::temp_dir().join(format!("estia-llama-key-{}", random_hex(6).unwrap()));
        write_private_file(&p, b"old").unwrap();
        write_private_file(&p, b"new").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"new");
        assert_eq!(fs::metadata(&p).unwrap().permissions().mode() & 0o777, 0o600);
        fs::remove_file(p).unwrap();
        assert_eq!(random_hex(32).unwrap().len(), 64);
    }

    #[test]
    fn configuration_from_the_environment_is_dropped() {
        for k in ["LLAMA_ARG_PORT", "LLAMA_ARG_CTX_SIZE", "LLAMA_API_KEY", "HF_TOKEN"] {
            assert!(env_is_dropped(k), "{k}");
        }
        for k in ["PATH", "HOME", "TMPDIR", "MTMD_BACKEND_DEVICE", "CUDA_VISIBLE_DEVICES", "LLAMA_ARGX"] {
            assert!(!env_is_dropped(k), "{k}");
        }
    }

    #[test]
    fn log_summary_prefers_error_lines() {
        let t = LogTail::default();
        for i in 0..70 {
            t.push(format!("0.00.{i:03} I srv line {i}"));
        }
        assert!(t.summary().ends_with("line 69"));
        assert!(!t.summary().contains("line 64"));
        t.push("0.01.000 E llama_model_load: error loading model".into());
        t.push("0.01.001 I srv cleaning up".into());
        assert_eq!(t.summary(), "0.01.000 E llama_model_load: error loading model");
    }

    #[test]
    fn log_summary_skips_crash_backtraces() {
        // What b11146 prints when an encoder-only model is started as a chat
        // server (macOS), and a Linux-style frame.
        let t = LogTail::default();
        for l in [
            "0.00.165.732 I cmn          init: llama threadpool init, n_threads = 8",
            "/Users/runner/work/llama.cpp/llama.cpp/src/llama-context.cpp:2260: GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max) failed",
            "WARNING: Using native backtrace. Set GGML_BACKTRACE_LLDB for more info.",
            "See: https://github.com/ggml-org/llama.cpp/pull/17869",
            "0   libggml-base.0.25.1.dylib           0x00000001029553f0 ggml_print_backtrace + 276",
            "9   dyld                                0x0000000190a9eb98 start + 6076",
        ] {
            t.push(l.into());
        }
        assert!(
            t.summary().starts_with("/Users/runner/work/llama.cpp/llama.cpp/src/llama-context.cpp:2260: GGML_ASSERT("),
            "{}",
            t.summary()
        );
        assert!(!t.summary().contains('|'), "{}", t.summary());

        let t = LogTail::default();
        for l in
            ["0.00.1 I srv loading", "0.00.2 W something odd", "#0  0x00007f0000001000 in abort ()", "#1 0x00007f0000002000 in main ()"]
        {
            t.push(l.into());
        }
        assert_eq!(t.summary(), "0.00.1 I srv loading | 0.00.2 W something odd");
    }

    #[test]
    fn cancel_targets_the_request_read_last() {
        let s = Shared::default();
        let first = s.next_seq();
        assert!(!s.is_cancelled(first));
        s.cancel();
        assert!(s.is_cancelled(first));
        // The next request starts clean, whatever came before it.
        let second = s.next_seq();
        assert!(!s.is_cancelled(second));
        assert!(!s.is_cancelled(0));
    }
}
