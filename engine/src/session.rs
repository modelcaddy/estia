//! A resident runner process and the one safe way to talk to it.
//!
//! A [`Session`] owns a child process that speaks the line protocol in
//! [`estia_proto`]: one JSON request per line on its stdin, one JSON
//! response per line on its stdout (several lines for a stream). The model
//! stays loaded inside the child across calls, which is the whole point —
//! warm is the normal path.
//!
//! What this type guarantees, and why each guarantee exists:
//!
//! - **Per-call deadline.** A child that wedges (reads the request, never
//!   writes) would otherwise block `read_line` forever while holding the
//!   process lock and freeze every caller. A persistent reader thread forwards
//!   lines over a channel so the wait is bounded with `recv_timeout`; on
//!   overrun the child is killed and the call fails.
//! - **Bounded respawn.** A broken pipe or EOF means the child died. The call
//!   respawns it once and retries; a second failure propagates. Never a loop.
//! - **Streams do not retry.** Once a token has reached the caller, a retry
//!   would duplicate it. A stream fails after a write failure (nothing sent
//!   yet, so it respawns once) or surfaces an error mid-stream, never re-runs.
//! - **Priority.** Calls queue through a gate that serves every waiting
//!   [`Priority::Interactive`] call before any [`Priority::Background`] one,
//!   first come first served within a class. A background batch cannot
//!   starve a chat turn; a call already running is never
//!   interrupted, the next slot just goes to the interactive caller.
//! - **Cancel.** A streaming call takes a [`CancelToken`]; flipping it writes
//!   `{"type":"cancel"}` to the child and the stream ends with
//!   [`SessionError::Cancelled`] carrying the partial text. An older runner
//!   that ignores the request finishes the generation and answers the cancel
//!   with an error line, which is drained so the next call is not desynced.
//! - **In-flight gate.** Idle shutdown never kills a child mid-call; the
//!   counter is re-checked under the process lock to close the race.
//! - **Drop reaps.** `std::process::Child` neither kills nor waits on drop.
//!   Dropping the process record kills the child, waits, and joins the reader.
//!
//! Serialized: one call at a time per child. This type decides *which* call
//! goes next (the gate) and makes each call safe; it does not decide what the
//! model does.

use crate::error::SessionError;
use estia_proto as proto;
use estia_proto::{Request, StreamEvent};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::any::Any;
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, SessionError>;

/// How to start (and restart) the runner process.
#[derive(Debug, Clone)]
pub struct Launch {
    program: PathBuf,
    args: Vec<OsString>,
}

impl Launch {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self { program: program.into(), args: Vec::new() }
    }

    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn program(&self) -> &Path {
        &self.program
    }

    pub fn args(&self) -> &[OsString] {
        &self.args
    }

    pub(crate) fn command(&self) -> Command {
        let mut c = Command::new(&self.program);
        c.args(&self.args);
        c
    }
}

#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Deadline for one response line. Generous by default: a large embed
    /// batch or a long generation must never false-trip (300 s covers roughly
    /// 20k embed units at ~15 ms each). For streams this is a *silence*
    /// deadline: every line resets it.
    pub call_timeout: Duration,
    /// How many times a call may respawn a dead child and retry. One.
    pub respawn_retries: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self { call_timeout: Duration::from_secs(300), respawn_retries: 1 }
    }
}

/// Who goes first when calls queue on one child. Interactive work — a chat
/// turn someone is watching — is served before background work such as a
/// bulk summarisation batch. Never preemptive: a running call finishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Priority {
    Interactive,
    #[default]
    Background,
}

/// Cooperative cancellation for a streaming call. Clone it, hand one copy to
/// the call, keep the other, and flip it from anywhere.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Everything a finished stream hands back.
#[derive(Debug, Clone, Default)]
pub struct StreamOutcome {
    pub text: String,
    /// The runner's `{"type":"meta"}` line, when it sent one.
    pub meta: Option<Value>,
}

/// Hooks for whoever wants to watch the session: a stats panel, a log, a
/// memory watcher. Every method has a no-op default.
pub trait SessionObserver: Send + Sync {
    /// A child process was started (first spawn or a respawn).
    fn on_spawn(&self, _pid: u32) {}
    /// A call is starting. Whatever is returned is held until the call ends
    /// and then dropped — the natural place for an RAII "in flight" counter.
    fn on_call_start(&self) -> Option<Box<dyn Any + Send>> {
        None
    }
    /// A call finished (successfully or not).
    fn on_call_end(&self, _elapsed: Duration) {}
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopObserver;
impl SessionObserver for NoopObserver {}

/// Shared write handle to the child's stdin: calls write requests through it
/// under the process lock; `cancel` writes through it without.
type SharedStdin = Arc<Mutex<ChildStdin>>;

/// The live child, its write pipe, and the reader thread forwarding its
/// stdout line by line.
struct Proc {
    child: Child,
    stdin: SharedStdin,
    line_rx: Receiver<String>,
    reader: Option<JoinHandle<()>>,
}

impl Drop for Proc {
    fn drop(&mut self) {
        // Kill → stdout closes → the reader's read_line hits EOF → join.
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(h) = self.reader.take() {
            let _ = h.join();
        }
    }
}

/// Decrements the in-flight counter on scope exit, including `?` returns and
/// unwinds, so a failed call can never leave the gate stuck above zero.
struct InFlightGuard<'a>(&'a AtomicUsize);
impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Everything one call keeps alive until it returns.
struct CallScope<'a> {
    _in_flight: InFlightGuard<'a>,
    _observer_token: Option<Box<dyn Any + Send>>,
    started: Instant,
    observer: &'a dyn SessionObserver,
}

impl Drop for CallScope<'_> {
    fn drop(&mut self) {
        self.observer.on_call_end(self.started.elapsed());
    }
}

// ── Priority gate ─────────────────────────────────────────────────────────────

struct GateState {
    busy: bool,
    next_ticket: u64,
    /// Tickets waiting, per priority class (index = `Priority as usize`).
    waiting: [BTreeSet<u64>; 2],
}

struct Gate {
    state: Mutex<GateState>,
    cv: Condvar,
}

impl Gate {
    fn new() -> Self {
        Self {
            state: Mutex::new(GateState { busy: false, next_ticket: 0, waiting: [BTreeSet::new(), BTreeSet::new()] }),
            cv: Condvar::new(),
        }
    }

    /// Wait until this caller is the one to run: nothing running, no
    /// interactive caller waiting ahead (for a background caller), and first
    /// in line within its own class.
    fn acquire(&self, prio: Priority) -> GateGuard<'_> {
        let class = prio as usize;
        let mut st = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let ticket = st.next_ticket;
        st.next_ticket += 1;
        st.waiting[class].insert(ticket);
        loop {
            let first_in_class = st.waiting[class].iter().next() == Some(&ticket);
            let no_interactive_ahead = prio == Priority::Interactive || st.waiting[Priority::Interactive as usize].is_empty();
            if !st.busy && first_in_class && no_interactive_ahead {
                st.busy = true;
                st.waiting[class].remove(&ticket);
                return GateGuard { gate: self };
            }
            st = self.cv.wait(st).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn waiting(&self, prio: Priority) -> usize {
        self.state.lock().unwrap_or_else(PoisonError::into_inner).waiting[prio as usize].len()
    }
}

struct GateGuard<'a> {
    gate: &'a Gate,
}

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        self.gate.state.lock().unwrap_or_else(PoisonError::into_inner).busy = false;
        self.gate.cv.notify_all();
    }
}

/// A long-lived runner process: model loaded once, kept resident across calls.
///
/// `Send + Sync`: every field is, so a `Session` can sit behind an `Arc` or a
/// `Mutex<Option<Session>>` and be borrowed by adapters that must be `Sync`.
pub struct Session {
    launch: Launch,
    cfg: SessionConfig,
    observer: Arc<dyn SessionObserver>,
    proc: Mutex<Proc>,
    /// The current child's stdin, reachable without the process lock so a
    /// cancel can be written while a stream holds it.
    stdin: Mutex<SharedStdin>,
    gate: Gate,
    in_flight: AtomicUsize,
    last_used: Mutex<Instant>,
}

impl Session {
    /// Start the runner. The child's stderr is inherited — runners route their
    /// own chatter there to keep the stdout line protocol clean. Whatever the
    /// runner loads lazily (a model, typically) loads on its first request,
    /// not here, so spawning is fast.
    pub fn spawn(launch: Launch, cfg: SessionConfig, observer: Arc<dyn SessionObserver>) -> Result<Self> {
        let proc = Self::spawn_proc(&launch, observer.as_ref())?;
        let stdin = Arc::clone(&proc.stdin);
        Ok(Self {
            launch,
            cfg,
            observer,
            proc: Mutex::new(proc),
            stdin: Mutex::new(stdin),
            gate: Gate::new(),
            in_flight: AtomicUsize::new(0),
            last_used: Mutex::new(Instant::now()),
        })
    }

    pub fn launch(&self) -> &Launch {
        &self.launch
    }

    pub fn config(&self) -> &SessionConfig {
        &self.cfg
    }

    /// Change the per-call deadline. Tests use this to avoid waiting the
    /// production value; a host may use it for a known-slow runner.
    pub fn set_call_timeout(&mut self, timeout: Duration) {
        self.cfg.call_timeout = timeout;
    }

    /// Pid of the current child (it changes on respawn).
    pub fn pid(&self) -> u32 {
        self.lock_proc().child.id()
    }

    /// Callers queued at `prio` right now (not counting the one running).
    pub fn waiting(&self, prio: Priority) -> usize {
        self.gate.waiting(prio)
    }

    fn spawn_proc(launch: &Launch, observer: &dyn SessionObserver) -> Result<Proc> {
        let mut child = launch
            .command()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| SessionError::Spawn { program: launch.program.display().to_string(), source: e })?;
        let stdin = child.stdin.take().ok_or(SessionError::NotPiped)?;
        let stdout = child.stdout.take().ok_or(SessionError::NotPiped)?;
        // Persistent reader: forward each line so calls can bound their wait.
        // Exits on EOF (child gone) or when the receiver is dropped (respawn
        // or teardown replaced this Proc).
        let (line_tx, line_rx) = mpsc::channel::<String>();
        let reader = std::thread::spawn(move || {
            let mut r = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match r.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if line_tx.send(std::mem::take(&mut line)).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        observer.on_spawn(child.id());
        Ok(Proc { child, stdin: Arc::new(Mutex::new(stdin)), line_rx, reader: Some(reader) })
    }

    fn lock_proc(&self) -> MutexGuard<'_, Proc> {
        // A panic while holding the lock must not brick every later call.
        self.proc.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Replace the child (after a death) and publish its stdin for cancels.
    fn respawn_into(&self, p: &mut Proc, cause: &SessionError) -> Result<()> {
        let fresh = Self::spawn_proc(&self.launch, self.observer.as_ref())
            .map_err(|e| SessionError::Respawn { cause: cause.to_string(), source: Box::new(e) })?;
        *self.stdin.lock().unwrap_or_else(PoisonError::into_inner) = Arc::clone(&fresh.stdin);
        *p = fresh;
        Ok(())
    }

    fn begin_call(&self) -> CallScope<'_> {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        CallScope {
            _in_flight: InFlightGuard(&self.in_flight),
            _observer_token: self.observer.on_call_start(),
            started: Instant::now(),
            observer: self.observer.as_ref(),
        }
    }

    fn touch(&self) {
        *self.last_used.lock().unwrap_or_else(PoisonError::into_inner) = Instant::now();
    }

    /// One request, one response line, at background priority. Counted as
    /// work: the in-flight gate is held, the observer is told, and
    /// `last_used` advances on success.
    pub fn call<R: Serialize + ?Sized>(&self, req: &R) -> Result<Value> {
        self.call_with(req, Priority::Background)
    }

    /// [`Session::call`] at an explicit priority.
    pub fn call_with<R: Serialize + ?Sized>(&self, req: &R, prio: Priority) -> Result<Value> {
        let _scope = self.begin_call();
        let _slot = self.gate.acquire(prio);
        let v = self.call_inner(req)?;
        self.touch();
        Ok(v)
    }

    /// [`Session::call`] followed by a typed parse of the payload.
    pub fn call_typed<R: Serialize + ?Sized, T: DeserializeOwned>(&self, req: &R) -> Result<T> {
        self.call_typed_with(req, Priority::Background)
    }

    pub fn call_typed_with<R: Serialize + ?Sized, T: DeserializeOwned>(&self, req: &R, prio: Priority) -> Result<T> {
        let v = self.call_with(req, prio)?;
        serde_json::from_value(v).map_err(SessionError::Parse)
    }

    /// One request, one response line, **not** counted as work: no in-flight
    /// gate, no observer, `last_used` untouched. For health checks, so a ping
    /// never keeps an idle child alive or shows up as a model call. Still
    /// queues through the gate at interactive priority so it cannot interleave
    /// with a call in progress.
    pub fn call_unobserved<R: Serialize + ?Sized>(&self, req: &R) -> Result<Value> {
        let _slot = self.gate.acquire(Priority::Interactive);
        self.call_inner(req)
    }

    /// Protocol v2 handshake. `Ok(Some)` for a v2 runner, `Ok(None)` for a v1
    /// runner (it answers `hello` with an unknown-type error), `Err` when the
    /// child cannot be talked to at all. Not counted as work.
    pub fn hello(&self) -> Result<Option<proto::HelloResp>> {
        match self.call_unobserved(&Request::Hello) {
            Ok(v) => serde_json::from_value(v).map(Some).map_err(SessionError::Parse),
            Err(SessionError::Runner(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn call_inner<R: Serialize + ?Sized>(&self, req: &R) -> Result<Value> {
        let line = serde_json::to_string(req).map_err(SessionError::Serialize)?;
        let timeout = self.cfg.call_timeout;
        let mut p = self.lock_proc();

        let mut last_err = match Self::exchange(&mut p, &line, timeout) {
            Ok(resp) => return Self::finish(&resp),
            Err(e) => e,
        };
        for _ in 0..self.cfg.respawn_retries {
            self.respawn_into(&mut p, &last_err)?;
            match Self::exchange(&mut p, &line, timeout) {
                Ok(resp) => return Self::finish(&resp),
                Err(e) => last_err = e,
            }
        }
        Err(SessionError::AfterRespawn(Box::new(last_err)))
    }

    fn finish(resp: &str) -> Result<Value> {
        Ok(proto::check_error(resp)?)
    }

    fn write_line(stdin: &SharedStdin, line: &str) -> Result<()> {
        let mut w = stdin.lock().unwrap_or_else(PoisonError::into_inner);
        w.write_all(line.as_bytes()).and_then(|_| w.write_all(b"\n")).and_then(|_| w.flush()).map_err(SessionError::Write)
    }

    /// Write one request line, wait up to `timeout` for one response line. On
    /// timeout the child is killed (so the caller's retry respawns it).
    fn exchange(p: &mut Proc, line: &str, timeout: Duration) -> Result<String> {
        Self::write_line(&p.stdin, line)?;
        match p.line_rx.recv_timeout(timeout) {
            Ok(resp) => Ok(resp),
            Err(RecvTimeoutError::Timeout) => {
                let _ = p.child.kill();
                let _ = p.child.wait();
                Err(SessionError::Timeout { secs: timeout.as_secs() })
            }
            Err(RecvTimeoutError::Disconnected) => Err(SessionError::Eof),
        }
    }

    /// Ask the child to stop the generation in flight. Safe to call from any
    /// thread at any time; a runner that is idle ignores it.
    fn send_cancel(&self) -> Result<()> {
        let stdin = Arc::clone(&self.stdin.lock().unwrap_or_else(PoisonError::into_inner));
        let line = serde_json::to_string(&Request::Cancel).map_err(SessionError::Serialize)?;
        Self::write_line(&stdin, &line)
    }

    /// Streaming call at background priority with no cancel token.
    pub fn stream<R, F>(&self, req: &R, on_token: F) -> Result<String>
    where
        R: Serialize + ?Sized,
        F: FnMut(&str),
    {
        self.stream_with(req, Priority::Background, None, on_token)
    }

    /// Streaming call: `on_token` fires for every non-empty token line as it
    /// arrives; the assembled text is returned at the end.
    ///
    /// A runner that cannot stream answers with a single `{"text":…}` line,
    /// surfaced as one `on_token` call, so callers never regress below the
    /// one-shot behaviour. The deadline is per line (silence), not per stream.
    /// A write failure respawns once (nothing was streamed yet); a failure
    /// mid-stream does not retry, because the caller may already have
    /// forwarded partial tokens. Flipping `cancel` sends a cancel to the child
    /// and ends the call with [`SessionError::Cancelled`] once the runner
    /// acknowledges (or, for a runner that ignores cancels, when the
    /// generation finishes anyway).
    pub fn stream_with<R, F>(&self, req: &R, prio: Priority, cancel: Option<&CancelToken>, on_token: F) -> Result<String>
    where
        R: Serialize + ?Sized,
        F: FnMut(&str),
    {
        self.stream_full(req, prio, cancel, on_token).map(|o| o.text)
    }

    /// [`Session::stream_with`] that also returns the runner's `meta` line.
    pub fn stream_full<R, F>(&self, req: &R, prio: Priority, cancel: Option<&CancelToken>, mut on_token: F) -> Result<StreamOutcome>
    where
        R: Serialize + ?Sized,
        F: FnMut(&str),
    {
        let _scope = self.begin_call();
        let _slot = self.gate.acquire(prio);
        let line = serde_json::to_string(req).map_err(SessionError::Serialize)?;
        let timeout = self.cfg.call_timeout;
        let mut p = self.lock_proc();

        if let Err(first) = Self::write_line(&p.stdin, &line) {
            self.respawn_into(&mut p, &first)?;
            Self::write_line(&p.stdin, &line)?;
        }

        // Poll in short ticks so a cancel is noticed within a tick even while
        // the runner is silent; the silence deadline is tracked separately.
        let tick = timeout.min(Duration::from_millis(250));
        let mut last_line = Instant::now();
        let mut cancel_sent = false;
        let mut full = String::new();
        let mut meta: Option<Value> = None;
        let outcome: Result<bool> = loop {
            if !cancel_sent && cancel.map(CancelToken::is_cancelled).unwrap_or(false) {
                cancel_sent = true;
                if let Err(e) = self.send_cancel() {
                    break Err(e);
                }
            }
            match p.line_rx.recv_timeout(tick) {
                Ok(raw) => {
                    last_line = Instant::now();
                    match proto::parse_stream_line(&raw) {
                        None | Some(StreamEvent::Other) | Some(StreamEvent::Keepalive) => continue,
                        Some(StreamEvent::Meta(v)) => meta = Some(v),
                        Some(StreamEvent::Error(e)) => break Err(SessionError::Runner(e)),
                        Some(StreamEvent::Token(t)) => {
                            if !t.is_empty() {
                                full.push_str(&t);
                                on_token(&t);
                            }
                        }
                        Some(StreamEvent::Cancelled) => break Ok(true),
                        Some(StreamEvent::Done) => break Ok(false),
                        Some(StreamEvent::Final(t)) => {
                            if full.is_empty() && !t.is_empty() {
                                full.push_str(&t);
                                on_token(&t);
                            }
                            break Ok(false);
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if last_line.elapsed() >= timeout {
                        let _ = p.child.kill();
                        let _ = p.child.wait();
                        break Err(SessionError::StreamSilence { secs: timeout.as_secs() });
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break Err(SessionError::StreamEof),
            }
        };

        let acknowledged = outcome?;
        if cancel_sent && !acknowledged {
            // The runner finished on its own after we asked it to stop: an
            // older runner that reads the cancel as an unknown request and
            // answers it with an error line. Drain that line so the next
            // call's response is its own.
            let _ = p.line_rx.recv_timeout(Duration::from_millis(500));
        }
        drop(p);
        self.touch();
        if cancel_sent {
            return Err(SessionError::Cancelled { partial: full });
        }
        Ok(StreamOutcome { text: full, meta })
    }

    /// Calls currently inside the session (0 ⇒ safe to tear down).
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// True when nothing is in flight and the last counted call finished at
    /// least `timeout` ago.
    pub fn is_idle(&self, timeout: Duration) -> bool {
        self.in_flight() == 0 && self.last_used.lock().unwrap_or_else(PoisonError::into_inner).elapsed() >= timeout
    }

    /// In-flight-gated soft teardown: if idle past `timeout`, kill the child in
    /// place (freeing its memory) and return `true`. The next call transparently
    /// respawns it through the broken-pipe path. Returns `false` while a call
    /// is in flight or before `timeout`.
    ///
    /// The counter is re-read under the process lock to close the race where a
    /// call increments it after the first check but before locking; even if
    /// that slips through, the killed pipe only triggers a respawn on that
    /// call, so correctness holds either way.
    pub fn maybe_shutdown(&self, timeout: Duration) -> bool {
        if !self.is_idle(timeout) {
            return false;
        }
        let mut p = self.lock_proc();
        if self.in_flight() != 0 {
            return false;
        }
        let _ = p.child.kill();
        let _ = p.child.wait();
        true
    }
}

#[cfg(test)]
mod bounds {
    use super::*;
    fn _assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    fn _session_is_send_sync() {
        _assert_send_sync::<Session>();
        _assert_send_sync::<CancelToken>();
    }
}
