//! One request per process.
//!
//! The older runners — `oneshot-runner.py` and both Swift runners —
//! read stdin to EOF, answer with a single `{"ok": …}` envelope (or stream
//! token lines then `{"ok":true,"done":true}`), and exit. Every call pays a
//! process start and a model load, which is why [`crate::Session`] exists;
//! this stays as the fallback path and for runners that cannot stay resident
//! (Apple Foundation Models today).
//!
//! Guarantees: a whole-call deadline (the child is killed on overrun), stdout
//! and stderr drained on threads so a chatty runner cannot deadlock on a full
//! pipe, a failure exit reported with the runner's stderr, and the same
//! stream-line classification as the resident path.

use crate::error::SessionError;
use crate::session::{Launch, SessionObserver};
use estia_proto as proto;
use estia_proto::StreamEvent;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ExitStatus, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

type Result<T> = std::result::Result<T, SessionError>;

#[derive(Debug, Clone)]
pub struct OneShotConfig {
    /// Hard ceiling on the whole call: process start, model load, generation.
    /// A wedged child (model-load deadlock, OOM thrash, never emitting output)
    /// must not strand the caller forever.
    pub call_timeout: Duration,
}

impl Default for OneShotConfig {
    fn default() -> Self {
        Self { call_timeout: Duration::from_secs(300) }
    }
}

/// A runner invoked once per request.
pub struct OneShot {
    launch: Launch,
    cfg: OneShotConfig,
    observer: Arc<dyn SessionObserver>,
}

impl OneShot {
    pub fn new(launch: Launch, cfg: OneShotConfig, observer: Arc<dyn SessionObserver>) -> Self {
        Self { launch, cfg, observer }
    }

    pub fn launch(&self) -> &Launch {
        &self.launch
    }

    /// Spawn, send the whole request, read one envelope back.
    pub fn call<R: Serialize + ?Sized>(&self, req: &R) -> Result<Value> {
        let body = serde_json::to_vec(req).map_err(SessionError::Serialize)?;
        let started = Instant::now();
        let _token = self.observer.on_call_start();
        let result = self.call_inner(&body);
        self.observer.on_call_end(started.elapsed());
        result
    }

    /// [`OneShot::call`] followed by a typed parse of the payload.
    pub fn call_typed<R: Serialize + ?Sized, T: DeserializeOwned>(&self, req: &R) -> Result<T> {
        let v = self.call(req)?;
        serde_json::from_value(v).map_err(SessionError::Parse)
    }

    /// Spawn, send the request, forward token lines as they arrive, return the
    /// assembled text once the child closes stdout. A runner that cannot
    /// stream answers with one `{"ok":true,"text":…}` envelope, surfaced as a
    /// single `on_token` call.
    pub fn stream<R, F>(&self, req: &R, on_token: F) -> Result<String>
    where
        R: Serialize + ?Sized,
        F: FnMut(&str),
    {
        let body = serde_json::to_vec(req).map_err(SessionError::Serialize)?;
        let started = Instant::now();
        let _token = self.observer.on_call_start();
        let result = self.stream_inner(&body, on_token);
        self.observer.on_call_end(started.elapsed());
        result
    }

    fn spawn(&self) -> Result<Child> {
        let child = self
            .launch
            .command()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| SessionError::Spawn { program: self.launch.program().display().to_string(), source: e })?;
        self.observer.on_spawn(child.id());
        Ok(child)
    }

    fn send_request(child: &mut Child, body: &[u8]) -> Result<()> {
        {
            let stdin = child.stdin.as_mut().ok_or(SessionError::NotPiped)?;
            stdin.write_all(body).map_err(SessionError::Write)?;
        }
        // Close stdin so the runner's `read()` sees EOF.
        drop(child.stdin.take());
        Ok(())
    }

    fn call_inner(&self, body: &[u8]) -> Result<Value> {
        let mut child = self.spawn()?;
        Self::send_request(&mut child, body)?;
        let stdout_reader = drain(child.stdout.take());
        let stderr_reader = drain(child.stderr.take());

        let status = wait_with_deadline(&mut child, self.cfg.call_timeout)?;
        let stdout_buf = joined(stdout_reader);
        let stderr_buf = joined(stderr_reader);

        if !status.success() {
            return Err(SessionError::Exited { status: status.to_string(), stderr: lossy_trimmed(&stderr_buf) });
        }
        Ok(proto::check_error(&String::from_utf8_lossy(&stdout_buf))?)
    }

    fn stream_inner<F: FnMut(&str)>(&self, body: &[u8], mut on_token: F) -> Result<String> {
        let mut child = self.spawn()?;
        Self::send_request(&mut child, body)?;

        let stdout = child.stdout.take().ok_or(SessionError::NotPiped)?;
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
        let stderr_reader = drain(child.stderr.take());

        let deadline = Instant::now() + self.cfg.call_timeout;
        let mut full = String::new();
        let outcome: Result<()> = loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let _ = child.kill();
                break Err(SessionError::Timeout { secs: self.cfg.call_timeout.as_secs() });
            }
            match line_rx.recv_timeout(remaining.min(Duration::from_millis(500))) {
                Ok(raw) => match proto::parse_stream_line(&raw) {
                    None | Some(StreamEvent::Other) | Some(StreamEvent::Keepalive) | Some(StreamEvent::Meta(_)) => {}
                    Some(StreamEvent::Token(t)) => {
                        if !t.is_empty() {
                            full.push_str(&t);
                            on_token(&t);
                        }
                    }
                    Some(StreamEvent::Refused(e)) => {
                        let _ = child.kill();
                        break Err(SessionError::Refused(e));
                    }
                    Some(StreamEvent::Error(e)) => {
                        let _ = child.kill();
                        break Err(SessionError::Runner(e));
                    }
                    // The terminal envelope. The child exits next; keep
                    // reading until it closes stdout so it is reaped cleanly.
                    // A one-shot runner has no cancel, so a cancelled marker
                    // is just another terminal.
                    Some(StreamEvent::Done) | Some(StreamEvent::Cancelled) => {}
                    Some(StreamEvent::Final(t)) => {
                        if full.is_empty() && !t.is_empty() {
                            full.push_str(&t);
                            on_token(&t);
                        }
                    }
                },
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break Ok(()),
            }
        };

        let _ = child.wait();
        let _ = reader.join();
        let stderr_buf = joined(stderr_reader);
        outcome?;
        if full.is_empty() {
            return Err(SessionError::NoOutput { stderr: lossy_trimmed(&stderr_buf) });
        }
        Ok(full)
    }
}

fn drain<R: Read + Send + 'static>(pipe: Option<R>) -> Option<JoinHandle<Vec<u8>>> {
    pipe.map(|mut r| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = r.read_to_end(&mut buf);
            buf
        })
    })
}

fn joined(handle: Option<JoinHandle<Vec<u8>>>) -> Vec<u8> {
    handle.and_then(|h| h.join().ok()).unwrap_or_default()
}

fn lossy_trimmed(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim().to_string()
}

/// Poll for exit; kill on overrun.
fn wait_with_deadline(child: &mut Child, timeout: Duration) -> Result<ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait().map_err(SessionError::Wait)? {
            Some(status) => return Ok(status),
            None => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(SessionError::Timeout { secs: timeout.as_secs() });
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}
