//! The threads around the request loop: the stdin reader (which acts on
//! cancels at once), and the watchdog (which stops llama-server and exits
//! when the adapter is signalled or its parent dies).

use crate::log;
use crate::server::{Shared, STOP_GRACE};
use serde_json::Value;
use std::io::{BufRead, BufReader, Read};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

/// A line from stdin for the request loop.
pub(crate) enum Incoming {
    Request {
        seq: u64,
        req: Value,
    },
    /// Not JSON: answered with an error so replies stay in step.
    Bad {
        error: String,
    },
}

/// Read stdin on its own thread. A `cancel` line is acted on here and never
/// queued or answered; every other line goes to the request loop in order,
/// numbered, so a cancel can tell which request it was meant for. The
/// channel closes at end of input.
pub(crate) fn spawn_stdin_reader(shared: Arc<Shared>, input: impl Read + Send + 'static) -> mpsc::Receiver<Incoming> {
    let (tx, rx) = mpsc::channel();
    let _ = std::thread::Builder::new().name("estia-llama-stdin".into()).spawn(move || {
        let mut r = BufReader::new(input);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match r.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let line = String::from_utf8_lossy(&buf);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let msg = match serde_json::from_str::<Value>(line) {
                Ok(req) if req.get("type").and_then(Value::as_str) == Some("cancel") => {
                    shared.cancel();
                    continue;
                }
                Ok(req) => Incoming::Request { seq: shared.next_seq(), req },
                Err(e) => {
                    shared.next_seq();
                    Incoming::Bad { error: format!("request is not JSON: {e}") }
                }
            };
            if tx.send(msg).is_err() {
                break;
            }
        }
    });
    rx
}

/// The last terminating signal received, or 0.
static SIGNAL: AtomicI32 = AtomicI32::new(0);

#[cfg(unix)]
extern "C" fn on_signal(sig: libc::c_int) {
    // Only an atomic store: the watchdog thread does the work.
    SIGNAL.store(sig, Ordering::SeqCst);
}

/// SIGTERM, SIGINT and SIGHUP stop llama-server and exit, via the watchdog.
pub(crate) fn install_signal_handlers() {
    #[cfg(unix)]
    for sig in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
        // SAFETY: the handler only stores to an atomic.
        unsafe {
            libc::signal(sig, on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t);
        }
    }
}

/// Every 100 ms: exit on a signal. Every 500 ms: exit if the parent has died
/// (this process was re-parented). Either way llama-server is stopped first.
/// This covers a parent killed outright, which closes no pipe the request
/// loop would notice while it waits on a long generation.
pub(crate) fn spawn_watchdog(shared: Arc<Shared>) {
    #[cfg(unix)]
    let parent = std::os::unix::process::parent_id();
    let _ = std::thread::Builder::new().name("estia-llama-watchdog".into()).spawn(move || {
        #[cfg(unix)]
        let mut n: u64 = 0;
        loop {
            std::thread::sleep(Duration::from_millis(100));
            let sig = SIGNAL.load(Ordering::SeqCst);
            if sig != 0 {
                log(format_args!("signal {sig}: stopping llama-server and exiting"));
                shared.close(STOP_GRACE);
                std::process::exit(128 + sig);
            }
            #[cfg(unix)]
            {
                n += 1;
                if n.is_multiple_of(5) && std::os::unix::process::parent_id() != parent {
                    log(format_args!("parent process exited: stopping llama-server and exiting"));
                    shared.close(STOP_GRACE);
                    std::process::exit(0);
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn reader_numbers_requests_and_takes_cancels_itself() {
        let shared = Arc::new(Shared::default());
        let input: &[u8] = b"{\"type\":\"hello\"}\n\n  \nnot json\n{\"type\":\"chat_stream\"}\n{\"type\":\"cancel\"}\n{\"type\":\"ping\"}";
        let rx = spawn_stdin_reader(Arc::clone(&shared), input);
        let mut got = Vec::new();
        while let Ok(m) = rx.recv_timeout(Duration::from_secs(5)) {
            got.push(m);
        }
        assert_eq!(got.len(), 4, "hello, bad line, chat_stream, ping; the cancel is not queued");
        let seqs: Vec<(u64, String)> = got
            .iter()
            .filter_map(|m| match m {
                Incoming::Request { seq, req } => Some((*seq, req["type"].as_str().unwrap().to_string())),
                Incoming::Bad { .. } => None,
            })
            .collect();
        assert_eq!(seqs, [(1, "hello".into()), (3, "chat_stream".into()), (4, "ping".into())]);
        assert!(matches!(&got[1], Incoming::Bad { error } if error.contains("not JSON")));
        // The cancel came after chat_stream (3) and before ping (4).
        assert!(shared.is_cancelled(3));
        assert!(!shared.is_cancelled(4));
    }
}
