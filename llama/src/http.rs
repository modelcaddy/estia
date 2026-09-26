//! A small HTTP/1.1 client for one `llama-server`, over a UNIX socket or
//! loopback TCP.
//!
//! One request per connection (`Connection: close`), so there is no pooling
//! and no pipelining to get wrong. Bodies may be `Content-Length`, chunked, or
//! delimited by the server closing the connection. Streams are server-sent
//! events on top of a chunked body ([`SseParser`]).
//!
//! Reads use a short socket timeout. Each time a read times out the caller's
//! idle hook runs, which is where the adapter sends protocol keepalives and
//! notices a cancel; the hook returns an error to abandon the request. A
//! cancel from another thread can also shut the socket down through a
//! [`Stream::try_clone`] of it, which ends a blocked read at once.

use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
#[cfg(unix)]
use std::os::unix::net::UnixStream;
#[cfg(unix)]
use std::path::PathBuf;
use std::time::Duration;

/// How long a read waits before the idle hook runs.
pub(crate) const READ_TICK: Duration = Duration::from_millis(200);
/// Writing a request (an embedding batch can be megabytes) must not hang.
const WRITE_TIMEOUT: Duration = Duration::from_secs(60);
/// A response head larger than this is not from llama-server.
const MAX_HEAD: usize = 64 * 1024;
/// Chunk-size lines and SSE lines are short; this only bounds bad input.
const MAX_LINE: usize = 16 * 1024 * 1024;

/// Called on every read timeout. `Err` abandons the request.
pub(crate) type Idle<'a> = dyn FnMut() -> io::Result<()> + 'a;

/// Where a `llama-server` listens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Endpoint {
    #[cfg(unix)]
    Unix(PathBuf),
    Tcp(SocketAddr),
}

impl Endpoint {
    pub(crate) fn connect(&self) -> io::Result<Stream> {
        match self {
            #[cfg(unix)]
            Endpoint::Unix(path) => Ok(Stream::Unix(UnixStream::connect(path)?)),
            Endpoint::Tcp(addr) => {
                let s = TcpStream::connect_timeout(addr, Duration::from_secs(5))?;
                let _ = s.set_nodelay(true);
                Ok(Stream::Tcp(s))
            }
        }
    }

    pub(crate) fn describe(&self) -> String {
        match self {
            #[cfg(unix)]
            Endpoint::Unix(path) => format!("unix:{}", path.display()),
            Endpoint::Tcp(addr) => format!("http://{addr}"),
        }
    }
}

/// A connected socket.
#[derive(Debug)]
pub(crate) enum Stream {
    #[cfg(unix)]
    Unix(UnixStream),
    Tcp(TcpStream),
}

impl Stream {
    pub(crate) fn try_clone(&self) -> io::Result<Stream> {
        match self {
            #[cfg(unix)]
            Stream::Unix(s) => Ok(Stream::Unix(s.try_clone()?)),
            Stream::Tcp(s) => Ok(Stream::Tcp(s.try_clone()?)),
        }
    }

    /// Close both directions. The server sees the connection end, and a read
    /// blocked on this socket in another thread returns.
    pub(crate) fn shutdown(&self) {
        let _ = match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.shutdown(Shutdown::Both),
            Stream::Tcp(s) => s.shutdown(Shutdown::Both),
        };
    }

    fn set_timeouts(&self, read: Duration, write: Duration) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Stream::Unix(s) => {
                s.set_read_timeout(Some(read))?;
                s.set_write_timeout(Some(write))
            }
            Stream::Tcp(s) => {
                s.set_read_timeout(Some(read))?;
                s.set_write_timeout(Some(write))
            }
        }
    }
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.read(buf),
            Stream::Tcp(s) => s.read(buf),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.write(buf),
            Stream::Tcp(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.flush(),
            Stream::Tcp(s) => s.flush(),
        }
    }
}

/// One outgoing request.
pub(crate) struct Request<'a> {
    pub method: &'a str,
    pub path: &'a str,
    /// Sent as `Authorization: Bearer <key>`. `/health` needs none.
    pub key: Option<&'a str>,
    /// A JSON body.
    pub body: Option<&'a [u8]>,
}

impl Request<'_> {
    fn head(&self) -> String {
        let mut h = format!("{} {} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nAccept: */*\r\n", self.method, self.path);
        if let Some(key) = self.key {
            h.push_str("Authorization: Bearer ");
            h.push_str(key);
            h.push_str("\r\n");
        }
        if let Some(body) = self.body {
            h.push_str("Content-Type: application/json\r\n");
            h.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        h.push_str("\r\n");
        h
    }
}

/// Send `req` on a fresh connection and return the connection, ready to read
/// the response.
pub(crate) fn send(mut stream: Stream, req: &Request) -> io::Result<Conn> {
    stream.set_timeouts(READ_TICK, WRITE_TIMEOUT)?;
    stream.write_all(req.head().as_bytes())?;
    if let Some(body) = req.body {
        stream.write_all(body)?;
    }
    stream.flush()?;
    Ok(Conn { stream, buf: Vec::new(), pos: 0 })
}

/// A connection with its read buffer.
pub(crate) struct Conn {
    stream: Stream,
    buf: Vec<u8>,
    pos: usize,
}

impl Conn {
    #[cfg(test)]
    pub(crate) fn from_stream(stream: Stream) -> Conn {
        stream.set_timeouts(READ_TICK, WRITE_TIMEOUT).unwrap();
        Conn { stream, buf: Vec::new(), pos: 0 }
    }

    fn available(&self) -> &[u8] {
        &self.buf[self.pos..]
    }

    fn consume(&mut self, n: usize) {
        self.pos += n;
        if self.pos == self.buf.len() {
            self.buf.clear();
            self.pos = 0;
        }
    }

    /// Read more bytes into the buffer. `Ok(0)` is end of stream. A read
    /// timeout runs `idle` and tries again.
    fn fill(&mut self, idle: &mut Idle) -> io::Result<usize> {
        if self.pos > 0 {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
        let mut tmp = [0u8; 16 * 1024];
        loop {
            match self.stream.read(&mut tmp) {
                Ok(n) => {
                    self.buf.extend_from_slice(&tmp[..n]);
                    return Ok(n);
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => idle()?,
                Err(e) => return Err(e),
            }
        }
    }

    /// One line without its `\r\n` or `\n`.
    fn read_line(&mut self, idle: &mut Idle, limit: usize) -> io::Result<Vec<u8>> {
        loop {
            if let Some(i) = self.available().iter().position(|&b| b == b'\n') {
                let mut line = self.available()[..i].to_vec();
                self.consume(i + 1);
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                return Ok(line);
            }
            if self.available().len() > limit {
                return Err(invalid("line too long"));
            }
            if self.fill(idle)? == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "connection closed mid-line"));
            }
        }
    }

    /// Read the status line and headers.
    pub(crate) fn read_head(mut self, idle: &mut Idle) -> io::Result<Response> {
        let status_line = self.read_line(idle, MAX_HEAD)?;
        let status = parse_status_line(&status_line)?;
        let mut headers = Vec::new();
        let mut total = status_line.len();
        loop {
            let line = self.read_line(idle, MAX_HEAD)?;
            if line.is_empty() {
                break;
            }
            total += line.len();
            if total > MAX_HEAD {
                return Err(invalid("response head too large"));
            }
            let text = String::from_utf8_lossy(&line);
            if let Some((name, value)) = text.split_once(':') {
                headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
            }
        }
        let head = Head { status, headers };
        let body = framing(&head)?;
        Ok(Response { head, conn: self, body })
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn parse_status_line(line: &[u8]) -> io::Result<u16> {
    let text = String::from_utf8_lossy(line);
    let mut parts = text.split_whitespace();
    let version = parts.next().unwrap_or("");
    if !version.starts_with("HTTP/1.") {
        return Err(invalid("not an HTTP/1.x response"));
    }
    parts.next().and_then(|s| s.parse::<u16>().ok()).ok_or_else(|| invalid("bad status line"))
}

/// Status and headers of a response. Header names are lower-cased.
#[derive(Debug, Clone)]
pub(crate) struct Head {
    pub status: u16,
    pub headers: Vec<(String, String)>,
}

impl Head {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Body {
    /// Chunked, waiting for the next chunk-size line.
    ChunkSize,
    /// Chunked, this many data bytes left in the current chunk.
    ChunkData(u64),
    /// Chunked, the CRLF after a chunk's data.
    ChunkEnd,
    /// `Content-Length`, this many bytes left.
    Length(u64),
    /// Until the server closes the connection.
    UntilClose,
    Done,
}

fn framing(head: &Head) -> io::Result<Body> {
    if matches!(head.status, 204 | 304) || (100..200).contains(&head.status) {
        return Ok(Body::Done);
    }
    if let Some(te) = head.header("transfer-encoding") {
        if te.to_ascii_lowercase().split(',').any(|t| t.trim() == "chunked") {
            return Ok(Body::ChunkSize);
        }
    }
    if let Some(len) = head.header("content-length") {
        let n = len.trim().parse::<u64>().map_err(|_| invalid("bad content-length"))?;
        return Ok(if n == 0 { Body::Done } else { Body::Length(n) });
    }
    Ok(Body::UntilClose)
}

/// A response whose body is read piece by piece.
pub(crate) struct Response {
    pub head: Head,
    conn: Conn,
    body: Body,
}

impl Response {
    /// The next piece of the decoded body; `Ok(None)` at its end.
    pub(crate) fn next_chunk(&mut self, idle: &mut Idle) -> io::Result<Option<Vec<u8>>> {
        loop {
            match self.body {
                Body::Done => return Ok(None),
                Body::ChunkSize => {
                    let line = self.conn.read_line(idle, 1024)?;
                    let text = String::from_utf8_lossy(&line);
                    let hex = text.split(';').next().unwrap_or("").trim();
                    let size = u64::from_str_radix(hex, 16).map_err(|_| invalid("bad chunk size"))?;
                    if size == 0 {
                        // Trailers, then the blank line that ends the body.
                        loop {
                            if self.conn.read_line(idle, MAX_HEAD)?.is_empty() {
                                break;
                            }
                        }
                        self.body = Body::Done;
                        return Ok(None);
                    }
                    self.body = Body::ChunkData(size);
                }
                Body::ChunkData(left) | Body::Length(left) => {
                    if self.conn.available().is_empty() && self.conn.fill(idle)? == 0 {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "connection closed mid-body"));
                    }
                    let take = (left as usize).min(self.conn.available().len());
                    let piece = self.conn.available()[..take].to_vec();
                    self.conn.consume(take);
                    let rest = left - take as u64;
                    self.body = match (self.body, rest) {
                        (Body::ChunkData(_), 0) => Body::ChunkEnd,
                        (Body::ChunkData(_), n) => Body::ChunkData(n),
                        (_, 0) => Body::Done,
                        (_, n) => Body::Length(n),
                    };
                    return Ok(Some(piece));
                }
                Body::ChunkEnd => {
                    if !self.conn.read_line(idle, 1024)?.is_empty() {
                        return Err(invalid("chunk not followed by CRLF"));
                    }
                    self.body = Body::ChunkSize;
                }
                Body::UntilClose => {
                    if self.conn.available().is_empty() && self.conn.fill(idle)? == 0 {
                        self.body = Body::Done;
                        return Ok(None);
                    }
                    let piece = self.conn.available().to_vec();
                    self.conn.consume(piece.len());
                    return Ok(Some(piece));
                }
            }
        }
    }

    /// The whole body, refusing more than `limit` bytes.
    pub(crate) fn read_all(&mut self, idle: &mut Idle, limit: usize) -> io::Result<Vec<u8>> {
        let mut all = Vec::new();
        while let Some(piece) = self.next_chunk(idle)? {
            all.extend_from_slice(&piece);
            if all.len() > limit {
                return Err(invalid("response body too large"));
            }
        }
        Ok(all)
    }
}

/// One server-sent event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SseEvent {
    /// The `event:` field, or `error` for llama-server's `error:` lines.
    pub event: Option<String>,
    /// The `data:` lines, joined with `\n`.
    pub data: String,
}

/// Incremental parser for `text/event-stream`. Feed it body bytes in any
/// split; it returns complete events.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    line: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl SseParser {
    pub(crate) fn push(&mut self, bytes: &[u8], out: &mut Vec<SseEvent>) -> io::Result<()> {
        for &b in bytes {
            if b == b'\n' {
                let mut line = std::mem::take(&mut self.line);
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                self.line_done(&String::from_utf8_lossy(&line), out);
            } else {
                if self.line.len() >= MAX_LINE {
                    return Err(invalid("event-stream line too long"));
                }
                self.line.push(b);
            }
        }
        Ok(())
    }

    /// End of stream: an event without its closing blank line still counts.
    pub(crate) fn finish(&mut self, out: &mut Vec<SseEvent>) {
        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            self.line_done(&String::from_utf8_lossy(&line), out);
        }
        self.dispatch(out);
    }

    fn line_done(&mut self, line: &str, out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(out);
            return;
        }
        if line.starts_with(':') {
            return; // comment (keepalive ping)
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "data" => self.data.push(value.to_string()),
            "event" => self.event = Some(value.to_string()),
            // llama-server reports a failure inside a stream as `error: {…}`.
            "error" => {
                self.event = Some("error".into());
                self.data.push(value.to_string());
            }
            _ => {}
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        if self.data.is_empty() && self.event.is_none() {
            return;
        }
        out.push(SseEvent { event: self.event.take(), data: self.data.join("\n") });
        self.data.clear();
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::thread;

    /// Write `parts` one by one (with small pauses, so they arrive split) to
    /// the far end of a socket pair and return the near end as a `Conn`.
    fn serve(parts: Vec<Vec<u8>>) -> Conn {
        let (a, mut b) = UnixStream::pair().unwrap();
        thread::spawn(move || {
            for p in parts {
                if b.write_all(&p).is_err() {
                    return;
                }
                thread::sleep(Duration::from_millis(2));
            }
        });
        Conn::from_stream(Stream::Unix(a))
    }

    fn no_idle() -> impl FnMut() -> io::Result<()> {
        || Ok(())
    }

    fn body_of(parts: Vec<&[u8]>) -> (u16, Vec<u8>) {
        let conn = serve(parts.into_iter().map(|p| p.to_vec()).collect());
        let mut idle = no_idle();
        let mut r = conn.read_head(&mut idle).unwrap();
        let body = r.read_all(&mut idle, 1 << 20).unwrap();
        (r.head.status, body)
    }

    #[test]
    fn content_length_body() {
        // The body arrives in two pieces; bytes past Content-Length are ignored.
        let (status, body) =
            body_of(vec![b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nContent-Type: application/json\r\n\r\n{\"a\":", b"true}garbage"]);
        assert_eq!(status, 200);
        assert_eq!(body, br#"{"a":true}"#);
    }

    #[test]
    fn chunked_body_split_anywhere() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n7;ext=1\r\n, world\r\n0\r\nX-Trailer: 1\r\n\r\n";
        // Every split point, including inside the head and inside chunk sizes.
        for cut in 1..raw.len() {
            let (status, body) = body_of(vec![&raw[..cut], &raw[cut..]]);
            assert_eq!(status, 200);
            assert_eq!(body, b"hello, world", "cut at {cut}");
        }
    }

    #[test]
    fn body_until_close_and_empty_bodies() {
        let (status, body) = body_of(vec![b"HTTP/1.1 503 Service Unavailable\r\n\r\nloading"]);
        assert_eq!((status, body.as_slice()), (503, &b"loading"[..]));
        let (status, body) = body_of(vec![b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"]);
        assert_eq!((status, body.len()), (200, 0));
    }

    #[test]
    fn truncated_chunked_body_is_an_error() {
        let conn = serve(vec![b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\na\r\nhel".to_vec()]);
        let mut idle = no_idle();
        let mut r = conn.read_head(&mut idle).unwrap();
        let err = r.read_all(&mut idle, 1 << 20).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn not_http_is_an_error() {
        let conn = serve(vec![b"SSH-2.0-OpenSSH\r\n\r\n".to_vec()]);
        let mut idle = no_idle();
        assert!(conn.read_head(&mut idle).is_err());
    }

    #[test]
    fn headers_are_case_insensitive() {
        let conn = serve(vec![b"HTTP/1.1 200 OK\r\nCONTENT-TYPE: text/event-stream\r\ncontent-length: 2\r\n\r\nok".to_vec()]);
        let mut idle = no_idle();
        let r = conn.read_head(&mut idle).unwrap();
        assert_eq!(r.head.header("content-type"), Some("text/event-stream"));
    }

    #[test]
    fn idle_hook_runs_while_waiting_and_can_abort() {
        let (a, _b) = UnixStream::pair().unwrap();
        let conn = Conn::from_stream(Stream::Unix(a));
        let mut calls = 0;
        let mut idle = || {
            calls += 1;
            if calls == 3 {
                Err(io::Error::other("stop"))
            } else {
                Ok(())
            }
        };
        let err = conn.read_head(&mut idle).err().unwrap();
        assert_eq!(err.to_string(), "stop");
        assert_eq!(calls, 3);
    }

    #[test]
    fn shutdown_from_another_handle_ends_a_blocked_read() {
        let (a, _b) = UnixStream::pair().unwrap();
        let stream = Stream::Unix(a);
        let other = stream.try_clone().unwrap();
        let conn = Conn::from_stream(stream);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            other.shutdown();
        });
        let mut idle = no_idle();
        let err = conn.read_head(&mut idle).err().unwrap();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn request_head_carries_key_and_length() {
        let body = br#"{"a":1}"#;
        let r = Request { method: "POST", path: "/tokenize", key: Some("k"), body: Some(body) };
        let h = r.head();
        assert!(h.starts_with("POST /tokenize HTTP/1.1\r\n"));
        assert!(h.contains("Authorization: Bearer k\r\n"));
        assert!(h.contains("Content-Length: 7\r\n"));
        assert!(h.contains("Connection: close\r\n"));
        assert!(h.ends_with("\r\n\r\n"));
        let r = Request { method: "GET", path: "/health", key: None, body: None };
        assert!(!r.head().contains("Authorization"));
    }

    fn sse(parts: &[&[u8]]) -> Vec<SseEvent> {
        let mut p = SseParser::default();
        let mut out = Vec::new();
        for part in parts {
            p.push(part, &mut out).unwrap();
        }
        p.finish(&mut out);
        out
    }

    #[test]
    fn sse_events_in_any_split() {
        let raw: &[u8] = b"data: {\"a\":1}\n\n: ping\n\ndata: {\"b\":2}\r\n\r\nevent: x\ndata: one\ndata:two\n\ndata: [DONE]\n\n";
        let want = vec![
            SseEvent { event: None, data: "{\"a\":1}".into() },
            SseEvent { event: None, data: "{\"b\":2}".into() },
            SseEvent { event: Some("x".into()), data: "one\ntwo".into() },
            SseEvent { event: None, data: "[DONE]".into() },
        ];
        assert_eq!(sse(&[raw]), want);
        for cut in 1..raw.len() {
            assert_eq!(sse(&[&raw[..cut], &raw[cut..]]), want, "cut at {cut}");
        }
        // Byte by byte.
        let bytes: Vec<&[u8]> = raw.chunks(1).collect();
        assert_eq!(sse(&bytes), want);
    }

    #[test]
    fn sse_error_lines_and_unterminated_last_event() {
        let got = sse(&[b"error: {\"code\":500}\n\ndata: tail"]);
        assert_eq!(
            got,
            vec![SseEvent { event: Some("error".into()), data: "{\"code\":500}".into() }, SseEvent { event: None, data: "tail".into() },]
        );
    }
}
