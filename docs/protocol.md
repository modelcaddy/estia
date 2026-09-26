# Runner protocol

A runner is the process that actually runs a model. The engine starts it as a
child process and talks to it over stdin and stdout. This page describes that
wire format. The Rust definitions are in `proto/src/lib.rs` (crate
`estia-proto`); the reference implementation is
`runners/mlx-python/estia-runner.py`.

The current version is **2** (`estia_proto::PROTOCOL_VERSION`). Every v1 line is
still valid in v2.

## Framing

- One JSON object per line on stdin (requests), one JSON object per line on
  stdout (responses).
- Each request line gets one response line, except streams, which answer with
  several lines and a terminal line.
- A runner must write nothing else to stdout. The Python runner redirects
  `sys.stdout` to stderr at start-up and writes responses through a saved
  handle, so library output cannot corrupt the stream. Stderr is free text.
- Inside a stream the engine skips blank lines and lines that are not JSON.
  Outside a stream the next line is taken as the response, so stray output
  there breaks the call.

## Errors

A runner reports a failed request with either envelope:

```json
{"error": "message"}
{"ok": false, "error": "message"}
```

Resident runners use the first, one-shot runners the second. The engine
accepts both (`check_error`, `parse_stream_line`).

## Resident and one-shot runners

**Resident.** The process stays up and serves many requests. Models are loaded
once and kept in memory. This is what the engine's `Session` drives, and it is
the only kind the `estia` CLI and server use. `estia-runner.py` is resident.

**One-shot.** The process reads stdin to EOF, answers once (or streams), and
exits. Every call pays a process start and a model load. The engine's `OneShot`
type drives these. `oneshot-runner.py`, the Apple Foundation Models runner and
the Swift MLX runner are one-shot.

## Handshake (v2)

Send `hello` once after spawn.

```json
{"type": "hello"}
```

```json
{"ok": true, "runner": "mlx-python", "version": "2.1.0", "protocol": 2,
 "capabilities": {"generate": true, "stream": true, "embed": true, "cancel": true,
                  "load": true, "chat": true, "tools": true, "prompt_cache": true,
                  "count_tokens": true, "structured": []}}
```

A v1 runner answers `hello` with an unknown-type error. The engine reads that as
"protocol v1, no capabilities". Every capability defaults to off, so a runner
that omits one is read conservatively. `structured` lists the output formats
the runner can constrain decoding to (`json`, `json_schema`); empty means the
engine validates and repairs the output itself.

`estia runner-check` performs this handshake and prints the result without
loading a model.

## Requests

`model_path` is always the absolute path of a model directory on disk. The
engine resolves ids to paths; runners never download anything.

### v1

| Request | Response |
|---|---|
| `{"type":"ping"}` | `{"ok":true}` |
| `{"type":"health"}` (one-shot runners) | `{"ok":true,"mlx_available":bool,"version":…,"detail":…}` |
| `{"type":"embed_batch","model_path":…,"inputs":["a","b"]}` | `{"embeddings":[[…],[…]]}` |
| `{"type":"embed","model_path":…,"input":"a"}` | `{"embedding":[…]}` |
| `{"type":"generate","model_path":…,"prompt":…,"max_tokens":N,"temperature":T}` | `{"text":"…"}` |
| `{"type":"generate_stream", …same fields…}` | a stream (below) |
| `{"type":"cancel"}` | nothing on its own; ends the stream in flight |

`max_tokens` and `temperature` are sent as `null` when unset. The one-shot
`generate` also carries `"json": true|false`; the resident one omits it.

### v2

| Request | Response |
|---|---|
| `{"type":"load","model_path":…,"kind":"generation"\|"embedding"}` | `{"ok":true,"loaded":true,"ms":N}` |
| `{"type":"unload","model_path":…}` | `{"ok":true,"unloaded":bool}` |
| `{"type":"chat","model_path":…,"messages":[…],"tools":[…],"cache_key":"…","format":{…},"max_tokens":N,"temperature":T}` | `{"text":"…","meta":{…}}` |
| `{"type":"chat_stream", …same fields…}` | a stream, with a `meta` line before the end |
| `{"type":"count_tokens","model_path":…,"text":"…"}` | `{"tokens":N}` |

`load` lets a caller pay the model load before the first request and measure
it. `unload` frees the memory.

`messages` are OpenAI-shaped: `{"role", "content", "name"?, "tool_call_id"?}`
with roles `system`, `user`, `assistant` and `tool`. The runner renders them
with the model's own chat template. `tools` are OpenAI tool schemas, declared
through the template where the model supports them. `tools`, `cache_key` and
`format` are omitted when unset.

`cache_key` names a conversation. The runner keeps that conversation's KV
cache and, on the next call with the same key, prefills only the part of the
prompt that is new. The Python runner keeps up to 8 such caches per process.

`format` is honoured only by a runner that lists it under
`capabilities.structured`. The Python runner accepts and ignores it.

### Apple Foundation Models

The Swift runners also accept `{"type":"apple_health"}` and
`{"type":"apple_generate","prompt":…,"temperature":T}`. These are one-shot and
not used by the CLI or the server.

## Streams

A streaming request (`generate_stream`, `chat_stream`) answers with:

```json
{"type": "token", "text": "The "}
{"type": "token", "text": "sea"}
{"type": "keepalive"}
{"type": "meta", "prompt_tokens": 28, "cached_tokens": 0, "generation_tokens": 29, "template": "native"}
{"done": true}
```

- `token` lines carry decoded text as it is produced. They may be empty.
- `keepalive` lines carry nothing. A runner sends them so a long prefill does
  not look like a hung process.
- `meta` (v2, chat only) reports token counts just before the end. `template`
  is `native` when the model's chat template rendered the messages and
  `manual` for the runner's fallback rendering.
- The stream ends with `{"done": true}` (one-shot runners: `{"ok": true,
  "done": true}`), or `{"done": true, "cancelled": true}` after a cancel, or an
  error line.
- A runner that cannot stream may answer with a single `{"text": "…"}` line
  instead. The engine treats it as the whole answer.

## Cancel

While a stream is open the engine may write `{"type":"cancel"}`. A runner that
honours it stops decoding and ends the stream with
`{"done": true, "cancelled": true}`. It never answers the cancel line itself.
The Python runner reads stdin on a separate thread for this, because the main
thread is busy in MLX while a stream runs.

A runner that does not understand cancel finishes the generation and then
answers the cancel line with an error. The engine drains that line so the next
call is not out of step.

## What the engine guarantees around a runner

These live in `engine/src/session.rs`:

- **Deadline.** Each response line must arrive within the call timeout
  (300 seconds by default). For a stream, every line resets it. On overrun the
  child is killed and the call fails.
- **One respawn.** If the child has died (broken pipe, EOF), a call respawns it
  once and retries. A second failure is returned to the caller.
- **Streams never retry.** Once a token has reached the caller, a retry would
  repeat it.
- **Priority.** Calls to one runner are serialised. Waiting `interactive`
  calls are served before `background` ones, first come first served within a
  class. A running call is never interrupted.
- **No orphans.** Dropping a session kills and reaps the child. The Python
  runner also exits on its own if its parent dies.

## Writing a runner

A minimal resident runner answers `ping`, `hello` (declare only what you
implement), and the requests its capabilities promise. The server's tests use
a stdlib-only Python fake that does exactly this; see `FAKE` in
`server/tests/api.rs`, and the `V2` and `V2_CHAT` runners in
`engine/tests/fake_runner.rs`.
