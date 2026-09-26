# Runner protocol

A runner is the process that actually runs a model. The engine starts it as a
child process and talks to it over stdin and stdout. This page describes that
wire format. The Rust definitions are in `proto/src/lib.rs` (crate
`estia-proto`); the reference implementation is
`runners/mlx-python/estia-runner.py`. A second runner, `estia-llama`, puts
llama.cpp behind the same protocol; see
[The llama.cpp runner](#the-llamacpp-runner).

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
the only kind the `estia` CLI and server use. `estia-runner.py` and
`estia-llama` are resident.

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
{"ok": true, "runner": "mlx-python", "version": "2.2.0", "protocol": 2,
 "capabilities": {"generate": true, "stream": true, "embed": true, "cancel": true,
                  "load": true, "chat": true, "tools": true, "prompt_cache": true,
                  "count_tokens": true, "structured": [],
                  "parses_tool_calls": false, "backend": "mlx-python"}}
```

A v1 runner answers `hello` with an unknown-type error. The engine reads that as
"protocol v1, no capabilities". Every capability defaults to off, so a runner
that omits one is read conservatively. `structured` lists the output formats
the runner can constrain decoding to (`json`, `json_schema`); empty means the
engine validates and repairs the output itself.

Two more fields say how to read what the runner returns:

- `parses_tool_calls` (default `false`): the runner parses the model's tool
  calls itself and returns them in `meta.tool_calls` (see [Streams](#streams)).
  A caller takes the calls from there and does not look for them in the text.
  A runner without it returns the model's raw text, calls included, and the
  caller parses them out of that, as Estia's server does in
  `server/src/toolcalls.rs`.
- `backend`: the backend id the runner's outputs belong to, `mlx-python` or
  `llama-cpp` (`estia_engine::Backend`). It is the part after `@` in an
  embedding fingerprint such as `embeddinggemma-300m-4bit@mlx-python`,
  because the same weights give different vectors on different backends.
  Runners that predate the field omit it; the MLX Python runner does not send
  it yet.

`estia runner-check` performs this handshake and prints the result without
loading a model.

## Requests

`model_path` is always an absolute path on disk: the model's directory for an
MLX model, the `.gguf` file for a GGUF model. The engine resolves ids to
paths; runners never download anything.

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

`messages` are OpenAI-shaped:
`{"role", "content", "name"?, "tool_call_id"?, "tool_calls"?}` with roles
`system`, `user`, `assistant` and `tool`. The runner renders them with the
model's own chat template. `tools` are OpenAI tool schemas, declared through
the template where the model supports them. `tools`, `cache_key` and `format`
are omitted when unset, and so are a message's optional fields.

An assistant turn that called tools keeps the calls in `tool_calls`, in
OpenAI's shape, with `arguments` as a JSON string:

```json
{"role": "assistant", "content": "",
 "tool_calls": [{"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Athens\"}"}}]}
{"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 24}"}
```

Chat templates render a `tool` result only after the assistant turn whose
call it answers, so a runner passes `tool_calls` to the template with the rest
of the message. Dropping them drops the results too.

`cache_key` names a conversation. The runner keeps that conversation's KV
cache and, on the next call with the same key, prefills only the part of the
prompt that is new. The Python runner keeps up to 8 such caches per process.

`format` is the engine's `OutputFormat` (`engine/src/structured.rs`):
`{"type": "json"}` for any JSON value, or
`{"type": "json_schema", "schema": {…}}` for JSON that matches a schema. Its
`type` is what `capabilities.structured` lists. It is honoured only by a
runner that lists that type, and Estia's server sends it only to such a
runner (and never together with `tools`). The Python runner accepts and
ignores it; for that runner the server puts the schema in the prompt instead.
The llama.cpp runner constrains decoding to it.

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
- `meta` (v2, chat only) reports on the generation just before the end. A
  non-streaming `chat` returns the same object as `{"text": "…", "meta": {…}}`.
  Every field is optional, and the engine ignores fields it does not know:

  | Field | Meaning |
  |---|---|
  | `prompt_tokens` | Tokens in the rendered prompt. |
  | `cached_tokens` | Prompt tokens served from the KV cache instead of prefilled. |
  | `generation_tokens` | Tokens generated. |
  | `template` | `native` when the model's chat template rendered the messages, `manual` for the runner's fallback rendering. |
  | `tool_calls` | Only from a runner that declares `parses_tool_calls`, and only when the model called tools: the calls, OpenAI-shaped like `Message.tool_calls` above. They come once, here, even in a stream. The `token` lines (or `text`) then carry only what the model wrote outside the calls, often nothing. |
  | `generation_tps` | Decode rate the runner measured, in tokens per second, over generation only (prefill excluded). |
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
- **No orphans.** Dropping a session (idle unload, shutdown, a deadline)
  sends the child SIGTERM, gives it 4 seconds to exit, then kills and reaps
  it. The grace lets a runner clean up: the llama.cpp runner stops its
  `llama-server` and removes its files. On Windows the child is killed at
  once. The Python runner also exits on its own if its parent dies.

## The llama.cpp runner

`estia-llama` (crate `llama/`) is a resident v2 runner for GGUF models. It
runs no model itself: it starts an upstream `llama-server` process and turns
each request into HTTP calls to it, over a private UNIX socket (loopback TCP
on Windows) with a random API key. The engine drives it like any other
resident runner, so the guarantees above apply unchanged. Why it is built this
way: [design/llama-backend.md](design/llama-backend.md).

### Command line

It runs as its own binary or, so a release stays one file, as a hidden
subcommand of the CLI. Both take the same arguments:

```text
estia-llama        --server <llama-server> --run-dir <dir> [--ctx <n>] [-- <llama-server args>…]
estia runner llama --server <llama-server> --run-dir <dir> [--ctx <n>] [-- <llama-server args>…]
```

| Argument | Meaning |
|---|---|
| `--server <path>` | The `llama-server` executable to start. Estia is tested against upstream build `b11146`, pinned with its SHA-256 in `engine/src/runtime/llama_pins.rs`. |
| `--run-dir <dir>` | A private directory for the runner's files, created with mode 0700 if missing. |
| `--ctx <n>` | Context length for generation models (`llama-server -c`). Default 8192, not the model's own maximum, which would size the KV cache for 128K tokens or more. |
| `-- <args>` | Passed to every `llama-server` it starts, after the runner's own flags, so they win. For example `-- -ngl 0` keeps a model off the GPU. Flags the runner owns or never allows are refused: where the server listens and how it authenticates (`--host`, `--port`, `--api-key`, …), its web UI and `/slots`, and features that download models or give the model tools (`-hf`, `--model-url`, `--tools`, `--agent`, `--mcp-servers-config`, …). |

Flags may also be written `--flag=value`. Anything else is an error, and
`estia-llama` exits with status 2 before reading stdin.

Then it speaks the protocol on stdin and stdout. Stdout carries only protocol
lines; `llama-server`'s output goes to the runner's stderr, each line prefixed
`[llama-server <pid>]`. `llama-server` does not inherit the variables it
would read as configuration (`LLAMA_ARG_*`, `LLAMA_API_KEY`) or `HF_TOKEN`.

The run directory holds, named after the runner's pid so that two runners can
share one:

- `llama-<pid>.sock`, the socket. If the path is longer than a socket path may
  be (about 104 bytes on macOS, 108 on Linux), the socket goes in a private
  directory under the system temp directory instead, and if that is too long
  too, the runner falls back to loopback TCP.
- `llama-<pid>.key`, the API key, mode 0600, passed with `--api-key-file` so
  it never appears in `ps`. Removed once the server is up.
- `llama-<pid>.json`, `{adapter_pid, server_pid, socket, addr, model, kind}`,
  so a host can find and stop a server whose runner was killed outright.

### Lifetime

- **stdin closes:** requests already read are answered, then `llama-server`
  is stopped (SIGTERM, SIGKILL after 3 seconds) and the runner exits with
  status 0.
- **SIGTERM, SIGINT, SIGHUP:** `llama-server` is stopped and the runner exits
  with 128 plus the signal number.
- **The parent dies:** the runner notices within half a second, stops
  `llama-server` and exits with status 0.
- **The runner is killed outright:** on Linux `llama-server` gets SIGKILL
  through `PR_SET_PDEATHSIG`. On macOS a small `/bin/sh` guard
  (`estia-llama-guard`) holds a pipe from the runner and kills the server when
  the pipe closes. The record in the run directory names it as well.
- **`llama-server` dies under a loaded model:** the call in flight gets an
  error line and the runner exits non-zero, so `Session` respawns it.

### Models

`model_path` names the model's `.gguf` file. The runner also accepts a
directory and then loads `<dir>/model.gguf`.

It holds one model at a time, in one `llama-server`. The engine keeps one
runner for generation and another for embeddings, as it does with the MLX
runner. `load` with
`"kind": "generation"` (the default) starts a chat server, and
`"kind": "embedding"` starts one with `--embedding`. A request for a model
that is not loaded loads it first; a request for a different model, or for the
same model as the other kind, stops the current server and starts a new one.
`unload` stops the server if it holds that model (or any model, when
`model_path` is left out). `ping` answers `{"ok": true}` and, once a model is
up, also checks that `llama-server` answers `/health`.

A model takes up to 10 minutes to become ready; streams get `keepalive` lines
while they wait. A `llama-server` that exits during the load fails that
request with the tail of its log, and the runner stays up.

### What it declares

```json
{"ok": true, "runner": "estia-llama", "version": "0.1.0", "protocol": 2,
 "capabilities": {"generate": true, "stream": true, "embed": true, "cancel": true,
                  "load": true, "chat": true, "tools": true, "prompt_cache": true,
                  "count_tokens": true, "structured": ["json", "json_schema"],
                  "parses_tool_calls": true, "backend": "llama-cpp"}}
```

`version` is the crate's version. Against the MLX Python runner, the
differences a caller sees:

- **Requests.** `chat` and `chat_stream` go to `llama-server`'s
  `/v1/chat/completions`, which renders the messages with the model's own chat
  template. `generate` and `generate_stream` send the prompt as one user turn,
  and honour `"json": true`. `count_tokens` counts with the model's tokenizer,
  special tokens included.
- **Structured output.** `format` constrains decoding (llama.cpp turns the
  schema into a grammar), so the engine's validate-and-repair pass should
  rarely have work to do.
- **Tool calls.** `llama-server` parses them, and the runner returns them in
  `meta.tool_calls` with `arguments` as a JSON string. `token` lines carry
  only the text outside the calls.
- **Sampling.** The runner sends every sampling value and never inherits
  `llama-server`'s defaults. `max_tokens` and `temperature` sent as `null`
  mean 256 and 0.0, as in the Python runner; top-k, top-p, min-p and the
  repeat penalty are off. Thinking is off
  (`chat_template_kwargs: {"enable_thinking": false}`).
- **Prompt cache.** One conversation's cache at a time, not eight: the runner
  runs one slot (`-np 1`) with `llama-server`'s RAM prompt cache off. A call
  whose `cache_key` matches the previous call's reuses the cached prefix; any
  other call, and any call without a key, starts cold. So `cached_tokens`
  never counts a prefix that another conversation left behind.
- **Meta.** `meta` carries `prompt_tokens` (the whole prompt, cached part
  included), `cached_tokens` and `generation_tokens` as `llama-server`
  reports them, `template: "native"`, `generation_tps`, and
  `finish_reason` (`stop`, `length` or `tool_calls`), which the engine does
  not read yet.
- **Embeddings.** Vectors are L2-normalised. Inputs longer than 512 tokens
  are cut to 512, as the Python runner does. Fingerprints end in
  `@llama-cpp`: vectors do not match the MLX ones for the same model, and the
  engine treats them as a different space.

## Writing a runner

A minimal resident runner answers `ping`, `hello` (declare only what you
implement), and the requests its capabilities promise. The server's tests use
a stdlib-only Python fake that does exactly this; see `FAKE` in
`server/tests/api.rs`, and the `V2` and `V2_CHAT` runners in
`engine/tests/fake_runner.rs`.
