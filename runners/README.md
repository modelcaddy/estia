# Runners

A runner is the process that runs a model. The engine starts it, writes JSON
requests to its stdin and reads JSON from its stdout. The wire types are in
`proto/src/lib.rs`.

| Runner | Kind | What it is for | Build |
|---|---|---|---|
| `mlx-python/estia-runner.py` | resident, protocol v2 | The backend the `estia` CLI and server use. Generation (`mlx-vlm`) and embeddings (`mlx-embeddings`), models kept loaded between requests, cancel, chat with tools and a per-conversation KV cache. | None. Runs on the Python runtime the engine installs. |
| `mlx-python/oneshot-runner.py` | one-shot | The older MLX runner: one process per request. Kept for hosts that still use the one-shot path. | None. Same Python runtime. |
| `apple/AppleRunner.swift` | one-shot | Apple Foundation Models only (`apple_health`, `apple_generate`). For a host that runs MLX through Python and also wants the Apple on-device model. | `swiftc`, see below |
| `mlx-swift/` | one-shot | Compiled MLX runner (`mlx-swift-lm`) for a host that must ship only signed code and cannot download an interpreter. Also answers the Apple calls. | Xcode and the Metal Toolchain, see [`mlx-swift/README.md`](mlx-swift/README.md) |

Every runner here needs macOS on Apple silicon. There is no runner for Linux
or Windows yet.

## Protocol

### Resident (protocol v2)

A long-lived process. One JSON request per line on stdin, one JSON response
per line on stdout; a stream answers with several lines. Models stay loaded
until `unload` or until the process exits. A failed request answers
`{"error":"…"}`.

| Request | Answer |
|---|---|
| `{"type":"hello"}` | `{"ok":true,"runner":"mlx-python","version":"2.1.0","protocol":2,"capabilities":{…}}` |
| `{"type":"load","model_path":…,"kind":"generation"\|"embedding"}` | `{"ok":true,"loaded":true,"ms":N}` |
| `{"type":"unload","model_path":…}` | `{"ok":true,"unloaded":true\|false}` |
| `{"type":"chat","model_path":…,"messages":[…],"tools":[…],"cache_key":…,"format":…,"max_tokens":…,"temperature":…}` | `{"text":"…","meta":{"prompt_tokens":N,"cached_tokens":N,"generation_tokens":N,"template":…}}` |
| `{"type":"chat_stream", …same fields…}` | token lines, then `{"type":"meta",…}`, then `{"done":true}` |
| `{"type":"cancel"}` | no answer of its own; the stream in flight ends with `{"done":true,"cancelled":true}` |
| `{"type":"count_tokens","model_path":…,"text":…}` | `{"tokens":N}` |

`tools`, `cache_key` and `format` are optional. `messages` use OpenAI roles
(system, user, assistant, tool) and are rendered with the model's own chat
template. `cache_key` keeps the KV cache for one conversation, so the next
turn only prefills the new suffix.

The v1 requests still work: `ping`, `generate`, `generate_stream`, `embed`
and `embed_batch`. The docstring at the top of `estia-runner.py` lists every
request and answer.

Stream lines are `{"type":"token","text":"…"}`, `{"type":"keepalive"}` and
`{"type":"meta",…}`, and the stream ends with `{"done":true}`. A keepalive
carries nothing. The runner sends it while output is being filtered, so the
engine's per-line silence deadline does not kill a healthy process.

The engine sends `hello` once after it starts the process. A runner that
answers `hello` with an error is treated as protocol v1 with no capabilities.
`estia-runner.py` declares `generate`, `stream`, `embed`, `cancel`, `load`,
`chat`, `tools`, `prompt_cache` and `count_tokens`. Its `structured` list is
empty: it cannot constrain decoding, so it ignores `format` and the engine
validates and repairs structured output instead.

### One-shot

A new process per request. The request is all of stdin, and the engine closes
stdin after writing it. The answer is one JSON object, `{"ok":true,…}` or
`{"ok":false,"error":"…"}`, and then the process exits. Every call pays for a
process start and a model load.

| Request | Python (`oneshot-runner.py`) | Swift MLX | Apple |
|---|---|---|---|
| `health` | yes | yes | |
| `generate` | yes | yes | |
| `generate_stream` | yes: token lines, then `{"ok":true,"done":true}` | | |
| `embed` | yes | yes | |
| `apple_health`, `apple_generate` | | yes | yes |

On the engine side, `Session` (`engine/src/session.rs`) drives a resident
runner and `OneShot` (`engine/src/oneshot.rs`) drives a one-shot runner. The
`estia` CLI and server only use the resident runner.

## How the engine finds the resident runner

The `estia` CLI looks for the script in this order (`find_runner` in
`cli/src/main.rs`):

1. `--runner <path>`, or the `ESTIA_RUNNER` environment variable.
2. `runners/mlx-python/estia-runner.py` next to the `estia` binary.
3. The same path two directories above the binary, only when the binary is
   in a cargo `target/<profile>/` directory. This finds the script when the
   binary is `target/debug/estia` or `target/release/estia` in this
   repository.
4. Otherwise, the copy compiled into the binary, written to
   `<data_dir>/engine/runners/mlx-python/estia-runner.py` (and rewritten when
   the binary's copy changes). This is what a `cargo install`ed binary uses.

The current directory is never searched: running `estia` inside a folder you
downloaded must not execute a script that folder contains. The compiled-in
copy comes from `cli/estia-runner.py`, a symbolic link to
`mlx-python/estia-runner.py` that keeps the script inside the `estia` crate's
package.

The interpreter is `--python` or `ESTIA_PYTHON` if set, else the installed
runtime's `<data_dir>/runtime/python/bin/python3`, else `python3` on `PATH`.
`estia runtime install` (and `estia setup`) installs that runtime: a
standalone Python 3.12 with `mlx-vlm`, `mlx-lm` and `mlx-embeddings`.

`estia status` shows the runner and interpreter it found. `estia runner-check`
does the `hello` handshake without loading a model.

`estia service install` copies the binary to `<data_dir>/engine/estia` and
the runner directory's `.py` files to `<data_dir>/engine/runners/mlx-python/`,
then points the service at that copy through `ESTIA_RUNNER`.

A host that uses the `estia-engine` crate directly passes the script path to
`EngineConfig::new(store, runtime, runner_path)`. It can set the interpreter
with `EngineConfig::with_python`. Without one, the engine uses the installed
runtime and returns `EngineError::NoInterpreter` if that is missing.

## Bundling runners in a host

- **Python runners.** Ship the `.py` files as plain resources. If you ship
  the `estia` binary, keep `runners/mlx-python/` next to it. A library host
  can put the script anywhere and pass its path to `EngineConfig::new`. The
  interpreter and MLX packages are not shipped: `PythonRuntime` downloads them
  on first use. The installer is behind the engine's `python-mlx` feature.
- **Apple runner.** Build it, ship the `estia-apple-runner` binary, sign it
  with the host, and call it through `OneShot`:

  ```bash
  cd runners/apple
  swiftc -O -parse-as-library AppleRunner.swift -o estia-apple-runner
  echo '{"type":"apple_health"}' | ./estia-apple-runner
  ```

  Calling Foundation Models needs the macOS 26 SDK to build and macOS 26 with
  Apple Intelligence to run. With an older SDK or macOS it still builds and
  runs, and `apple_health` reports the model as unavailable.
- **Swift MLX runner.** For a host that must ship only signed code, such as a
  sandboxed Mac app. Ship the binary and its `*.bundle` directories side by
  side, sign it with the host, and build `estia-engine` without `python-mlx`
  (the default) so no downloader for executable code is compiled in. It also
  answers the Apple calls, so such a host does not need the Apple runner. See
  [`mlx-swift/README.md`](mlx-swift/README.md).

Compiled runner binaries and build directories are git-ignored.
