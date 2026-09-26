# Changelog

## 0.1.0 — unreleased

First public release. Estia was developed by ModelCaddy and moved into this
repository with its history.

### Backend

- MLX on Apple Silicon macOS, through a resident Python runner
  (`runners/mlx-python/estia-runner.py`) using `mlx-vlm` for generation and
  `mlx-embeddings` for embeddings. The default on Apple Silicon.
- llama.cpp, new and not yet run with the Gemma 4 GGUF files: upstream
  `llama-server` behind the `estia-llama` adapter. The default everywhere
  else. See [llama.cpp backend](#llamacpp-backend) below.
- Runner protocol v2: newline-delimited JSON over stdin/stdout with a `hello`
  handshake and capabilities, `load` / `unload`, `chat` and `chat_stream`
  rendered by the model's own chat template, a per-conversation KV cache
  (`cache_key`), `count_tokens`, and `cancel`. v1 runners still work.
- One-shot runners kept alongside: the older Python runner, an Apple
  Foundation Models runner and a compiled Swift MLX runner. The CLI and server
  do not use them.

### Engine (`estia-engine`)

- Runner sessions with per-line deadlines, one respawn on a dead child, a
  priority gate that serves interactive calls before background ones, and
  cancel.
- A built-in model registry: three Gemma 4 generation models and four
  embedding models, grouped into families.
- A Hugging Face downloader with resume, parallel chunks, SHA-256 checks and
  pause, and an on-disk model store.
- A Python runtime installer: a pinned `python-build-standalone` build plus
  the MLX packages, installed under the data directory (feature `python-mlx`).
- Roles (`text`, `fast`, `vision`, `code`, `embed`, and any other name) bound
  to model families, checked against capabilities, with fallbacks.
- Structured output: a JSON repair ladder, JSON Schema validation and a retry
  hint.
- Embedding fingerprints (`<model id>@<backend>`) and task prefixes per model.
- `RemoteEngine`, `RemoteGen` and `RemoteEmbed` for using a server from Rust.

### Server (`estia-server`)

- OpenAI-compatible `/v1/chat/completions` (streaming, tools, `response_format`,
  prompt cache keyed by `user`), `/v1/embeddings` and `/v1/models`.
- Native `/engine/*` routes: health, role defaults, model list, pulls and
  deletes with job progress over server-sent events, generation against a
  JSON Schema, embeddings with a fingerprint check, runtime install, stats and
  jobs.
- Bearer tokens with scopes (`generate`, `embed`, `models:read`,
  `models:write`, `admin`), stored as SHA-256 hashes. A running server picks
  up tokens minted or revoked with `estia token` on the next request, without
  a restart.
- LAN mode with pairing (request, operator approval, one-time token pickup)
  and Bonjour advertisement as `_estia._tcp`.
- Idle unload of resident models (15 minutes by default).
- A static browser test client at `/client`.

### CLI (`estia`)

- `setup`, `service` (launchd on macOS, systemd `--user` on Linux), `serve`,
  `status`, `dashboard`.
- `pair`, `discover`, `remote-check`, `token`.
- `models`, `pull`, `rm`, `roles`, `runtime`.
- `run`, `chat`, `embed`, `tokens`, `bench`, `runner-check`.

### llama.cpp backend

From [docs/design/llama-backend.md](docs/design/llama-backend.md), which has
the status of each slice and the live evidence.

- **`estia-llama`**, a new crate: an adapter that speaks runner protocol v2 on
  stdin and stdout and runs one upstream `llama-server` per model on a private
  UNIX socket with a random API key. Streaming, cancel (the HTTP connection is
  closed and `llama-server` stops the task), JSON and JSON Schema output
  constrained by a grammar, tool calls parsed by `llama-server` and returned
  in `meta.tool_calls`, `count_tokens`, embeddings, a prompt-cache slot reused
  only by the cache key that filled it, explicit sampling, thinking off.
  `llama-server` is stopped when the adapter exits, is signalled, loses its
  parent, or (on macOS, through a guard process) is killed. The `estia` binary
  runs it as `estia runner llama`; the crate also builds a small
  `estia-llama` binary.
- **Protocol:** `Message.tool_calls`, `meta.tool_calls`,
  `meta.generation_tps`, and the capabilities `parses_tool_calls` and
  `backend`. Old runners and engines ignore them.
- **Backends in the engine:** `Backend` (`mlx-python`, `llama-cpp`),
  `EngineConfig::with_backend` and `with_llama(LlamaLaunch)`, resolution of a
  role or family to the backend's artifact, and a check that refuses an
  artifact or fingerprint of the other backend. Stale `llama-server`
  processes whose adapter died are stopped when an engine starts.
- **Stopping a runner** (idle unload, shutdown, a missed deadline) now sends
  SIGTERM and waits up to 4 s before SIGKILL, on both backends. The llama.cpp
  adapter uses the time to stop its `llama-server` and remove its socket and
  record from `run/`.
- **Models:** GGUF artifacts for the three Gemma 4 families (Google's QAT
  Q4_0 files), EmbeddingGemma 300M Q8_0 and Nomic Embed Text v1.5 Q8_0,
  pinned to commits with explicit file lists and SHA-256 hashes. Embedding
  models have one artifact per format; fingerprints are
  `<artifact id>@<backend>`, so MLX fingerprints do not change.
- **`estia import`** registers a GGUF file of your own, reading its kind,
  context length, width and pooling from its metadata; copied or linked.
- **Runtime:** `estia runtime install --backend llama [--variant …]` (engine
  feature `llama-runtime`) downloads the pinned llama.cpp build b11146 for the
  platform and accelerator, checks its SHA-256, unpacks it and runs
  `--version`. `ESTIA_LLAMA_SERVER` uses your own `llama-server`;
  `ESTIA_LLAMA_ARGS` adds arguments, minus the ones that would open it up.
- **CLI and server:** a global `--backend` / `ESTIA_BACKEND`, saved to
  `config.json` by `setup`. `/engine/health` gains `backend` and a `backends`
  list with `active`, `supported`, and for llama.cpp `build`, `variant` and
  `server`. `/engine/models` and `/v1/models` mark artifacts `runnable` on the
  running backend and list imports. `POST /engine/runtime/install` takes
  `{backend, variant}`. `x_estia` gains `generation_tps`, and streamed
  `tool_calls` carry `index`.
- **The `embed` role** can be bound to any embedding model, built-in or
  imported.
- **CI:** a `llama` job on Linux and macOS runs the adapter's integration
  tests against the pinned `llama-server` with two small models, and fails if
  a test skipped or a process was left behind.

### Tool results and JSON Schemas reach the model

- An assistant message's `tool_calls` now reach the runner as structured
  data, so the model's chat template renders the `{"role": "tool"}` results
  after them. Before, the calls reached the runner as text and Gemma's
  template dropped the results.
- The MLX runner (2.2.0) passes the calls and results to the template, and
  when Gemma 4 ends its turn with no text right after a tool result, asks
  once more in a new model turn. It also reports `generation_tps`.
- `response_format` reaches the model on both backends: as `format` to a
  runner that constrains decoding (llama.cpp), otherwise as an instruction
  with the schema in the system prompt (MLX), or after a raw prompt on
  `/engine/generate` and `estia run --schema`. The output is still validated.
  `structured::with_prompt_hint`, `prompt_with_hint` and `prompt_hint` are
  public for hosts that run the engine in process.
- `examples/python/tools.py` sends results as `{"role": "tool"}`;
  `structured.py`, `quickstart.sh` and `in_process.rs` no longer put the
  schema in the prompt. The guide drops both workarounds.
- The `/client` page names the backend, describes the right runtime, and
  leaves models the backend cannot run out of its pickers.

### Logging and request ids

See [docs/logging.md](docs/logging.md).

- **Request ids.** Every response carries an `X-Request-Id` header. A client
  may send its own (1 to 64 ASCII letters, digits, `.`, `_`, `:`, `-`);
  anything else is replaced with 16 random hex digits. JSON error bodies carry
  the id as `error.request_id`. A streamed chat completion that fails after it
  started ends with `{"error": {"message", "type", "request_id"}}`; a failed
  `/engine/generate` stream sends `{"error": "…", "request_id": "…"}`.
- **Access log.** One line per request, target `estia_server::access`: method,
  path (never the query string), status, duration, peer, the token's name,
  and for model work the model asked for and served, load time, token counts,
  time to first token, tokens per second and how it finished. A stream is
  logged when it ends, so a stream the client abandons shows
  `finish=cancelled`. 5xx responses and failed streams are `warn`; successful
  polls (`health`, `stats`, `pairings`, `jobs`, `pair/{id}`) are `debug`. A
  401 or 403 records its reason. Estia never logs prompts, completions,
  embedding inputs, vectors, tokens, cache keys or a 422 message.
- **Lifecycle events** for startup, runner start, handshake, restart, timeout
  and cancel, model load, ready, release and unload, pulls, runtime installs,
  role changes, pairing, token changes seen by the server, Bonjour, connection
  limits and shutdown. Events emitted while a request is handled carry its id
  in a `request` span. Pulls and installs carry `job_id`.
- **Filters and formats.** `serve --log-level` / `ESTIA_LOG` and
  `--log-format text|json` / `ESTIA_LOG_FORMAT`. The default is `info` for
  Estia and `warn` for libraries. `RUST_LOG` is still read, before
  `ESTIA_LOG`. An invalid `ESTIA_LOG` stops `serve`; an invalid `RUST_LOG`
  directive is skipped with a warning. Other commands log Estia's warnings and
  the runner's stderr.
- **No prints in the libraries.** `estia-engine` and `estia-server` emit only
  `tracing` events and install no subscriber; a program that embeds them sees
  nothing until it installs one. The hand-written stderr logger and the `log`
  dependency are gone; `mdns-sd` output still arrives, through `tracing-log`.
- **Runner stderr** is read by Estia and logged line by line (target
  `estia_engine::runner`) instead of passing straight through, so JSON logs
  stay one object per line.
- **Service.** `service install --log-level/--log-format` checks the value and
  writes it into the service definition. `service logs` gains `-f/--follow`
  and `-n/--lines`.

Behaviour changes that come with it:

- The server loads a model with an explicit `load` right after starting its
  runner, so the load time can be logged. The first request waits the same
  total time as before.
- When `serve` mints the first admin token itself, it prints the token only
  if stderr is a terminal. Otherwise it logs a warning without the token,
  because stderr is then a log file; run `estia token new local --replace` to
  get one.
- On Linux, `service logs` reads the journal. It used to read files that
  systemd never writes.

### For builders

- [docs/building-clients.md](docs/building-clients.md): a guide for putting an
  app, assistant or tool on top of Estia: roles, tokens and scopes, pairing,
  discovery, the prompt cache, streaming and cancelling, structured output,
  tools, embeddings and fingerprints, priorities, limits, errors and request
  ids, Host and Origin rules, web front-ends, and Rust.
- [examples/](examples/README.md): a curl walkthrough
  (`curl/quickstart.sh`); Python chat, retrieval over notes, JSON Schema
  extraction, tools, pairing, and a chat with no SDK; JavaScript chat and
  embeddings; and two Rust examples in `engine/examples/`, `remote_client`
  (`RemoteEngine`) and `in_process` (`Engine` with no server). Each was run
  against a live engine and prints the request id with any error.
- [ROADMAP.md](ROADMAP.md): where Estia is today and what comes next, with the
  check that closes each item.
- [docs/design/llama-backend.md](docs/design/llama-backend.md): the design for
  a llama.cpp backend (upstream `llama-server` as a child process behind a
  protocol-v2 adapter), now with its implementation status.

### Hardening before release

These change behaviour for anyone who used the pre-release builds inside
ModelCaddy.

- **Host and Origin checks.** The server answers only to a `Host` that is an
  IP literal, `localhost` / `*.localhost`, a `*.local` name, this machine's
  hostname, or a name allowed with `serve --allow-host` (also on
  `service install`) or `ESTIA_ALLOWED_HOSTS`. A state-changing request whose
  `Origin` is not the origin it was sent to is refused. Both are 403, on every
  route, before authentication. This closes DNS rebinding and cross-site posts
  from web pages, including against `--no-auth`.
- **Pairing names.** At most 64 characters of letters, digits, single spaces
  and `. _ - ' ’ ( )`; control, escape, invisible and bidi characters are
  refused with 400 instead of being stored and printed. Unknown scopes are
  400. The CLI also escapes anything unprintable it shows from the engine or
  the network.
- **Pairing caps.** At most 4 pending requests per client address, besides 24
  overall; either cap now answers 429 (`rate_limit_error`) instead of 400.
- **Admin approvals.** `estia pair approve` refuses a request for `admin`
  unless given `--allow-admin`, and warns about `models:write`. `pair list`
  columns are now ID, STATUS, SCOPES, FROM, NAME.
- **Deny revokes.** Denying an approved pairing revokes the token it minted,
  collected or not; the response carries `revoked` and `token_name`.
- **Pairing and token files.** Every change to `pairings.json` and
  `tokens.json` holds a lock (`pairings.lock`, `tokens.lock`), so concurrent
  requests, approvals and polls lose nothing and a token is handed out exactly
  once. Both files are written 0600 from the start. A `pairings.json` that
  does not parse is an error rather than an empty list. Pairing storage
  failures are 500; `estia pair request` keeps polling through them.
- **Token names are unique.** `estia token new` refuses an existing name;
  `--replace` rotates it (keeping its scopes unless `--scopes` is given).
  `token revoke` of an unknown name and `roles rm` of an unbound role exit 1.
- **Request caps.** `max_tokens` above 8192 is lowered to 8192;
  `max_attempts` is held to 1–3; more than 256 embedding inputs is a 400.
- **Connection limits.** A 10-second limit to send request headers (which
  also closes idle keep-alive connections), 32 connections per client address
  (loopback exempt), and a total cap below the open-files limit, which the
  server raises at start. The server now speaks HTTP/1.1 only.
- **Prompt cache per token.** Cache keys are scoped to the caller's token, so
  `cached_tokens` cannot reveal another client's conversation.
- **Unknown routes are 404.** A path that matches no route gets a JSON 404
  instead of 401; `/client/` redirects to `/client`.
- **Runner lookup.** The CLI no longer looks for `estia-runner.py` under the
  current directory, and looks two levels above the binary only when it runs
  from a cargo `target/<profile>/` directory.
- **Service files.** Paths in the systemd unit are quoted and escaped; paths
  with control characters are refused for both launchd and systemd.
- **Bonjour on macOS.** The `dns-sd` registration ends with the engine, even
  when the engine is killed, instead of advertising a dead engine.
- **IPv6 binds.** `serve --bind` takes `localhost`, `::1` and host names;
  `status`, `dashboard` and the one-engine-per-data-directory check find an
  engine bound to IPv6.
- **Runtime install.** `mlx-lm` is installed explicitly: `mlx-vlm` 0.7 no
  longer depends on it, and a clean `estia setup` failed verification.
- **Test page.** The `/client` page moved to `server/client/index.html`, inside
  the server crate that embeds it; `clients/web/index.html` links to it. The
  CLI's embedded runner is likewise `cli/estia-runner.py`, a link to
  `runners/mlx-python/estia-runner.py`.
- `status` hints lead with loopback (`estia serve`,
  `service install --local`); the dashboard clock shows local time.

### Distribution

- A release tarball for `aarch64-apple-darwin`, built on GitHub Actions: the
  `estia` binary, the runner scripts, `LICENSE`, `NOTICE`, `README.md` and
  `THIRD_PARTY_LICENSES` (the licence texts of the Rust crates compiled into
  the binary, generated by cargo-about).
- Every crate carries `LICENSE`, `NOTICE` and the README.
- Minimum supported Rust version: 1.89.

### Known limits

[ROADMAP.md](ROADMAP.md) says which of these are planned to change.

- The llama.cpp backend has run end to end only on an Apple Silicon Mac with
  small test models. The Gemma 4 GGUF files, Linux (outside CI) and NVIDIA
  GPUs are untested, and Estia does not build for Windows.
- No TLS: LAN traffic, tokens included, is plain HTTP.
- Image input is not passed through the API.
- Role sampling settings (`temperature`, `max_tokens`, `pin`) are stored but
  not applied.
- A non-streaming request is not cancelled when its client disconnects, and
  there is no time limit on reading a request body.
