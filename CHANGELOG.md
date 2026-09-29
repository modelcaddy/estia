# Changelog

Changes to Estia, newest first. [docs/versioning.md](docs/versioning.md) says
what the numbers mean: before 1.0, a new minor version (0.4 to 0.5) may break
things and a patch version (0.4.0 to 0.4.1) does not.

Estia was developed by ModelCaddy inside its app and became a standalone
repository in 0.3.0; this repository's history starts there, and the earlier
commits stay in ModelCaddy's own repository. The 0.1.0, 0.2.0 and 0.3.0
sections are retroactive: those versions were never tagged or published, and
builds made before 0.4.0 report version 0.0.1 or 0.1.0 whatever they contain.
The sections group that history by milestone, each dated by its last commit,
and use today's names (the project took the name Estia in 0.3.0).

## Unreleased

### Image input

- Chat messages can carry images: OpenAI `image_url` parts with a `data:` URL
  (PNG, JPEG, WebP, GIF), in `user` messages, up to 8 per request and 20 MB
  each. Remote URLs are refused (the engine never fetches for a client), as
  are unknown content-part types, which used to be replaced by a text marker.
- The Gemma 4 models read them on both backends, so they advertise `vision`
  again and the `vision` role requires it. On MLX the runner (2.4.0) passes
  images to mlx-vlm's vision tower, after converting to RGB and cropping to
  the content when it is a small island on a plain background (a measured
  gain on rendered pages). On llama.cpp each GGUF artifact now downloads its
  image projector (`mmproj.gguf`) and the adapter starts `llama-server
  --mmproj`; `estia import --mmproj <file>` does the same for your own
  models.
- A request with images for a model or runner that cannot read them is a 400
  that says so. Image turns bypass the prompt cache.
- `estia chat --image <file>` attaches images from the terminal.
- Protocol: `Message.images` (`{mime, data}` base64) and
  `capabilities.images`.

### Local apps connect without a token

- `serve` writes a same-user secret (`local-access.secret`, 0600, new on every
  start) and `POST /engine/local-token` trades it, from loopback only, for a
  token named `local-<app>` with `generate` and `embed` (at most
  `models:read` besides). An app on the engine's machine joins it with nothing
  to type; asking again replaces the app's token.
- For Rust hosts: `estia_engine::find_local_engine` and
  `LocalEngine::claim_token`, and `default_data_dir`.
- LAN discovery moved from `estia-server` into `estia-engine`
  (`discovery::discover`, feature `discovery`), so a host app can find engines
  without the server crate. `estia_server::discover` still works.

### Fixed

- The Rust remote client (`RemoteEngine`) panicked when built, used or
  dropped on a Tokio worker thread, which is where a host's async code calls
  it. It now moves its blocking HTTP work to a plain thread when called
  inside a runtime.
- `/engine/generate` dropped images sent in the protocol's own message form
  (`images: [{mime, data}]`, what `RemoteEngine` sends), so the model answered
  as if there were none. They are taken and checked like `image_url` parts.

### Memory budget

- The engine classifies the machine: `constrained` (16 GB or less, no fan,
  or 4 cores or fewer), `standard`, or `capable` (32 GB or more with a fan).
  `ESTIA_DEVICE_TIER` and `ESTIA_FAKE_RAM_GB` override the detection, to try
  a small machine's policy on a big one.
- Resident models stay within a memory budget: 50, 60 or 70 % of RAM by
  tier, or `serve --memory-budget` / `ESTIA_MEMORY_BUDGET` (`off` turns it
  off). Before a model loads, idle models are unloaded, least recently used
  first, until it fits. A model larger than the budget answers 503
  `insufficient_memory`; one that fits only by unloading a model in use
  answers 503 `engine_busy`. Loads are serialised so two never count the
  same free memory.
- A constrained machine keeps one generation model at a time and unloads an
  idle model after 3 minutes instead of 15. `serve --idle-unload-minutes`
  still wins, and now defaults to the tier's window.
- MLX runner 2.5.0: loads models lazily, reading in only the language model,
  so the Gemma 4 vision and audio towers stay on disk until an image arrives
  (0.9 GB less for E2B and E4B); caps wired memory to the tier's limit (a
  quarter of RAM on a constrained machine, instead of MLX's two thirds) and
  its buffer cache (256 MB there), from `ESTIA_MLX_WIRED_LIMIT_BYTES` and
  `ESTIA_MLX_CACHE_LIMIT_BYTES`, which the engine sets.
- `estia recommend` prints the tier, the budget, which Gemma 4 families fit,
  and the `text`, `fast` and `vision` bindings to use; `--apply` sets them.
  `estia status` shows the machine and its policy.
- `/engine/stats` has a `memory` object: tier, RAM, budget, what resident
  models use, and the caps.
- `service install` passes `--memory-budget` and `--idle-unload-minutes` on.
- For hosts: `estia_engine::machine` (profile, tier, `MemoryPolicy`,
  `recommend`), `EngineConfig::memory` / `with_memory`, `Launch::env`, and
  `idle_for` on sessions. `MemoryPolicy::unlimited()` is the old behaviour.

## 0.4.0 — 2026-09-27

The first tagged release.

The llama.cpp backend, version numbers that say which build is running, a
guide to running and testing Estia, and the fixes from an acceptance run
against a live engine.

### llama.cpp backend

From [docs/design/llama-backend.md](docs/design/llama-backend.md), which has
the status of each slice and the live evidence.

- **A second backend**, new and not yet run with the Gemma 4 GGUF files:
  upstream `llama-server` behind the `estia-llama` adapter. It is the default
  everywhere except Apple Silicon, where MLX stays the default.
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

### Versions and build information

See [docs/versioning.md](docs/versioning.md).

- **Version 0.4.0** for every crate in the workspace, and in the version
  requirements between them. The history before it is numbered 0.1.0 to
  0.3.0 in this file.
- **`estia --version`** prints one line, `estia 0.4.0 (<commit>, <date>)`.
- **`estia version`** prints the version, commit, build date, target and
  build profile, the Rust compiler, the HTTP API version, the runner protocol
  version, the backends and whether each runs on this machine, the engine
  features, the pinned llama.cpp build and the version of the MLX runner
  compiled into the binary. `estia version --json` prints the same as one
  JSON object. It reads no data directory, so it works before `setup`.
- **`/engine/health`** gains `build`, `{"commit": …, "date": …}`, the build
  of the server that answered. The other fields are unchanged.
- **The `estia serving` log line** gains `commit`.
- **Where the commit comes from:** git, with `+dirty` when a compiled-in file
  differs from the commit; `.cargo_vcs_info.json` in a crate built from its
  published package; or `ESTIA_BUILD_COMMIT` at build time. Otherwise
  `unknown`. `SOURCE_DATE_EPOCH` sets the date. The build never fails for
  want of git.

### Running and testing

- **[docs/running-and-testing.md](docs/running-and-testing.md)**, a guide:
  what a machine needs, installing from source or a release archive, a first
  run with MLX on a Mac and with llama.cpp on Linux, the service, where the
  logs are and how to follow one request by its id, testing from a phone or
  another computer, a troubleshooting table, and the test suites and CI for
  contributors. It says what was run for it and what was not; on Linux, only
  CI has run.
- **`scripts/smoke-test.sh`** checks a running engine the way a client sees
  it: health and build, auth, models, chat, streaming, the prompt cache, JSON
  Schema output, embeddings, the fingerprint check, the JSON 404, the `Host`
  check and request ids (12 checks; `--quick` runs 3). It needs bash, curl
  and python3, takes the token from `--token-file`, `--token` or
  `ESTIA_TOKEN`, tags its requests with `X-Request-Id`s that share one prefix
  so the engine's log lines for a run are easy to find, and exits 0 (all
  passed), 1 (a check failed) or 2 (could not start).
- The README gains a "Run and test" section and the `version` command; its
  build instructions name the Linux packages (a C compiler, `pkg-config`,
  the OpenSSL headers). `examples/README.md` points to the smoke test.
- **Release workflow:** the packaged binary must report the tag's commit,
  without `+dirty` (`estia version --json`), or the release stops. It
  releases only a commit on which CI passed, and a final version only when
  `CHANGELOG.md` has its section, which the release notes link. It builds
  with a read-only token; a separate job, the only one that can write,
  publishes the tarball after checking its SHA-256 again. Third-party
  actions are pinned to commit SHAs.
- A second `estia serve` on a data directory that already has an engine now
  says "one engine per data directory" instead of "one engine per machine",
  which was wrong: engines with different data directories can run side by
  side on one machine.

### Fixes from the acceptance run

An acceptance run on 2026-09-27 drove a live MLX engine on an M1 Pro (32 GB)
through its API, its examples and its performance. What it found, fixed and
re-tested live on a scratch engine; the ids are the run's finding ids.

- **One load per model** (perf-F1). Requests that arrive together for a
  model that is not loaded share one load: the first starts the runner and
  loads the weights, the others wait for it and then use the same session.
  Before, each started a runner of its own and loaded the weights again. A
  load that fails hands its error to everyone waiting on it; the next
  request tries again.
- **`finish_reason: "length"`** (api-F2) when the answer used up
  `max_tokens`, on `/v1/chat/completions` (streamed and not) and on
  `/engine/generate`, which gains `finish_reason` (`null` for a raw
  `prompt`). Runners report it in the new `meta.finish_reason`; the MLX
  runner (2.3.0) and the llama.cpp adapter both send it, and for a runner
  that does not, the server infers `length` from the token count. Before,
  a cut-off answer said `stop`. The access log's `finish` can say `length`.
- **Cancel during prefill** (api-F3). A client that closes a stream while a
  long prompt is still being read now frees the model within about a
  quarter of a second: the MLX runner prefills in chunks of about 0.25 s
  and checks for a cancel between them, and the server cancels as soon as
  the client goes instead of when the first token fails to send. With an
  8.5K-token prompt closed after 1 s, the next short request on the same
  model answered in 0.3 s on `gemma4-e2b` and `gemma4-e4b`; before, it
  waited for the whole prefill (about 5 s on e2b, 21 s on e4b). Chunking
  costs the 12B about 10% on a cold prefill; e2b loses nothing.
- **Request bodies up to 32 MiB** (api-F4), room for a full embed batch;
  axum's default refused 2 MB. `estia serve --max-body-bytes`, `estia
  service install --max-body-bytes` or `ESTIA_MAX_BODY_BYTES` change it (the
  flag wins). A larger body gets a JSON 413 that says the limit and how to
  raise it, with `request_id`; malformed JSON (400), a wrong content type
  (415) and missing fields (422) are JSON errors too, not `text/plain`.
- **`/engine/embed` with `inputs: []`** is a 400 (api-F5), as on
  `/v1/embeddings`, and loads no model.
- **Unique ids** (api-F6): `chatcmpl-` and 24 random hex digits for
  completions, `call_` and 24 for tool calls, set once per response. Before,
  completion ids repeated within a second and tool calls were numbered
  `call_0`, `call_1` in every response. An id a runner supplies is kept.
- **The prompt cache survives what used to reset it** (api-F7, perf-F2). The
  MLX runner keeps each conversation's cache itself: it records the tokens
  the cache holds and snapshots the sliding-window layers at the end of the
  history. On `gemma4-12b`, whose prompt ends in an empty thought channel
  the template drops from history, the second turn of a 1.7K-token
  conversation reused 1665 of 1707 tokens and took 1.9 s instead of 20 s.
  Sending the same messages again (a regenerate) reuses all but the last
  few tokens (1700 of 1707 on the 12B) and gives the same answer at
  temperature 0. A turn after a cancel reuses what was prefilled before the
  cancel. The snapshot costs about 330 MB per 12B conversation.
- **Trailing text after JSON** (api-F1). Structured output that closes its
  object or array and then goes on (`Hope this helps!`, a stray `}`, a
  second object) is used, with `strip_trailing_text` in `repairs`, instead
  of failing with 422. A code fence after a preamble is honoured. A lone
  number or string must still be the whole output, and truncated JSON is
  not repaired. A fallback tool call written as JSON and followed by a
  sign-off is still a call. `estia_engine::structured::first_json_value` is
  public.
- **Capabilities say what reaches the model** (api-F8). Image parts are not
  passed to the model, so no Gemma 4 artifact advertises `vision` any more:
  they list `text` and `tools`, on `/engine/models` and, new,
  `/v1/models` (`x_estia.capabilities`). The `vision` role needs only text
  until a model advertises vision, so the default role table, which binds
  `vision` to `gemma4-e4b`, still passes its own check.
- **Load time and error messages** (api-F9). `x_estia.load_ms` (and
  `load_ms` on `/engine/generate` and `/engine/embed`) is how long a request
  waited for its model to load, `null` when it was loaded; `ms` is the work
  itself (`/engine/embed`'s used to include the load). The 404 for a model
  that is not installed names no path on the engine's machine and says to
  run `estia pull <id>`; the path goes to the log. Unknown names get
  ``unknown model or role `…` ``. The 500s for a missing runner script or
  llama adapter name no paths either.
- **`encoding_format: "base64"`** on `/v1/embeddings` (examples-1): each
  vector as its little-endian f32 bytes in standard base64, what OpenAI's
  JavaScript SDK asks for by default. `float` stays the default; anything
  else is a 400. Before, the SDK decoded the engine's arrays as base64 and
  got 192 wrong numbers instead of 768.
- **Memory per model in `/engine/stats`**: `models` lists each loaded model
  with its runner's `pid` and `memory_bytes`, the physical footprint of the
  runner and every process it started (so a llama.cpp model's
  `llama-server` is counted). It includes the Metal buffers an MLX runner
  keeps its weights in, which `ps` leaves out: 4.0 GB for `gemma4-e2b`
  where `ps` shows about 100 MB. `loaded` is unchanged. For hosts:
  `estia_engine::procmem` (`phys_footprint`, `tree_footprint`,
  `children_of`), `GenSession::pid` and `EmbedSession::pid`, and
  `Session::current_pid`, which never waits for a call in progress.
- **Examples and docs** (examples-2 to 6). `remote_client.rs` reads the
  engine's embedding fingerprint instead of assuming the MLX one, so it runs
  against a llama.cpp engine. The curl snippets use `$ESTIA_TOKEN`. Every
  `estia pair` subcommand, argument and option has help text. `pair.py` and
  `estia pair request` stop waiting at 290 s, before the engine drops an
  undecided request at 300 s, and say so, instead of ending on a 404. The
  API reference documents the fields Estia accepts and ignores
  (`tool_choice`, `stop`, `n`), `/engine/stats`, and the new fields and
  limits above.
- **MLX runner 2.3.0** carries the runner side of these: chunked prefill,
  `meta.finish_reason`, the conversation cache, and a cancelled stream that
  records exactly what its cache holds.

### Licences

- `estia pull` and `estia setup` print a model's licence before they
  download it. For the Gemma Terms of Use (EmbeddingGemma) they add the links
  to the terms and the prohibited-use policy; a vendor licence Estia has no
  links for points to the model's Hugging Face card. `estia models` has a
  licence column, and the Models table in `/client` a Licence column.
- The MLX runtime install pins `mlx-embeddings` to 0.1.0. It is GPL-3.0;
  Estia does not ship it, and `estia runtime install` fetches it from PyPI.
  The pin does not force a reinstall.

### Other fixes

- `estia import --label`, `--query-prefix` and `--doc-prefix` each have
  their own help text.
- `estia setup` no longer writes a `backend` key into `config.json` when the
  backend is only this machine's default; running it again changes nothing.
  An explicit `--backend` (or `ESTIA_BACKEND`) is still saved.
- `estia dashboard` shows `loaded : none` instead of a blank list.
- CI and release builds use a pinned Rust (1.95.0) instead of the newest
  stable, so a new Rust release cannot fail CI with new lints; it is bumped on
  purpose (docs/versioning.md).

### Known limits

[ROADMAP.md](ROADMAP.md) says which of these are planned to change.

- The llama.cpp backend has run end to end only on an Apple Silicon Mac with
  small test models. The Gemma 4 GGUF files, Linux (outside CI) and NVIDIA
  GPUs are untested, and Estia does not build for Windows.
- No TLS: LAN traffic, tokens included, is plain HTTP.
- Image input is not passed through the API: image parts in a message are
  dropped, and no model advertises `vision`.
- A small model can stray into another script. Asked in Greek, `gemma4-e2b`
  wrote one syllable in Hebrew letters (`Ολύבותρος` for Olympus); the model
  chose that token (probability 0.30), and Estia passes tokens through
  unchanged (perf-F10).
- Role sampling settings (`temperature`, `max_tokens`, `pin`) are stored but
  not applied.
- A non-streaming request is not cancelled when its client disconnects, and
  there is no time limit on reading a request body.

## 0.3.0 — 2026-09-26

Ready to stand alone: the Estia name, hardening, logs, examples and
documentation.

### The Estia name

- Crates `estia-proto`, `estia-engine`, `estia-server` and `estia` (the
  binary), `ESTIA_*` environment variables, the `estia_` token prefix,
  `x_estia` in responses, `_estia._tcp` on Bonjour, the launchd label
  `com.modelcaddy.estia` and the systemd unit `estia.service`. Tokens minted
  before the rename still verify.

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
  protocol-v2 adapter).
- The README, [docs/api.md](docs/api.md), [docs/protocol.md](docs/protocol.md),
  CONTRIBUTING.md and SECURITY.md, written for Estia on its own.

### Hardening before release

These change behaviour from 0.2.0.

- **Host and Origin checks.** The server answers only to a `Host` that is an
  IP literal, `localhost` / `*.localhost`, a `*.local` name, this machine's
  hostname, or a name allowed with `serve --allow-host` (also on
  `service install`) or `ESTIA_ALLOWED_HOSTS`. A state-changing request whose
  `Origin` is not the origin it was sent to is refused. Both are 403, on every
  route, before authentication. This closes DNS rebinding and cross-site posts
  from web pages, including against `--no-auth`.
- **Revoked tokens.** A token revoked with `estia token revoke` kept working
  in a running server until it restarted. The server now reads `tokens.json`
  again whenever the file changes.
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
  from a cargo `target/<profile>/` directory. A binary with no runner beside
  it writes the copy compiled into it to `<data_dir>/engine/runners/`.
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
  `runners/mlx-python/estia-runner.py`. On phones, fields no longer zoom the
  page on focus and every tab fits the screen.
- `status` hints lead with loopback (`estia serve`,
  `service install --local`); the dashboard clock shows local time.

### Distribution

- A release workflow that builds a tarball for `aarch64-apple-darwin` on
  GitHub Actions when a `v*` tag is pushed: the `estia` binary, the runner
  scripts, `LICENSE`, `NOTICE`, `README.md` and `THIRD_PARTY_LICENSES` (the
  licence texts of the Rust crates compiled into the binary, generated by
  cargo-about).
- CI on macOS and Linux: rustfmt, clippy, build, tests, the minimum Rust
  version, packaging and cargo-deny.
- Every crate carries `LICENSE`, `NOTICE` and the README.
- Minimum supported Rust version: 1.89.

## 0.2.0 — 2026-09-26

The LAN daemon: an HTTP server over the engine, pairing, Bonjour discovery, a
login service, and a client for remote engines. Built from 2026-09-09.

### Runner protocol v2

- Newline-delimited JSON over stdin/stdout, as before, plus a `hello`
  handshake with capabilities, `load` / `unload`, `chat` and `chat_stream`
  rendered by the model's own chat template, a per-conversation KV cache
  (`cache_key`), and `count_tokens`. v1 runners still work.

### Server (`estia-server`)

- OpenAI-compatible `/v1/chat/completions` (streaming, tools, `response_format`,
  prompt cache keyed by `user`), `/v1/embeddings` and `/v1/models`.
- Native `/engine/*` routes: health, role defaults, model list, pulls and
  deletes with job progress over server-sent events, generation against a
  JSON Schema, embeddings with a fingerprint check, runtime install, stats and
  jobs.
- Bearer tokens with scopes (`generate`, `embed`, `models:read`,
  `models:write`, `admin`), stored as SHA-256 hashes. A running server picks
  up tokens minted with `estia token` on the next request, without a restart.
- LAN mode with pairing (request, operator approval, one-time token pickup)
  and Bonjour advertisement as `_estia._tcp`.
- Idle unload of resident models (15 minutes by default).
- A static browser test client at `/client`.

### Engine (`estia-engine`)

- `RemoteEngine`, `RemoteGen` and `RemoteEmbed` for using a server from Rust.

### CLI (`estia`)

- `setup`, `service` (launchd on macOS, systemd `--user` on Linux), `serve`,
  `dashboard`.
- `pair`, `discover`, `remote-check`, `token`.
- `chat`, `tokens`, `runner-check`.

## 0.1.0 — 2026-09-08

The engine and the CLI.

### Backend

- MLX on Apple Silicon macOS, through a resident Python runner
  (`runners/mlx-python/estia-runner.py`) using `mlx-vlm` for generation and
  `mlx-embeddings` for embeddings.
- The runner protocol (version 1): newline-delimited JSON over stdin/stdout,
  with `cancel`.
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

### CLI (`estia`)

- `status`, `models`, `pull`, `rm`, `roles`, `runtime`.
- `run`, `embed`, `bench`.
