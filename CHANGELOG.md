# Changelog

## 0.1.0 — unreleased

First public release. Estia was developed by ModelCaddy and moved into this
repository with its history.

### Backend

- MLX on Apple Silicon macOS, through a resident Python runner
  (`runners/mlx-python/estia-runner.py`) using `mlx-vlm` for generation and
  `mlx-embeddings` for embeddings. This is the only working backend.
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

- No backend for Linux or Windows yet; a llama.cpp backend is planned.
- No TLS: LAN traffic, tokens included, is plain HTTP.
- Image input is not passed through the API.
- Role sampling settings (`temperature`, `max_tokens`, `pin`) are stored but
  not applied.
