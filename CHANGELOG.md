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
  `models:write`, `admin`), stored as SHA-256 hashes.
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

### Known limits

- No backend for Linux or Windows yet; a llama.cpp backend is planned.
- No TLS: LAN traffic, tokens included, is plain HTTP.
- Image input is not passed through the API.
- Role sampling settings (`temperature`, `max_tokens`, `pin`) are stored but
  not applied.
