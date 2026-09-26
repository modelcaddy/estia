# HTTP API

`estia serve` answers on one port (27200 by default) with two groups of
routes:

- `/v1/*` follows OpenAI's request and response shapes, so OpenAI SDKs and
  tools work with only a base URL and a key.
- `/engine/*` covers what OpenAI's shapes cannot express: health, role
  defaults, model downloads with progress, generation against a JSON Schema,
  embeddings with a fingerprint check, pairing, stats and jobs.

The code is in `server/src/` (`openai.rs`, `engine_api.rs`, and
`required_scope` in `lib.rs`). The `/engine/*` contract is versioned:
`/engine/health` reports `api_version` (currently `1`).

## Authentication

Send `Authorization: Bearer <token>`. Tokens are minted with `estia token new`,
by `estia setup` or the first `estia serve`, or through pairing.

| Scope | Allows |
|---|---|
| `generate` | chat completions and `/engine/generate` |
| `embed` | `/v1/embeddings` and `/engine/embed` |
| `models:read` | listing models, roles, jobs and stats |
| `models:write` | downloading and deleting models |
| `admin` | everything, including roles, runtime install and pairing decisions |

A missing or unknown token gets `401`; a token without the needed scope gets
`403`. `estia serve --no-auth` turns authentication off, and is refused unless
the server is bound to loopback.

## Routes

| Method | Path | Scope | What it does |
|---|---|---|---|
| POST | `/v1/chat/completions` | `generate` | Chat completion, streaming or not |
| POST | `/v1/embeddings` | `embed` | Embeddings |
| GET | `/v1/models` | `models:read` | Bound roles, then generation and embedding models |
| GET | `/engine/health` | none | Version, bind address, auth, backends, loaded models |
| GET | `/engine/defaults` | `models:read` | The role table |
| PUT | `/engine/defaults` | `admin` | Replace the role table (applied now, saved to `config.json`) |
| GET | `/engine/models` | `models:read` | Every known model with install state and size |
| POST | `/engine/models/pull` | `models:write` | Start a download; returns a job id |
| DELETE | `/engine/models/{id}` | `models:write` | Remove an installed model and any partial download |
| GET | `/engine/models/{id}/progress` | `models:read` | Server-sent events for the running download of `id` |
| POST | `/engine/generate` | `generate` | Generate from a prompt or messages, optionally against a JSON Schema |
| POST | `/engine/embed` | `embed` | Embeddings with task prefixes and a fingerprint |
| GET | `/engine/stats` | `models:read` | Uptime, loaded models, queue depth, job count |
| GET | `/engine/jobs` | `models:read` | All jobs |
| GET | `/engine/jobs/{id}` | `models:read` | One job |
| GET | `/engine/jobs/{id}/events` | `models:read` | Server-sent events for one job until it ends |
| POST | `/engine/runtime/install` | `admin` | Install the Python MLX runtime as a job |
| POST | `/engine/pair` | none | Ask for a token (pairing request) |
| GET | `/engine/pair/{id}` | none | Poll a pairing request; returns the token once |
| GET | `/engine/pairings` | `admin` | Pending and recent pairing requests |
| POST | `/engine/pairings/{id}/approve` | `admin` | Approve: mint a token with the requested scopes |
| POST | `/engine/pairings/{id}/deny` | `admin` | Deny |
| GET | `/client` | none | The browser test client (a static page) |
| GET | `/` | none | Redirects to `/client` |

Model ids, families and roles are explained in the README under
[Roles](../README.md#roles). Wherever a generation request takes `model`, it
accepts an artifact id, a family or a role, in that order of precedence.
Embedding requests take an embedding model id or `embed`.

## Errors

Errors raised by the handlers use OpenAI's shape:

```json
{"error": {"message": "model `gemma4-e2b-it-4bit-mlx` is not downloaded at …", "type": "not_found_error", "code": 404}}
```

| Status | When |
|---|---|
| 400 | Empty `messages` or `input`, neither `prompt` nor `messages`, unknown `response_format` type or embedding `task`, unknown scope, empty pairing name, a pairing that cannot be approved or denied |
| 401 | Missing or unknown token |
| 403 | Token lacks the scope |
| 404 | Unknown model, model not downloaded, unknown job, unknown pairing id on poll |
| 422 | Structured output still invalid after the retry; embedding fingerprint mismatch |
| 500 | Runner failure |

A body that is not JSON, lacks `Content-Type: application/json`, or does not
fit the route's fields is rejected before the handler runs, with a plain-text
400, 415 or 422 from the HTTP framework.

## POST /v1/chat/completions

Supported request fields:

| Field | Notes |
|---|---|
| `model` | Role, family or artifact id |
| `messages` | `system`, `user`, `assistant`, `tool`. `content` may be a string or an array of parts. Text parts are joined. Other parts, such as images, are replaced by a marker like `[image_url omitted]`: image input is not passed to the model yet. An assistant message's `tool_calls` are passed back to the model as JSON text. |
| `tools` | OpenAI tool schemas. The model answers in its own call syntax and the server converts it to `tool_calls`. Gemma 4's native syntax and a `{"tool_call": {"name", "arguments"}}` object are recognised. |
| `response_format` | `text`, `json_object`, or `json_schema` with `json_schema.schema`. See [Structured output](#structured-output). |
| `max_completion_tokens`, `max_tokens` | Default 1024 |
| `temperature` | Default 0.2 |
| `stream` | Server-sent events, ending with `data: [DONE]` |
| `user` | Used as the prompt-cache key |
| `priority` | Estia extension: `interactive` (default) or `background` |

Other OpenAI fields (`tool_choice`, `n`, `top_p`, `stop`, and so on) are
accepted and ignored.

**Prompt cache.** Requests with the same cache key reuse the runner's KV cache,
so a conversation that grows by appending only prefills its new turns. The key
is `user` when given. Otherwise the server derives one from the model, the
leading system messages and the first user message, which stays the same for
every turn of a conversation.

**Response.** Standard fields, plus:

- `usage.prompt_tokens_details.cached_tokens`: prompt tokens served from the
  cache.
- `x_estia`: `family`, `backend`, `cached_tokens`, `template` (`native` or
  `manual`), `repaired` and `repairs` (for JSON output), `ms`.
- `finish_reason` is `tool_calls` when tool calls were parsed, otherwise
  `stop`.

**Streaming.** Prose streams as `delta.content` chunks. When `tools` are
declared, the server holds the first characters back to tell a tool call from
prose; a tool call, or any JSON-mode output, is sent in the final chunk
instead of token by token. The final chunk carries `finish_reason`, `usage` and
`x_estia` (`cached_tokens`, `template`, `ms`). If the client disconnects, the
generation is cancelled in the runner.

## POST /v1/embeddings

| Field | Notes |
|---|---|
| `model` | `embed` (currently `embeddinggemma-300m-4bit`) or an embedding model id |
| `input` | A string or an array of strings |
| `task` | Estia extension: `document` (default), `query`, `clustering`, or `none`. The model's own prefix for that task is prepended. `none` sends the inputs unchanged. |
| `expect_fingerprint` | Estia extension: refuse with 422 unless the server would produce vectors with this fingerprint |
| `priority` | Estia extension: `interactive` (default) or `background` |

The response has OpenAI's `data[].embedding` float arrays. `encoding_format` is
ignored; vectors are always floats. `usage` is reported as zero. `x_estia`
carries `fingerprint`, `dims` and `task`.

A **fingerprint** is `<model id>@<backend>`, for example
`embeddinggemma-300m-4bit@mlx-python`. The same model run by two backends gives
vectors that cannot be compared, so store the fingerprint with your index and
send it back as `expect_fingerprint`.

## GET /v1/models

Bound roles come first, each with `x_estia: {"role": true, "family": …}`. Then
every generation model (`x_estia`: `family`, `format`, `installed`,
`context_length`) and every embedding model (`x_estia`: `kind`, `dims`,
`installed`). Models that are not downloaded are listed too; check
`installed`.

## GET /engine/health

Open, so clients can check an engine before they have a token.

```json
{"ok": true, "engine": "estia", "version": "0.0.1", "api_version": 1, "protocol_version": 2,
 "bind": "127.0.0.1:27200", "auth_required": true, "uptime_s": 3,
 "backends": [{"id": "mlx-python", "runtime_installed": false}], "loaded": []}
```

`version` is the crate version of the server that answered.

## GET, PUT /engine/defaults

The role table, as stored under `roles` in `config.json`:

```json
{"fast": {"family": "gemma4-e2b", "pin": false},
 "text": {"family": "gemma4-e4b", "pin": false},
 "vision": {"family": "gemma4-e4b", "pin": false}}
```

`PUT` replaces the whole table. Each binding is checked against the family's
capabilities; a bad one fails the request with 400 and nothing changes. On
success the table applies at once and is written to `config.json`. A binding
may also carry `temperature` and `max_tokens`; they are stored but not applied
yet.

## Models and jobs

`GET /engine/models` returns `{"generation": […], "embedding": […],
"models_dir": …}`. Each entry has `id`, `kind`, `format`, `label`, `repo_id`,
`revision`, `license`, `installed`, `bytes_on_disk` and `required_disk_bytes`.
Generation entries add `family`, `context_length`, `capabilities`,
`partial_bytes` and `pulling` (the running job id, if any). Embedding entries
add `dims`, `arch`, `multilingual` and `fingerprint`.

`POST /engine/models/pull` with `{"id": "gemma4-e2b-it-4bit-mlx"}` answers
`202 {"job_id": …}`, or `{"job_id": …, "already_running": true}` if that model
is already downloading. Downloads come from Hugging Face, resume after an
interruption, and check SHA-256 where the repository publishes one.

`POST /engine/runtime/install` works the same way for the Python runtime.

A job looks like:

```json
{"id": "job_1_12345", "kind": "pull", "model_id": "gemma4-e2b-it-4bit-mlx", "status": "running",
 "progress": {"phase": "downloading", "file_name": "model.safetensors", "file_index": 3,
              "file_count": 9, "bytes_downloaded": 1200000000, "total_bytes": 3580000000}}
```

`status` is `running`, `done` or `failed`. A finished job carries `result` or
`error`. A runtime job reports `setup` (`phase`, `message`, and bytes where
known) instead of `progress`. The events routes send the job object each time
it changes and close when it ends. Jobs live in memory and are lost on
restart.

## POST /engine/generate

The native generation route. It returns the engine's own facts instead of
OpenAI's envelope.

| Field | Notes |
|---|---|
| `model` | Default `text` |
| `prompt` or `messages` | One is required. The runner treats `prompt` as a single user turn. `messages` go through the model's chat template and get tools, the prompt cache and token counts. |
| `tools` | As in chat completions (with `messages`) |
| `cache_key` | Prompt-cache key; derived from `messages` when absent |
| `format` | Same shape as OpenAI's `response_format` |
| `max_attempts` | For structured output. Default 2, which is one retry. |
| `max_tokens`, `temperature`, `priority`, `stream` | As above. Streaming is ignored when `format` asks for JSON. |

Response:

```json
{"text": "…", "json": {…}, "repaired": false, "repairs": [], "attempts": 1, "tool_calls": [],
 "meta": {"prompt_tokens": 28, "cached_tokens": 0, "generation_tokens": 29, "template": "native"},
 "model": "gemma4-e2b-it-4bit-mlx", "family": "gemma4-e2b", "backend": "mlx-python", "ms": 1038}
```

`meta` is `null` for a raw `prompt`. With `stream: true` the events are
`{"token": "…"}` and then `{"done": true, "text": …, "meta": …, "model": …,
"family": …, "backend": …, "ms": …}`, or `{"error": "…"}`. There is no
`[DONE]` sentinel on this route.

## POST /engine/embed

| Field | Notes |
|---|---|
| `model` | Default `embed` |
| `inputs` | A string or an array of strings |
| `task`, `expect_fingerprint`, `priority` | As in `/v1/embeddings` |

Response: `{"vectors": [[…]], "fingerprint": …, "model": …, "dims": …, "task": …,
"backend": …, "ms": …}`.

## Structured output

The MLX runner cannot constrain decoding, so the server enforces JSON after
generation:

1. Parse the output. If that fails, repair it step by step: strip code fences
   and any preamble, fix invalid backslash escapes, fix unescaped interior
   quotes. Each step taken is reported in `repairs`.
2. With a schema, validate against it (JSON Schema).
3. If the output is still unusable, ask the model once more with the
   validator's complaint appended (`/engine/generate`: up to `max_attempts`).
4. If that fails too, answer 422.

A streaming chat completion with JSON output gets no retry: if the output
cannot be used, the stream ends with an error event and `data: [DONE]`.

## Pairing

```text
client                                   engine                      operator
POST /engine/pair {name, scopes}  ──►    pending, expires in 300 s
  ◄── 202 {"id": "772ca7c0bf6ada4b", …}
GET /engine/pair/{id}             ──►    {"status": "pending", "token": null}
                                                            ◄── estia pair approve {id}
GET /engine/pair/{id}             ──►    {"status": "approved", "token": "estia_…"}
GET /engine/pair/{id}             ──►    {"status": "approved", "token": null}
```

`scopes` defaults to `generate`, `embed`, `models:read`. Unknown scopes are
refused. At most 24 requests may be pending at once. The token is returned on
the first poll after approval and never again. The operator decides with the
CLI on the engine's machine (it edits the data directory directly and needs no
token) or through the `admin` routes above.

## Clients

- Any OpenAI SDK: set the base URL to `http://<host>:27200/v1` and the API key
  to the token.
- Rust: `estia_engine::RemoteEngine`, with `RemoteGen` and `RemoteEmbed`,
  talks to a daemon through the `/engine/*` routes with the same method shapes
  as local sessions.
- Browser: `/client`, source in `clients/web/index.html`, uses only the routes
  above.
