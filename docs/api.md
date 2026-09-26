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
the server is bound to loopback. A revoked token (`estia token revoke`, or a
denied pairing) gets `401` from the next request on; the server re-reads
`tokens.json` and needs no restart.

## Host and Origin checks

Before authentication, on every route including health, pairing, `/client`
and unknown paths, the server checks where a request is addressed:

- **`Host`** must be an IP literal (v4 or v6, any port), `localhost` or
  `*.localhost`, a `*.local` name, this machine's hostname (short or fully
  qualified), or a name allowed with `estia serve --allow-host <name>` or
  `ESTIA_ALLOWED_HOSTS=<name>,<name>`. In both, a port is ignored,
  `*.example.com` allows the subdomains of `example.com` but not the name
  itself, and `*` disables the check. A request with no `Host` at all (not a
  browser) is let through.
- **`Origin`**, when present on a request other than GET, HEAD or OPTIONS,
  must have the same scheme, host and port as the request's `Host` (default
  ports filled in). `Origin: null` never matches.

Either failure is a `403` with type `permission_error`; the message for an
unknown host says how to allow it. This blocks DNS rebinding (a site pointing
its own name at the engine) and cross-site form posts. Clients that reach the
engine by IP address or `.local` name and send no `Origin`, which covers the
SDKs, curl and `RemoteEngine`, are unaffected. A reverse proxy must forward the
original `Host`, or its own name must be allowed.

## Limits

| Limit | Value | Over it |
|---|---|---|
| `max_tokens` / `max_completion_tokens` | 8192 | Lowered to 8192 |
| `max_attempts` on `/engine/generate` | 3 | Held to 1–3 |
| Inputs per `/v1/embeddings` or `/engine/embed` request | 256 | 400 |
| Pending pairing requests | 24 overall, 4 per client address | 429, type `rate_limit_error` |
| Time to send a request's headers | 10 s | Connection closed; also closes a keep-alive connection idle that long |
| Open connections per client address | 32 (IPv6 counted per /64; loopback exempt) | New connections are closed at once |
| Open connections in total | Below the open-files limit, which the server raises at start (at most 4096) | New connections wait in the listen backlog |

The server speaks HTTP/1.1 only.

## Routes

| Method | Path | Scope | What it does |
|---|---|---|---|
| POST | `/v1/chat/completions` | `generate` | Chat completion, streaming or not |
| POST | `/v1/embeddings` | `embed` | Embeddings |
| GET | `/v1/models` | `models:read` | Roles from the role table, then generation and embedding models |
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
| POST | `/engine/pairings/{id}/deny` | `admin` | Deny a pending request, or take back an approved one and revoke its token |
| GET | `/client` | none | The browser test client (a static page) |
| GET | `/client/` | none | Redirects (308) to `/client` |
| GET | `/` | none | Redirects to `/client` |

Any other path gets a `404` in the error shape below, with or without a
token.

Model ids, families and roles are explained in the README under
[Roles](../README.md#roles). Wherever a generation request takes `model`, it
accepts an artifact id, a family or a role, in that order of precedence.
Embedding requests take an embedding model id or `embed`.

## Errors

Errors raised by the handlers use OpenAI's shape, plus the request id (see
[Request ids](#request-ids)):

```json
{"error": {"message": "model `gemma4-e2b-it-4bit-mlx` is not downloaded at …", "type": "not_found_error", "code": 404, "request_id": "0938df2480a78ff1"}}
```

| Status | When |
|---|---|
| 400 | Empty `messages` or `input`, more than 256 embedding inputs, neither `prompt` nor `messages`, unknown `response_format` type or embedding `task`, unknown scope, a pairing name that breaks the [name rules](#pairing), an unknown pairing id on approve or deny, a pairing that cannot be approved or denied |
| 401 | Missing, unknown or revoked token |
| 403 | Token lacks the scope; `Host` not allowed or cross-origin write (type `permission_error`, see [Host and Origin checks](#host-and-origin-checks)) |
| 404 | Unknown path, unknown model, model not downloaded, unknown job, unknown pairing id on poll |
| 422 | Structured output still invalid after the retry; embedding fingerprint mismatch |
| 429 | `POST /engine/pair` while 24 requests are pending, or 4 from the same address (type `rate_limit_error`) |
| 500 | Runner failure; the pairing store could not be read or written (details in the server's log, under the request id) |

A body that is not JSON, lacks `Content-Type: application/json`, or does not
fit the route's fields is rejected before the handler runs, with a plain-text
400, 415 or 422 from the HTTP framework. These responses still carry the
`X-Request-Id` header.

## Request ids

Every response, on every route and every status, carries an `X-Request-Id`
header. The id also appears:

- in every JSON error body, as `error.request_id`;
- in the last event of a streamed chat completion that fails after it
  started: `data: {"error": {"message", "type", "request_id"}}`, then
  `data: [DONE]`;
- in the error event of a failed `/engine/generate` stream:
  `{"error": "…", "request_id": "…"}`;
- on every line the engine logs while it handles the request (see
  [logging.md](logging.md#request-ids)).

A client may send its own `X-Request-Id`. The server keeps it when it is 1 to
64 characters of ASCII letters, digits, `.`, `_`, `:` and `-`, and otherwise
replaces it with 16 random hex digits. Log the id next to any error you show
or report, so the operator can find the request. The OpenAI SDKs expose it:
`response._request_id` and `error.request_id` in Python, `error.requestID` in
JavaScript.

## POST /v1/chat/completions

Supported request fields:

| Field | Notes |
|---|---|
| `model` | Role, family or artifact id |
| `messages` | `system`, `user`, `assistant`, `tool`. `content` may be a string or an array of parts. Text parts are joined. Other parts, such as images, are replaced by a marker like `[image_url omitted]`: image input is not passed to the model yet. An assistant message's `tool_calls` are passed back to the model as JSON text. |
| `tools` | OpenAI tool schemas. The model answers in its own call syntax and the server converts it to `tool_calls`. Gemma 4's native syntax and a `{"tool_call": {"name", "arguments"}}` object are recognised. |
| `response_format` | `text`, `json_object`, or `json_schema` with `json_schema.schema`. See [Structured output](#structured-output). |
| `max_completion_tokens`, `max_tokens` | Default 1024, at most 8192 (larger values are lowered) |
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

Cache entries belong to the token that made them: the server combines the key
with a hash of the caller's token before it reaches the runner. Two tokens that
send the same `user`, or the same opening messages, never share an entry, so
`cached_tokens` cannot tell one client anything about another's conversation.
With `--no-auth` every request shares one namespace.

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
`x_estia` (`cached_tokens`, `template`, `ms`). A stream that fails after it
started ends with `data: {"error": {"message", "type", "request_id"}}` and
`data: [DONE]`. If the client disconnects, the generation is cancelled in the
runner. A non-streaming request is not cancelled when its client disconnects:
the generation runs to the end.

## POST /v1/embeddings

| Field | Notes |
|---|---|
| `model` | `embed` (currently `embeddinggemma-300m-4bit`) or an embedding model id |
| `input` | A string or an array of at most 256 strings |
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

The roles in the role table come first, each with
`x_estia: {"role": true, "family": …}`. These are the generation roles
(`text`, `fast`, `vision` and any others bound with `estia roles set` or
`PUT /engine/defaults`). `embed` is not listed as a role: it is fixed to
`embeddinggemma-300m-4bit` and is not in the table, but `"model": "embed"`
works on the embedding routes. Then come every generation model (`x_estia`:
`family`, `format`, `installed`, `context_length`) and every embedding model
(`x_estia`: `kind`, `dims`, `installed`). Models that are not downloaded are
listed too; check `installed`.

## GET /engine/health

Open, so clients can check an engine before they have a token.

```json
{"ok": true, "engine": "estia", "version": "0.1.0", "api_version": 1, "protocol_version": 2,
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
| `cache_key` | Prompt-cache key; derived from `messages` when absent. Scoped to the caller's token, as in chat completions. |
| `format` | Same shape as OpenAI's `response_format` |
| `max_attempts` | For structured output. Default 2, which is one retry; held between 1 and 3. |
| `max_tokens`, `temperature`, `priority`, `stream` | As above (`max_tokens` at most 8192). Streaming is ignored when `format` asks for JSON. |

Response:

```json
{"text": "…", "json": {…}, "repaired": false, "repairs": [], "attempts": 1, "tool_calls": [],
 "meta": {"prompt_tokens": 28, "cached_tokens": 0, "generation_tokens": 29, "template": "native"},
 "model": "gemma4-e2b-it-4bit-mlx", "family": "gemma4-e2b", "backend": "mlx-python", "ms": 1038}
```

`meta` is `null` for a raw `prompt`. With `stream: true` the events are
`{"token": "…"}` and then `{"done": true, "text": …, "meta": …, "model": …,
"family": …, "backend": …, "ms": …}`, or `{"error": "…", "request_id": "…"}`.
There is no `[DONE]` sentinel on this route.

## POST /engine/embed

| Field | Notes |
|---|---|
| `model` | Default `embed` |
| `inputs` | A string or an array of at most 256 strings |
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
refused with 400; a scope listed twice counts once.

`name` is trimmed and must then be 1 to 64 characters of: letters and digits
in any script, single spaces, and `. _ - ' ’ ( )`. Control characters, escape
sequences, invisible and bidi formatting characters, emoji, colons and runs of
spaces are refused with 400 rather than stripped. The operator reads names in
a terminal, and the token an approval mints is named `pair:<name>:<id>`.

At most 24 requests may be pending at once, and at most 4 from one client
address. Past either cap a new request gets 429 (`rate_limit_error`) until one
is approved, denied or expires. Expired requests do not count.

The token is returned on the first poll after approval and never again. A
poll can return 500 if the engine cannot read or write `pairings.json`;
retry it, since the token is only handed out once its collection is saved.

The operator decides with the CLI on the engine's machine (it edits the data
directory directly and needs no token) or through the `admin` routes above.
`estia pair approve` refuses a request for `admin` unless given
`--allow-admin`; the approve route itself grants exactly the requested scopes.

`POST /engine/pairings/{id}/deny` on a pending request denies it. On an
approved request it also revokes the minted token, whether or not the device
has collected it, and answers
`{"id", "name", "status": "denied", "revoked": true, "token_name": "pair:<name>:<id>"}`.
Denying twice changes nothing. `GET /engine/pairings` rows carry `revoked`
too. A pairing is dropped from the store 5 minutes after it was requested (10
if an approved token is still uncollected); denying it after that is a 400
that points to `estia token revoke`.

## Clients

[building-clients.md](building-clients.md) is the guide for writing one, and
[`examples/`](../examples/README.md) has runnable programs in shell, Python,
JavaScript and Rust.

- Any OpenAI SDK: set the base URL to `http://<host>:27200/v1` and the API key
  to the token.
- Rust: `estia_engine::RemoteEngine`, with `RemoteGen` and `RemoteEmbed`,
  talks to a daemon through the `/engine/*` routes with the same method shapes
  as local sessions.
- Browser: `/client`, source in `server/client/index.html`, uses only the
  routes above.
