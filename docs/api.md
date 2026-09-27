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
| Request body | 32 MiB. Change it with `estia serve --max-body-bytes <bytes>` (or `estia service install --max-body-bytes <bytes>`) or `ESTIA_MAX_BODY_BYTES=<bytes>`; the flag wins. | 413, type `invalid_request_error`; the message gives the limit and how to raise it |
| Pending pairing requests | 24 overall, 4 per client address | 429, type `rate_limit_error` |
| Time to send a request's headers | 10 s | Connection closed; also closes a keep-alive connection idle that long |
| Open connections per client address | 32 (IPv6 counted per /64; loopback exempt) | New connections are closed at once |
| Open connections in total | Below the open-files limit, which the server raises at start (at most 4096) | New connections wait in the listen backlog |

The body limit applies to every route. 32 MiB holds a full batch of 256
embedding inputs of about 100 KB each; a client that sends more per request
should split it, as it must past 256 inputs anyway.

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
| GET | `/engine/stats` | `models:read` | Uptime, loaded models with their runner's pid and memory, queue depth, job count |
| GET | `/engine/jobs` | `models:read` | All jobs |
| GET | `/engine/jobs/{id}` | `models:read` | One job |
| GET | `/engine/jobs/{id}/events` | `models:read` | Server-sent events for one job until it ends |
| POST | `/engine/runtime/install` | `admin` | Install a backend's runtime as a job: Python and MLX, or the pinned llama.cpp build |
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
accepts an artifact id, a family or a role, in that order of precedence. A
family or role resolves to the artifact in the running backend's format
(`gemma4-e2b-it-4bit-mlx` on `mlx-python`, `gemma4-e2b-it-qat-q4_0-gguf` on
`llama-cpp`); an artifact id in the other format is a 400. Embedding requests
take an embedding model id, an imported embedding model's id, or `embed`.

The backend is one of:

| Id | What runs the models |
|---|---|
| `mlx-python` | MLX on Apple Silicon, through the Python runner |
| `llama-cpp` | upstream `llama-server`, through the `estia-llama` adapter |

## Errors

Every error uses OpenAI's shape, plus the request id (see
[Request ids](#request-ids)):

```json
{"error": {"message": "unknown model or role `writer`", "type": "not_found_error", "code": 404, "request_id": "0938df2480a78ff1"}}
```

| Status | When |
|---|---|
| 400 | Empty `messages`, `input` or `inputs`, more than 256 embedding inputs, neither `prompt` nor `messages`, unknown `response_format` type, embedding `task` or `encoding_format`, an artifact id in the other backend's format, pulling an imported model, unknown scope, a pairing name that breaks the [name rules](#pairing), an unknown pairing id on approve or deny, a pairing that cannot be approved or denied |
| 401 | Missing, unknown or revoked token |
| 403 | Token lacks the scope; `Host` not allowed or cross-origin write (type `permission_error`, see [Host and Origin checks](#host-and-origin-checks)) |
| 404 | Unknown path; a `model` that names no model, family or role (``unknown model or role `…` ``); a model with no artifact for the running backend; a model that is not installed (the message names the `estia pull` that fixes it); unknown job; unknown pairing id on poll |
| 413 | Request body over the [limit](#limits) (32 MiB by default) |
| 422 | Structured output still invalid after the retry; embedding fingerprint mismatch |
| 429 | `POST /engine/pair` while 24 requests are pending, or 4 from the same address (type `rate_limit_error`) |
| 500 | Runner failure, including a request the model's chat template refuses (llama-server's message is passed on); the pairing store could not be read or written (details in the server's log, under the request id) |

A body that is not JSON, lacks `Content-Type: application/json`, or does not
fit the route's fields is rejected before the handler runs, with a 400, 415 or
422 from the HTTP framework. A body over the limit is a 413 at the same stage.
These answers come in the same JSON shape, with type `invalid_request_error`,
the framework's text as `message`, and the request id.

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
| `messages` | `system`, `user`, `assistant`, `tool`. `content` may be a string or an array of parts: `text` parts (joined) and, in `user` messages, `image_url` parts (see [Images](#images)). Any other part type is a 400. An assistant message's `tool_calls` reach the model's chat template as structured calls, so the `{"role": "tool", "tool_call_id": …}` results after it are rendered too. |
| `tools` | OpenAI tool schemas, rendered by the model's chat template. The model's calls come back as `tool_calls`. On `llama-cpp`, llama-server parses them; on `mlx-python` the server parses Gemma 4's native syntax and a `{"tool_call": {"name", "arguments"}}` object. |
| `response_format` | `text`, `json_object`, or `json_schema` with `json_schema.schema`. See [Structured output](#structured-output). |
| `max_completion_tokens`, `max_tokens` | Default 1024, at most 8192 (larger values are lowered) |
| `temperature` | Default 0.2 |
| `stream` | Server-sent events, ending with `data: [DONE]` |
| `user` | Used as the prompt-cache key |
| `priority` | Estia extension: `interactive` (default) or `background` |

Other OpenAI fields are accepted and ignored, on either backend. Three of
them change what a client gets back, so plan for them:

- `tool_choice`: `"none"` does not stop the model from calling a tool you
  declared, and `"required"` or a named function does not force a call. To
  rule tools out, leave `tools` out of the request.
- `stop`: stop sequences are not applied. Generation ends at the end of the
  model's turn or at `max_tokens`; cut the text yourself if you need to.
- `n`: one choice comes back, whatever `n` asks for. Send separate requests
  for more.

`top_p` and the other sampling fields are ignored too.

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
- `x_estia`: `family`, `backend` (`mlx-python` or `llama-cpp`),
  `cached_tokens`, `template` (`native` or `manual`), `generation_tps` (the
  runner's decode speed, when it reports one), `repaired` and `repairs` (for
  JSON output), `ms` (generation time) and `load_ms` (how long the request
  waited for the model to load; `null` when it was already loaded).
- `finish_reason` is `tool_calls` when tool calls were parsed, `length` when
  the output used up `max_tokens` and was cut off, otherwise `stop`. A
  `length` answer is incomplete: raise `max_tokens`, or ask for less. With
  JSON output it usually fails validation too.

**Streaming.** Prose streams as `delta.content` chunks. When `tools` are
declared on `mlx-python`, the server holds the first characters back to tell a
tool call from prose; on `llama-cpp` prose streams at once, because
llama-server separates the calls itself. Tool calls, each with an `index`, and
any JSON-mode output are sent in the final chunk instead of token by token.
The final chunk carries `finish_reason`, `usage` and `x_estia` (`backend`,
`cached_tokens`, `template`, `generation_tps`, `ms`, `load_ms`). A stream that
fails after it started ends with
`data: {"error": {"message", "type", "request_id"}}` and `data: [DONE]`.

If the client disconnects, the generation is cancelled in the runner, and the
next request to that model starts without waiting for it. This holds while the
prompt is still being read (prefill), before the first token: a long prompt
is abandoned part-way instead of being read to the end. A non-streaming
request is not cancelled when its client disconnects: the generation runs to
the end, and later requests to that model wait for it.

## POST /v1/embeddings

| Field | Notes |
|---|---|
| `model` | `embed` (the model the `embed` role is bound to, else `embeddinggemma-300m-4bit`, or `embeddinggemma-300m-q8_0-gguf` on llama.cpp) or an embedding model id |
| `input` | A string or an array of at most 256 strings |
| `task` | Estia extension: `document` (default), `query`, `clustering`, or `none`. The model's own prefix for that task is prepended. `none` sends the inputs unchanged. |
| `encoding_format` | `float` (default) or `base64`, as OpenAI. Anything else is a 400. |
| `expect_fingerprint` | Estia extension: refuse with 422 unless the server would produce vectors with this fingerprint |
| `priority` | Estia extension: `interactive` (default) or `background` |

The response has OpenAI's shape. With `float`, each `data[].embedding` is an
array of numbers. With `base64` it is one string: the vector's 32-bit floats
as little-endian bytes, base64-encoded, which is what OpenAI sends. The OpenAI
SDKs (Python and JavaScript) ask for `base64` when you pass no
`encoding_format`, and decode it back into an array of numbers. `usage`
is reported as zero. `x_estia` carries `fingerprint`, `dims`, `task` and
`load_ms`.

A **fingerprint** is `<artifact id>@<backend>`, for example
`embeddinggemma-300m-4bit@mlx-python` on MLX,
`embeddinggemma-300m-q8_0-gguf@llama-cpp` for the same model on llama.cpp, and
`<id>@llama-cpp` for an imported model. The same model run by two backends
gives vectors that cannot be compared, so store the fingerprint with your index
and send it back as `expect_fingerprint`. A mismatch is a 422, before any
work is done, whose message starts with `fingerprint mismatch:` and names
both fingerprints:

```text
fingerprint mismatch: this host serves `embeddinggemma-300m-4bit@mlx-python`, you expected `embeddinggemma-300m-q8_0-gguf@llama-cpp` — re-embed before mixing
```

`/engine/embed` says the same more briefly:
``fingerprint mismatch: host serves `…`, expected `…` ``.

## GET /v1/models

The roles in the role table come first, each with
`x_estia: {"role": true, "family": …}`. These are the roles bound with
`estia roles set` or `PUT /engine/defaults` (`text`, `fast`, `vision`, and
`embed` once it is bound). Unbound, `"model": "embed"` still works on the
embedding routes. Then come every generation model, built-in and imported
(`x_estia`: `family`, `format`, `backend`, `runnable`, `imported`,
`installed`, `context_length`, `capabilities`), and every embedding model (`x_estia`: `kind`,
`dims`, `runnable`, `artifact`, `format`, `fingerprint`, `imported`,
`installed`). `runnable` says whether the running backend can load it: MLX
artifacts are listed on a llama.cpp engine but not runnable, and the other way
round. Models that are not downloaded are listed too; check `installed`.

## GET /engine/health

Open, so clients can check an engine before they have a token.

From an MLX engine, just after it started (the server sorts the keys; they
are regrouped here):

```json
{"ok": true, "engine": "estia", "version": "0.4.0",
 "build": {"commit": "8642bfc4e+dirty", "date": "2026-09-26"},
 "api_version": 1, "protocol_version": 2,
 "bind": "127.0.0.1:27391", "auth_required": true, "uptime_s": 0, "backend": "mlx-python",
 "backends": [{"id": "mlx-python", "active": true, "supported": true, "runtime_installed": true},
              {"id": "llama-cpp", "active": false, "supported": true, "runtime_installed": false,
               "build": "b11146", "variant": null, "server": null}],
 "loaded": []}
```

On a llama.cpp engine with the runtime installed, the `llama-cpp` entry comes
first, for example `"variant": "metal", "server": "installed"`, and `loaded`
lists the models in memory.

`version` is the crate version of the server that answered, and `build` the
commit and UTC date its binary was built from, as `estia version` prints them
([versioning.md](versioning.md)). `backend` is the backend this engine runs.
`backends` lists both, the active one first:
`supported` says whether it can run on this machine and `runtime_installed`
whether its runtime is installed. For `llama-cpp`, `build` is the pinned
llama.cpp build, `variant` the installed variant (`cpu`, `metal`, `vulkan`,
`cuda-12`, `cuda-13`, `rocm`, or null), and `server` where `llama-server`
comes from: `installed`, `custom` (`ESTIA_LLAMA_SERVER`), or null when there
is none.

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

`GET /engine/models` returns `{"backend": …, "generation": […], "embedding":
[…], "models_dir": …}`. Each entry has `id`, `kind`, `format`, `runnable`,
`imported`, `label`, `repo_id`, `revision`, `license`, `installed`,
`bytes_on_disk`, `required_disk_bytes` and `pulling` (the running job id, if
any). Generation entries add `family`, `backend`, `context_length`,
`capabilities` and `partial_bytes`. `capabilities` lists what reaches the
model through this API: `["text", "tools", "vision"]` for the Gemma 4
families; an imported GGUF lists `vision` only when it was imported with its
projector (`estia import --mmproj`). Embedding entries describe the model and
its artifact for the running backend: `artifact`, `dims`, `arch`,
`multilingual`, `fingerprint`, and `artifacts`, every artifact of the model
(`id`, `format`, `backend`, `installed`). For an imported model linked with
`estia import --link`, `bytes_on_disk` is the link's size.

`POST /engine/models/pull` with `{"id": "gemma4-e2b"}` answers
`202 {"job_id": …}`, or `{"job_id": …, "already_running": true}` if that model
is already downloading. `id` is an artifact id (that artifact, in either
format), or a family, role, embedding model id or `embed` (the running
backend's artifact of it). Imported models cannot be pulled (400). Downloads
come from Hugging Face, resume after an interruption, and check SHA-256 where
the repository publishes one; GGUF artifacts download only their listed files,
pinned to a commit.

`POST /engine/runtime/install` works the same way for a backend's runtime.
Its body is optional: `{"backend": "llama-cpp", "variant": "cpu"}`. `backend`
defaults to the engine's own (`mlx` and `llama` work too); `variant` is for
llama.cpp only and defaults to probing the machine. A backend that cannot run
on this machine is a 400.

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
| `prompt` or `messages` | One is required. The runner treats `prompt` as a single user turn. `messages` go through the model's chat template and get tools, the prompt cache and token counts. On `llama-cpp`, a `prompt` with a JSON `format` is sent through chat as one user turn, so decoding is constrained and `meta` is filled in. |
| `tools` | As in chat completions (with `messages`) |
| `cache_key` | Prompt-cache key; derived from `messages` when absent. Scoped to the caller's token, as in chat completions. |
| `format` | Same shape as OpenAI's `response_format` |
| `max_attempts` | For structured output. Default 2, which is one retry; held between 1 and 3. |
| `max_tokens`, `temperature`, `priority`, `stream` | As above (`max_tokens` at most 8192). Streaming is ignored when `format` asks for JSON. |

Response:

```json
{"text": "…", "finish_reason": "stop", "json": {…}, "repaired": false, "repairs": [], "attempts": 1, "tool_calls": [],
 "meta": {"prompt_tokens": 28, "cached_tokens": 0, "generation_tokens": 29, "generation_tps": 76.1, "template": "native"},
 "model": "gemma4-e2b-it-4bit-mlx", "family": "gemma4-e2b", "backend": "mlx-python", "ms": 1038, "load_ms": null}
```

`finish_reason` is as in chat completions: `stop`, `length` or `tool_calls`.
`ms` is generation time and `load_ms` the wait for the model to load (`null`
when it was loaded already). `meta` and `finish_reason` are `null` for a raw
`prompt` that went to the runner's `generate`. With `stream: true` the events
are `{"token": "…"}` and then `{"done": true, "text": …, "finish_reason": …,
"meta": …, "model": …, "family": …, "backend": …, "ms": …, "load_ms": …}`, or
`{"error": "…", "request_id": "…"}`. There is no `[DONE]` sentinel on this
route.

## POST /engine/embed

| Field | Notes |
|---|---|
| `model` | Default `embed` |
| `inputs` | A string or an array of at most 256 strings |
| `task`, `expect_fingerprint`, `priority` | As in `/v1/embeddings` |

Response: `{"vectors": [[…]], "fingerprint": …, "model": …, "dims": …, "task": …,
"backend": …, "ms": …, "load_ms": …}`. `ms` is the embedding itself, and
`load_ms` the wait for the model to load (`null` when it was loaded already).

## GET /engine/stats

From an MLX engine with two models loaded (keys regrouped):

```json
{"uptime_s": 30, "loaded": ["embeddinggemma-300m-4bit", "gemma4-e2b-it-4bit-mlx"],
 "models": [{"id": "embeddinggemma-300m-4bit", "kind": "embedding", "pid": 19640, "memory_bytes": 530566784},
            {"id": "gemma4-e2b-it-4bit-mlx", "kind": "generation", "pid": 19564, "memory_bytes": 3961055488}],
 "queue": {"interactive": 0, "background": 0}, "jobs": 0}
```

`models` has one entry per resident model: the process that runs it and the
physical memory that process and its children hold, in bytes. On macOS that
is the physical footprint, the figure in Activity Monitor's Memory column.
It includes the Metal buffers an MLX runner keeps its weights and KV cache
in, which `ps` and `top` leave out of the resident size: the `gemma4-e2b`
runner above showed 102 MB in `ps` against a 3.8 GB footprint. On
Linux it is the resident size (`VmRSS`), which counts shared pages and
leaves out swapped ones. For a llama.cpp model the figure includes the
`llama-server` the adapter started. `pid` changes when a runner that died is
replaced. `memory_bytes` is `null` when the memory cannot be read (a runner
that just died, or a platform other than macOS and Linux). Reading stats never
waits for a generation in progress.

`queue` counts the generation calls waiting, by priority.

## Images

A `user` message's content can include images, as OpenAI `image_url` parts:

```json
{"model": "vision", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "What is the total on this invoice?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
]}]}
```

- **Inline only.** The URL must be a `data:` URL with base64 data. A remote
  URL (`https://…`, `file://…`) is a 400: the engine never fetches a URL on a
  client's behalf.
- **Formats:** PNG, JPEG, WebP and GIF (its first frame). The format is read
  from the bytes; a declared type that disagrees is ignored.
- **Limits:** 20 MB per image and 8 images per request. Over the size limit is
  413; more images, or an image in a `system`, `assistant` or `tool` message,
  is 400.
- **Models:** the model must read images: the `vision` role (Gemma 4 E4B by
  default), any Gemma 4 family or artifact, or an import with a projector.
  Anything else is a 400 that says so, rather than an answer that ignores the
  image. `/engine/generate` takes the same `messages`.
- **Preparation:** the MLX runner converts each image to RGB and, when its
  content is a small island on a plain background (a rendered page, a
  screenshot with margins), crops to it; small text reads much more
  reliably that way.
- **Prompt cache:** a request with images is never served from the prompt
  cache, and `cached_tokens` is 0.

## Structured output

How the model is held to the requested shape depends on the backend:

- **`llama-cpp`**: the server passes `format` to the adapter, which sends
  llama-server a `response_format`. Decoding is constrained by a grammar built
  from the JSON Schema, so the output parses and fits the schema's structure;
  `attempts: 1` and `repaired: false` are the norm. The grammar cannot stop
  `max_tokens` from cutting the output short, though: a long string that runs
  into the limit leaves unfinished JSON, which fails validation and is
  retried. Bound long fields with `maxLength`, or leave room in `max_tokens`.
  llama.cpp supports a subset of JSON Schema. With `tools` in the same request
  no format is passed (llama-server refuses a grammar together with tools).
- **`mlx-python`**: the runner cannot constrain decoding. Unless `tools` are
  declared, the server adds an instruction with the schema to the system
  prompt (after your own system message, or as a new one), or after a raw
  `prompt`. The prompt-cache key is derived before the instruction is added.

Either way the server then enforces JSON after generation:

1. Parse the output. If that fails, repair it step by step: strip code fences
   and any preamble (`strip_fences_or_preamble`), drop text after a complete
   object or array, such as a sign-off or a stray `}` (`strip_trailing_text`),
   fix invalid backslash escapes (`repair_escapes`), fix unescaped interior
   quotes (`repair_quotes`). Each step taken is reported in `repairs`. A lone
   number, string or boolean must be the whole output, and JSON cut off
   before it closes is not repaired.
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
