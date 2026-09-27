# Building on Estia

This guide is for developers who want to put their own app, assistant or tool
on top of an Estia engine. It explains what a client needs to know, with a
short snippet for each idea and a link to a complete example in
[`examples/`](../examples/README.md). Every example runs against a live
engine.

The full request and response shapes are in [api.md](api.md).

## Start here

1. Run the engine: `estia serve` (or `estia service install --local`).
2. Mint a token for your app with only the scopes it needs:
   `estia token new myapp --scopes generate,embed,models:read`.
   It prints the token once.
3. Point any OpenAI SDK at `http://127.0.0.1:27200/v1` with the token as the
   API key.
4. Send a role such as `fast`, `text` or `embed` as the model.

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:27200/v1", api_key=os.environ["ESTIA_TOKEN"])
r = client.chat.completions.create(model="fast", messages=[{"role": "user", "content": "Name one sea."}])
print(r.choices[0].message.content)
```

The first request to a model starts a runner and loads the weights, which
takes a few seconds; requests that arrive meanwhile wait for that same load.
Later requests reuse it. `x_estia.load_ms` says how long a request waited for
a load (`null` when the model was ready), and `x_estia.ms` how long the
generation took. The engine unloads a model after 15 idle minutes by default
(`estia serve --idle-unload-minutes`), so the next request pays the load
again.

## Two APIs on one port

| | `/v1/*` | `/engine/*` |
|---|---|---|
| Shape | OpenAI's | Estia's own |
| Use it for | chat, embeddings and model lists through any OpenAI SDK | health, pairing, model downloads, role table, stats, and generation or embedding with the engine's facts at the top level |
| Estia facts | in an `x_estia` object that standard clients can ignore | top-level fields (`fingerprint`, `attempts`, `json`, `meta`) |

Most apps use `/v1/*` for inference and `/engine/health` plus pairing from
`/engine/*`. `/engine/health` needs no token and reports `api_version`
(currently `1`); check it before anything else.

## Roles, not model ids

A role is a name you send as `model`. The operator binds each role to a model
family (`estia roles set fast gemma4-e2b`), so your app keeps working when the
operator swaps a model.

| Role | Default family | When unbound |
|---|---|---|
| `text` | `gemma4-e4b` | error |
| `fast` | `gemma4-e2b` | falls back to `text` |
| `vision` | `gemma4-e4b` | falls back to `text` |
| `code` | none | falls back to `text` |
| `embed` | none | EmbeddingGemma 300M (`embeddinggemma-300m-4bit`) |

`model` also accepts a family (`gemma4-e2b`) or an artifact id
(`gemma4-e2b-it-4bit-mlx`). An artifact id wins over a family, and a family
over a role. Prefer roles: an artifact id ties your app to one download and
one backend. The same family has an MLX artifact and a GGUF artifact
(`gemma4-e2b-it-qat-q4_0-gguf`); a role or family answers from the one the
engine's backend runs, while the other backend's artifact id is a 400.

**Backends.** An engine runs its models on MLX (`mlx-python`, the default on
Apple Silicon Macs) or llama.cpp (`llama-cpp`, the default elsewhere).
`/engine/health` says which, in `backend`, and every chat completion repeats
it in `x_estia.backend`. The API is the same on both. The differences an app
may notice: artifact ids (above), embedding fingerprints (see
[Embeddings](#embeddings)), and how closely the model follows a JSON Schema
(see [Structured output](#structured-output)).

The response tells you what answered: `model` is the artifact id and
`x_estia.family` the family. `GET /v1/models` lists the role table first, each
entry with `"x_estia": {"role": true, "family": …}`. An unknown role is a 404:

```json
{"error": {"code": 404, "message": "unknown model or role `writer`", "type": "not_found_error", "request_id": "5f0c1d2e3a4b6c7d"}}
```

Examples: the chat examples read `ESTIA_MODEL` and default to `fast`.

## Tokens and scopes

Every route except `/engine/health`, the two pairing routes and `/client`
needs `Authorization: Bearer <token>`, on loopback too.

| Scope | Allows |
|---|---|
| `generate` | `/v1/chat/completions`, `/engine/generate` |
| `embed` | `/v1/embeddings`, `/engine/embed` |
| `models:read` | `/v1/models`, `/engine/models`, `/engine/defaults`, stats, jobs |
| `models:write` | model downloads and deletes |
| `admin` | everything, including roles, runtime install and pairing decisions |

Give each app its own token with the least it needs. A chat app needs
`generate`; a search indexer needs `embed`. Named tokens can be revoked one by
one without touching other apps:

```bash
estia token new notes-app --scopes embed,generate
estia token revoke notes-app              # refused from its next request
estia token new notes-app --replace       # rotate: the old token stops working now
```

`estia token new` prints only the token, so
`export ESTIA_TOKEN=$(estia token new myapp --scopes generate)` works. Keep
tokens out of source code: read them from the environment, a file with mode
0600, or the platform's keychain.

A missing or unknown token is 401 (`authentication_error`). A valid token
without the scope is 403 (`permission_error`, "token lacks the `generate`
scope"). A running engine picks up tokens minted or revoked with
`estia token` on the next request, without a restart.

Examples: each example names its scopes at the top and exits with a clear
message when `ESTIA_TOKEN` is missing, unknown or lacks a scope.

## Pairing, for apps on other devices

An app on a phone, tablet or another computer cannot run `estia token new`.
It pairs instead: it asks for a token, the operator approves on the engine's
machine, and the app collects the token once.

```text
POST /engine/pair {"name": "kitchen tablet", "scopes": ["generate"]}
  → 202 {"id": "bb706d7dccbe7777", "status": "pending", "expires_in": 300, ...}
GET /engine/pair/bb706d7dccbe7777 → {"status": "pending", "token": null}
        operator: estia pair approve bb706d7dccbe7777
GET /engine/pair/bb706d7dccbe7777 → {"status": "approved", "token": "estia_..."}   (once)
GET /engine/pair/bb706d7dccbe7777 → {"status": "approved", "token": null}
```

What your app should do:

- Ask for the smallest set of scopes. The operator sees them before approving,
  and `estia pair approve` refuses `admin` without `--allow-admin`.
- Use a name the operator will recognise: at most 64 letters, digits, single
  spaces and `. _ - ' ’ ( )`. Anything else is a 400.
- Show the pairing id and the approve command, then poll every 2 seconds or so.
- Save the token the moment it arrives. It is handed out exactly once. Store it
  in the keychain, or in a file created with mode 0600.
- Handle the outcomes: `denied`; a 404 once the request has expired, 5
  minutes after it was made; 429 when 24 requests are pending, or 4 from your
  address; 500 when the engine could not read or write its pairing file (poll
  again).
- Stop polling a little before the 5 minutes are up, and tell the user the
  request is about to expire and they can ask again. The examples stop at 290
  seconds.

If the operator later denies the pairing, the token is revoked and your app
gets 401. Treat 401 as "pair again".

Examples: [`python/pair.py`](../examples/python/pair.py) (standard library
only), [`remote_client.rs`](../engine/examples/remote_client.rs) with
`--pair`.

## Finding an engine on the network

A LAN engine (`estia serve --lan` or `estia service install`) advertises the
Bonjour service `_estia._tcp`. The instance name is the machine's short
hostname, and the TXT record has three keys:

| Key | Meaning |
|---|---|
| `api_version` | Version of the `/engine/*` contract (`1`) |
| `engine_version` | The server's version, as `version` in `/engine/health` (`0.4.0`) |
| `protocol_version` | Runner protocol version (`2`) |

```text
$ dns-sd -B _estia._tcp
  Add  ...  local.  _estia._tcp.  MacBook-Pro-5
$ dns-sd -L MacBook-Pro-5 _estia._tcp local.
  MacBook-Pro-5._estia._tcp.local. can be reached at MacBook-Pro-5.local.:27200
  api_version=1 engine_version=0.4.0 protocol_version=2
```

In an app, use the platform's browser for service type `_estia._tcp`: the
Network framework on Apple platforms, `NsdManager` on Android, a Zeroconf
library elsewhere. Check `api_version` before pairing. Discovery is a
convenience: always let the user type an address too, since `estia status`
prints the URL and pairing by address always works.

## The prompt cache

The runner keeps each conversation's KV cache, so a conversation that grows by
appending only prefills its new turn. Send a stable conversation id as `user`
on `/v1/chat/completions` (`cache_key` on `/engine/generate`):

```python
conversation_id = f"chat-{uuid.uuid4().hex[:12]}"   # one per conversation, reused every turn
r = client.chat.completions.create(model="fast", messages=history, user=conversation_id)
print(r.usage.prompt_tokens_details.cached_tokens)  # also in x_estia.cached_tokens
```

In a run of `examples/python/chat.py` on `gemma4-e2b`, the second turn had 78
prompt tokens, 64 of them served from the cache.

Rules that follow from how it works:

- Append to the history. Editing or reordering earlier turns breaks the shared
  prefix and the turn prefills from scratch.
- The cache belongs to your token. The engine mixes a hash of the caller's
  token into the key, so two apps sending the same `user` never share an
  entry, and `cached_tokens` cannot reveal another client's conversation.
- Without `user`, the engine derives a key from the model, the leading system
  messages and the first user message. Two of your conversations that open the
  same way then share one entry. Send `user`.
- Sending the same messages again (a regenerate) reuses all of the prompt but
  its last few tokens, the ones that open the model's turn.
- The runner keeps a small number of conversation caches per loaded model (8
  today), least recently used out first. An idle unload or a runner restart
  drops them all. After a cancelled turn, the next one reuses whatever part of
  the prompt was prefilled before the cancel.

Examples: [`python/chat.py`](../examples/python/chat.py),
[`javascript/chat.mjs`](../examples/javascript/chat.mjs),
[`curl/quickstart.sh`](../examples/curl/quickstart.sh) steps 3 and 4.

## Streaming and cancelling

`"stream": true` on `/v1/chat/completions` returns server-sent events: one
`data: {...}` line per piece, then `data: [DONE]`. The last chunk before
`[DONE]` carries `finish_reason`, `usage` and `x_estia`
(`cached_tokens`, `template`, `ms`). `finish_reason` is `length` when the
answer used up `max_tokens` and was cut off, `tool_calls` when the model
called tools, and `stop` otherwise. Treat `length` as an incomplete answer.

To cancel, close the connection. The engine sees the client go and stops the
generation in the runner, so the next request does not wait behind it. That
works before the first token too: a long prompt that is still being read
(prefill) is abandoned part-way, not read to the end first.

```python
stream = client.chat.completions.create(model="fast", messages=history, stream=True, user=conversation_id)
try:
    for chunk in stream:
        ...
except KeyboardInterrupt:
    stream.close()          # closes the connection; the engine cancels
```

In JavaScript, call `stream.controller.abort()`. With openai-node 6 the
`for await` loop then ends without throwing, so check
`stream.controller.signal.aborted` afterwards. In Rust, flip a `CancelToken`.

With `tools` declared on MLX, the engine holds the first characters back to
tell a tool call from prose; on llama.cpp prose streams at once. Either way
the tool calls arrive whole in the final chunk. JSON output
(`response_format`) is also sent in the final chunk. If a stream fails after
it started, it ends with a `data: {"error": {"message", "type", "request_id"}}`
event and `data: [DONE]`; the Python SDK raises that as `openai.APIError`.

Only streams can be cancelled. A non-streaming request keeps running in the
runner after its client disconnects, and the next call to that model waits for
it. If a person may give up on a long answer, stream it.

`/engine/generate` streams differently: `{"token": "…"}` events, then one
`{"done": true, "text": …, "meta": …}` or `{"error": "…"}`, with no `[DONE]`.

Examples: all chat examples cancel on Ctrl-C;
[`remote_client.rs`](../engine/examples/remote_client.rs) cancels with a
`CancelToken`.

## Structured output

Ask for JSON that matches a schema with `response_format`:

```python
r = client.chat.completions.create(
    model="fast",
    messages=[{"role": "system", "content": "Extract the event described in the user's text."},
              {"role": "user", "content": text}],
    response_format={"type": "json_schema", "json_schema": {"name": "event", "schema": schema}},
)
event = json.loads(r.choices[0].message.content)
```

The model sees the schema; you do not need to repeat it in the prompt. On
llama.cpp, decoding is constrained to the schema, so the output parses and has
the schema's structure, unless `max_tokens` cuts it short: give long string
fields a `maxLength`, or leave room in `max_tokens`. On MLX, which cannot constrain decoding, the engine
adds the schema to the system prompt. Either way the engine then checks the
output: it parses it, repairs common defects (code fences, preambles, text
after the closing brace, bad escapes, unescaped quotes), validates, and
retries once with the validator's
complaint. `x_estia.repaired` and `x_estia.repairs` say what it fixed. With
`gemma4-e2b` on MLX, the event extraction in `structured.py`, with no schema
in its prompt, returned valid JSON in 10 runs out of 10, none repaired.

When the output still does not validate, the answer is a 422 with type
`invalid_request_error` and a message that starts with
`structured output failed after retry:` and lists the problems. The Python SDK
raises `openai.UnprocessableEntityError`. Retrying the same request rarely
helps: simplify the schema, give an example in the prompt, try a larger role,
or fall back to text. A streamed JSON request gets no retry.

`/engine/generate` takes the same object as `format`, retries up to
`max_attempts` (1 to 3, default 2), and returns the parsed value as `json`
with `attempts`.

Examples: [`python/structured.py`](../examples/python/structured.py)
(`--impossible` shows the 422), `quickstart.sh` steps 8 and 10,
[`in_process.rs`](../engine/examples/in_process.rs).

## Tools

Declare tools with OpenAI tool schemas. The model answers in its own call
syntax and the engine turns it into `tool_calls` with
`finish_reason: "tool_calls"`. Your code runs the function and sends the
result back.

```python
r = client.chat.completions.create(model="fast", messages=messages, tools=TOOLS)
msg = r.choices[0].message
if msg.tool_calls:
    messages.append({"role": "assistant", "content": msg.content or "",
                     "tool_calls": [c.model_dump() for c in msg.tool_calls]})
    for call in msg.tool_calls:
        result = FUNCTIONS[call.function.name](**json.loads(call.function.arguments))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    # ...and ask again with the longer history
```

Send each result as a `{"role": "tool"}` message with the `tool_call_id` of
the call it answers, after the assistant message that made the calls. The
engine passes the assistant's `tool_calls` to the model's chat template as
structured calls, so the template renders the results where the model
expects them. With `gemma4-e2b` on MLX, `tools.py` answered from the tool's
result for five questions out of five.

On llama.cpp the calls are parsed by `llama-server` rather than by Estia, and
the model's chat template must support tools: with a template that does not,
`llama-server` refuses a `tool` message and the request fails with a 500.
Gemma 4's tool calls on llama.cpp have not been tested yet; with a small test
model and a tool-capable template, the calls and results were checked to
reach the template.

`tool_choice` is accepted and ignored: `"none"` does not stop the model from
calling a declared tool, and `"required"` does not force a call. To rule tools
out for a turn, send the request without `tools`. Bound the loop (the example
stops after 4 steps), and decide in your code which calls to run: only tools
you declare can be called, and nothing runs on the engine.

Examples: [`python/tools.py`](../examples/python/tools.py), `quickstart.sh`
step 9.

## Embeddings

```python
docs = client.embeddings.create(model="embed", input=passages, encoding_format="float",
                                extra_body={"task": "document", "priority": "background"})
fingerprint = docs.model_extra["x_estia"]["fingerprint"]    # store it with the vectors
query = client.embeddings.create(model="embed", input=[question], encoding_format="float",
                                 extra_body={"task": "query", "expect_fingerprint": fingerprint})
```

**Task prefixes.** Embedding models expect a different prefix for what you
store and what you search with. Send `task: "document"` when indexing and
`task: "query"` when searching; the engine adds the model's own prefix.
`document` is the default, so a query sent without `task` is embedded as a
document. `clustering` exists too, and `none` sends your text unchanged, for
clients that add prefixes themselves (`RemoteEngine` does).

**Fingerprints.** Every response carries a fingerprint, `<artifact id>@<backend>`,
for example `embeddinggemma-300m-4bit@mlx-python` on MLX and
`embeddinggemma-300m-q8_0-gguf@llama-cpp` for the same model on llama.cpp.
Vectors with different fingerprints cannot be compared, even for the same
model on two backends.
Store the fingerprint with your index and send it back as
`expect_fingerprint`. If the engine would produce different vectors, it
answers 422 before doing any work:

```text
fingerprint mismatch: this host serves `embeddinggemma-300m-4bit@mlx-python`, you expected `embeddinggemma-300m-q8_0-gguf@llama-cpp` — re-embed before mixing
```

That is `/v1/embeddings`; `/engine/embed` words it more briefly
(``fingerprint mismatch: host serves `…`, expected `…` ``). Check for status
422 and a message that starts with `fingerprint mismatch:`, not for the whole
text.

**Re-embedding.** Any change of embedding model or backend on the host changes
the fingerprint. On that 422, rebuild the index from your source texts with
the new fingerprint, then switch to it. Keep the source texts, not only the
vectors. Moving an engine from MLX to llama.cpp, or binding `embed` to
another model, is such a change. `GET /engine/models` lists each embedding
model's `fingerprint`, so an app can check before it searches.

Other facts: at most 256 inputs per request (more is a 400, so batch);
`encoding_format` may be `"float"` (arrays of numbers) or `"base64"` (what
the OpenAI SDKs ask for when you pass nothing, and decode for you), so either
way the SDK hands you numbers; `usage` is reported as zero; `embed` means
EmbeddingGemma 300M (768 dimensions) unless the operator binds it to another
model. The snippets pass `"float"` anyway: engines from before base64
support (0.4.0 and earlier) always send arrays, and the JavaScript SDK, unless
it asked for `"float"`, decodes those arrays as base64 into wrong numbers (192
of them for a 768-dimension vector). The native `POST /engine/embed` takes
`inputs` and returns `vectors`, `fingerprint` and `dims` at the top level.

Examples: [`python/rag.py`](../examples/python/rag.py) (explains the choice of
route), [`javascript/embed.mjs`](../examples/javascript/embed.mjs),
`quickstart.sh` steps 6, 7 and 10.

## Priorities

Chat, `/engine/generate`, `/v1/embeddings` and `/engine/embed` take a
`priority` field: `interactive` (the default over HTTP) or `background`. Each
loaded model serves one call at a time. When calls queue, every waiting
interactive call goes before any background one. A running call is never
interrupted.

Mark bulk work, such as indexing or summarising a backlog, as `background`, so
a person waiting on a chat reply goes first:

```python
client.embeddings.create(model="embed", input=batch, encoding_format="float",
                        extra_body={"task": "document", "priority": "background"})
```

`GET /engine/stats` (`models:read`) shows how many generation calls wait at
each priority, and for each loaded model the runner's pid and the memory it
holds (`memory_bytes`). Use that figure, not `ps`: an MLX runner keeps its
weights in Metal buffers, which the resident size in `ps` and `top` leaves
out, so `ps` can show 100 MB for a model that holds 3.8 GB. In Rust,
`Priority::default()` is `Background`, and `GenHandle::generate` uses it; pass
`Priority::Interactive` for calls a person waits on.

## Limits

| Limit | Value | Over it |
|---|---|---|
| `max_tokens` | default 1024, at most 8192 | Lowered to 8192 without an error |
| `temperature` | default 0.2 | |
| Inputs per embedding request | 256 | 400 |
| Request body | 32 MiB, unless the operator changed it (`estia serve --max-body-bytes`, `ESTIA_MAX_BODY_BYTES`) | 413, with the limit in the message |
| `max_attempts` on `/engine/generate` | 1 to 3 | Held to that range |
| Pending pairing requests | 24, and 4 per client address | 429 `rate_limit_error` |
| Time to send request headers | 10 s | Connection closed |
| Open connections per client address | 32 (loopback exempt) | New connections closed |

There is no rate limit per token. The server speaks HTTP/1.1 only; reuse
connections (the SDKs do).

## Errors and request ids

Errors use OpenAI's shape:

```json
{"error": {"message": "token lacks the `generate` scope", "type": "permission_error", "code": 403, "request_id": "afba215093ed7162"}}
```

| Status | Meaning | What your app does |
|---|---|---|
| 400 | Bad request: empty messages, over 256 inputs, bad pairing name | Fix the request |
| 401 | Missing, unknown or revoked token | Ask for a new token or pair again |
| 403 | Token lacks the scope, or the Host/Origin check refused the request | Mint a token with the scope; see below for Host and Origin |
| 404 | Unknown role or model (``unknown model or role `…` ``), model not installed, expired pairing | Use a role; ask the operator to run the `estia pull` the message names |
| 413 | Request body over the limit (32 MiB by default) | Send less per request: fewer embedding inputs, a shorter history |
| 422 | Structured output invalid after the retry; fingerprint mismatch | See the sections above |
| 429 | Too many pending pairing requests | Wait, then pair again |
| 500 | Runner failure | Retry once; report it with the request id |

A body that is not JSON, or a missing `Content-Type: application/json`, gets a
400, 415 or 422 from the HTTP framework, in the same JSON shape (type
`invalid_request_error`).

Every response carries an `X-Request-Id` header, and error bodies repeat it as
`error.request_id`. Log it next to any error you show or report, so the
operator can find the same request in the engine's log. The OpenAI SDKs read
the header for you: `e.request_id` in Python, `e.requestID` in JavaScript. The
examples print it with every error. `RemoteEngine` errors do not include it.

You can also send your own id, for example your app's turn or job id, so your
log and the engine's line up:

```bash
curl -s http://127.0.0.1:27200/v1/chat/completions -H "Authorization: Bearer $ESTIA_TOKEN" \
  -H 'Content-Type: application/json' -H 'X-Request-Id: myapp-turn-42' \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "Name one sea."}]}'
```

The engine keeps an id of 1 to 64 ASCII letters, digits, `.`, `_`, `:` and
`-`, and replaces anything else with a random one. The operator finds every
line about the request with `grep myapp-turn-42` on the engine's log.
[logging.md](logging.md) describes the log, and what it never contains
(prompts, completions, vectors, tokens).

Set `max_retries=0` (Python) or `maxRetries: 0` (JavaScript) for a local
engine. The SDKs' default retries repeat requests that will fail the same way.

## Host and Origin rules

Before authentication, the engine checks where each request is addressed:

- `Host` must be an IP address, `localhost` or `*.localhost`, a `*.local`
  name, this machine's hostname, or a name the operator allowed with
  `estia serve --allow-host <name>` (or `ESTIA_ALLOWED_HOSTS`). Anything else
  is a 403 whose message says how to allow the name.
- Any request other than GET, HEAD or OPTIONS that carries an `Origin`
  header must come from the same scheme, host and port it is sent to.
  `Origin: null` never matches.

This stops a web page from reaching the engine through DNS rebinding or a
cross-site post. Native apps, command-line tools and servers send no `Origin`
and reach the engine by IP address or `.local` name, so they are not affected.
If your users reach the engine by a custom name, such as `studio.lan`, the
operator adds `--allow-host studio.lan`.

## Web front-ends and CORS

Estia sends no CORS headers and refuses cross-origin writes. A page served
from anywhere else, such as a dev server on `http://localhost:5173`, cannot
call the engine directly:

- A POST from that page carries `Origin: http://localhost:5173`, and the
  engine answers 403 ("cross-origin request refused").
- A request with an `Authorization` header first sends a CORS preflight
  (`OPTIONS`). The engine answers it without `Access-Control-Allow-*` headers,
  so the browser stops there.

Your options, best first:

1. **Call Estia from your own backend.** The browser talks to your server; your
   server calls Estia with a token that never reaches the browser. Server-side
   HTTP clients send no `Origin`, so nothing needs configuring. Stream the
   engine's events through to the page if you want live text.
2. **Desktop apps (Tauri, Electron and similar): call from the native side.**
   A webview's `fetch` sends an origin such as `tauri://localhost`, which the
   engine refuses. Make the request from Rust (`RemoteEngine`) or from the Node
   main process instead.
3. **Serve the page and the API from one origin through a reverse proxy.** The
   proxy serves your static files and forwards `/v1/` and `/engine/` to Estia.
   It must pass the browser's `Host` through unchanged, and the operator must
   allow that name with `--allow-host`. If the proxy rewrites `Host` to the
   engine's address, the browser's `Origin` no longer matches and writes get
   403. The page then holds a token, so give it a narrow one (`generate` only)
   and serve the page only to people you would give that token to.

`--allow-host '*'` does not help: it turns off the `Host` check (for a proxy
that checks `Host` itself) but adds no CORS headers. The bundled `/client`
page works because the engine serves it from its own origin.

## Running on the LAN

```bash
estia serve --lan                # or: estia service install
estia status                     # prints the URL other devices should use
```

- Authentication cannot be turned off on a LAN bind. Devices pair for a token.
- Traffic is plain HTTP, tokens included. Serve the LAN only on a network you
  trust. Otherwise keep the engine on loopback and reach it through a tunnel,
  for example `ssh -N -L 27200:127.0.0.1:27200 you@studio.local`, then use
  `http://127.0.0.1:27200` on the client.
- Devices reach the engine by IP address or `<hostname>.local`; any other name
  needs `--allow-host`.
- Only one engine runs per data directory, and one model call runs at a time
  per loaded model. Many devices can share an engine; they queue.

## From Rust

The `estia-engine` crate gives you two ways in. Until the crates are
published, depend on this repository by path or by git.

**A server somewhere else: `RemoteEngine`.** It uses the `/engine/*` routes
and returns the engine's own facts. It is blocking; call it from a thread or
`spawn_blocking`.

```rust
let engine = Arc::new(RemoteEngine::new("http://127.0.0.1:27200", Some(token))?);
let (text, meta) = engine.chat_stream("fast", &messages, None, Some("conv-1"), Some(200), None,
                                      Priority::Interactive, None, |piece| print!("{piece}"))?;
```

HTTP failures come back as `SessionError::Runner` with text such as
`remote engine 401 Unauthorized: unknown token`.

For embeddings, ask for the role `embed`, as in any other client. The first
response reports the fingerprint, which names the model and the backend, so
the same code runs against an MLX or a llama.cpp engine. Store it with your
vectors and hand it to `RemoteEmbed`, which sends it as `expect_fingerprint`
on every call. `RemoteEngine::embed_batch` sends `task: none`, so add the
model's prefix yourself; `find_embed_model` turns the fingerprint into the
built-in model and its prefixes:

```rust
let (_, fingerprint) = engine.embed_batch("embed", &[first_text], None, Priority::Background)?;
let spec = find_embed_model(&fingerprint);    // None for a model imported on the engine
let query_prefix = spec.map(|s| s.prefix(EmbedTask::Query)).unwrap_or("");
let emb = EmbedHandle::Remote(RemoteEmbed::new(Arc::clone(&engine), "embed", fingerprint));
```

`GenHandle` and `EmbedHandle` wrap a local session or a remote one behind the
same methods, so the rest of your code does not care where the model runs.

**Models in your own process: `Engine`.** No server and no HTTP: your program
starts the runner processes itself. It needs the Python runtime and the models
that `estia setup` installs, and the runner script.

```rust
let cfg = EngineConfig::new(ModelStore::new(dir.join("models")), PythonRuntime::new(dir.join("runtime")), runner_script);
let engine = Engine::new(cfg);
let gen = engine.spawn_gen_session("gemma4-e2b-it-4bit-mlx")?;
let out = gen.chat_with(&messages, None, Some("conv-1"), None, Some(200), None, Priority::Interactive)?;
```

The in-process engine does not read the CLI's `config.json`; pass your own
role table with `EngineConfig::with_roles`, and a `LlamaLaunch` with
`EngineConfig::with_llama` to run llama.cpp. Structured output is yours to
enforce: `structured::with_prompt_hint` shows the model the schema (for a
runner that cannot constrain decoding), then `structured::enforce` and
`Structured::retry_hint`.

`estia-engine` reports what its runners do (start, handshake, model load,
cancel, restart, idle release) as `tracing` events, and turns each runner's
stderr into events too. They are discarded until your program installs a
`tracing` subscriber; [logging.md](logging.md#embedding-the-crates) shows one.

Examples: [`remote_client.rs`](../engine/examples/remote_client.rs),
[`in_process.rs`](../engine/examples/in_process.rs).

## Current limits worth knowing

[ROADMAP.md](../ROADMAP.md) says which of these are planned to change, and in
what order.

- The llama.cpp backend is new. It has run end to end only on an Apple
  Silicon Mac with small test models; the Gemma 4 GGUF files, Linux and
  Windows are untested ([design/llama-backend.md](design/llama-backend.md)
  has the status). Ask by role, not artifact id, and store embedding
  fingerprints, and your app works on either backend unchanged.
- Image parts in messages are replaced by a text marker.
- A non-streaming request is not cancelled when its client disconnects.
- `tool_choice`, `n`, `stop` and `top_p` are accepted and ignored, on both
  backends. Plan for it: leave `tools` out rather than sending
  `tool_choice: "none"`; expect one choice whatever `n` says; and cut the text
  at your stop sequence yourself, since generation runs on to the end of the
  model's turn or to `max_tokens`. Role sampling settings (`temperature`,
  `max_tokens`) are stored but not applied.
- No TLS.
