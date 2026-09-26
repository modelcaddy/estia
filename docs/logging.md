# Logging

Estia writes one log line per HTTP request and one per lifecycle event:
startup, runner processes, model loads, pulls, pairing, shutdown. Lines are
text for people or JSON for log shippers. Every request has an id that shows
up in the response, in error bodies and on every line the request caused.

The library crates (`estia-engine`, `estia-server`) only emit
[`tracing`](https://docs.rs/tracing) events. The `estia` binary decides where
they go and which ones are kept.

## Where logs go

| How Estia runs | Where the lines are | How to read them |
|---|---|---|
| `estia serve` in a terminal | stderr, coloured | the terminal |
| `estia serve` with stderr redirected | stderr, no colour | the file or pipe |
| Service on macOS (launchd) | `<data dir>/logs/estia.err.log` | `estia service logs`, `estia service logs -f` |
| Service on Linux (`systemd --user`) | the journal | `estia service logs -f`, or `journalctl --user -u estia -f` |
| Your own program embedding the crates | wherever your subscriber writes | see [Embedding](#embedding-the-crates) |

`estia service logs` prints the last 40 lines (`-n` to change it). With
`-f` it keeps printing new lines until Ctrl-C: on macOS it runs `tail -F` on
the log files, on Linux `journalctl --user -u estia -f`.

On macOS, `estia.out.log` (stdout) normally stays empty; log lines go to
`estia.err.log`. The files are not rotated. To clear them, stop the service
(`estia service stop`), delete them, and start it again.

## Levels and filters

`estia serve` logs Estia's own events from `info` up and other libraries'
events from `warn` up. Change that with `--log-level` or the `ESTIA_LOG`
environment variable. The value is a comma-separated list of
[filter directives](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html#directives):
a bare level, or `target=level`.

| Setting | Effect |
|---|---|
| (nothing) | `info` for Estia, `warn` for libraries |
| `ESTIA_LOG=debug` | `debug` for everything, Estia included |
| `ESTIA_LOG=estia=debug` | `debug` for Estia's crates only |
| `ESTIA_LOG=estia_server=debug` | `debug` for the HTTP server only; the rest keeps its default |
| `ESTIA_LOG=estia_server::access=off` | no access log; everything else as before |
| `ESTIA_LOG=estia_server::access=debug` | also log successful polls (see [Levels of access lines](#levels-of-access-lines)) |
| `ESTIA_LOG=estia_engine::runner=warn` | hide what the runner prints to stderr |
| `ESTIA_LOG=mdns_sd=debug` | Bonjour library details (sockets, interfaces) |
| `ESTIA_LOG=warn` | only warnings and errors |

```bash
estia serve --log-level estia_server=debug
ESTIA_LOG=debug estia serve
estia service install --local --log-level estia_server=debug --log-format json
```

`RUST_LOG` is still read, with the same syntax. The filter is built in this
order, and a later directive for the same target replaces an earlier one:
the defaults, then `RUST_LOG`, then `--log-level` / `ESTIA_LOG`. An invalid
directive in `--log-level` or `ESTIA_LOG` stops `estia serve` with an error;
an invalid one in `RUST_LOG`, which other programs read too, is skipped with
a warning.

`estia service install --log-level … --log-format …` writes those flags into
the service definition. `ESTIA_LOG` in the shell you install from is not
copied. Run `service install` again to change them.

Other commands (`run`, `chat`, `pull`, `setup`, and so on) print their own
output and log only Estia's warnings and the runner's stderr. They read
`ESTIA_LOG`, `ESTIA_LOG_FORMAT` and `RUST_LOG` the same way, so
`ESTIA_LOG=info estia run` also shows the runner starting and its handshake.

### Targets

| Target | What it covers |
|---|---|
| `estia_server` | startup, shutdown, models ready and released, Bonjour, connection limits |
| `estia_server::access` | the access log: one line per request |
| `estia_server::engine_api` | model pulls and removals, runtime installs, role table changes |
| `estia_server::pairing` | pairing requested, approved, denied, collected, expired |
| `estia_server::tokens` | tokens added to or removed from `tokens.json` while the server runs |
| `estia_engine::session` | runner processes: started, failed and restarted, timed out, cancelled, stopped |
| `estia_engine::resident` | runner handshake, model load and unload |
| `estia_engine::runner` | each line a runner writes to stderr: Python warnings and tracebacks, or on llama.cpp the adapter's own lines (`[estia-llama] …`) and `llama-server`'s log (`[llama-server <pid>] …`) |
| `estia_engine::models::hf` | download retries and resumes |
| `estia_engine::engine` | idle release through `Engine::reap_idle` (programs that embed the engine) |
| `estia` | the CLI's own notices |

A directive names a target or a prefix of one: `estia_server` covers all
four `estia_server…` targets, and `estia` covers every Estia crate.

## Formats

`--log-format text` (the default) or `--log-format json`; the environment
variable is `ESTIA_LOG_FORMAT`.

Text puts the time (UTC), the level, the span, the target, the message and
the fields on one line. Strings that came from a client (`caller`, `asked`,
`error`, device names) are quoted and escaped, so they cannot move the
cursor or recolour your terminal. The startup line of a 0.4.0 engine, from a
live run (paths shortened):

```text
2026-09-26T21:56:28.815248Z  INFO estia_server: estia serving version=0.4.0 commit=8642bfc4e+dirty api_version=1 protocol_version=2 url=http://127.0.0.1:27391 bind=127.0.0.1:27391 auth=true lan=false advertise=false idle_unload_s=900 allowed_hosts=- data_dir=…/data backend=mlx-python runner=…/data/engine/runners/mlx-python/estia-runner.py pid=88202
```

Lines from requests, from a live run of an earlier build (they have the same
shape in 0.4.0):

```text
2026-09-26T07:29:11.339963Z  INFO request{request_id=204209c3ff0ef7df}: estia_engine::session: runner started model=gemma4-e2b-it-4bit-mlx pid=53949 program=/tmp/estia-demo/data/runtime/python/bin/python3
2026-09-26T07:29:14.970672Z  INFO request{request_id=204209c3ff0ef7df}: estia_engine::resident: model loaded model=gemma4-e2b-it-4bit-mlx kind=generation load_ms=2635
2026-09-26T07:29:15.269041Z  INFO estia_server::access: request_id=204209c3ff0ef7df method=POST path=/v1/chat/completions status=200 duration_ms=3929 peer=127.0.0.1:52593 caller="builder" asked="fast" model=gemma4-e2b-it-4bit-mlx stream=false load_ms=3630 max_tokens=16 prompt_tokens=16 cached_tokens=0 completion_tokens=2 tokens_per_s=6.7 finish=stop
2026-09-26T07:29:16.989466Z  INFO estia_server::access: request_id=demo-cancel-1 method=POST path=/v1/chat/completions status=200 duration_ms=1536 peer=127.0.0.1:52598 caller="builder" asked="fast" model=gemma4-e2b-it-4bit-mlx stream=true max_tokens=600 time_to_first_token_ms=233 finish=cancelled
2026-09-26T07:29:19.618055Z  INFO estia_server::access: request_id=d50b24389a9704f0 method=GET path=/engine/health status=403 duration_ms=0 peer=127.0.0.1:52605 error_type=permission_error error="host `evil.example` not allowed (DNS-rebinding guard)"
2026-09-26T07:29:19.744314Z  INFO estia_server::access: request_id=ff64bace8b072210 method=POST path=/v1/embeddings status=401 duration_ms=0 peer=127.0.0.1:52611 caller="embedonly" error_type=authentication_error error="revoked token"
```

JSON writes one object per line. The event's fields sit at the top level
next to `timestamp` (UTC, RFC 3339), `level`, `message` and `target`. An
event emitted while a request was being handled also carries
`"span": {"name": "request", "request_id": "…"}`. The startup line of a 0.4.0
engine (paths shortened):

```json
{"timestamp":"2026-09-26T21:57:47.670222Z","level":"INFO","message":"estia serving","version":"0.4.0","commit":"8642bfc4e+dirty","api_version":1,"protocol_version":2,"url":"http://127.0.0.1:27391","bind":"127.0.0.1:27391","auth":true,"lan":false,"advertise":false,"idle_unload_s":900,"allowed_hosts":"-","data_dir":"…/data","backend":"mlx-python","runner":"…/data/engine/runners/mlx-python/estia-runner.py","pid":88888,"target":"estia_server"}
```

Other events, from a live run of an earlier build (paths shortened; the MLX
runner was then version 2.1.0):

```json
{"timestamp":"2026-09-26T07:29:52.035818Z","level":"INFO","message":"runner handshake","model":"gemma4-e2b-it-4bit-mlx","pid":54146,"runner":"mlx-python","runner_version":"2.1.0","protocol":2,"capabilities":"generate,stream,embed,cancel,load,chat,tools,prompt_cache,count_tokens","target":"estia_engine::resident","span":{"request_id":"e63b82ec3d6d1695","name":"request"}}
{"timestamp":"2026-09-26T07:29:54.259940Z","level":"INFO","message":"model loaded","model":"gemma4-e2b-it-4bit-mlx","kind":"generation","load_ms":2223,"target":"estia_engine::resident","span":{"request_id":"e63b82ec3d6d1695","name":"request"}}
{"timestamp":"2026-09-26T07:29:55.081943Z","level":"INFO","request_id":"json-stream-1","method":"POST","path":"/v1/chat/completions","status":200,"duration_ms":540,"peer":"127.0.0.1:52658","caller":"builder","asked":"fast","model":"gemma4-e2b-it-4bit-mlx","stream":true,"max_tokens":60,"prompt_tokens":16,"cached_tokens":0,"completion_tokens":36,"time_to_first_token_ms":144,"tokens_per_s":66.7,"finish":"stop","target":"estia_server::access"}
{"timestamp":"2026-09-26T07:29:55.542889Z","level":"INFO","message":"pairing requested","id":"d4650fbed2ae965a","name":"laptop","scopes":"generate,embed","from":"127.0.0.1","target":"estia_server::pairing","span":{"request_id":"a43995f8678d7f17","name":"request"}}
{"timestamp":"2026-09-26T07:30:07.873609Z","level":"INFO","request_id":"60821612923d0aa3","method":"POST","path":"/v1/embeddings","status":200,"duration_ms":2518,"peer":"127.0.0.1:52682","caller":"embedonly2","asked":"embed","model":"embeddinggemma-300m-4bit","load_ms":2461,"inputs":2,"dims":768,"target":"estia_server::access"}
{"timestamp":"2026-09-26T07:30:07.883858Z","level":"INFO","request_id":"a7e58cf390b5f3d8","method":"POST","path":"/v1/chat/completions","status":403,"duration_ms":0,"peer":"127.0.0.1:52686","caller":"embedonly2","error_type":"permission_error","error":"token lacks the `generate` scope","target":"estia_server::access"}
{"timestamp":"2026-09-26T07:31:49.662579Z","level":"INFO","message":"released idle model","model":"gemma4-e2b-it-4bit-mlx","idle_s":60,"target":"estia_server"}
{"timestamp":"2026-09-26T07:32:02.129452Z","level":"WARN","message":"runner failed; starting a new one","model":"gemma4-e2b-it-4bit-mlx","pid":54718,"exit":"signal 9","cause":"runner closed stdout (EOF)","target":"estia_engine::session","span":{"request_id":"after-crash-1","name":"request"}}
{"timestamp":"2026-09-26T07:32:12.111349Z","level":"INFO","message":"estia stopped","uptime_s":142,"loaded":"gemma4-e2b-it-4bit-mlx","target":"estia_server"}
```

Access lines have no `message`; all they say is in their fields.

## Request ids

Every response carries an `X-Request-Id` header. JSON error bodies carry
the same id as `error.request_id`:

```bash
curl -si http://127.0.0.1:27200/v1/models -H 'Authorization: Bearer estia_wrong'
# HTTP/1.1 401 Unauthorized
# x-request-id: 0938df2480a78ff1
# {"error":{"code":401,"message":"unknown token","request_id":"0938df2480a78ff1","type":"authentication_error"}}
```

To use your own id, send `X-Request-Id`. Estia keeps it when it is 1 to 64
characters of ASCII letters, digits, `.`, `_`, `:` and `-`; anything else is
replaced with 16 random hex digits. Use it to tie your application's logs to
Estia's:

```bash
curl -s http://127.0.0.1:27200/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -H 'X-Request-Id: myapp-turn-42' \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "Name one sea."}]}'
```

The OpenAI SDKs read the header for you. In Python (checked with `openai`
3.19):

```python
import openai
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:27200/v1", api_key=TOKEN)
r = client.chat.completions.create(model="fast", messages=[{"role": "user", "content": "Say hi."}])
print(r._request_id)          # e58cdaee70ef0520
try:
    client.models.list()
except openai.APIStatusError as e:
    print(e.request_id, e.body)   # the id, and the error object with request_id in it
```

A streamed chat completion that fails after it has started sends
`{"error": {"message", "type", "request_id"}}` as its last event before
`data: [DONE]`. A failed `/engine/generate` stream sends
`{"error": "…", "request_id": "…"}`.

The access line has the id as `request_id`. Every other line written while
the request ran (runner started, model loaded, pairing approved, a runner
restarted) carries it in the `request` span. To find everything about one
request:

```bash
grep 0938df2480a78ff1 "<data dir>/logs/estia.err.log"                         # text
jq -c 'select(.request_id == "0938df2480a78ff1" or .span.request_id == "0938df2480a78ff1")' estia.err.log   # json
```

Model pulls and runtime installs outlive the request that started them. Their
lines carry `job_id` instead, and the access line of the starting request
carries the same `job_id`.

## The access log

One line per request, with target `estia_server::access`, written when the
response has been sent. For a streamed response that is when the stream
ends, so the line has the token counts and how the stream ended. A field that
does not apply to the request is left out.

| Field | Meaning |
|---|---|
| `request_id` | See [Request ids](#request-ids) |
| `method`, `path` | The request line. The path never includes the query string; a path over 200 bytes is cut. |
| `status` | HTTP status. A stream that failed after it started still shows 200; see `finish`. Absent when the client disconnected before a non-streaming response was ready; that generation still ran to the end, so the line has its token counts and no `finish`. |
| `duration_ms` | From the request's arrival to the end of the response |
| `peer` | Client address and port |
| `caller` | The name of the token used (never the token) |
| `error_type` | For error responses: the `type` of the error body |
| `error` | For 401 and 403: the reason: `missing bearer token`, ``Authorization is not `Bearer <token>` ``, `unknown token`, `revoked token` (removed from `tokens.json` while the server ran), ``token lacks the `generate` scope``, ``host `x` not allowed (DNS-rebinding guard)``, ``cross-origin POST from Origin `…` ``. For other errors: the error message, except 422, whose message can quote model output. For a stream that failed: the runner's error. |
| `asked` | The `model` the client sent: a role, family or model id |
| `model` | The model that served it |
| `stream` | Whether the generation streamed |
| `load_ms` | This request started the runner and loaded the model; how long that took |
| `max_tokens` | The token limit in force, after capping |
| `prompt_tokens`, `cached_tokens`, `completion_tokens` | From the runner. `cached_tokens` came from the prompt cache. Absent for raw `prompt` generations on `/engine/generate` and for cancelled streams. |
| `time_to_first_token_ms` | Streams only: from the request's arrival to the first text the runner streamed back. The runner holds back its last 24 characters to filter control markers, so this is later than the model's first token. For prose it is when the client sees text; when tools are declared, the server holds text back itself until it can tell prose from a tool call. |
| `tokens_per_s` | `completion_tokens` divided by the time the runner call took, prompt processing included |
| `finish` | `stop`, `tool_calls`, `cancelled` (the client went away mid-stream) or `error` |
| `attempts` | Structured-output runner calls, when more than one |
| `inputs`, `dims` | Embeddings: number of inputs, vector size |
| `job_id` | Pulls and runtime installs: the job the request started or found running |
| `pairing_id` | Pairing routes: the pairing the request was about |

### Levels of access lines

- `warn`: responses with status 500 or above, and generations that failed
  after the response started (`finish=error`).
- `debug`: successful polls, which clients repeat every few seconds:
  `GET /engine/health`, `/engine/stats`, `/engine/pairings`, `/engine/jobs`,
  `/engine/jobs/{id}` and `/engine/pair/{id}`. A failed poll is `info`.
- `info`: everything else.

## Lifecycle events

| Event (message) | Level | Fields |
|---|---|---|
| `estia serving` | info | `version`, `commit`, `api_version`, `protocol_version`, `url`, `bind`, `auth`, `lan`, `advertise`, `idle_unload_s`, `allowed_hosts`, `data_dir`, `backend`, `runner`, `pid` |
| `runner started` | info | `model`, `pid`, `program` |
| `runner handshake` | info | `model`, `pid`, `runner`, `runner_version`, `protocol`, `capabilities` |
| `model loaded` | info | `model`, `kind`, `load_ms` (the runner's own measure) |
| `generation model ready`, `embedding model ready` | info | `model`, `family` or `dims`, `ready_ms` (start, handshake and load together) |
| `model load failed` | warn | `model`, `kind`, `error` |
| `released idle model` | info | `model`, `idle_s` (`serve --idle-unload-minutes`) |
| `runner failed; starting a new one` | warn | `model`, `pid`, `exit` (`code N` or `signal N`), `cause` |
| `runner did not answer in time; stopping it`, `runner stream silent too long; stopping it` | warn | `model`, `pid`, `secs` |
| `runner exited mid-stream` | warn | `model`, `pid`, `exit` |
| `generation cancelled` | info | `model`, `pid`, `acknowledged` (the runner confirmed the cancel) |
| `runner stopped` | debug | `model`, `pid`, `exit` |
| a runner's stderr line | info | `model`, `pid`; target `estia_engine::runner` |
| `model pull started` | info | `job_id`, `model`, `repo`, `revision` |
| `model pull progress` | info | `job_id`, `model`, `percent` (10, 20, … 90), `bytes`, `total_bytes` |
| `model pull finished` | info | `job_id`, `model`, `files`, `bytes`, `secs` |
| `model pull failed` | warn | `job_id`, `model`, `error` |
| `download attempt failed; retrying` | warn | `model`, `file`, `attempt`, `error` |
| `resuming a partial download` | info | `model`, `bytes_on_disk` |
| `model removed` | info | `model`, `removed` |
| `runtime install started`, `runtime install phase`, `runtime install finished`, `runtime install failed` | info, warn on failure | `job_id`, `phase`, `step`, `python`, `mlx_lm`, `secs`, `error` |
| `role table replaced` | info | `roles` |
| `pairing requested` | info | `id`, `name`, `scopes`, `from` |
| `pairing approved` | info | `id`, `name`, `scopes`, `token_name` |
| `pairing denied` | info | `id`, `name`, `revoked`, `token_name` |
| `pairing token collected` | info | `id`, `name` |
| `pairing request expired undecided` | info | `id`, `name` |
| `approved pairing expired before its token was collected; …` | warn | `id`, `name`, `token_name`. The token still works: revoke it with `estia token revoke`. |
| `token revoked (removed from tokens.json)` | info | `name` |
| `token added (found in tokens.json)` | info | `name`, `scopes` |
| `advertising over Bonjour` | info | `service`, `instance`, `port` |
| `re-registered the Bonjour advertisement` | info | `reason` (`address change`, `wake from sleep`, `heartbeat`, `registrar exited`) |
| `Bonjour advertisement failed; …`, `Bonjour re-registration failed` | warn | `error`, `reason` |
| `registering via dns-sd failed; falling back to our own multicast sockets` | warn | `error` (macOS) |
| `connection refused: this client already holds its share of connections` | warn | `peer`, `per_client`, `refused`; at most one line every 10 seconds |
| `accept failed` | warn | `error` |
| `shutdown requested` | info | `signal` |
| `shutting down: …`, `stopped the Bonjour advertisement` | info | |
| `requests still in flight after 10 s; closing them` | warn | |
| `estia stopped` | info | `uptime_s`, `loaded` (models that were loaded) |
| `estia stopped with an error` | error | `error` |

Pairings decided with `estia pair approve` or `deny` on the engine's machine,
and tokens minted or revoked with `estia token`, happen in another process
that edits the data directory. The server does not see them as they happen.
It logs `pairing token collected` when a device picks up its token, and
`token added` or `token revoked` when it next reads `tokens.json`, which it
does on the next request that carries a token.

When `estia serve` mints the first admin token itself (no `tokens.json`
yet), it prints the token only if stderr is a terminal. Otherwise it logs a
warning and does not print the token, because stderr is then a log file; run
`estia token new local --replace` to get one.

## What is never logged

- Prompts, messages, system prompts, tool schemas and tool-call arguments.
- Completions, including JSON output and the text of a 422 structured-output
  error, which can quote it.
- Embedding inputs and vectors.
- Bearer tokens, their hashes and pairing tokens. Only token names appear.
- Request bodies and query strings.
- Prompt-cache keys (`user`, `cache_key`).

What is logged and may matter to you: client IP addresses, token and device
names, the `model` a client asked for, host names and origins of refused
requests, file paths under the data directory, and 5xx error messages, which
come from the engine or the runner. Runner stderr is passed through as the
runner wrote it: the runner itself does not print prompts, but a Python
library it loads could print anything. On llama.cpp, `llama-server`'s log is
passed through too, at `info` whatever its own level: several lines per
request (slot and timing lines) and more at each model load. It printed no
prompt text in our runs. The CLI's own commands (`run`, `chat`, `embed`, …)
hide it unless `ESTIA_LOG` asks for it. Hide it with
`ESTIA_LOG=estia_engine::runner=off`.

## Embedding the crates

`estia-engine` and `estia-server` install no subscriber. Without one, their
events go nowhere. Add `tracing-subscriber` to your program and install one
before you start the server:

```toml
[dependencies]
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
```

```rust
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

tracing_subscriber::registry()
    .with(EnvFilter::new("warn,estia=info"))
    .with(fmt::layer().json().flatten_event(true).with_writer(std::io::stderr))
    .init();
```

The access log needs the router from `estia_server::router`, which installs
the request-id middleware as its outermost layer. `estia_server::access`
exposes the header name and the id rules.
