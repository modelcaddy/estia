# Estia

Estia is a local LLM inference engine written in Rust. It runs language and
embedding models on your own machine and serves them over an OpenAI-compatible
HTTP API, to programs on that machine and, after pairing, to other devices on
your network. Clients ask for roles such as `text`, `fast` or `embed`, and the
engine decides which model answers.

Estia is developed by ModelCaddy and runs inside its apps.

## Status

Estia is early software. Version 0.1.0 has not been released. Expect breaking
changes before 1.0.

- **Apple Silicon Macs only, for now.** The only working backend is MLX, run by
  a Python runner that Estia installs for itself. Linux and Windows need a
  llama.cpp backend. It is planned and not written yet.
- **No TLS.** LAN traffic is plain HTTP, including bearer tokens, prompts and
  outputs. That is fine on a home network you control. Do not serve the LAN on
  shared or public Wi-Fi.
- **One machine, tokens only.** One engine runs on one machine. Access is by
  bearer token with scopes. There are no user accounts.
- **A fixed model list.** The registry knows three Gemma 4 generation models
  and four embedding models. Adding a model means adding a registry entry in
  code.
- **Text only through the API.** Image parts in chat messages are replaced by
  a text marker before they reach the model.

## Install from source

You need:

- A recent stable Rust toolchain ([rustup](https://rustup.rs)) and, on macOS,
  the Xcode Command Line Tools.
- An Apple Silicon Mac to run models. `estia setup` downloads Python and the
  MLX packages; nothing else needs to be installed first.
- About 10 GB of free disk for the runtime and the default models.
- `python3` on `PATH`, only if you want to run the test suite.

```bash
git clone https://github.com/modelcaddy/estia
cd estia
cargo install --path cli     # puts `estia` in ~/.cargo/bin
# or
cargo build --release        # binary at target/release/estia
```

The CLI needs the runner script `runners/mlx-python/estia-runner.py`. It uses
the copy beside the binary or in the checkout when there is one; otherwise it
writes the copy compiled into the binary to `<data_dir>/engine/runners/` and
uses that, so a binary installed with `cargo install` works from any
directory. `ESTIA_RUNNER` (or `--runner`) overrides both.

`estia service install` copies the binary and the runner scripts into the data
directory, so an installed service keeps working if the checkout moves. Run it
again after a rebuild to upgrade the service.

Build and test:

```bash
cargo test                                                   # needs python3 for the fake-runner tests; skips them without it
cargo test -p estia-engine --test model_store -- --ignored   # live Hugging Face download tests
cargo test -p estia-server -- --ignored                      # Bonjour advertise and discover on this machine
cargo deny check                                             # licence and advisory policy in deny.toml (cargo install cargo-deny)
```

[CONTRIBUTING.md](CONTRIBUTING.md) covers formatting, linting and pull requests.

## Quick start

```bash
estia setup
```

`setup` installs the Python runtime (about 700 MB), downloads the models for
the `text`, `fast` and `embed` roles (about 9 GB), writes the role table to
`config.json` and mints the first admin token. The token is printed once. Keep
it. Running `setup` again is safe: it skips what is already there.

Start the server in the foreground:

```bash
estia serve                    # http://127.0.0.1:27200, this machine only
```

Or run it as a service that starts at login and restarts if it crashes:

```bash
estia service install --local  # this machine only
estia service install          # also serves the LAN; see the next section
```

Open <http://127.0.0.1:27200/client> in a browser and paste the token. The
page has tabs to chat, embed, pull models, set roles and approve devices. It is
a static page served by the engine and uses only the public API.

From the command line:

```bash
TOKEN=estia_...                # the token setup printed

curl -s http://127.0.0.1:27200/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "Name one sea."}]}'

curl -s http://127.0.0.1:27200/v1/embeddings \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model": "embed", "input": ["the sea at dawn", "quarterly tax filing"]}'
```

With the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:27200/v1", api_key="estia_...")
r = client.chat.completions.create(
    model="fast",
    messages=[{"role": "user", "content": "Name one sea."}],
    user="conv-1",  # prompt-cache key: later turns of conv-1 prefill only what is new
)
print(r.choices[0].message.content)
```

The first request to a model starts a runner process and loads the weights,
which takes a few seconds. Later requests reuse it. The server releases a model
after 15 idle minutes (`serve --idle-unload-minutes`, 0 to keep it loaded).

Give other programs their own tokens with only the scopes they need:

```bash
estia token new editor --scopes generate,embed
```

## Using it from other devices

In these examples the engine runs on a Mac at `192.168.1.20`.

### Serve the LAN

```bash
estia serve --lan              # or: estia service install
estia status                   # prints the URL other devices should use
```

`--lan` binds `0.0.0.0` and advertises the engine over Bonjour. Authentication
stays on: every route except health, pairing and `/client` needs a token, and
`--no-auth` is refused on any non-loopback bind, so a LAN server always requires a token.

### Pair a device

A device asks for a token and names the scopes it wants. You approve the
request on the engine's machine. The device then collects its token, once.

From a browser on the device, open <http://192.168.1.20:27200/client>, enter a
name, tick the scopes and press **Request pairing**. The page shows a pairing
id and picks up the token when you approve it.

From a terminal on the device:

```bash
estia pair request --engine http://192.168.1.20:27200 --name laptop --scopes generate,embed,models:read
```

It prints the pairing id, waits up to 5 minutes (`--wait-seconds`), and prints
the token when the request is approved.

With curl only:

```bash
curl -s -X POST http://192.168.1.20:27200/engine/pair \
  -H 'Content-Type: application/json' \
  -d '{"name": "laptop", "scopes": ["generate", "embed", "models:read"]}'
# {"id":"772ca7c0bf6ada4b","status":"pending","expires_in":300,...}

curl -s http://192.168.1.20:27200/engine/pair/772ca7c0bf6ada4b
# poll until "status":"approved"; that response carries the token, once
```

Approve on the engine's machine:

```bash
estia pair list
estia pair approve 772ca7c0bf6ada4b     # or: estia pair deny 772ca7c0bf6ada4b
```

A client holding an `admin` token can approve from anywhere: with
`estia pair approve <id> --engine http://192.168.1.20:27200 --token <admin token>`,
from the Admin tab of `/client`, or with `POST /engine/pairings/<id>/approve`.
Read the requested scopes before you approve. A device can ask for `admin`.

Pending requests expire after 5 minutes. At most 24 can wait at once.

The paired device then uses the engine like a local one:

```bash
estia remote-check --engine http://192.168.1.20:27200 --token <token>
```

```python
client = OpenAI(base_url="http://192.168.1.20:27200/v1", api_key="<token>")
```

### Discovery

A LAN engine advertises the Bonjour service `_estia._tcp` with TXT keys
`api_version`, `engine_version` and `protocol_version`. The instance name is
the machine's short hostname; change it with `serve --name`, or turn
advertising off with `--no-advertise`.

```bash
estia discover                 # browses for 3 seconds (--seconds)
dns-sd -B _estia._tcp          # macOS
avahi-browse -rt _estia._tcp   # Linux with Avahi
```

Discovery is a convenience. Pairing by address always works, and the URL from
`estia status` is enough.

On macOS the engine registers through `dns-sd`, so mDNSResponder sends the
multicast. A service started by launchd is a plain executable without the
local-network permission, and its own multicast would fail silently. Going
through mDNSResponder avoids that.

## Roles

A role is a name a client sends as `model`. The engine maps each role to a
model family, so clients do not hard-code model names.

| Role | Needs | Default binding | When unbound |
|---|---|---|---|
| `text` | text | `gemma4-e4b` | error |
| `fast` | text | `gemma4-e2b` | falls back to `text` |
| `vision` | vision | `gemma4-e4b` | falls back to `text` |
| `code` | text | none | falls back to `text` |
| `embed` | embedding | `embeddinggemma-300m-4bit` | cannot be rebound yet |

Role names are open. Bind any name to a generation family:

```bash
estia roles                              # list bindings
estia roles set writer gemma4-12b-qat
estia roles set code gemma4-e2b
estia roles rm code                      # back to the fallback
```

A binding is checked against the family's capabilities: `vision` needs a
vision-capable family and every other name needs text. Bindings are stored in
`config.json`. A running server reads that file when it starts, so restart it
after `estia roles set`, or change roles live with `PUT /engine/defaults`
(admin scope) or the Setup tab of `/client`.

Bound roles are listed first in `GET /v1/models`, with
`"x_estia": {"role": true, "family": ...}`. A request's `model` can also be a
family (`gemma4-e2b`) or an artifact id (`gemma4-e2b-it-4bit-mlx`). An artifact
id wins over a family, and a family over a role.

A binding can carry `temperature`, `max_tokens` and `pin`. They are stored but
the server does not apply them yet.

### Built-in models

| Id | Family or kind | Disk (approx.) | Licence |
|---|---|---|---|
| `gemma4-e4b-it-4bit-mlx` | `gemma4-e4b` | 5.2 GB | Gemma Terms of Use |
| `gemma4-12b-it-qat-4bit-mlx` | `gemma4-12b-qat` | 6.8 GB | Gemma Terms of Use |
| `gemma4-e2b-it-4bit-mlx` | `gemma4-e2b` | 3.6 GB | Gemma Terms of Use |
| `embeddinggemma-300m-4bit` | embedding, 768 dims (default) | 0.25 GB | Gemma Terms of Use |
| `multilingual-e5-small-mlx` | embedding, 384 dims | 0.3 GB | MIT |
| `nomicai-modernbert-embed-base-6bit` | embedding, 768 dims, English | 0.13 GB | Apache-2.0 |
| `nomic-embed-text-v1.5` | embedding, 768 dims, English | 0.6 GB | Apache-2.0 |

All are MLX weights from Hugging Face. `estia models` shows what is installed;
`estia pull <id>` and `estia rm <id>` add and remove them.

## HTTP API

| Method | Path | Scope |
|---|---|---|
| POST | `/v1/chat/completions` | `generate` |
| POST | `/v1/embeddings` | `embed` |
| GET | `/v1/models` | `models:read` |
| GET | `/engine/health` | none |
| GET | `/engine/defaults` | `models:read` |
| PUT | `/engine/defaults` | `admin` |
| GET | `/engine/models` | `models:read` |
| POST | `/engine/models/pull` | `models:write` |
| DELETE | `/engine/models/{id}` | `models:write` |
| GET | `/engine/models/{id}/progress` | `models:read` |
| POST | `/engine/generate` | `generate` |
| POST | `/engine/embed` | `embed` |
| GET | `/engine/stats` | `models:read` |
| GET | `/engine/jobs`, `/engine/jobs/{id}`, `/engine/jobs/{id}/events` | `models:read` |
| POST | `/engine/runtime/install` | `admin` |
| POST | `/engine/pair` | none |
| GET | `/engine/pair/{id}` | none |
| GET | `/engine/pairings` | `admin` |
| POST | `/engine/pairings/{id}/approve`, `/engine/pairings/{id}/deny` | `admin` |
| GET | `/client`, `/` | none |

`admin` includes every other scope.

`/v1/chat/completions` supports streaming, `tools` (the model's native call
syntax is parsed into `tool_calls`), and `response_format` with `json_object`
or `json_schema` (validated after generation, repaired where possible, retried
once). `user` is used as the prompt-cache key.

Estia adds a few fields that standard clients can ignore:

- Requests: `priority` (`interactive` or `background`) on chat and
  embeddings; `task` (`document`, `query`, `clustering` or `none`, which
  picks the model's own input prefix) and `expect_fingerprint` (refuse with 422
  unless the vectors would match) on embeddings.
- Responses: an `x_estia` object. Chat completions carry `family`, `backend`,
  `cached_tokens`, `template`, `repaired`, `repairs` and `ms`. Embeddings carry
  `fingerprint`, `dims` and `task`. `/v1/models` entries carry `role`,
  `family`, `installed`, `dims` and similar facts.

Request and response shapes for every route are in [docs/api.md](docs/api.md).

## CLI

`estia <command> --help` has the details. Every command accepts `--data-dir`,
`--runner` and `--python`.

| Command | What it does |
|---|---|
| `setup` | Install the runtime, pull models for the roles (`--roles`, default `text,fast,embed`), write `config.json`, mint the first admin token |
| `service install [--local] [--port]` | Run the server at login under launchd (macOS) or systemd `--user` (Linux); LAN unless `--local` |
| `service uninstall`, `start`, `stop`, `restart`, `status`, `logs` | Manage that service |
| `serve` | Run the server in the foreground (`--lan`, `--port`, `--bind`, `--name`, `--no-advertise`, `--idle-unload-minutes`, `--no-auth`) |
| `status` | Data directory, runtime, runner, installed models, roles, running server, pending pairings, service |
| `dashboard` | Live terminal view of a running server (`--token` for stats and pairings, `--once` for one snapshot) |
| `token new <name> [--scopes]`, `token list`, `token revoke <name>` | Bearer tokens; a new token defaults to `admin` |
| `pair list`, `pair approve <id>`, `pair deny <id>` | Decide pairing requests; add `--engine` and `--token` to act on another engine |
| `pair request --engine <url>` | Ask an engine for a token and wait for approval |
| `discover` | Find engines on the LAN |
| `remote-check --engine <url>` | Health, one generation and one embedding against a server |
| `models`, `pull <id>`, `rm <id>` | List, download and remove models |
| `roles`, `roles set <role> <family>`, `roles rm <role>` | Show and change role bindings |
| `runtime status`, `runtime install`, `runtime remove` | The Python MLX runtime |
| `run` | Generate from the prompt on stdin (`--model`, `--schema`, `--json`) |
| `chat` | Chat through the model's template; stdin is text or a JSON array of messages (`--system`, `--cache-key`, `--tools`, `--two-turns`) |
| `embed` | Embed each line of stdin and print the vectors as JSON |
| `tokens` | Count the tokens of stdin |
| `bench` | Time a model load, two generations and a batch of 32 embeddings |
| `runner-check` | Handshake with the runner and print its capabilities; loads no model |

`run`, `chat`, `embed`, `tokens` and `bench` start their own runner process.
They do not go through a running server.

## Configuration

### Data directory

| OS | Default |
|---|---|
| macOS | `~/Library/Application Support/estia` |
| Other | `~/.local/share/estia` |

Override it with `--data-dir` or `ESTIA_DATA_DIR`. One server runs per data
directory: `serve` refuses to start while another server for the same
directory answers.

| Path | Contents |
|---|---|
| `models/<id>/` | Downloaded models. An interrupted download waits in `models/<id>.download/` and resumes. |
| `runtime/python/` | Python and the MLX packages |
| `config.json` | `{"roles": {...}}`, the role table |
| `tokens.json` | Token names, SHA-256 hashes of the tokens, scopes, creation times |
| `pairings.json` | Recent pairing requests |
| `engine.json` | PID, address and port of the running server; removed when it exits cleanly |
| `engine/` | The copy of the binary and runner scripts that the service runs |
| `logs/` | `estia.out.log` and `estia.err.log` from the launchd service |

### Environment variables

| Variable | Meaning |
|---|---|
| `ESTIA_DATA_DIR` | Data directory (same as `--data-dir`) |
| `ESTIA_RUNNER` | Path to `estia-runner.py` (same as `--runner`) |
| `ESTIA_PYTHON` | Python interpreter for the runner (same as `--python`). Default: the installed runtime, else `python3` on `PATH`. |
| `ESTIA_MDNS_REFRESH_SECS` | How often the built-in Bonjour advertiser re-registers, in seconds (default 300). A testing aid; not used when macOS `dns-sd` does the advertising. |
| `RUST_LOG` | Library logging to stderr, for example `RUST_LOG=mdns_sd=debug` |

`estia service install` writes the data directory and the path of its own copy
of the runner into the service definition. If you use a non-default data
directory, set `ESTIA_DATA_DIR` before installing.

### Service files and logs

- macOS: `~/Library/LaunchAgents/com.modelcaddy.estia.plist`. Output goes to
  `<data dir>/logs/estia.err.log` and `estia.out.log`; `estia service logs`
  prints their tails.
- Linux: `~/.config/systemd/user/estia.service`. Output goes to the journal:
  `journalctl --user -u estia`.
- `estia serve` in a terminal logs to stderr.

## Security model

- **Loopback by default.** `serve` binds `127.0.0.1`. Binding any other address
  needs `--lan`, and `--lan` refuses to run without authentication.
- **Tokens on every route.** Everything except `/engine/health`, the two
  pairing routes and the static `/client` page needs a bearer token, on
  loopback too. `--no-auth` turns this off and is allowed on loopback only.
- **Scopes.** `generate`, `embed`, `models:read`, `models:write`, `admin`.
  `admin` includes the others. Give each client the least it needs.
- **Hashed at rest.** Tokens are 24 random bytes, shown once when minted.
  `tokens.json` stores only their SHA-256 hashes. A running server picks up
  tokens minted by the CLI without a restart. It can keep accepting a token
  removed with `estia token revoke` until it restarts, so restart it after a
  revoke.
- **Pairing.** Anyone who can reach the port can file a pairing request, so
  approvals are manual and requests expire after 5 minutes. An approved token
  waits in `pairings.json` in plain text until the device's next poll collects
  it, then it is removed. Whoever can write the data directory can approve
  pairings and mint tokens.
- **No TLS.** Tokens, prompts and outputs cross the network unencrypted. Serve
  the LAN only on a network you trust. For anything else, keep the server on
  loopback and reach it through a tunnel you trust, such as SSH port
  forwarding.
- **No rate limiting**, apart from the cap of 24 pending pairing requests.
- **The browser client** keeps its token in the page's `localStorage`.
- **Downloads.** The Python build is checked against a SHA-256 pinned in the
  source. Model weights are checked against the SHA-256 Hugging Face publishes
  for large files; small files get a size check. The MLX packages come from
  PyPI with a version floor, not a pin.

To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Architecture

```text
  OpenAI SDKs, curl, /client, RemoteEngine
                  │  HTTP + bearer token
                  ▼
  estia-server ── estia-engine ── Session ──► python3 estia-runner.py   (one process per loaded model)
                                               JSON lines on stdin/stdout
```

| Crate | Path | What it holds |
|---|---|---|
| `estia-proto` | `proto/` | Wire types for the runner protocol |
| `estia-engine` | `engine/` | Runner sessions (priority gate, cancel, deadlines, one respawn), one-shot runs, the model registry, the Hugging Face downloader (resume, parallel chunks, SHA-256) and on-disk store, the Python runtime installer (feature `python-mlx`), roles, structured output, and `RemoteEngine` for talking to a server from Rust |
| `estia-server` | `server/` | The axum server: `/v1/*`, `/engine/*`, tokens, pairing, Bonjour |
| `estia` | `cli/` | The `estia` binary |

`clients/web/index.html` is the `/client` page, compiled into the server.

**Runners** are the processes that run models. The engine talks to them with a
newline-delimited JSON protocol, currently version 2: one request per line on
stdin, one response per line on stdout, and `{"type":"token"}` lines for
streams. Version 2 adds a `hello` handshake with capabilities, explicit
`load` and `unload`, `chat` and `chat_stream` rendered by the model's own chat
template with a per-conversation KV cache, and `count_tokens`. A `cancel`
line stops a stream in flight. The full description is in
[docs/protocol.md](docs/protocol.md).

A **resident** runner stays up and keeps its model in memory across requests.
A **one-shot** runner reads one request, answers, and exits, paying a process
start and a model load every time. The server and the CLI use only the
resident runner.

| Runner | Kind | Notes |
|---|---|---|
| `runners/mlx-python/estia-runner.py` | resident | The working backend. `mlx-vlm` for generation, `mlx-embeddings` for embeddings. |
| `runners/mlx-python/oneshot-runner.py` | one-shot | The older Python runner |
| `runners/apple/AppleRunner.swift` | one-shot | Apple Foundation Models; builds with plain `swiftc` |
| `runners/mlx-swift/` | one-shot | Compiled MLX runner for hosts that must ship a signed binary and cannot download an interpreter |

See [runners/README.md](runners/README.md) for how each is built.

The **Python runtime** is a pinned `python-build-standalone` build (Python
3.12.7 for arm64 macOS) with `mlx-vlm` and `mlx-embeddings` installed from
PyPI, all under `<data dir>/runtime/`. No system Python, Homebrew or pip is
needed.

**Structured output.** The MLX runner cannot constrain decoding, so the engine
enforces JSON after the fact: parse, repair common defects (code fences,
preambles, bad escapes, unescaped quotes), validate against the JSON Schema,
and retry once with the validator's complaint.

## Benchmarks

Measured numbers, with dates and machines, are in [BENCH.md](BENCH.md). One
example: on an Apple Silicon Mac with 32 GB, `gemma4-e2b` through the HTTP API
took 5.4 s for the first request (process start, load and prompt) and 422 ms
for the second turn of the same conversation, with 31 of its 49 prompt tokens
served from the cache. Run `estia bench` on your own machine before quoting a
number.

## Licence

Estia is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). The licence covers the code, not the names "Estia" or
"ModelCaddy".

Models and the Python packages Estia downloads come under their own licences.
The Gemma models are under the Gemma Terms of Use. `GET /engine/models` lists
each model's licence.
