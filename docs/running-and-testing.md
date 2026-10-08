# Running and testing Estia

This guide is for three kinds of reader:

- **You want to run Estia** on your own machine: [What you need](#what-you-need),
  [Install](#install), the first run [on a Mac](#first-run-on-a-mac-with-mlx) or
  [on Linux](#first-run-on-linux-with-llamacpp),
  [Run it as a service](#run-it-as-a-service) and [Logs](#logs).
- **You want to check that an engine works**, from its own machine or from
  another device: [Test it](#test-it) and [Troubleshooting](#troubleshooting).
- **You are changing Estia's code**: [For contributors](#for-contributors).

Commands and output in this guide come from real runs of Estia 0.4.0 (from a
development build before the release) on an Apple Silicon Mac with the MLX
backend, unless a section says otherwise. So sample output shows a `+dirty`
commit and MLX runner 2.2.0, where the release reports its tagged commit and
runner 2.3.0. The
sample runs used a second engine on port 27381 with its own data directory,
next to an installed one; the commands show the default port, 27200. Paths in
sample output are shortened or replaced with example ones. Not run for this guide: the `estia service`
commands, serving the LAN (`--lan`), downloads (`setup`, `pull`,
`runtime install`), and everything on Linux except what CI runs, which
[the Linux section](#what-has-run-on-linux) lists.

The README has the reference material: every command, the HTTP API, roles,
models and the security model. [docs/building-clients.md](building-clients.md)
is for people writing apps on top of Estia.

## What you need

| Machine | Backend | State |
|---|---|---|
| Apple Silicon Mac | MLX (the default there) | The main platform. Everything in this guide ran here. |
| Apple Silicon Mac | llama.cpp with Metal (`--backend llama`) | Ran end to end with two small test models. The Gemma 4 GGUF files have not been run yet. |
| Intel Mac | llama.cpp, CPU build | Should work. Not run. |
| Linux, x64 or arm64 | llama.cpp: CPU, Vulkan, CUDA or ROCm | Partly checked: see [What has run on Linux](#what-has-run-on-linux). |
| Windows | none | Estia does not build for Windows yet. |

To build from source:

- Rust 1.89 or newer ([rustup](https://rustup.rs)).
- macOS: the Xcode Command Line Tools (`xcode-select --install`).
- Linux: a C compiler, `pkg-config` and the OpenSSL headers, which the HTTPS
  client links against. On Debian and Ubuntu:
  `sudo apt-get install build-essential pkg-config libssl-dev`.

To run models, nothing else needs to be installed first: `estia setup`
downloads what the backend needs. Plan for disk space:

| What | MLX | llama.cpp |
|---|---|---|
| Runtime | Python and the MLX packages, about 700 MB | `llama-server`, 11 to 31 MB for CPU, Metal and Vulkan; more for CUDA and ROCm |
| `fast` (Gemma 4 E2B) | 3.6 GB | 3.35 GB |
| `text` (Gemma 4 E4B) | 5.2 GB | 5.15 GB |
| `embed` (EmbeddingGemma 300M) | 0.25 GB | 0.33 GB |

`estia setup` makes the `text`, `fast` and `embed` roles ready: about 9 to 10
GB with the runtime. `estia setup --roles fast,embed` needs about 4.5 GB on
MLX and 3.7 GB on llama.cpp.

To test: `curl` and `python3` for [the smoke test](#the-smoke-test), and
Python 3.9 or Node 18 for [the examples](../examples/README.md).

## Install

### From source

```bash
git clone https://github.com/modelcaddy/estia
cd estia
cargo install --locked --path cli    # puts `estia` in ~/.cargo/bin
estia --version
```

Or `cargo build --release`, which leaves the binary at
`target/release/estia`. A cold release build takes a few minutes.

On an Apple Silicon Mac the MLX backend needs its runner script,
`runners/mlx-python/estia-runner.py`. A binary built from the checkout finds
it, and every binary also carries a copy it writes into the data directory
when it needs one, so there is nothing to copy by hand. The README's
[Install](../README.md#install) section has the lookup order.

### From a release archive

Releases are published on
[GitHub](https://github.com/modelcaddy/estia/releases) as
`estia-<version>-<target>.tar.gz`, each with a `.sha256` file beside it. The
release workflow builds one target today, `aarch64-apple-darwin` (Apple
Silicon Macs). There are no Linux or Intel Mac archives yet; build from source
there.

```bash
V=0.4.0                      # the version you want
T=aarch64-apple-darwin
curl -fLO https://github.com/modelcaddy/estia/releases/download/v$V/estia-$V-$T.tar.gz
curl -fLO https://github.com/modelcaddy/estia/releases/download/v$V/estia-$V-$T.tar.gz.sha256
shasum -a 256 -c estia-$V-$T.tar.gz.sha256      # prints: estia-…tar.gz: OK
tar -xzf estia-$V-$T.tar.gz
./estia-$V-$T/estia --version
```

The folder holds `estia`, `runners/mlx-python/`, `LICENSE`, `NOTICE`,
`README.md` and `THIRD_PARTY_LICENSES`. Keep `runners/` next to the binary.
Add the folder to your `PATH`, or run the binary by its path.

`--version` on a release binary names the commit the release was tagged at,
without `+dirty`; the release workflow checks that before it publishes.

The binary is not signed or notarized. A download made with `curl` runs as it
is. A download saved by a browser is quarantined, and macOS refuses to open
it; clear the flag with `xattr -dr com.apple.quarantine estia-$V-$T`.

These commands were run as written against the public v0.4.0 release on
2026-09-27: the checksum matched and `--version` printed
`estia 0.4.0 (8058df07a, 2026-09-27)`, the tagged commit.

### Which build is this?

```bash
estia --version              # one line: version, commit and build date
estia version                # also the target, API and protocol versions, backends and features
estia version --json         # the same, for scripts
curl -s http://127.0.0.1:27200/engine/health   # a running server: "version" and "build"
```

```text
$ estia --version
estia 0.4.0 (8642bfc4e+dirty, 2026-09-26)
```

The commit ends in `+dirty` when the binary was built from a checkout with
uncommitted changes to the code, as this one was. `/engine/health` reports
the same as `"build": {"commit": …, "date": …}`, so you can tell which build
a running server is from any device; the smoke test prints it too. An
installed service runs its own copy of the binary, so after a rebuild it
keeps the old build until you run `estia service install` again. Quote
`estia version` in bug reports. [docs/versioning.md](versioning.md) explains
every field and how version numbers change.

## First run on a Mac with MLX

### 1. Set up

```bash
estia setup                        # runtime, the text, fast and embed models, config.json, a token
estia setup --roles fast,embed     # or only the small model and embeddings, about 4.5 GB
```

`setup` installs Python and the MLX packages under the data directory
(`~/Library/Application Support/estia`), downloads the models, writes the
role table and mints the first admin token. The token is printed once: keep
it. Running `setup` again is safe; it skips what is already there.

### 2. Serve

```bash
estia serve                        # http://127.0.0.1:27200, this machine only
```

It logs one line when it is ready:

```text
2026-09-26T21:44:51.934211Z  INFO estia_server: estia serving version=0.4.0 commit=8642bfc4e+dirty api_version=1 protocol_version=2 url=http://127.0.0.1:27381 bind=127.0.0.1:27381 auth=true lan=false advertise=false idle_unload_s=900 allowed_hosts=- data_dir=…/data backend=mlx-python runner=…/data/engine/runners/mlx-python/estia-runner.py pid=81009
```

Stop it with Ctrl-C. Only one server runs per data directory.

### 3. Try it in a browser

Open <http://127.0.0.1:27200/client>, paste the token into the **Token**
field and press **Save**. The page has tabs to chat, embed, pull models, set
roles and approve devices. It is a static
page served by the engine, and uses only the public API.

### 4. Try it with curl

```bash
export ESTIA_TOKEN=estia_...       # the token setup printed
curl -s http://127.0.0.1:27200/v1/chat/completions \
  -H "Authorization: Bearer $ESTIA_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model": "fast", "messages": [{"role": "user", "content": "Name one sea in Europe."}]}' \
  | python3 -m json.tool
```

```json
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "The Mediterranean Sea is a sea in Europe.",
                "role": "assistant"
            }
        }
    ],
    "created": 1790463803,
    "id": "chatcmpl-f968156af98da95039e913a0",
    "model": "gemma4-e2b-it-4bit-mlx",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 10,
        "prompt_tokens": 15,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
        "total_tokens": 25
    },
    "x_estia": {
        "backend": "mlx-python",
        "cached_tokens": 0,
        "family": "gemma4-e2b",
        "generation_tps": 86.91659075404218,
        "load_ms": null,
        "ms": 318,
        "repaired": null,
        "repairs": null,
        "template": "native"
    }
}
```

`model` is a role: `fast` answered from `gemma4-e2b-it-4bit-mlx`. The first
request to a model starts a runner and loads the weights, which took 4.6
seconds in this run; that request's `x_estia.load_ms` says so. The request
shown came later, so `load_ms` is `null` and it took 0.32 seconds (`ms`).
Requests that arrive while a model is loading wait for that one load. The
server releases a model after 15 idle minutes, 3 on a constrained machine
(`serve --idle-unload-minutes`); `estia recommend` shows this machine's tier.

### 5. Run the smoke test

```bash
scripts/smoke-test.sh --quick      # reads ESTIA_TOKEN; see "The smoke test" below
```

For anything but a quick try, give each program its own token with only the
scopes it needs:

```bash
estia token new smoke --scopes generate,embed,models:read
```

## First run on Linux with llama.cpp

This path has not been run end to end on Linux yet: see
[What has run on Linux](#what-has-run-on-linux). `estia setup --roles
fast,embed`, below, is the smaller first try, and reports are welcome.

On Linux the default backend is llama.cpp: upstream's `llama-server`, run by
Estia behind a small adapter. The steps are the same as on a Mac; what
differs is what `setup` downloads.

```bash
estia setup --roles fast,embed     # llama.cpp for this machine and the GGUF models, about 3.7 GB
estia serve
curl -s http://127.0.0.1:27200/engine/health    # "backend":"llama-cpp"
```

The same, one step at a time:

```bash
estia runtime install --backend llama              # probes the machine: CUDA, then Vulkan, then CPU
estia runtime install --backend llama --variant cpu     # or pick one: cpu, vulkan, cuda-12, cuda-13, rocm
estia runtime status
estia pull fast                                    # gemma4-e2b-it-qat-q4_0-gguf, 3.35 GB
estia pull embed                                   # embeddinggemma-300m-q8_0-gguf, 0.33 GB
estia serve
```

To run a GGUF file you already have instead of the built-in models:

```bash
estia import ~/models/my-model-Q4_K_M.gguf --id my-model
estia roles set fast my-model
```

If `llama-server` does not start for lack of a library: the CPU build needs
`libgomp.so.1` (OpenMP) and OpenSSL 3, plus glibc 2.34 or newer. On Ubuntu:
`sudo apt-get install libgomp1 openssl`. To use a `llama-server` of your own,
set `ESTIA_LLAMA_SERVER=/path/to/llama-server`. `ESTIA_LLAMA_ARGS="-ngl 0"`
keeps everything on the CPU.

On a Mac, the same backend runs with `--backend llama`:
`estia --backend llama setup`, then `estia serve`.

### What has run on Linux

Checked in CI, on GitHub's Ubuntu 24.04 x64 runners, on every push:

- The workspace builds, passes clippy, and `cargo test --workspace` passes.
- The llama.cpp adapter's integration tests pass against the pinned
  `llama-server` build (b11146, CPU) with two small models (tinygemma3 and
  all-MiniLM-L6-v2), and leave no process behind.

Not run on Linux yet:

- `estia runtime install` (the download and the variant probe), `estia setup`,
  `estia serve` answering real requests, and `estia service install` under
  systemd.
- The Gemma 4 GGUF files, on any platform.
- GPU builds (CUDA, Vulkan, ROCm), Linux on arm64, and discovery through
  Avahi.

Reports of runs on Linux are welcome: open an issue with the output of
`estia version` and of [the smoke test](#the-smoke-test).

## Run it as a service

A service starts the server at login and restarts it if it crashes: launchd
on macOS, `systemd --user` on Linux.

```bash
estia service install --local      # this machine only
estia service install              # also serves the LAN, for other devices
estia service status
estia service logs                 # the last 40 lines (-n to change it)
estia service logs -f              # follow new lines until Ctrl-C
estia service restart
estia service stop                 # and: estia service start
estia service uninstall
```

- `install` copies the binary and the runner scripts into
  `<data dir>/engine/`, so the service keeps working if the checkout or
  download moves. Run `install` again after you upgrade or rebuild.
- `install` takes `--port`, `--allow-host`, `--log-level`, `--log-format`,
  `--max-body-bytes` and the global `--backend`, and passes them to
  `estia serve`. `ESTIA_LOG` and `ESTIA_MAX_BODY_BYTES` in your shell are not
  passed on.
- To use a data directory other than the default, set `ESTIA_DATA_DIR`
  before `install`; the service definition records it.
- There is one Estia service per user account. The definition is
  `~/Library/LaunchAgents/com.modelcaddy.estia.plist` on macOS and
  `~/.config/systemd/user/estia.service` on Linux.
- On Linux, systemd stops a user's services when that user logs out, unless
  lingering is on: `loginctl enable-linger "$USER"`.

These commands were not run for this guide.

## Logs

### Where they are

| How Estia runs | Where the lines go | How to read them |
|---|---|---|
| `estia serve` in a terminal | stderr, coloured | the terminal |
| `estia serve` with stderr redirected | stderr, no colour | the file or pipe |
| Service on macOS | `<data dir>/logs/estia.err.log` | `estia service logs -f` |
| Service on Linux | the journal | `estia service logs -f`, or `journalctl --user -u estia -f` |

On macOS the data directory is `~/Library/Application Support/estia`, so the
service log is `~/Library/Application Support/estia/logs/estia.err.log`. The
file is not rotated.

### What they say

One line per HTTP request (the access log) and one per lifecycle event:
startup, runner started, model loaded or released, pulls, pairing, shutdown.
Estia never logs prompts, completions, embedding inputs, vectors, bearer
tokens, request bodies, query strings or prompt-cache keys. Token names, client
addresses and the model asked for are logged.

From the run above, the first chat request, which started the runner and
loaded the model before it answered (timestamps removed):

```text
INFO request{request_id=2f9d0363dd408b46}: estia_engine::session: runner started model=gemma4-e2b-it-4bit-mlx pid=81060 program=…/runtime/python/bin/python3
INFO request{request_id=2f9d0363dd408b46}: estia_engine::resident: runner handshake model=gemma4-e2b-it-4bit-mlx pid=81060 runner=mlx-python runner_version=2.2.0 protocol=2 capabilities=generate,stream,embed,cancel,load,chat,tools,prompt_cache,count_tokens
INFO request{request_id=2f9d0363dd408b46}: estia_engine::resident: model loaded model=gemma4-e2b-it-4bit-mlx kind=generation load_ms=3990
INFO request{request_id=2f9d0363dd408b46}: estia_server: generation model ready model=gemma4-e2b-it-4bit-mlx family=gemma4-e2b ready_ms=5376
INFO estia_server::access: request_id=2f9d0363dd408b46 method=POST path=/v1/chat/completions status=200 duration_ms=6202 peer=127.0.0.1:55582 caller="smoke" asked="fast" model=gemma4-e2b-it-4bit-mlx stream=false load_ms=5376 max_tokens=1024 prompt_tokens=13 cached_tokens=0 completion_tokens=2 tokens_per_s=2.4 finish=stop
```

`caller` is the name of the token that made the request, never the token.

### Levels and filters

The default is `info` for Estia and `warn` for the libraries it uses. Change it
with `--log-level` or `ESTIA_LOG`: a level, or `target=level` directives,
comma-separated.

```bash
ESTIA_LOG=debug estia serve                           # everything at debug
ESTIA_LOG=estia_server=debug estia serve              # the HTTP server only
ESTIA_LOG=estia_server::access=debug estia serve      # also health and other polls
ESTIA_LOG=estia_server::access=off estia serve        # no access log
ESTIA_LOG=estia_engine::runner=warn estia serve       # hide what the runner prints
estia serve --log-level warn                          # warnings and errors only
estia service install --local --log-level estia_server=debug
```

[docs/logging.md](logging.md) lists every target and event.

### JSON

`--log-format json` (or `ESTIA_LOG_FORMAT=json`) writes one JSON object per
line, for log shippers and `jq`. From a run with
`ESTIA_LOG=estia_server::access=debug`:

```json
{"timestamp":"2026-09-26T21:47:36.609297Z","level":"DEBUG","request_id":"guide-health-1","method":"GET","path":"/engine/health","status":200,"duration_ms":0,"peer":"127.0.0.1:55921","target":"estia_server::access"}
{"timestamp":"2026-09-26T21:47:36.617243Z","level":"INFO","request_id":"guide-401-1","method":"GET","path":"/v1/models","status":401,"duration_ms":0,"peer":"127.0.0.1:55922","error_type":"authentication_error","error":"unknown token","target":"estia_server::access"}
```

### Following one request

Every response has an `X-Request-Id` header, and every error body repeats it
as `error.request_id`. A client may send its own id (1 to 64 letters, digits,
`.`, `_`, `:` and `-`), and the engine keeps it. The access line carries the
id as `request_id`; every other line logged while the request ran carries it
in a `request{request_id=…}` span. So one search finds all of a request's
lines:

```bash
grep 2f9d0363dd408b46 ~/Library/Application\ Support/estia/logs/estia.err.log      # text
jq -c 'select(.request_id == "guide-401-1" or .span.request_id == "guide-401-1")' estia.log   # json
```

The smoke test sends its own ids, such as `smoke-81158-31903-embed`, and
prints the prefix it used, so one search finds a whole run. For the embedding
check of the run below (runner output left out):

```text
$ grep smoke-81158-31903-embed estia.log
2026-09-26T21:45:21.914182Z  INFO request{request_id=smoke-81158-31903-embed}: estia_engine::session: runner started model=embeddinggemma-300m-4bit pid=81202 program=…/runtime/python/bin/python3
2026-09-26T21:45:23.378277Z  INFO request{request_id=smoke-81158-31903-embed}: estia_engine::resident: runner handshake model=embeddinggemma-300m-4bit pid=81202 runner=mlx-python runner_version=2.2.0 protocol=2 capabilities=generate,stream,embed,cancel,load,chat,tools,prompt_cache,count_tokens
2026-09-26T21:45:25.092274Z  INFO request{request_id=smoke-81158-31903-embed}: estia_engine::resident: model loaded model=embeddinggemma-300m-4bit kind=embedding load_ms=1713
2026-09-26T21:45:25.092316Z  INFO request{request_id=smoke-81158-31903-embed}: estia_server: embedding model ready model=embeddinggemma-300m-4bit artifact=embeddinggemma-300m-4bit dims=768 ready_ms=3178
2026-09-26T21:45:25.150139Z  INFO estia_server::access: request_id=smoke-81158-31903-embed method=POST path=/v1/embeddings status=200 duration_ms=3236 peer=127.0.0.1:55643 caller="smoke" asked="embed" model=embeddinggemma-300m-4bit load_ms=3178 inputs=2 dims=768
```

## Test it

### The smoke test

`scripts/smoke-test.sh` checks a running engine the way a client sees it. It
needs `bash`, `curl` and `python3`, and nothing from the repository besides
itself, so you can copy it to another machine.

```bash
scripts/smoke-test.sh [--url URL] [--token TOKEN | --token-file FILE]
                      [--model ROLE] [--embed ROLE] [--quick]
```

| Option | Default |
|---|---|
| `--url` | `ESTIA_URL`, else `http://127.0.0.1:27200` |
| `--token`, `--token-file` | `ESTIA_TOKEN`. The token needs `generate`, `embed` and `models:read`. |
| `--model` | `ESTIA_MODEL`, else `fast` |
| `--embed` | `embed` |
| `--quick` | Only health, one chat and one embedding |

It reads only: it pulls no models and changes no settings. It does run the
model, and the first model call loads it. The token goes to curl through a
file, so it does not show in the process list.

| Check | Passes when |
|---|---|
| health | `GET /engine/health` answers; prints the version, build, backend and loaded models |
| auth | A request with no token gets 401 with a request id (skipped on a `--no-auth` engine) |
| models | `GET /v1/models` lists models; prints the roles |
| chat | A chat completion returns text and token counts |
| stream | A streamed completion sends at least one piece of text, then `data: [DONE]` |
| prompt cache | The second turn of a conversation (same `user`) has `cached_tokens` above 0. Skipped, with a note, if the engine reports 0 on both turns, which a runner without prompt-cache support does; both built-in backends cache. |
| json schema | An answer with `response_format` `json_schema` parses and fits the schema |
| embeddings | Two vectors of the model's width, each of length 1, with a fingerprint |
| fingerprint check | A wrong `expect_fingerprint` gets 422 |
| unknown route | A path that matches no route gets a JSON 404 |
| host check | A request for `Host: evil.example` gets 403 (the DNS-rebinding guard) |
| request id | An `X-Request-Id` the client sends comes back unchanged |

It exits 0 when no check failed, 1 when one did, and 2 when it could not start
(a bad option, a missing tool, or no token for an engine that needs one).

A full run against `estia serve` on an Apple Silicon Mac:

```text
$ scripts/smoke-test.sh --url http://127.0.0.1:27381 --token-file ~/estia-dev/smoke.token
Estia smoke test
  engine  http://127.0.0.1:27381
  model   fast (chat), embed (embeddings)
  token   from ~/estia-dev/smoke.token
  ids     requests carry X-Request-Id smoke-81158-31903-<check>

The first check that uses a model loads it, which can take several seconds.

  PASS  health              118 ms  Estia 0.4.0 (8642bfc4e+dirty, 2026-09-26), backend mlx-python, API v1, token required, loaded: gemma4-e2b-it-4bit-mlx
  PASS  auth                  1 ms  no token: 401 authentication_error "missing bearer token (Authorization: Bearer …)", request id 8962614f8dc27097
  PASS  models                1 ms  13 entries; roles: fast, text, vision; installed: 5
  PASS  chat                358 ms  "The Mediterranean Sea is a sea in Europe." (gemma4-e2b, 21 prompt + 10 completion tokens)
  PASS  stream              222 ms  2 text chunks, then [DONE], 11 completion tokens: "One, two, three, four, five."
  PASS  prompt cache        363 ms  turn 2: 35 of 52 prompt tokens came from the cache
  PASS  json schema         338 ms  {"city": "Athens", "country": "Greece"}
  PASS  embeddings          3.24 s  2 vectors of 768 dims, unit length, fingerprint embeddinggemma-300m-4bit@mlx-python
  PASS  fingerprint check     1 ms  a wrong expect_fingerprint got 422 invalid_request_error
  PASS  unknown route         1 ms  GET /v1/no-such-route got 404 not_found_error
  PASS  host check            1 ms  Host: evil.example got 403 permission_error
  PASS  request id            1 ms  sent X-Request-Id smoke-81158-31903-rid, got the same id back

All checks passed: 12 passed, 0 skipped.
The engine's log lines for this run carry smoke-81158-31903 (docs/logging.md says where the log is).
$ echo $?
0
```

With `--quick`:

```text
$ scripts/smoke-test.sh --url http://127.0.0.1:27381 --token-file ~/estia-dev/smoke.token --quick
Estia smoke test
  engine  http://127.0.0.1:27381
  model   fast (chat), embed (embeddings)
  token   from ~/estia-dev/smoke.token
  mode    quick: health, one chat, one embedding
  ids     requests carry X-Request-Id smoke-81222-30937-<check>

The first check that uses a model loads it, which can take several seconds.

  PASS  health              115 ms  Estia 0.4.0 (8642bfc4e+dirty, 2026-09-26), backend mlx-python, API v1, token required, loaded: embeddinggemma-300m-4bit, gemma4-e2b-it-4bit-mlx
  PASS  chat                363 ms  "The Mediterranean Sea is a sea in Europe." (gemma4-e2b, 21 prompt + 10 completion tokens)
  PASS  embeddings           22 ms  2 vectors of 768 dims, unit length, fingerprint embeddinggemma-300m-4bit@mlx-python

All checks passed: 3 passed, 0 skipped.
The engine's log lines for this run carry smoke-81222-30937 (docs/logging.md says where the log is).
$ echo $?
0
```

With a token the engine does not know. The checks that need a token are
skipped after the first refusal, since they would fail for the same reason:

```text
$ scripts/smoke-test.sh --url http://127.0.0.1:27381 --token estia_this_is_not_a_real_token
Estia smoke test
  engine  http://127.0.0.1:27381
  model   fast (chat), embed (embeddings)
  token   from --token
  ids     requests carry X-Request-Id smoke-81246-25910-<check>

The first check that uses a model loads it, which can take several seconds.

  PASS  health              113 ms  Estia 0.4.0 (8642bfc4e+dirty, 2026-09-26), backend mlx-python, API v1, token required, loaded: embeddinggemma-300m-4bit, gemma4-e2b-it-4bit-mlx
  PASS  auth                  1 ms  no token: 401 authentication_error "missing bearer token (Authorization: Bearer …)", request id 0ed244d981866c2e
  FAIL  models                1 ms  the engine refused the token: unknown token (request id smoke-81246-25910-models)
                                    Check --token, --token-file or ESTIA_TOKEN. To mint one on the engine's machine: estia token new smoke --scopes generate,embed,models:read
  SKIP  chat                     -  skipped: the engine refused the token (see above)
  SKIP  stream                   -  skipped: the engine refused the token (see above)
  SKIP  prompt cache             -  skipped: the engine refused the token (see above)
  SKIP  json schema              -  skipped: the engine refused the token (see above)
  SKIP  embeddings               -  skipped: the engine refused the token (see above)
  SKIP  fingerprint check        -  skipped: the engine refused the token (see above)
  PASS  unknown route         1 ms  GET /v1/no-such-route got 404 not_found_error
  PASS  host check            1 ms  Host: evil.example got 403 permission_error
  PASS  request id            1 ms  sent X-Request-Id smoke-81246-25910-rid, got the same id back

1 of 12 checks failed, 5 passed, 6 skipped.
Each FAIL line says what went wrong, and the line under it what to try.
The engine's log lines for this run carry smoke-81246-25910 (docs/logging.md says where the log is).
$ echo $?
1
```

### From another device

The engine must serve the LAN: `estia serve --lan`, or `estia service install`
without `--local`. On the engine's machine, `estia status` prints the URL
other devices should use. In these examples it is `https://192.168.1.20:27200`.

A LAN engine serves HTTPS with a certificate of its own, made on its first LAN
start. Other devices pin it, so on the engine's machine note what they will
compare against:

```console
$ estia tls fingerprint
fingerprint : 78cd9cd1cca5b77a0f427954f9a1e44e3b5ce0124e9115d9b65afe40b5100d72
short       : 78cd 9cd1 cca5 b77a   (what people compare by eye)
in full     : 78cd 9cd1 cca5 b77a 0f42 7954 f9a1 e44e 3b5c e012 4e91 15d9 b65a fe40 b510 0d72
curl        : curl -k --pinnedpubkey sha256//KcRfPaV4WCo5XHv9YWvSNn/ZDixnWYjJSznFuOON8mk= https://…
file        : /Users/you/Library/Application Support/estia/tls/cert.pem
created     : 2026-10-07
```

Plain HTTP still works from the engine's own machine (`http://127.0.0.1:27200`);
from any other device it gets `https_required`.

**A laptop, with a terminal.** If `estia` is installed on the laptop, it pairs,
shows a code, asks once you have approved whether the engine shows the same
code, and saves the token in a file only you can read:

```console
$ (umask 077 && estia pair request --engine https://192.168.1.20:27200 --name laptop \
    --scopes generate,embed,models:read > ~/.estia-token)
engine certificate 78cd 9cd1 cca5 b77a (not trusted yet: the code checks it)
pairing request `upvvrnkhkdnxwpvn` sent as `laptop`.

    code: 982 541

on the engine's machine, the operator runs:
  estia pair approve upvvrnkhkdnxwpvn
which shows the engine's code for this request: they approve only if it is 982 541 too. If the engine shows another code, stop: something between this device and the engine answered in its place.
waiting up to 290s …
approved. Does the engine show 982 541 for this request? [y/N] y
pin: 78cd9cd1cca5b77a0f427954f9a1e44e3b5ce0124e9115d9b65afe40b5100d72
  (this engine's certificate: pass it as --pin or ESTIA_PIN to remote-check, dashboard and pair --engine)
```

On the engine's machine, approve: it shows the engine's code and asks
whether the laptop shows the same. Leave it on screen; the laptop asks the
same question the other way once you have approved:

```console
$ estia pair list
engine certificate  78cd 9cd1 cca5 b77a  (estia tls fingerprint shows all of it)
ID                 STATUS                 CODE      SCOPES                       FROM             NAME
upvvrnkhkdnxwpvn   Pending                982 541   generate,embed,models:read   192.168.1.30     laptop
$ estia pair approve upvvrnkhkdnxwpvn
pairing upvvrnkhkdnxwpvn asks for scopes [generate,embed,models:read] from 192.168.1.30, under the name `laptop`

    code: 982 541

Does the device show 982 541? [y/N] y
approved upvvrnkhkdnxwpvn with scopes [generate,embed,models:read] from 192.168.1.30, name `laptop`, the codes match — the client collects its token on its next poll

on the device, confirm that this engine shows 982 541: the device trusts this engine only after that
```

If the laptop shows a different code, answer no and deny the request
(`estia pair deny <id>`); if you answer no on the laptop, it does not keep
the token: deny the request here too. In a script, `estia pair approve <id>
--code <the laptop's code>` approves without the question, and a wrong code
is refused (`the code does not match …`). On the laptop the same holds the
other way: when `estia pair request` reads from something that is not a
terminal, a `y` is refused, and it takes the six digits the engine shows for
the request (`estia pair list`) instead, pinning only if they equal its own
code; with `--json` it prints the id and its code as a `request` event on
stderr, and a `confirm` event just before it reads the answer. Without `estia`, pair with
`examples/python/pair.py`, which needs only Python, prints the same code,
asks the same question and prints an `export ESTIA_PIN=…` line. Then check
the engine from the laptop, pinning its certificate (`--pinnedpubkey` is the
`curl` line of `estia tls fingerprint`):

```bash
export ESTIA_PIN=78cd9cd1cca5b77a0f427954f9a1e44e3b5ce0124e9115d9b65afe40b5100d72
estia remote-check --engine https://192.168.1.20:27200 --token "$(cat ~/.estia-token)"   # health, one generation, one embedding
scripts/smoke-test.sh --url https://192.168.1.20:27200 --pinnedpubkey 'sha256//KcRf…8mk=' --token-file ~/.estia-token
```

**A phone or tablet, in the browser.**

1. Open `https://192.168.1.20:27200/client`. The browser warns that it does
   not know the certificate. Open its details (the warning page, or the
   padlock) and compare the SHA-256 fingerprint with `estia tls fingerprint`.
   Accept it only if they match.
2. Under **Pair this device**, enter a name for the device. The scopes
   `generate`, `embed` and `models:read` are ticked already; leave `admin`
   and `models:write` off unless the device needs them. Press
   **Request pairing**. The page shows a pairing id. A browser cannot show a
   pairing code, so the request has none (`CODE` is `none`).
3. On the engine's machine, check the request and approve it, saying that you
   compared the certificate by hand (never `--no-code` for a device that
   shows a code: deny that request instead):

   ```bash
   estia pair list                                # ID, STATUS, CODE, SCOPES, FROM, NAME
   estia pair approve dexuhpfwucdypprw --no-code
   ```

4. The page picks up its token and keeps it in the browser. Use the Chat
   tab.

The same pairing without a code, by hand with curl on the engine's machine
(plain HTTP over loopback, so `CODE` is `-`):

```text
$ curl -s -X POST http://127.0.0.1:27200/engine/pair -H 'Content-Type: application/json' \
    -d '{"name": "laptop", "scopes": ["generate", "embed", "models:read"]}'
{"check":"plain","engine_nonce":null,"expires_in":300,"how":"the operator approves with `estia pair approve <id>`; poll GET /engine/pair/<id> until approved","id":"dexuhpfwucdypprw","status":"pending","tls":{…}}
$ estia pair approve dexuhpfwucdypprw
approved dexuhpfwucdypprw with scopes [generate,embed,models:read] from 127.0.0.1, name `laptop` — the client collects its token on its next poll
$ curl -s http://127.0.0.1:27200/engine/pair/dexuhpfwucdypprw
{"id":"dexuhpfwucdypprw","status":"approved","tls":{…},"token":"estia_..."}
$ curl -s http://127.0.0.1:27200/engine/pair/dexuhpfwucdypprw
{"id":"dexuhpfwucdypprw","status":"approved","tls":{…},"token":null}
$ estia pair deny dexuhpfwucdypprw
denied dexuhpfwucdypprw (name `laptop`); its token pair:laptop:dexuhpfwucdypprw was revoked
```

The token is handed out once. `estia pair deny` takes it back, even after it
was collected: the engine refuses it from the next request on.

**Checking that nothing crosses in clear.** On the engine's machine, watch the
port while another device chats:

```bash
sudo tcpdump -i en0 -A -s0 'tcp port 27200'      # Linux: -i eth0 or the LAN interface
```

The handshake shows the certificate (it is public); after it, nothing
readable: no `Authorization` header, no token, no prompt, no output.
`server/tests/tls.rs` checks the same with a relay that records every byte.

**A laptop, with the OpenAI SDK.** Any OpenAI client works with the engine's
URL plus `/v1` and the token as the API key; over HTTPS, give its HTTP client
the pin with `examples/python/estia_tls.py`. Checked with the Python SDK 3.19
over plain HTTP (the pinned form below has not been run with the SDK yet):

```python
import os
from openai import OpenAI

import estia_tls  # examples/python/estia_tls.py

url = os.environ["ESTIA_URL"]
client = OpenAI(base_url=url + "/v1", api_key=os.environ["ESTIA_TOKEN"],
                http_client=estia_tls.openai_http_client(url))   # https: pinned to ESTIA_PIN
r = client.chat.completions.create(
    model="fast",
    messages=[{"role": "user", "content": "Name one sea."}],
    user="laptop-test",          # the prompt-cache key for this conversation
)
print(r.choices[0].message.content)
print("request id:", r._request_id)
e = client.embeddings.create(model="embed", input=["the sea at dawn"])
print(len(e.data[0].embedding), "dims")
```

**Finding the engine by name.** A LAN engine advertises itself over Bonjour as
`_estia._tcp`:

```bash
estia discover                   # NAME, HOST, URL, API, CERT (UNVERIFIED), ENGINE
dns-sd -B _estia._tcp            # macOS
avahi-browse -rt _estia._tcp     # Linux with Avahi
```

Discovery is a convenience. When it finds nothing, use the URL from
`estia status`; see [Troubleshooting](#troubleshooting).

### The examples

[examples/](../examples/README.md) has small programs, each run against a live
engine: a curl walkthrough of every call a client makes
(`examples/curl/quickstart.sh`), Python (chat, retrieval over notes, JSON
Schema extraction, tools, pairing, and a chat with no SDK), JavaScript, and
two Rust programs. They read the same `ESTIA_URL` and `ESTIA_TOKEN` as the
smoke test. The curl walkthrough is the next step after the smoke test: it
shows each request and response in full.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401` with `missing bearer token`, `unknown token` or `revoked token` | No token, a wrong one, or one that was revoked or replaced | Send `Authorization: Bearer <token>`. On the engine's machine, `estia token list` shows the names; `estia token new <name> --scopes …` mints one. |
| `403` with `this engine does not answer to the host name …` | The client reached the engine by a name the engine does not know. This guards against DNS rebinding. | Use the IP address, `localhost` or `<machine>.local`, or allow the name: `estia serve --allow-host studio.lan` (or `service install --allow-host …`, or `ESTIA_ALLOWED_HOSTS`). |
| `403` with ``token lacks the `generate` scope`` | The token was minted without that scope | Mint one with the scopes it needs: `estia token new app --scopes generate,embed`, or rotate it with `estia token new app --replace --scopes …`. |
| `403` with `cross-origin POST from Origin …` | A web page on another origin posted to the engine | Serve the page from the engine's own origin or through a proxy; see [Web front-ends and CORS](building-clients.md#web-front-ends-and-cors). |
| `no Python interpreter for the resident runner: the runtime is not installed`, or `no llama-server: the llama.cpp runtime is not installed under …` | The backend's runtime is missing | `estia runtime install` (MLX) or `estia runtime install --backend llama`, or `estia setup`. `estia runtime status` shows what is there. |
| `404` with ``model `gemma4-e2b-it-4bit-mlx` is not installed on this engine; run `estia pull …` `` | The role points to a model that is not installed | Run the `estia pull` the message names, or `estia pull fast` (a role, family or model id). `estia models` marks what is installed; `estia roles` shows the bindings. The engine's log has the path it looked in. |
| `404` with ``unknown model or role `…` `` | The `model` field names no role, family or model | Ask for a role (`fast`, `text`, `embed`). `GET /v1/models` lists the names. |
| `413` with `request body is larger than this engine accepts (… bytes, 32.0 MiB)` | A request over the body limit, 32 MiB by default: usually a large embedding batch, or a chat history with long documents pasted in | Send less per request (fewer embedding inputs per call). If the engine should take more, raise the limit: `estia serve --max-body-bytes <bytes>` (for the service, `estia service install --max-body-bytes <bytes>`), or `ESTIA_MAX_BODY_BYTES=<bytes>` in the engine's environment. |
| `400` with ``… is a gguf artifact and this engine runs mlx-python`` (or the reverse) | A model id for the other backend | Ask by role or family instead of the artifact id. |
| The first requests after installing llama.cpp on a Mac take about 25 s each | macOS checks the unsigned `llama-server` the first few times it starts (Gatekeeper). Later starts take a fraction of a second. | Wait for it. It happens only after an install. |
| `Error: an engine is already running for this data directory (pid …)` | Another `estia serve`, or the service, is using the same data directory | Use that engine, or stop it (`estia service stop`, or Ctrl-C), or give this one its own `--data-dir`. |
| `Error: Address already in use` | Another program holds the port | Pick another port with `--port`, or find the program: `lsof -nP -iTCP:27200 -sTCP:LISTEN`. |
| `estia discover` finds nothing | The engine serves loopback only (`serve` without `--lan`, or `service install --local`), advertising is off (`--no-advertise`), or the network does not pass multicast between devices (guest networks and some routers isolate clients) | Connect by address: `estia status` on the engine's machine prints the URL. |
| A request you gave up on still holds the model, and the next one waits | A known limit: a non-streaming request is not cancelled when its client disconnects; the generation runs to the end | Use `"stream": true` for anything a user may cancel; closing a stream cancels the generation. Keep `max_tokens` modest. |
| `ps` or `top` shows an MLX runner using about 100 MB while the Mac is short of memory | On macOS, a process's resident size (RSS) leaves out the Metal buffers MLX keeps the weights and KV cache in. In one run, `ps` showed 102 MB for a `gemma4-e2b` runner whose footprint was 3.8 GB, 3.4 GB of it Metal buffers. | Read the runner's physical footprint instead: `GET /engine/stats` reports it per loaded model, as does the Memory column in Activity Monitor (`footprint <pid>` in a terminal). To free memory, let idle models unload (`--idle-unload-minutes`), lower `--memory-budget`, or stop the engine. |
| A request answers 503 `insufficient_memory` | The model is larger than the memory budget for this machine's tier (`estia recommend` prints both). | Use a smaller model, or raise the budget: `estia serve --memory-budget 10GB` (`service install --memory-budget` for the service). |
| `estia serve` says it minted an admin token but did not print it | stderr was not a terminal, so the token would have gone to a log file | `estia token new local --replace` prints a new one. |
| `400` of type `https_required`: `this engine serves other devices over HTTPS only` | A client on another device used `http://` with a LAN engine (an app paired with 0.4, or a typed URL) | Use `https://` and pair again so the app pins the certificate. Then revoke the old token (`estia token list`, `estia token revoke pair:<name>:<id>`): it crossed the network in clear. |
| `the engine at … presented certificate …, but this app pinned …` (`CertificateMismatch`) | The certificate is not the one the device paired with: the operator ran `estia tls rotate`, the data directory was replaced, or something on the network is answering in the engine's place | Ask the operator. If they rotated it, pair again. If not, do not send anything to that address. |
| The device and `estia pair approve` show different codes, or `the code does not match` on `estia pair approve --code` | The device saw another certificate than the engine's | Answer no, deny the request (`estia pair deny <id>`), and answer no on the device. Look for what is between them (a proxy, a captive portal, another machine on the address). |
| `this request has a code` / `this request came over https with no code` on approve | The flags follow the request: a device that pairs over https shows a code; a browser or curl does not | For a code, run `estia pair approve <id>` in a terminal and compare, or give the code the device shows with `--code` in a script; for none, `--no-code` after comparing the fingerprint. `estia pair list` shows which in the CODE column. A device that shows a code while its request has none (`none`) met something that removed the code: deny it. |
| `pairing not confirmed` on the device | The user did not answer yes to "Does the engine show … ?" | If the engine shows another code, or none, something between the device and the engine answered in its place: do not use the token, and deny the request on the engine. Otherwise pair again and answer `y`. |
| `… serves https with its own certificate: give its fingerprint with --pin` | `remote-check`, `dashboard` or `pair --engine` was given an `https://` engine without a pin | `--pin <fingerprint>` or `ESTIA_PIN`: what `estia pair request` printed, or `estia tls fingerprint` on the engine's machine. |
| A browser warns about the engine's certificate | It is self-signed: no authority vouches for it | Compare the SHA-256 in the browser's certificate viewer with `estia tls fingerprint`, and accept it only if they match. |

When you report a problem, include `estia version`, the smoke test's output,
and the request id of a failing request.

## For contributors

[CONTRIBUTING.md](../CONTRIBUTING.md) covers pull requests and commit style.
This section is about running and testing.

### Tests

```bash
cargo test --workspace --locked
```

This runs every crate's tests, about 190 of them. Once built, they take
about 10 seconds on an Apple Silicon Mac. The engine's tests drive a fake
runner written in Python, so they need `python3` on `PATH`; without it they
print `skip:` and pass. No model and no network are needed.

Some tests are `#[ignore]`d because they use the network:

```bash
cargo test -p estia-engine --test model_store -- --ignored   # downloads from Hugging Face
cargo test -p estia-server -- --ignored                      # Bonjour advertise and discover on this machine
```

### The llama.cpp adapter against a real llama-server

`llama/tests/adapter.rs` drives the adapter against upstream `llama-server`
when three variables are set, and otherwise skips its six tests while still
reporting them as passed:

| Variable | What |
|---|---|
| `ESTIA_LLAMA_SERVER` | A `llama-server` binary, ideally the pinned build (b11146) |
| `ESTIA_LLAMA_TEST_MODEL` | A small chat GGUF with a chat template: tinygemma3 (47 MB) |
| `ESTIA_LLAMA_TEST_EMBED_MODEL` | An embedding GGUF with 384 dimensions: all-MiniLM-L6-v2 Q8_0 (25 MB) |

Get the same files CI uses. The model URLs and hashes are in the `llama` job
of `.github/workflows/ci.yml`; the `llama-server` archives and their hashes
are in `engine/src/runtime/llama_pins.rs`. On an Apple Silicon Mac:

```bash
mkdir -p ~/estia-test-assets && cd ~/estia-test-assets
curl -fLO https://github.com/ggml-org/llama.cpp/releases/download/b11146/llama-b11146-bin-macos-arm64.tar.gz
curl -fLO https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/c287502cd9e278dac8eed805c112cce5d0081e0b/tinygemma3-Q8_0.gguf
curl -fLO https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/544f204f2eaa2d71361ffc74d6df7170285b286a/all-MiniLM-L6-v2-Q8_0.gguf
shasum -a 256 *.tar.gz *.gguf
# 1ad3f9eff80edb9dbef4259ad564d1720612ef7eea48fa4afed0e54f5f3d5711  llama-b11146-bin-macos-arm64.tar.gz
# 263215c3cadd6e16740741a7624ab4cbb6c8e777688bd5331ecfbf5681c2f8ed  all-MiniLM-L6-v2-Q8_0.gguf
# 7566ae7219c93ea2ecc692a931ee122d30c55261d0e2c3347acb8b939d2e9abd  tinygemma3-Q8_0.gguf
tar -xzf llama-b11146-bin-macos-arm64.tar.gz       # llama-b11146/llama-server
```

On Linux x64 the archive is `llama-b11146-bin-ubuntu-x64.tar.gz`. Then, from
the repository:

```bash
export ESTIA_LLAMA_SERVER=~/estia-test-assets/llama-b11146/llama-server
export ESTIA_LLAMA_TEST_MODEL=~/estia-test-assets/tinygemma3-Q8_0.gguf
export ESTIA_LLAMA_TEST_EMBED_MODEL=~/estia-test-assets/all-MiniLM-L6-v2-Q8_0.gguf
cargo test -p estia-llama --locked -- --show-output
```

With `--show-output`, a skipped test prints a line starting with `skipping`;
CI fails the job on one. The first run after unpacking can be slow on macOS
while the unsigned binary is checked. On this machine the six adapter tests
took 37 seconds. tinygemma3 answers gibberish: these tests check the protocol,
not answer quality.

The hashes above were checked on files downloaded earlier; the `curl` lines
were not re-run for this guide.

### Formatting, lints and policy

CI fails a pull request on any of these:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo deny check                    # cargo install cargo-deny --locked
rustup toolchain install 1.89.0 --profile minimal
cargo +1.89.0 check --workspace --all-targets --all-features --locked   # the minimum Rust version
```

`cargo deny --offline check` uses the advisory database already on disk
instead of fetching it.

### What CI runs

`.github/workflows/ci.yml`, on every push to `main` and every pull request:

| Job | Runs on | What |
|---|---|---|
| rustfmt | Ubuntu | `cargo fmt --all --check` |
| test | macOS 15 (Apple Silicon), Ubuntu | clippy with `-D warnings`, build, `cargo test --workspace` |
| llama | Ubuntu (CPU build), macOS 15 (Metal build) | Downloads the pinned `llama-server` and the two small models, checks their SHA-256, runs the adapter tests with `--show-output`, and fails on a skipped test or on a `llama-server` left running |
| msrv | Ubuntu | `cargo check` with the `rust-version` from `Cargo.toml` |
| licences and packages | Ubuntu | Builds `THIRD_PARTY_LICENSES` with cargo-about, checks that every crate packages `LICENSE`, `NOTICE` and `README.md`, and runs `cargo package --workspace` |
| cargo-deny | Ubuntu | Licences, bans, sources and security advisories, per `deny.toml` |

`.github/workflows/release.yml` runs when a `v*` tag is pushed: it builds the
release archive, unpacks it in a clean directory to check that the binary
finds its runner and reports the tag's commit, and publishes it. The release
checklist is in [docs/versioning.md](versioning.md#release-checklist).

CI does not run a model through `estia serve`. Before a release, run the smoke
test against a real engine on each backend you changed.

### Running a development engine beside an installed one

Give the development engine its own data directory and port, so it does not
touch your installed engine's tokens, config or service:

```bash
cargo build --release
DEV=~/estia-dev
mkdir -p "$DEV"
(umask 077 && target/release/estia --data-dir "$DEV" token new smoke --scopes generate,embed,models:read > "$DEV/smoke.token")
target/release/estia --data-dir "$DEV" serve --port 27381
# in another terminal
scripts/smoke-test.sh --url http://127.0.0.1:27381 --token-file ~/estia-dev/smoke.token
```

To skip downloading models and the runtime again, make `models/` and
`runtime/` in the development directory symbolic links to the installed
engine's. Then anything that writes there (`estia pull`, `estia rm`,
`estia runtime install`) changes the installed engine too.

`estia --version` and `/engine/health` tell you which build answered (see
[Which build is this?](#which-build-is-this)).
