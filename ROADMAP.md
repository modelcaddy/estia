# Roadmap

This page says where Estia is today and what is planned next, in order. Each
item gives the reason for it and an exit check: something anyone can run to
see that the item is done. Plans change; [CHANGELOG.md](CHANGELOG.md) records
what actually shipped.

Last reviewed 2026-09-27, against version 0.4.0.

## Where Estia is today

Estia works, on one kind of machine, with limits a builder should know before
depending on it. Each limit below was checked against the code or a live run.

| Area | Today |
|---|---|
| Platforms | Models run on Apple Silicon Macs (MLX, or llama.cpp with Metal). The llama.cpp backend should also run on Linux, but has run there only in CI: the adapter's integration tests pass against the pinned `llama-server` on GitHub's Ubuntu 24.04 x64 runners. Estia does not build for Windows. Releases ship one tarball, `aarch64-apple-darwin`. |
| Backend | Two. MLX, through a Python runner (`runners/mlx-python/estia-runner.py`) and a Python runtime of about 700 MB that Estia installs; `mlx-vlm` is held below 0.7 because the runner has not been run on 0.7. And llama.cpp, new: upstream `llama-server` behind the `estia-llama` adapter, with a pinned build (b11146) that Estia downloads and checks. |
| Models | A registry of three Gemma 4 families and four embedding models, with MLX and GGUF artifacts. On llama.cpp, `estia import` adds a GGUF file without a code change. |
| Input | Text only. Image parts become a marker such as `[image_url omitted]`. The `vision` role exists, but the runner loads Gemma with no image input. |
| Tools | Tool calls are parsed from Gemma's own syntax on MLX and by `llama-server` on llama.cpp. `{"role": "tool"}` results reach the model through its chat template. Gemma 4's tool calls on llama.cpp are untested. |
| Structured output | llama.cpp constrains decoding to the JSON Schema. On MLX the engine puts the schema in the system prompt. Both then validate, repair and retry. |
| Roles | `temperature`, `max_tokens` and `pin` are stored with a role and not applied. `embed` can be bound to any embedding model; unbound it means EmbeddingGemma 300M. |
| Network | Plain HTTP. On the LAN, tokens, prompts and outputs are unencrypted. No CORS headers; cross-origin writes get 403. |
| Scheduling | One call at a time per loaded model. Waiting calls go interactive first, then background, and otherwise in arrival order. There is no per-client queue, quota or rate limit. |
| Request lifetime | A non-streaming request runs to the end after its client disconnects, and the next call waits for it. In a live run, a 600-token request abandoned after 1 s kept the model for 8.3 s, and a 5-token request behind it took 7.4 s. Request headers must arrive within 10 s, but a request body has no time limit: a connection that sent headers and then stalled was still open after 40 s. |
| Clients | OpenAI SDKs work unchanged on `/v1/*`. There is no Estia SDK for pairing, discovery or `/engine/*`. The Rust `RemoteEngine` returns errors as text, without the HTTP status as a field or the request id. The crates are not on crates.io. |
| Runners | Protocol v2 is documented in [docs/protocol.md](docs/protocol.md). The engine's tests drive a fake runner. The llama.cpp adapter has integration tests against a real `llama-server`; nothing checks the MLX runner against the protocol. |
| Packaging | Build from source or unpack the macOS tarball. `estia service install` sets up launchd or systemd `--user`. No Homebrew formula, no `.deb`, no Windows build. |

## Next

The llama.cpp backend is the major item in progress. The smaller items after
it fix problems builders hit today; they are short and do not wait for it.

### 1. llama.cpp backend

**Why.** It is the only way off Apple Silicon. llama.cpp publishes builds for
macOS, Linux and Windows, for CPU, Metal, Vulkan, CUDA (NVIDIA), ROCm and HIP
(AMD) and SYCL (Intel). One backend brings Linux servers, Windows desktops,
Intel Macs, CPU-only machines and discrete GPUs. It also constrains decoding
to a JSON Schema, and its download is 12 to 33 MB for CPU, Metal or Vulkan,
against about 700 MB for the Python runtime.

**What.** From [docs/design/llama-backend.md](docs/design/llama-backend.md):

- Run upstream `llama-server` as a child process, one per loaded model, on a
  private socket with a random API key.
- A small Rust adapter speaks runner protocol v2 on one side and
  `llama-server`'s HTTP API on the other. Sessions, the priority gate,
  deadlines, respawn and cancel do not change.
- Pin one llama.cpp build per Estia release and check its SHA-256. Pick the
  variant by probing the machine, with CPU as the fallback that always works.
- Models: Google's QAT Q4_0 GGUF files for the three Gemma 4 families, and
  EmbeddingGemma Q8_0 for `embed`. They join the existing families, so roles
  do not change.
- MLX stays the default on Apple Silicon. llama.cpp is the default
  everywhere else.

**For builders.** The HTTP API, scopes, pairing and roles stay the same.
`x_estia.backend` reads `llama-cpp`. Artifact ids differ per backend, so ask by
role or family. Embedding fingerprints end in `@llama-cpp`: an index built on
one backend must be re-embedded before it is searched from the other, and
`expect_fingerprint` turns a mismatch into a 422.

**Slices.** Each has its own exit check in the design doc.

| Slice | Delivers |
|---|---|
| L1 | The adapter, with a `llama-server` the user supplies; `estia runner-check --backend llama` |
| L2 | GGUF artifacts in the registry and store; the backend named in `/engine/health` and `x_estia` |
| L3 | Tool calls, and `format` passed to runners that declare `structured` |
| L4 | Embeddings with the `@llama-cpp` fingerprint |
| L5 | The runtime installer: pinned builds, variant probe, `estia setup` on Linux |
| L6 | Prompt-cache isolation between tokens, and more than one cache slot |
| L7 | CI on Linux and macOS with a tiny model; release archives for Linux and Intel macOS |
| L8 | Benchmarks against MLX on the same Mac; a decision on the Mac default |
| L9 | Images and audio (after the protocol change in item 6) |

**Exit check.** On a fresh Ubuntu 24.04 machine with no GPU:
`estia setup --roles fast,embed`, then `estia serve`, then
`examples/curl/quickstart.sh` exits 0 and reports `"backend":"llama-cpp"`. On a
machine with an NVIDIA GPU, setup picks the CUDA build and
`llama-server --list-devices` lists the GPU. On an Apple Silicon Mac, nothing
changes unless the operator picks the llama backend.

**Progress (2026-09-27).** Slices L1 to L7 have their code. What is missing
are the live checks that need a Gemma 4 GGUF download (3.35 GB for the
smallest) or a Linux or NVIDIA machine, and the Linux and Intel macOS
release archives.

| Slice | State |
|---|---|
| L1 | Done. The adapter's integration tests drive a real `llama-server` through the protocol. |
| L2 | Built. GGUF artifacts, pinned revisions and the explicit-file downloader (tested on a 453 KB file). The Gemma 4 GGUF files have not been pulled or run. |
| L3 | Built. `format` reaches runners that constrain decoding; tool calls and results travel as structured data. Gemma 4's tool calls on llama.cpp are untested. |
| L4 | Built. Embeddings, fingerprints and the 422, checked with a 25 MB MiniLM. EmbeddingGemma Q8_0 (334 MB) has not been pulled, nor its dense layers checked. |
| L5 | Built. The installer, pinned hashes and the variant probe, checked on macOS arm64. No fallback to the next build when `--list-devices` shows no GPU; no Linux or NVIDIA run. |
| L6 | Partly. A slot is reused only by the cache key that filled it, and cache keys are per token. One slot per model, not several. |
| L7 | Partly. The `llama` CI job runs on GitHub (Ubuntu x64 with the CPU build, macOS 15 with Metal) and passed on 2026-09-26: the adapter's 6 integration tests against the pinned `llama-server`, none skipped. No Linux or Intel macOS release archives. |
| L8, L9 | Not started. |

Checked end to end on an Apple Silicon Mac with the two small test models:
`estia runtime install --backend llama`, `estia import`, and
`estia serve --backend llama` answering chat, streams, cancel, JSON Schema
output (10 of 10 `attempts: 1`, `repaired: false`), embeddings with
`@llama-cpp` fingerprints and the 422 on a mismatch, `/engine/health` naming
the backend, and the `/client` page in a browser; after SIGTERM no
`llama-server` was left and the run directory was empty. MLX still answered
chat and embeddings on the same build. The design's
[status block](docs/design/llama-backend.md#status) has the details.

### 2. Tool results and JSON Schemas reach the model

**Why.** Both are bugs, and both push builders into workarounds that
`docs/building-clients.md` and the examples have to explain. A tool loop is
the first thing many assistants build.

**What.**

- Add `tool_calls` to the runner protocol's `Message`, and pass an assistant's
  calls to the runner as structured data, so the model's chat template renders
  the tool turn and the `{"role": "tool"}` result after it.
- Pass `response_format` to the runner as `format`. For a runner that cannot
  constrain decoding (MLX), the engine adds the schema to the system prompt.
  For one that can (llama.cpp, slice L3), decoding follows the schema. The
  engine still validates.

**Exit check.** `examples/python/tools.py`, changed to send results as
`{"role": "tool"}`, answers from the tool's result, and the prompt token count
grows when the tool message is added. `examples/python/structured.py`, with
the schema removed from its prompt, returns valid JSON in 10 of 10 runs on
`fast`. The workarounds are removed from the guide and the examples.

**Status (2026-09-27): done on MLX; on llama.cpp the mechanics are in place
and Gemma 4 is untested.**

- Tool results: `Message.tool_calls` carries an assistant's calls to the
  runner as structured data, and both runners hand them to the chat template.
  On MLX with `gemma4-e2b`, `tools.py` now sends `{"role": "tool"}` and
  answered from the result for 5 questions out of 5, in each of two
  separate runs. Gemma 4's template leaves the model's turn open
  after a tool result, and the 4-bit E2B model often ended the turn there
  with no text (3 of those 5 questions, at every temperature tried), so the
  MLX runner now asks once more in a new model turn when that happens. On
  llama.cpp the structured calls and results reach `llama-server` and its
  template: with a tool-capable template passed through `ESTIA_LLAMA_ARGS`,
  the prompt grew from 190 tokens (question) to 267 (with the call and a short
  result) and 317 (a longer result). tinygemma3, the only model run, never
  calls a tool, so a tool call parsed by `llama-server` and Gemma 4's answer
  from a result are unchecked.
- JSON Schemas: on llama.cpp the adapter constrains decoding (ten
  `/engine/generate` calls and `structured.py`: `attempts: 1`,
  `repaired: false`). A grammar cannot stop `max_tokens` from cutting a long
  string short; that output fails validation and is retried like any other.
  On MLX the engine adds the schema to the system prompt; `structured.py`
  with the schema removed from its prompt returned valid JSON in 10 runs out
  of 10 on `fast` (`gemma4-e2b`), none repaired.
- The workarounds are gone from `tools.py`, `structured.py`,
  `quickstart.sh`, `in_process.rs` and the guide.

### 3. Request lifetime: body timeout, cancel on disconnect

**Why.** A client that gives up on a non-streaming request still holds the
model until the generation ends, and everyone behind it waits. A client that
sends headers and then stalls holds a connection for as long as it likes.

**What.** Watch for the client going away during a non-streaming generation
and cancel it in the runner, as streams already do. Add a time limit for
reading a request body.

**Exit check.** A non-streaming request with `max_tokens: 600`, abandoned
after 1 s, is logged with `finish=cancelled`, and a short request sent right
after it is not delayed by the rest of the 600 tokens. A connection that sends
headers and then stops sending its body is closed within the limit, and the
limit is in [docs/api.md](docs/api.md#limits).

### 4. mlx-vlm 0.7

**Why.** The runtime installs `mlx-vlm>=0.6.13,<0.7`. 0.7 changed its
dependency set (it no longer pulls in `mlx-lm`), and the runner has not been
run on it. The longer the cap stays, the further the Mac backend falls behind
upstream fixes and model support.

**What.** Run the runner on 0.7, fix what breaks, and move the cap.

**Exit check.** A clean `estia runtime install` resolves `mlx-vlm` 0.7.x.
`estia runner-check` reports the same capabilities as today. `estia bench`
passes for all three Gemma 4 families and the default embedding model, and
`examples/curl/quickstart.sh` exits 0.

## Soon

### 5. TLS on the LAN

**Why.** A LAN engine sends tokens, prompts and outputs in plain text. Today
the advice is to serve the LAN only on a network you trust, or use an SSH
tunnel.

**What.** On a LAN bind, the engine makes a self-signed certificate and serves
HTTPS. Pairing hands the device the certificate's SHA-256 fingerprint with
its token, and the device pins it from then on. The operator sees a short code
derived from the fingerprint in `estia pair list` and the device shows the
same code, so a swapped certificate during pairing can be caught. Loopback
stays plain HTTP.

**Exit check.** `estia serve --lan` answers HTTPS on the LAN address. A packet
capture of a chat from a paired device shows no token, prompt or output in
clear text. A device that paired with one engine refuses a server with a
different certificate at the same address. `estia remote-check`,
`remote_client.rs` and the Python examples work with the pinned certificate,
and the guide shows how.

### 6. Image input through `/v1`

**Why.** The `vision` role exists and Gemma 4 E2B and E4B are multimodal, but
image parts in a message are replaced by a text marker.

**What.** Content parts in the runner protocol's `Message` (the protocol
carries a string today). Images as `data:` URLs only; a remote URL is a 400,
so the engine never fetches from the network on a client's behalf. Limits on
image size and count. The MLX runner passes images to `mlx-vlm`; the llama
backend loads the model's projector (slice L9).

**Exit check.** `/v1/chat/completions` with `"model": "vision"` and a base64
PNG describes the image, on MLX and on llama.cpp. A remote `image_url` gets
400. An image over the size limit gets 400. The marker is gone from
[docs/api.md](docs/api.md).

### 7. Role options and `embed` rebinding

**Why.** A role can store `temperature`, `max_tokens` and `pin`, and the
server ignores all three, so an operator's setting silently does nothing.
(`embed` rebinding landed with the llama.cpp work: `estia roles set embed
<model>` works, without the warning described below.)

**What.** A request's own value wins, then the role's, then the engine
default. `pin` keeps a role's model loaded through idle unload. `embed` can
be bound to any installed embedding model; `estia roles set embed` warns that
the fingerprint changes and indexes must be re-embedded.

**Exit check.** After `PUT /engine/defaults` gives `fast` `max_tokens: 20`, a
request to `fast` that sends no `max_tokens` stops at 20 tokens. A pinned
role's model is still loaded after `--idle-unload-minutes 1` has passed.
`estia roles set embed multilingual-e5-small-mlx` makes `/v1/embeddings` with
`"model": "embed"` return 384 dimensions and the fingerprint
`multilingual-e5-small-mlx@mlx-python`.

### 8. Fair sharing between clients

**Why.** Many devices can share one engine, and each loaded model serves one
call at a time. Today one client that sends a burst of requests at the same
priority makes everyone else wait behind all of them.

**What.** A queue per token, served in turn within each priority. Optional
per-token limits on waiting calls and on `max_tokens`, answered with 429 and
`Retry-After`. Queue depth per token in `/engine/stats`.

**Exit check.** Token A sends 5 requests at once and token B sends 1 just
after: B's request is served second, not sixth. A token over its limit gets
429 with `Retry-After`. `/engine/stats` shows the waiting calls per token
name.

### 9. A conformance suite for runners

**Why.** The llama adapter will be the second resident runner, and anyone who
writes a runner for another backend needs a way to show that it follows
protocol v2. The engine's tests use a fake runner; they test the engine, not
the runner.

**What.** A suite that drives any runner command through the protocol:
`hello`, `load`, `chat`, `chat_stream`, `cancel` mid-stream, `count_tokens`,
`embed_batch` with 256 inputs, `unload`, malformed input, and a check that no
child processes are left. It grows out of the L1 and L7 tests.

**Exit check.** `estia runner-check --conformance` (or a test crate with the
same checks) passes for the MLX runner and the llama adapter in CI. A runner
that ignores `cancel` fails with a message that names the check.

### 10. Thin SDKs for Python and TypeScript

**Why.** The OpenAI SDKs cover chat and embeddings. Everything else a builder
writes by hand: pairing and saving the token, discovery, checking
`api_version`, reading `x_estia`, guarding embeddings with a fingerprint, and
surfacing the request id. Most of `examples/python/pair.py` is that kind of
code.

**What.** Small packages that do those things and hand `/v1/*` to the
official OpenAI SDK instead of reimplementing it. In Rust, `RemoteEngine`
errors become a type with the HTTP status and the request id, and the crates
are published.

**Exit check.** The Python and JavaScript examples, rewritten on the SDKs,
are shorter and give the same results in the same live runs. The packages
install from PyPI and npm. A `RemoteEngine` 401 can be matched on its status
instead of its text, and carries the request id.

## Later

### 11. Packages: Homebrew, `.deb`, Windows installer

**Why.** Building from source asks too much of people who only want to run
an engine.

**What.** In order: a Homebrew tap for macOS, which can come as soon as there
is a tagged release, since the macOS tarball already exists; a `.deb` for
Ubuntu and Debian once the llama backend ships Linux archives (slice L7); a
Windows installer once Estia builds on Windows and has a way to run as a
service there.

**Exit check.** `brew install` of the tap's formula gives a working `estia`
whose tarball hash the formula checks. `apt install ./estia_<version>_amd64.deb`
on Ubuntu 22.04 and 24.04, then `estia setup --roles fast,embed`, gives a
running engine. On Windows 11, the installer puts `estia.exe` on `PATH`, and
`estia serve` answers `/engine/health`.

### 12. An optional CORS allow-list

**Why.** A web app served from another origin cannot call the engine, by
design. The guide sends builders through their own backend or a reverse
proxy. Some want a browser page to talk to a local engine directly.

**What.** `serve --allow-origin <origin>`, repeatable and off by default. The
engine answers CORS preflight for listed origins only, echoes the exact
origin (never `*`), and lets writes from those origins through the Origin
check. Tokens are still required, and the page still holds one, so the guide
keeps recommending a narrow token.

**Exit check.** A page on an allowed origin can stream a chat with `fetch`. A
page on any other origin still gets 403 and no `Access-Control-Allow-*`
headers. Without the flag, behaviour is exactly as today.

### 13. The llama.cpp default on Apple Silicon

**Why.** llama.cpp on a Mac would replace a 700 MB Python runtime with a
12 MB download. Published numbers put its generation 12 to 18 percent behind
MLX on Gemma 4 E4B.

**What.** Decide from slice L8's benchmarks, run on the same Mac as the MLX
rows in [BENCH.md](BENCH.md).

**Exit check.** The decision and the numbers behind it are in the README and
BENCH.md.

## Smaller items

These are too small for a section but are on the list:

- Report a true decode rate in the access log. `tokens_per_s` today divides
  by the whole runner call, prompt processing included. The runner could pass
  on the rate `mlx-vlm` measures.
- Answer `/v1/models` differently for artifacts the active backend cannot
  load (open question 5 in the llama design).

## Helping

Open an issue before starting on an item, so work is not done twice. The
llama design lists [open questions](docs/design/llama-backend.md#open-questions)
that need answers before or during its slices.
[CONTRIBUTING.md](CONTRIBUTING.md) covers building, testing and sending a
change.
