# Design: a llama.cpp backend

Status: written 2026-09-26 as a proposal; slices L1 to L7 are now built, with
the live checks that need large downloads or other machines still open. See
[Status](#status).

Estia runs models only through MLX on Apple Silicon today. This page decides
how Estia should run them through [llama.cpp](https://github.com/ggml-org/llama.cpp),
which would add Linux, Windows, Intel Macs, CPU-only machines, and NVIDIA,
AMD and Intel GPUs. It is written for contributors, and for builders who want
to know what will change for an app that sits on top of Estia.

Facts about upstream projects were checked on 2026-09-26 against the sources
listed at the end. Facts about Estia were checked against this repository at
commit `09ba222`. Where something was not verified, the text says so.

## Status

Updated 2026-09-27. The code for L1 to L7 is in the tree: the `estia-llama`
crate, GGUF artifacts and imports in the engine, the `llama-runtime`
installer, `--backend` in the CLI and server, and the CI job. Where the
implementation differs from the text below, the text is kept and the
difference noted here.

| Slice | State | Evidence |
|---|---|---|
| L1 adapter | Done | `cargo test -p estia-llama` with `ESTIA_LLAMA_SERVER`, `ESTIA_LLAMA_TEST_MODEL` and `ESTIA_LLAMA_TEST_EMBED_MODEL` set: 6 integration tests against `llama-server` b11146 pass (load, stream, cancel, JSON Schema, a 256-input batch, cache isolation, cleanup on SIGTERM and on a killed parent), plus 39 unit tests |
| L2 registry and store | Built; Gemma 4 GGUF not run | Artifacts, pinned commits, explicit file lists, `model.gguf` naming; file names, sizes and hashes match the Hugging Face API. The downloader was tested on a 453 KB file. The 3.35 GB `gemma4-e2b-it-qat-q4_0-gguf` was not pulled |
| L3 tools and structured output | Built; Gemma 4 tools untested | `format` goes to runners that list it under `structured`; `meta.tool_calls` and `parses_tool_calls`; explicit sampling; thinking off. JSON Schema output on tinygemma3: 10 of 10 `attempts: 1`, `repaired: false`. Tool results reach the template (checked with a tool-capable template, below); a parsed tool call from `llama-server` is unit-tested only |
| L4 embeddings | Built; EmbeddingGemma GGUF not run | 384-dimension unit vectors from all-MiniLM-L6-v2 Q8_0, fingerprint `minilm@llama-cpp`, 422 on an MLX fingerprint, 256 inputs in one batch |
| L5 runtime install | Built; macOS arm64 only | `estia runtime install --backend llama` verifies the SHA-256, unpacks, runs `--version` (`build 11146`). Probe order and CUDA pairing are unit-tested; no GPU fallback after `--list-devices` yet |
| L6 cache isolation | Partly | One slot; reused only by the cache key whose request filled it and succeeded. Several slots: not done |
| L7 CI and releases | Partly | The `llama` job (ubuntu-latest CPU, macos-15 Metal) passed on GitHub on 2026-09-26 (commit `8642bfc`): the 6 adapter integration tests ran against `llama-server` b11146 on both, none skipped. No new release archives |
| L8, L9 | Not started | |

Differences from the text below:

- The adapter declares `runner: "estia-llama"` and `version` is the crate
  version, without the build number.
- One model per adapter: a request for another model, or the same file as
  the other kind, replaces the running `llama-server`.
- `generate` sends the prompt as one user turn through
  `/v1/chat/completions`, as the table says; `/completion` is not used.
- The prompt cache keeps one conversation per model (`-np 1`), not eight as
  the Python runner does.
- The embedding role can be rebound (`estia roles set embed <model>`), and
  `estia import` registers a user's own GGUF file; neither was in the plan.
- On MLX, where decoding cannot be constrained, the server now puts the JSON
  Schema in the system prompt.
- The engine now stops every runner with SIGTERM and a 4 s grace before
  SIGKILL (idle unload, shutdown, deadlines), so the adapter always gets to
  stop its `llama-server` and remove its socket and record.
- On macOS the first starts of a freshly installed `llama-server` are slow:
  in the run below, the install's own `--version` check and the next three
  starts took 24 to 30 s each (macOS checking the unsigned binary; almost no
  CPU), and later starts 0.7 s. The check during install did not make the
  next start fast.

Live run on an Apple M1 Pro (macOS 15.7.3), late on 2026-09-26, scratch data
directory, port 27361, with the two CI test models. The archive was put in
the installer's resume directory first, so the install did not download it
again; the hash check, unpack and `--version` ran as usual. An earlier run
downloaded it (11 MB).

```text
$ estia --data-dir <d> runtime install --backend llama              # 29.6 s
  "variant": "metal", "version": "version: 0.5.0-dev (build 11146, commit 7fe450e19)", "reason": "Apple Silicon: Metal"
$ estia --data-dir <d> --backend llama import tinygemma3-Q8_0.gguf --id tinygemma3 --link
imported tinygemma3 (generation, gemma3) · context 32768 (the file declares 131072) · template yes
$ estia --data-dir <d> --backend llama import all-MiniLM-L6-v2-Q8_0.gguf --id minilm --link
imported minilm (embedding, bert) · 384-dim, mean pooling · fingerprint minilm@llama-cpp
$ estia --data-dir <d> roles set fast tinygemma3 && estia --data-dir <d> roles set embed minilm
$ estia --data-dir <d> serve --backend llama --port 27361
INFO estia serving ... backend=llama-cpp runner=.../estia → .../runtime/llama/b11146-metal/llama-server
INFO runner handshake model=tinygemma3 runner=estia-llama protocol=2 capabilities=generate,stream,embed,cancel,load,chat,tools,prompt_cache,count_tokens,structured:json,structured:json_schema
INFO [estia-llama] started llama-server 50996 for .../models/tinygemma3/model.gguf (generation) on unix:.../run/llama-50995.sock
INFO model loaded model=tinygemma3 kind=generation load_ms=25924        # first start after install; later 700 to 900 ms
```

- `GET /engine/health`: `"backend": "llama-cpp"`, first entry
  `{"id": "llama-cpp", "active": true, "build": "b11146", "variant": "metal", "server": "installed", "runtime_installed": true}`.
- Chat, non-streaming and streaming: `x_estia.backend: "llama-cpp"`,
  `generation_tps` 110 to 160, final chunk with `usage`, then `[DONE]`. The
  same conversation sent again with the same `user` reported 18 of 23 prompt
  tokens cached; another `user` right after it, 0.
- Cancel: a `max_tokens: 4096` stream dropped by `curl --max-time 1` was logged
  `generation cancelled ... acknowledged=true` and `finish=cancelled`,
  `llama-server` logged `stop: cancel task`, and the next request answered in
  39 ms.
- JSON Schema: ten `/engine/generate` calls at temperature 0.7 with a schema
  of an integer, a boolean, an enum and a string with `maxLength`, each
  `attempts: 1`, `repaired: false`; one with a raw `prompt` too;
  `/v1/chat/completions` with `json_schema` and with `json_object` returned
  valid JSON, `repaired: false`. `examples/python/structured.py` returned
  schema-valid JSON three times out of three. With a plain `string` field the
  tiny model often wrote until `max_tokens` inside the string: the grammar
  keeps the output on the schema's path but cannot close it early, so the
  first attempt was cut off, and the retry passed or ended in a 422.
- Tools: a request with `tools` reaches `llama-server` (tinygemma3 answers in
  prose; its Gemma 3 template has no tool support, and a follow-up with a
  `tool` message is refused with HTTP 500 `Conversation roles must
  alternate`). With `ESTIA_LLAMA_ARGS="--chat-template-file <a Hermes-style
  template with tools>"`, the same `/v1/chat/completions` round trip was
  accepted: the question with the tool declared was 190 prompt tokens, the
  follow-up with the assistant's `tool_calls` and a short `tool` result 267,
  and with a longer result 317, so the calls and results reach the template.
  A tool call parsed by `llama-server` and returned in `meta.tool_calls` was
  not produced live (the tiny model never calls a tool).
- Embeddings: 384-dimension vectors of norm 1.0, fingerprint
  `minilm@llama-cpp`; a 256-input batch in order in 0.54 s; 257 inputs → 400;
  `expect_fingerprint: "embeddinggemma-300m-4bit@mlx-python"` → 422
  `fingerprint mismatch`. An MLX artifact id → 400; an embedding model with no
  GGUF artifact → 404; pulling an import → 400.
- `/v1/models` lists the MLX artifacts with `runnable: false` and the GGUF
  ones, built-in and imported, with `runnable: true`.
- `/client` in a browser (`--no-auth`): the Connect tab shows
  `backend llama-cpp (llama.cpp b11146, metal)`, the Setup tab describes the
  llama.cpp runtime and marks MLX models "(other backend)", the model picker
  leaves them out, and the Chat tab streamed an answer from
  `fast (role → tinygemma3)`.
- SIGTERM on `estia serve`: `estia stopped ... loaded=minilm,tinygemma3`,
  each adapter logged `signal 15: stopping llama-server and exiting`,
  `pgrep llama-server` found nothing, and `run/` was empty.
- MLX on the same build, port 27362: chat (`Athens`, `generation_tps` 98),
  embeddings (768 dimensions, `embeddinggemma-300m-4bit@mlx-python`, 422 on a
  `@llama-cpp` fingerprint), `structured.py` valid in 10 runs of 10, and
  `tools.py` answered from the tool result for 5 questions of 5.

What remains:

- **Gemma 4 GGUF (L2, L3).** Pull `gemma4-e2b-it-qat-q4_0-gguf` (3.35 GB),
  run the BENCH.md `get_weather` request and check the `tool_calls` entry and
  the answer after a `tool` message; run ten JSON Schema generations.
  Also check whether Gemma 4 on llama.cpp ends its turn right after a tool
  result, as the 4-bit MLX E2B does (the MLX runner now retries in a new
  turn; the adapter has no such retry).
- **EmbeddingGemma GGUF (L4).** Pull `embeddinggemma-300m-q8_0-gguf`, check 768
  dimensions and the fingerprint, and list its tensors for the dense layers.
- **Linux and GPUs (L5, L7).** A GPU fallback after `--list-devices`; an
  NVIDIA machine picking CUDA; `estia setup` and `estia serve` on a real
  Linux machine (CI runs only the adapter's tests); release archives for Linux
  and Intel macOS.
- **More slots (L6), benchmarks and the Mac default (L8), images (L9).**
- **Smaller:** `service install` does not copy `ESTIA_LLAMA_SERVER` or
  `ESTIA_LLAMA_ARGS` into the service definition; a template error from
  `llama-server` is a 500, not a 400; the adapter creates no Windows job
  object; `finish_reason` from the adapter is not read, so `length` never
  reaches clients; a plain `load` can take longer than the engine's 300 s
  line deadline without a keepalive (the adapter waits up to 600 s).

## Summary

1. **Run upstream `llama-server` as a child process** (option A below). Estia
   downloads a pinned llama.cpp release build for the platform and
   accelerator, checks its SHA-256, and starts one `llama-server` per loaded
   model on a private socket with a random API key.
2. **Keep the runner protocol as the only boundary.** A small Rust adapter
   speaks protocol v2 on stdin and stdout, like `estia-runner.py`, and turns
   each request into `llama-server` HTTP calls. `Session`, the priority gate,
   deadlines, respawn and cancel do not change. The `estia` binary can run the
   adapter itself, so a release is still one binary plus the downloaded
   llama.cpp build.
3. **Do not link llama.cpp into Estia** (option B) **and do not use
   llama-cpp-python** (option C). B would put CUDA, ROCm, Vulkan and SYCL
   toolchains into Estia's build. C needs a Python runtime on every platform
   and brings its own chat and tool handling. Both lag upstream by weeks.
4. **Models:** Google's own QAT Q4_0 GGUF files for the three Gemma 4
   families, and the ggml-org EmbeddingGemma Q8_0 GGUF for `embed`. They join
   the existing families as `format: gguf` artifacts. Roles do not change.
5. **Backend choice:** MLX stays the default on Apple Silicon. llama.cpp is
   the default everywhere else, and an option on Apple Silicon until a
   benchmark on the same Mac says otherwise.
6. **Embeddings from llama.cpp are a different vector space.** Their
   fingerprint ends in `@llama-cpp`. An index built on one backend must be
   re-embedded before it can be searched from the other.

### What changes for builders

- Nothing in the HTTP API, token scopes, pairing or roles. A client that
  sends `"model": "fast"` or `"model": "embed"` works on either backend.
- `x_estia.backend` reads `llama-cpp` instead of `mlx-python`.
- Artifact ids differ per backend (`gemma4-e2b-it-4bit-mlx` versus
  `gemma4-e2b-it-qat-q4_0-gguf`). Ask by role or family, not by artifact id,
  if your app should run on both.
- Embedding fingerprints differ per backend. Store the fingerprint with your
  index and send it back as `expect_fingerprint`; a mismatch is a 422, not
  silently worse results.
- Structured output gets stricter: llama.cpp constrains decoding to the JSON
  Schema, so `repaired` should be rare. The engine still validates.

## Where Estia is today

From the code:

- One backend, `mlx-python`: `runners/mlx-python/estia-runner.py`, started by
  `Engine::launch` in `engine/src/engine.rs` as `<python> <script>`.
- Runner protocol v2 (`proto/src/lib.rs`, [protocol.md](../protocol.md)):
  `hello`, `load`, `unload`, `chat` and `chat_stream` (messages, tools,
  `cache_key`, `format`, `max_tokens`, `temperature`), `count_tokens`,
  `embed_batch`, `cancel`, `keepalive` and `meta` lines.
- The registry already has `Format::Gguf` and
  `find_family_default(family, format)`, but no GGUF artifacts
  (`engine/src/models/registry.rs`).
- MLX is hard-coded in the server and CLI: `resolve_generation` uses
  `Format::Mlx` and `embed_backend()` returns `BACKEND_MLX_PYTHON`
  (`server/src/lib.rs`); the `backend` field and the `/engine/health`
  `backends` list are the literal `mlx-python` (`server/src/engine_api.rs`);
  the CLI uses `Format::Mlx` and `BACKEND_MLX_PYTHON` in several commands
  (`cli/src/main.rs`).
- The server never passes `format` to the runner: every `chat_with` and
  `chat_stream_with` call in `server/src/openai.rs` and
  `server/src/engine_api.rs` passes `None`. JSON is enforced after
  generation by `engine/src/structured.rs`.
- The downloader keeps `.json`, `.txt`, `.model`, `.safetensors`, `.py`,
  `.jinja` and tokenizer files (`should_download_hf_file` in
  `engine/src/models/hf.rs`). It would skip a `.gguf` file.
- Embedding fingerprints are `<model id>@<backend>`
  (`EmbedModel::fingerprint_for` in `engine/src/models/embed.rs`).
- The Python runtime is about 700 MB and pinned to `aarch64-apple-darwin`
  (`engine/src/runtime/python.rs`).
- CI builds and tests on `macos-15` and `ubuntu-latest`. Releases ship
  `aarch64-apple-darwin` only. Estia has not been built on Windows in CI.

From a live run on this machine (Apple M1 Pro, 32 GB, macOS 15.7.3), with a
scratch data directory:

```text
$ estia --data-dir <scratch> runner-check
protocol : v2 (mlx-python 2.1.0)
{"generate": true, "stream": true, "embed": true, "cancel": true, "load": true,
 "chat": true, "tools": true, "prompt_cache": true, "count_tokens": true, "structured": []}
```

The llama adapter must declare the same capabilities, and can add
`structured: ["json", "json_schema"]`.

## llama.cpp today

**Licence.** MIT. The release workflow copies `LICENSE` into every archive.

**Cadence.** Every push to `master` publishes a build release named
`bNNNNN`, unless the commit message says `[no release]`. The API listing
shows about a hundred build releases a week in September 2026: b10244 was
published on 2026-08-03 and b11193 on 2026-09-26. Since August 2026 there are
also version tags (v0.1.0 was tagged on 2026-08-17) and versioned releases,
v0.2.0 (2026-08-21) through v0.5.0 (2026-09-23).
A versioned release has release notes and one file, `nightly-tag.txt`, which
names its build: v0.5.0 is b11146. GitHub's "latest release" API returns
v0.5.0, which has no binaries, so an installer must name a build tag.

**Prebuilt binaries.** Build b11193 has 35 assets. The ones that matter here:

| Platform | Accelerator | Asset | Size |
|---|---|---|---|
| macOS arm64 | Metal and CPU | `llama-b11193-bin-macos-arm64.tar.gz` | 11.8 MB |
| macOS x64 | CPU (Metal is off in this build) | `llama-b11193-bin-macos-x64.tar.gz` | 11.3 MB |
| Linux x64 | CPU | `llama-b11193-bin-ubuntu-x64.tar.gz` | 17.0 MB |
| Linux arm64 | CPU | `llama-b11193-bin-ubuntu-arm64.tar.gz` | 13.6 MB |
| Linux x64 | Vulkan | `llama-b11193-bin-ubuntu-vulkan-x64.tar.gz` | 31.0 MB |
| Linux arm64 | Vulkan | `llama-b11193-bin-ubuntu-vulkan-arm64.tar.gz` | 24.8 MB |
| Linux x64 | CUDA 12.8 | binaries + `cudart-…-cuda-12.8-x64` | 170.5 + 594.4 MB |
| Linux x64 | CUDA 13.4 | binaries + `cudart-…-cuda-13.4-x64` | 151.3 + 440.2 MB |
| Linux arm64 | CUDA 13.4 | binaries + `cudart-…-cuda-13.4-arm64` | 147.1 + 552.5 MB |
| Linux x64 | ROCm 10.0 (AMD) | `llama-b11193-bin-ubuntu-rocm-10.0-x64.tar.gz` | 240.2 MB |
| Linux x64 | SYCL (Intel), fp16 / fp32 | `…-ubuntu-sycl-fp16-x64`, `…-fp32-x64` | 55.3 / 55.0 MB |
| Linux x64 | OpenVINO 2026.4 | `…-ubuntu-openvino-2026.4-x64` | 109.1 MB |
| Windows x64 | CPU | `llama-b11193-bin-win-cpu-x64.zip` | 18.6 MB |
| Windows arm64 | CPU | `llama-b11193-bin-win-cpu-arm64.zip` | 12.1 MB |
| Windows x64 | Vulkan | `llama-b11193-bin-win-vulkan-x64.zip` | 32.5 MB |
| Windows x64 | CUDA 12.4 | binaries + `cudart-llama-bin-win-cuda-12.4-x64.zip` | 262.4 + 391.4 MB |
| Windows x64 | CUDA 13.4 | binaries + `cudart-llama-bin-win-cuda-13.4-x64.zip` | 151.7 + 423.5 MB |
| Windows x64 | ROCm/HIP 10.0 (AMD) | `llama-b11193-bin-win-rocm-10.0-x64.zip` | 256.7 MB |
| Windows x64 | SYCL (Intel) | `llama-b11193-bin-win-sycl-x64.zip` | 120.2 MB |

Every asset has a SHA-256 digest in the GitHub releases API, and each build
has a GitHub build attestation (`actions/attest` in the release workflow).

**How they are built** (`.github/workflows/release.yml`):

- Linux builds use `GGML_BACKEND_DL=ON`, `GGML_NATIVE=OFF` and
  `GGML_CPU_ALL_VARIANTS=ON`. The CPU code is compiled for many instruction
  sets and one is picked at run time; GPU backends are shared libraries loaded
  at run time. Binaries find their libraries through an `$ORIGIN` rpath, so
  the CUDA runtime archive is extracted into the same directory.
- Windows GPU zips are the Windows CPU build with one GPU backend DLL added
  (the release job "injects" `llama-server` and the CPU backend into them).
- CUDA builds leave `CMAKE_CUDA_ARCHITECTURES` unset and use llama.cpp's
  default list of GPU architectures. ROCm builds name their targets: 23 on
  Linux, from `gfx908` to `gfx1201`.
- Build images, which set the oldest glibc a Linux binary can need: x64 CPU
  and Vulkan build on `ubuntu-22.04`; arm64 CPU and Vulkan on
  `ubuntu-24.04-arm`; CUDA inside `nvidia/cuda:*-devel-ubuntu24.04`; ROCm and
  SYCL on `ubuntu-24.04`. Not verified: the glibc symbol versions the
  binaries actually require.
- macOS builds set `CMAKE_OSX_DEPLOYMENT_TARGET=13.3`.

**Signing.** The release workflow has no code-signing or notarization step for
macOS. A reporter on macOS 26.3.1 saw a freshly downloaded, unsigned
`llama-server` (b10534) take about 37 seconds on its first run while macOS
checked it; later runs were fast
([openclaw#138672](https://github.com/openclaw/openclaw/issues/138672)).

## What llama-server offers against what the protocol needs

The adapter answers the runner protocol and makes these calls. "Adapter"
means the answer comes from the adapter without a call.

| Protocol v2 request | llama-server call | Notes |
|---|---|---|
| `hello` | adapter | Capabilities: `generate`, `stream`, `embed`, `cancel`, `load`, `chat`, `tools`, `prompt_cache`, `count_tokens`, `structured: ["json", "json_schema"]`. `runner: "llama-server"`, `version: "<adapter version>+b<build>"`. |
| `ping` | adapter, plus `GET /health` when a model is loaded | `/health` answers 503 while the model loads and 200 when ready. It needs no API key. |
| `load` (generation) | start `llama-server -m <dir>/model.gguf …`, poll `GET /health` | Reports `ms` from spawn to the first 200. |
| `load` (embedding) | start `llama-server --embedding …` | A separate process, as the MLX runner has a separate embed session today. |
| `unload` | stop the process | llama-server has `--sleep-idle-seconds`, but Estia's own idle unload already drops the session. |
| `chat`, `chat_stream` | `POST /v1/chat/completions` with `messages`, `stream`, `stream_options.include_usage`, `max_tokens`, `temperature`, `cache_prompt`, `id_slot`, `chat_template_kwargs` | The chat template comes from the GGUF file and runs through llama.cpp's Jinja engine (`--jinja` is on by default). |
| `tools` | `tools` in the same request | llama.cpp detects Gemma 4's template by its `'<\|tool_call>call:'` string and parses calls with a Gemma 4 parser (`common/chat.cpp`). The response carries OpenAI `tool_calls`, not raw text. See [Tool calls](#tool-calls). |
| `format` (`json`, `json_schema`) | `response_format` `json_object` or `json_schema` | Decoding is constrained by a grammar built from the schema. llama.cpp supports a subset of JSON Schema; the engine keeps validating. llama-server refuses a custom `grammar` together with `tools`. |
| `cache_key` | `id_slot` and `cache_prompt` | llama-server reuses the longest matching prefix held by a slot. It has no per-key caches. See [Prompt cache](#prompt-cache-and-isolation). |
| `meta` line | final chunk: `usage.prompt_tokens`, `usage.prompt_tokens_details.cached_tokens`, `usage.completion_tokens` | Streams carry `usage` in a last chunk when `stream_options.include_usage` is true; every final chunk also has `timings` (`cache_n`, `prompt_n`, `predicted_n`). `template` is always `native`. |
| `count_tokens` | `POST /tokenize` with `add_special: true` | Count the returned ids. The Python runner counts with the tokenizer's default special tokens; `add_special: true` should match. |
| `generate`, `generate_stream` (v1, raw prompt) | chat with one user message | The Python runner also treats `prompt` as one user turn. |
| `embed_batch`, `embed` | `POST /v1/embeddings` with `input: [...]` | Vectors are L2-normalised. Each input must fit the physical batch size (`-ub`). |
| `cancel` | close the HTTP connection | llama-server polls the connection; when it is closed, the response reader posts cancel tasks for the request (`server_response_reader::stop` in `tools/server/server-queue.cpp`). The adapter then ends the stream with `{"done": true, "cancelled": true}`. |
| `keepalive` line | adapter timer, in streams only | llama-server also sends SSE comment pings every 30 s by default. In a stream the adapter sends its own keepalive every few seconds while it waits, so Estia's 300 s silence deadline cannot trip during a long prefill. A non-stream `chat` writes nothing until its answer, because outside a stream the engine takes the next line as the response. |
| images, audio (not in the protocol yet) | `image_url` and `input_audio` content parts | Gemma 4 E2B and E4B take image and audio input in llama.cpp through their `mmproj` file (`docs/multimodal.md`). Needs a protocol change first. |

Other llama-server features, and whether Estia would use them:

- **Router mode** (one process that loads and unloads several models): not
  used. The router starts each model as a child `llama-server` on a free
  `127.0.0.1` TCP port, and removes API-key options from the arguments it
  gives the child (`unset_reserved_args` in `tools/server/server-models.cpp`).
  Estia already does loading, unloading and idle release itself.
- **Metrics** (`--metrics`, Prometheus text): off by default. Useful later for
  `/engine/stats`.
- **Context shift**: off by default. A request whose prompt does not fit the
  context is refused ("exceeds the available context size"), and generation
  stops when the context is full. Estia should count tokens first or pass the
  error on.
- **Built-in tools, agent mode and MCP servers** (`--tools`, `--agent`,
  `--mcp-servers-*`): never. They give the model file and shell access.
- **Speculative decoding**: ggml-org publishes MTP draft files for Gemma 4.
  Later, if measured to help.

## Options

- **A.** Download upstream `llama-server` builds and run one per loaded model.
  Talk to it over loopback HTTP.
- **B.** Write a Rust runner that links llama.cpp through the
  [`llama-cpp-2`](https://crates.io/crates/llama-cpp-2) crate and speaks the
  JSON-lines protocol.
- **C.** Add [`llama-cpp-python`](https://pypi.org/project/llama-cpp-python/)
  to the existing Python runner.

| | A: `llama-server` | B: `llama-cpp-2` | C: `llama-cpp-python` |
|---|---|---|---|
| Platforms and accelerators | Every row of the asset table above, built and published by upstream | Whatever Estia's CI builds. Each accelerator needs its toolchain in CI (CUDA toolkit, ROCm, Vulkan SDK, oneAPI) on Linux and Windows | PyPI has only a source archive (0.3.35), so a plain `pip install` compiles llama.cpp. The project's GitHub releases carry wheels for CPU (several platforms), CUDA 11.8 to 13.2, ROCm, HIP, Vulkan and Metal, installed by URL or extra index |
| Build work for Estia | None: download, verify, extract | High: `llama-cpp-sys-2` features `cuda`, `vulkan`, `rocm`, `metal`, `mtmd` compile llama.cpp with CMake | Picking the right wheel per machine. Linux and Windows also need a Python runtime, which Estia does not install there today |
| Lag behind upstream | None: pin any build | Crate releases come a few days to four weeks apart (0.1.157 on 2026-09-22); the repository's llama.cpp submodule points at a commit from 2026-09-21 | 0.3.35 (2026-08-17) vendors llama.cpp from 2026-08-16; no release since |
| Fit with protocol v2 | Full: Jinja templates, Gemma 4 tool-call parser, schema grammars, prompt cache, tokenizer, embeddings, `mmproj`, cancel | Low-level API. The docs.rs page for the latest release lists `apply_chat_template` and `json_schema_to_grammar` but no wrapper for the Jinja chat parser that llama-server uses for tools. Estia would rewrite slots, caching and parsing | Renders templates with Python's `jinja2` and has its own tool-call handlers (`llama_chat_format.py`); it does not use llama.cpp's Gemma 4 parser |
| Download size | 12 to 33 MB (CPU, Metal, Vulkan); about 575 to 765 MB with CUDA libraries | Similar, built by Estia | Python runtime (about 700 MB on macOS today) plus the package |
| Crash isolation | Separate process | Separate process | Separate process |
| Licence | MIT | MIT or Apache-2.0 crate over MIT llama.cpp | MIT |

**How others do it.** Ollama starts upstream `llama-server` per model as a
subprocess, on a free port on `127.0.0.1`, with `--no-webui` and `--offline`,
and leaves GPU layers and threads to `llama-server` unless the user sets them
(`llm/llama_server.go`). The Ollama 0.34.2 app installed on the machine this
page was written on bundles `llama-server`, `libllama.0.4.1.dylib`, and
`libggml-cpu-*` variant libraries. LM Studio runs llama.cpp (and MLX on Apple
Silicon) and manages its inference runtimes apart from the app with
`lms runtime ls|get|select|update|remove`. A user report shows it keeping one
llama.cpp build per accelerator on Windows (`nvidia-cuda12-avx2`,
`vulkan-avx2`, `avx2`, …) under `.lmstudio/extensions/backends/`
([Deguffer#64](https://github.com/BootBlock/Deguffer/issues/64)).

**Why A.** It is the only option that covers every platform on day one without
Estia owning a GPU build matrix, and the only one where tool parsing,
templates and grammars come from the same code upstream tests. Ollama
already runs `llama-server` this way.

**What A costs.** One HTTP hop on loopback per call. Two processes per loaded
model (adapter and server). Estia inherits upstream's flag and API changes,
so it must pin a build and test before each bump. `llama-server`'s defaults
suit a standalone server, not a child process, so the adapter sets many flags
explicitly.

## Design

### Processes

```text
estia serve ── Session ──►  estia runner llama  ──HTTP──►  llama-server -m model.gguf
               (unchanged)  (adapter, protocol v2)          UNIX socket, or 127.0.0.1 + API key
```

One adapter and one `llama-server` per loaded artifact: the same
one-process-per-model shape the MLX runner has. `Session` sees an ordinary
resident runner.

### The adapter

- A new workspace crate, `estia-llama`: a library and a small binary. The
  `estia` CLI runs it in a hidden mode (`estia runner llama`), so the release
  stays one file. A host that embeds `estia-engine` can call the library from
  its own binary, or ship the small binary.
- The engine starts it with `Launch::new(<adapter>)` and arguments naming the
  `llama-server` binary. `Launch` already takes a program and arguments;
  `EngineConfig` needs a backend field in place of the Python interpreter and
  script it assumes today.
- It speaks HTTP/1.1 to one `llama-server`, including server-sent events, over
  TCP or a UNIX socket.
- It writes `llama-server`'s output to its own stderr. Stdout carries only
  protocol lines.
- It exits when its parent dies and stops its `llama-server` on the way out,
  as the Python runner's orphan watchdog does.

### llama-server flags

| Flag | Why |
|---|---|
| `-m <artifact dir>/model.gguf` | The weights. The store saves each artifact's files under fixed names. |
| `--mmproj <artifact dir>/mmproj.gguf` | Only once the API passes images. Leaving it out saves about 1 GB of memory for E2B and E4B. |
| `--host <dir>/<name>.sock` on macOS and Linux | A UNIX socket: no TCP port to find or guard. `--host` treats a value ending in `.sock` as a socket path (`server-http.cpp`; see also the v0.5.0 notes). |
| `--host 127.0.0.1 --port <free port>` on Windows | Whether UNIX sockets work on Windows is not verified. |
| `--api-key-file <file, mode 0600>` | A random key per process. `/health` stays open; every other route needs the key. A file keeps the key out of `ps`. |
| `--no-ui` | No web interface. |
| `--no-slots` | `/slots` exposes per-slot state. |
| `--offline` | No network access for model downloads. |
| `-c <registry context_length>` | The GGUF files declare 131,072 tokens (E2B, E4B) and 262,144 (12B). Left at the default, the KV cache would be sized for that. |
| `-np 1` at first, more later | Estia sends one call at a time per runner. More slots only keep more conversations cached ([L6](#slice-plan)). |
| `--cache-ram 0` | The default keeps up to 8192 MiB of prompt cache in RAM, shared across callers. |
| `--embedding --pooling <model default> -b N -ub N` | Embedding processes only. `-ub` must hold the longest input. |
| `--metrics` | Optional, for stats. |

Never passed: `--tools`, `--agent`, `--mcp-servers-config`,
`--mcp-servers-json`, `--media-path`, `-hf`, `--models-dir`.

GPU layers, threads and flash attention are left to `llama-server`, which
picks them automatically and, with `--fit` (on by default), sizes the context
and offload to the device's free memory. Ollama also leaves GPU layers and
threads to `llama-server` unless the user sets them.

### Sampling, templates and thinking

- The adapter sends every sampling value and never inherits `llama-server`'s
  defaults (temperature 0.8, top_k 40, top_p 0.95, min_p 0.05). When the
  request leaves them null it uses the Python runner's defaults:
  `max_tokens` 256 and `temperature` 0.0. `llama-server`'s own default for
  `max_tokens` is unlimited.
- Gemma 4's GGUF chat template has an `enable_thinking` switch. The adapter
  sends `chat_template_kwargs: {"enable_thinking": false}` so output matches
  the MLX runner, until Estia exposes reasoning.

### Tool calls

When a request declares tools, `llama-server` always parses the model's tool
calls and returns them as OpenAI `tool_calls` (it sets `parse_tool_calls`
itself, in `tools/server/server-common.cpp`). Estia's server instead parses
the model's raw text (`server/src/toolcalls.rs`). Proposal:

- Add an optional `tool_calls` array (OpenAI shape) to the runner's `meta`
  line and a capability flag that says the runner parses calls itself. The
  server uses them when present and parses text otherwise. Old runners and
  old engines are unaffected: unknown fields are ignored.
- Content deltas still stream as `token` lines. The adapter collects
  `delta.tool_calls` chunks and sends them once, in `meta`. The server already
  holds tool calls back to the final chunk.

### Prompt cache and isolation

Estia keeps prompt caches per token: the server mixes a hash of the caller's
token into `cache_key`, so `cached_tokens` cannot tell one client about
another's conversation ([api.md](../api.md)). `llama-server` reuses any
matching prefix held by a slot, whoever sent it, and so does its RAM prompt
cache. Left alone, client B would see `cached_tokens > 0`, and a faster first
token, when client A had sent the same opening.

The adapter keeps the guarantee:

- `--cache-ram 0`.
- Each slot remembers the `cache_key` that last filled it. A request goes to
  the slot that holds its key (`id_slot`). A request with a new key takes the
  least recently used slot and is sent with `cache_prompt: false`, so nothing
  from the slot's previous owner is reused or reported.
- A request with no `cache_key` is sent with `cache_prompt: false`.

### Embeddings

- One `llama-server --embedding` per embedding model.
- Fingerprint: `<GGUF artifact id>@llama-cpp`, for example
  `embeddinggemma-300m-q8_0-gguf@llama-cpp`. The accelerator and build number
  are not part of it: Estia treats the same weights and pooling as one space,
  accepting small floating-point differences between CPU and GPU kernels. The
  build goes into `x_estia` for diagnosis.
- Vectors will not match the MLX ones: different quantisation (MLX 4-bit
  against Q8_0), different kernels, and in some GGUF conversions no projection
  layers (see below). The existing fingerprint rule already treats them as
  different spaces.
- The Python runner truncates EmbeddingGemma inputs at 512 tokens. The adapter
  should do the same, or the difference must be documented.

### Crashes and orphans

- `llama-server` dies: the adapter fails the call in flight and exits;
  `Session` respawns once, as today.
- The adapter is killed hard and cannot clean up. On Linux, start
  `llama-server` with a parent-death signal (`PR_SET_PDEATHSIG`). On Windows,
  put it in a job object that kills on close. macOS has neither, so the
  adapter records `<data dir>/run/llama-<pid>.json` and the engine stops stale
  servers when it starts.

## Model artifact plan

### Generation

New artifacts in the existing families:

| Family | New artifact id | Repository, revision | Files | Size | Licence |
|---|---|---|---|---|---|
| `gemma4-e2b` | `gemma4-e2b-it-qat-q4_0-gguf` | `google/gemma-4-E2B-it-qat-q4_0-gguf` @ `675cff42a74c` | `gemma-4-E2B_q4_0-it.gguf`, `gemma-4-E2B-it-mmproj.gguf` | 3.35 GB + 0.99 GB | Apache-2.0 |
| `gemma4-e4b` | `gemma4-e4b-it-qat-q4_0-gguf` | `google/gemma-4-E4B-it-qat-q4_0-gguf` @ `4b4a2c1d584b` | `gemma-4-E4B_q4_0-it.gguf`, `gemma-4-E4B-it-mmproj.gguf` | 5.15 GB + 0.99 GB | Apache-2.0 |
| `gemma4-12b-qat` | `gemma4-12b-it-qat-q4_0-gguf` | `google/gemma-4-12B-it-qat-q4_0-gguf` @ `29d097773436` | `gemma-4-12b-it-qat-q4_0.gguf`, `mmproj-gemma-4-12b-it-qat-q4_0.gguf` | 6.98 GB + 0.18 GB | Apache-2.0 |

The MLX artifacts of the same families are 3.6, 5.2 and 6.8 GB.

Why these files:

- They are Google's own, not gated, and Apache-2.0 according to their model
  cards and Google's Gemma 4 licence page.
- They are quantisation-aware trained for 4-bit, so Q4_0 should lose less than
  a Q4_0 made after training. The 12B MLX artifact is already a conversion of
  Google's QAT checkpoint, so this keeps the same lineage.
- Each repository holds one weights file and one projector. Nothing to choose.
- Their GGUF metadata carries the chat template with Gemma 4's
  `<|tool_call>call:` syntax, which is what llama.cpp detects (checked through
  the Hugging Face API).

Alternatives, to compare in [L8](#slice-plan) before any switch: ggml-org
Q4_0 (E2B 2.84 GB, E4B 4.59 GB, not QAT), unsloth QAT `UD-Q4_K_XL` (E2B
2.62 GB, E4B 4.22 GB, 12B 6.72 GB), and unsloth or bartowski `Q4_K_M` (E2B
3.11 or 3.46 GB).

### Embeddings

| Estia model | GGUF artifact | Repository, revision | File | Size | Licence |
|---|---|---|---|---|---|
| EmbeddingGemma 300M (default `embed`) | `embeddinggemma-300m-q8_0-gguf` | `ggml-org/embeddinggemma-300M-GGUF` @ `0f741b5a6585` | `embeddinggemma-300M-Q8_0.gguf` | 334 MB | Gemma Terms of Use (base model) |
| Nomic Embed Text v1.5 | `nomic-embed-text-v1.5-q8_0-gguf` | `nomic-ai/nomic-embed-text-v1.5-GGUF` @ `0188c9bf4097` | `nomic-embed-text-v1.5.Q8_0.gguf` | 146 MB | Apache-2.0 |
| Multilingual E5 Small, Nomic ModernBERT | none planned | | | | |

llama.cpp's converter handles the BERT, NomicBERT, ModernBERT and Gemma
embedding architectures, but only community GGUFs exist for E5 Small and
ModernBERT. Estia keeps those two only so old indexes can be identified.

**Projection layers.** EmbeddingGemma ends with two dense layers
(768 → 3072 → 768) after mean pooling. llama.cpp's converter includes them only
with `--sentence-transformers-dense-modules`. The Hugging Face API reports
307,581,696 parameters for ggml-org's Q8_0 file and 302,863,104 for ggml-org's
`embeddinggemma-300M-qat-q4_0-GGUF`. The difference, 4,718,592, is exactly
2 × 768 × 3072. So the QAT Q4_0 file appears to lack the dense layers and the
Q8_0 file to have them. The MLX artifact has them (`dense.0` and `dense.1` in
its weight map). Use Q8_0, and confirm by listing its tensors in L4.

### Registry and store changes

- `Artifact` gets an explicit file list, because GGUF repositories often hold
  many quantisations. The store saves the files as `model.gguf` and
  `mmproj.gguf` in `models/<artifact id>/`, so `model_path` stays a directory
  and the adapter never guesses.
- GGUF artifacts pin `revision` to a commit, not `main`.
- GGUF artifacts download their listed files, not `should_download_hf_file`'s
  pattern. SHA-256 comes from the Hugging Face LFS `oid`, as today.
- `EmbedModel` splits like generation models: prefixes, dimensions and the
  multilingual flag stay on the model; each format gets its own artifact with
  id, repository, files and size. `find_embed_model` resolves either id.
- `/v1/models` and `/engine/models` should show which artifacts the running
  backend can load. Either hide the others or add a `runnable` field (open
  question).

## Backend selection

| Host | Default | Also possible |
|---|---|---|
| macOS arm64 | `mlx-python` | `llama-cpp` with Metal |
| macOS x64 | `llama-cpp`, CPU | none (upstream's x64 build has Metal off) |
| Linux x64 | `llama-cpp`: CUDA, then ROCm, then Vulkan, then CPU | SYCL, OpenVINO when asked for |
| Linux arm64 | `llama-cpp`: CUDA, then Vulkan, then CPU | |
| Windows x64 | `llama-cpp`: CUDA, then HIP, then Vulkan, then CPU | SYCL when asked for |
| Windows arm64 | `llama-cpp`, CPU | CUDA 13.4 arm64 exists upstream |

Rules:

1. One backend per engine, that is per data directory. `estia setup` chooses
   it and writes it to `config.json`. `--backend` and `ESTIA_BACKEND`
   override it.
2. Probe, do not guess. Cheap signals give a candidate order (an NVIDIA driver
   library, `/dev/kfd` for AMD, a Vulkan loader). Estia installs a candidate
   and runs `llama-server --list-devices`. If no GPU device is listed, it
   falls back to the next candidate. The CPU build always works, and it picks
   its instruction set at run time.
3. Of the CUDA builds, try the newest first and fall back to the older one if
   the probe fails, which covers older drivers.
4. `ESTIA_LLAMA_SERVER=/path/to/llama-server` uses a binary the user provides:
   a distribution package, or a build for a GPU the prebuilt binaries miss.
   Estia checks `--version` and warns outside the tested range.
5. Roles resolve to the family's artifact in the active backend's format
   (`find_family_default(family, backend.format())`), so one role table works
   on every backend.

## Runtime install

A `LlamaRuntime` in `engine/src/runtime/llama.rs`, behind a `llama-runtime`
feature for the same reason `python-mlx` exists: a host that must not download
executable code builds without it.

- One pinned build per Estia release, preferably the build a versioned
  llama.cpp release names (v0.5.0 is b11146). The SHA-256 of every asset
  Estia may fetch is compiled in, taken from the GitHub asset digests. Never
  "latest".
- Layout: `<data dir>/runtime/llama/<build>-<variant>/` holding the binaries,
  their libraries and, for CUDA, the extracted runtime archive. A `.version`
  stamp and an atomic swap at the end, as the Python installer does.
- Checks: hash, extract, remove the macOS quarantine attribute, run
  `llama-server --version` with a first-run timeout of at least 120 s (the
  macOS check above), then `--list-devices`.
- CLI and API: `estia runtime install --backend llama [--variant cpu|vulkan|cuda-13|cuda-12|rocm]`,
  `runtime status`, `runtime remove`. `POST /engine/runtime/install` takes a
  `backend`. `/engine/health` lists
  `{"id": "llama-cpp", "runtime_installed": true, "build": "b11146", "variant": "vulkan"}`.
- Download sizes: 12 to 33 MB for CPU, Metal and Vulkan; 591 MB for CUDA 13.4
  on Linux x64 with its runtime libraries; 765 MB for CUDA 12.8; 240 MB for
  ROCm. The Python MLX runtime is about 700 MB.

## Services

- **Linux.** `estia service install` already writes a systemd `--user` unit.
  CPU, Vulkan and CUDA need nothing more. ROCm needs the user to be allowed to
  open `/dev/kfd` and `/dev/dri/renderD*`; AMD's install guide does this with
  the `render` and `video` groups or a udev rule. `estia status` should say
  so when the probe finds an AMD GPU it cannot open.
- **Windows.** Estia has no service support there. `estia serve` in a terminal
  would work once Estia builds on Windows. Out of scope for this page.

## CI

- A new job on `ubuntu-latest` downloads the pinned llama.cpp CPU build
  (17 MB), `ggml-org/tinygemma3-GGUF` (a 47 MB Gemma 3 model with a chat
  template, plus a 1 MB projector; WTFPL) and `all-MiniLM-L6-v2` Q8_0 from
  `second-state/All-MiniLM-L6-v2-Embedding-GGUF` (25 MB, Apache-2.0). It
  caches them with `actions/cache`, keyed by build tag and file hashes.
- It runs adapter tests through the real protocol: `hello`, `load`,
  `chat_stream`, a cancel mid-stream, `count_tokens`, output constrained to a
  JSON Schema, a 256-input `embed_batch`, `unload`, and a check that no
  `llama-server` process is left behind.
- The same job runs on `macos-15` with the macOS arm64 build.
- A PR that bumps the pinned build must pass this job.
- The tiny model cannot test Gemma 4's template or tool calls. A manual or
  nightly job with `gemma4-e2b-it-qat-q4_0-gguf` (3.35 GB) covers those.

## Performance expectations

Nothing has been measured with llama.cpp in this repository. Published
numbers, and one MLX baseline measured while writing this page:

| Hardware | Runtime | Model | Prompt processing | Generation | Source |
|---|---|---|---|---|---|
| AMD Ryzen 7 7700, 8 threads, CPU only | llama.cpp | Gemma 4 E2B Q8_0 | ≈273 tok/s (pp512) | ≈22.7 tok/s (tg128 at depth 512) | gemma4.c README |
| RTX 4070 Ti, all layers on the GPU | llama.cpp build 8641 | Gemma 4 E4B Q8_0 | ≈6,757 tok/s (pp512) | ≈69.7 tok/s (tg128) | alfonsofortunato.com |
| RTX 4070 Ti, 33 layers on the GPU | same | same | ≈3,157 tok/s | ≈27.9 tok/s | same page |
| M2 Air, 16 GB | llama.cpp / MLX | Gemma 4 E4B, Q4_K_M / 4-bit | | 24 / 27 tok/s | birjob.com |
| M3 Max, 64 GB | llama.cpp / MLX | same | | 78 / 92 tok/s | birjob.com |
| M1 Pro, 32 GB, other jobs running | MLX, `estia chat` | `gemma4-e2b-it-4bit-mlx` | first token 297 to 888 ms for a 24-token prompt | ≈73 to 76 tok/s (318 tokens, three runs) | measured here, 2026-09-26 |

The last row came from:

```bash
echo "Write a 300-word description of a lighthouse at night." \
  | estia --data-dir <scratch> chat --model gemma4-e2b --max-tokens 400 --temperature 0
# turn 1: first token 789 ms · total 4987 ms · prompt 24 tokens (0 cached) · generated 318 · template native
```

Generation speed is 318 tokens over the time from first token to end. The
machine had a load average between 7 and 19 from other work.

What to expect:

- **Apple Silicon.** In the published E4B numbers MLX generates 12 to 18
  percent faster than llama.cpp. Speed alone is no reason to change the Mac
  default.
- **CPU only, x86.** E2B is usable for short answers at about 20 tokens a
  second on a current 8-core desktop. E4B is slower. Q4_0 reads fewer bytes
  per token than Q8_0 and should generate faster; measure it.
- **NVIDIA consumer GPU.** E4B at about 70 tokens a second with the whole
  model on the GPU. Whether every layer fits on the GPU matters more than
  which card it is.

Each row is one source, with different builds and quantisations. L8 replaces
them with rows in [BENCH.md](../../BENCH.md).

## Risks

- **Upstream churn.** A hundred builds a week. Flags get renamed and removed
  (the current help lists `--draft-max` and other `--draft-*` options as
  removed). Mitigation: pin one build, run the
  conformance job before every bump, and have the adapter refuse builds
  outside the tested range unless the user overrides.
- **macOS first run.** Upstream binaries are not signed with a Developer ID or
  notarized, and the first run can stall for tens of seconds. Mitigation:
  run `--version` during install with a long timeout. A host that must ship
  only signed code (a sandboxed Mac app) has to build and sign `llama-server`
  itself, as with the Swift MLX runner today.
- **glibc.** The arm64, CUDA, ROCm and SYCL builds come from Ubuntu 24.04
  images and may not start on older distributions. Mitigation: the install
  probe falls back to the x64 CPU build (Ubuntu 22.04 image), and
  `ESTIA_LLAMA_SERVER` accepts a distribution's own build. Later, Estia could
  publish its own builds for older systems.
- **Drivers.** A CUDA 13 build needs a newer driver than a CUDA 12 build. ROCm
  builds cover a fixed list of GPUs. Vulkan needs a loader and a driver.
  Mitigation: the probe and fallback order in [Backend selection](#backend-selection).
- **Server defaults are open.** Without flags, `llama-server` reflects any
  CORS origin with credentials, serves a web interface, exposes `/slots`,
  fetches remote `image_url` URLs, and in router mode runs children on
  loopback TCP. Mitigation: the flag table above, a private socket or a
  random API key, and an adapter that forwards only `data:` URLs once images
  are supported.
- **Cache leaks across clients.** Prefix reuse would reveal to one token what
  another token sent. Mitigation: the slot rules in
  [Prompt cache](#prompt-cache-and-isolation), with a test in L6.
- **Behaviour differences from MLX.** Sampling defaults, thinking, context
  length and truncation, tool-call parsing, embedding truncation. Mitigation:
  the adapter sets each one explicitly; L3, L4 and L8 compare outputs.
- **Memory.** Two `llama-server` processes (generation and embedding) each
  size themselves to the memory free when they start. Estia does no memory
  accounting today. Ollama reads buffer sizes from `llama-server`'s log; the
  adapter could report them later.
- **Socket path length.** A UNIX socket path is limited to about 104 bytes on
  macOS and 108 on Linux. Deep data directories will not fit. Mitigation:
  short names under `<data dir>/run/`, then a private directory under
  `$TMPDIR`, then TCP with an API key.

## Open questions

1. Should `llama-cpp` become the default on Apple Silicon? It would replace a
   700 MB Python runtime with a 12 MB download, at 12 to 18 percent slower
   generation in published numbers. Decide after L8.
2. Tool calls: is the `meta.tool_calls` extension right, or should the adapter
   render the prompt itself (`/apply-template`) and call `/completion` to get
   raw text? The second keeps Estia's parser but depends on Gemma's special
   tokens appearing in raw output, which is not verified.
3. How should Estia expose reasoning (`reasoning_content`, `enable_thinking`)?
   Same question for MLX.
4. Images and audio: the protocol's `Message.content` is a string. Content
   parts need a protocol change, and a rule that remote URLs are refused.
5. Does `/v1/models` hide artifacts the running backend cannot load, or mark
   them?
6. Which JSON Schema keywords does llama.cpp's grammar conversion reject, and
   what should the engine do then: fall back to validate-and-repair, or 400?
7. Windows: UNIX sockets in `llama-server`, a service manager, and whether
   Estia builds there at all.
8. Should EmbeddingGemma on llama.cpp truncate at 512 tokens like the MLX
   runner, or use the model's full 2048?

## Slice plan

Each slice ends with a check someone else can run.

**L1. Adapter with a user-supplied `llama-server`.** Crate `estia-llama`:
`hello`, `ping`, `load`, `unload`, `chat` and `chat_stream` (text only),
`count_tokens`, `cancel`, keepalives, private socket and API key, orphan
handling. Engine: a backend choice in `EngineConfig`, and
`estia runner-check --backend llama`.
*Exit check:* `estia runner-check --backend llama` prints the capabilities
listed above. The crate's integration test, given `ESTIA_LLAMA_SERVER` and the
tinygemma3 GGUF, drives the adapter through protocol v2: `load` returns `ms`;
`chat_stream` streams tokens and then a `meta` line with prompt, cached and
generated counts; a cancel sent after the first token of a
`max_tokens: 4096` request ends with `{"done": true, "cancelled": true}` long
before 4096 tokens, and the `llama-server` log shows the task cancelled; after
`kill -TERM` on the adapter, no `llama-server` child is left.

**L2. GGUF artifacts in the registry and store.** Explicit file lists, pinned
revisions, fixed file names, the three Gemma 4 GGUF artifacts, `Backend` in
place of the hard-coded `Format::Mlx` and `mlx-python` in server and CLI,
`/engine/health` naming the backend.
*Exit check:* `estia pull gemma4-e2b-it-qat-q4_0-gguf` downloads exactly the
listed files and verifies their SHA-256; `estia models` shows the artifact
with `format: gguf`; `/v1/chat/completions` with `"model": "fast"` answers
from it on a llama engine and from the MLX artifact on an MLX engine;
`cargo test` passes on Linux and macOS.

**L3. Tools and structured output.** Tool calls through `meta.tool_calls`;
the server passes `format` to runners that declare `structured`; explicit
sampling; thinking off.
*Exit check:* the `get_weather` request from BENCH.md on
`gemma4-e2b-it-qat-q4_0-gguf` returns an OpenAI `tool_calls` entry named
`get_weather` with arguments `{"city": "Athens"}`; ten `/engine/generate`
calls with a JSON Schema all return `attempts: 1` and `repaired: false`.

**L4. Embeddings.** GGUF embedding artifacts, the `EmbedModel` split, the
`@llama-cpp` fingerprint, pooling and batch flags, the truncation decision.
*Exit check:* `/v1/embeddings` on a llama engine returns 768-dimension unit
vectors with fingerprint `embeddinggemma-300m-q8_0-gguf@llama-cpp`; sending
the MLX fingerprint as `expect_fingerprint` gets 422; a 256-input batch
succeeds; the Q8_0 file's tensor list shows the two dense layers.

**L5. Runtime installer and variant choice.** `LlamaRuntime`, pinned hashes,
CUDA runtime pairing, probe and fallback, macOS quarantine and first-run
timeout, `estia runtime install --backend llama`, `estia setup` choosing
`llama-cpp` on Linux.
*Exit check:* on a fresh Ubuntu 24.04 machine without a GPU,
`estia setup --roles fast,embed` installs the CPU build and the models, and
`estia remote-check` passes against `estia serve`; on a machine with an
NVIDIA GPU, the CUDA variant is chosen and `--list-devices` shows a CUDA
device; an archive with one byte changed is refused.

**L6. Cache isolation and more slots.** Slot per `cache_key`,
`cache_prompt: false` on a change of owner, configurable slot count.
*Exit check:* an integration test where token A sends prompt P twice (the
second reports `cached_tokens > 0`), then token B sends P (reports
`cached_tokens: 0`); with two slots, two alternating conversations both keep
getting cache hits.

**L7. CI and release artifacts.** The CI job described above; release archives
for `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` and
`x86_64-apple-darwin`, each with `THIRD_PARTY_LICENSES`.
*Exit check:* CI is green on Linux and macOS; the Linux archive's `estia`
runs `estia --version` in Ubuntu 22.04 and Debian 12 containers.

**L8. Benchmarks and defaults.** BENCH.md rows for llama.cpp on the same
Apple Silicon Mac as the MLX rows, an x86 CPU, and an NVIDIA GPU; quality
spot checks of the QAT Q4_0 files against the alternatives; the Mac default
decided.
*Exit check:* the rows exist with dates, machines, builds and commands, and
the decision is recorded in the README.

**L9. Images and audio** (after a protocol change). Content parts in the
protocol, `--mmproj` on load, `data:` URLs only.
*Exit check:* `/v1/chat/completions` with a base64 image on
`gemma4-e4b-it-qat-q4_0-gguf` describes the image; a remote image URL gets
400.

## Other findings

These are outside this design but turned up while checking facts.

- **Gemma 4 licence.** The registry (`engine/src/models/registry.rs`) and the
  README list the three Gemma 4 artifacts as "Gemma Terms of Use". Google's
  Gemma 4 licence page and the `google/gemma-4-*` model cards say Apache 2.0.
  The mlx-community cards say `gemma`. Worth checking and correcting.
  EmbeddingGemma is under the Gemma Terms of Use (`google/embeddinggemma-300m`
  is gated and tagged `license: gemma`).
- **`format` never reaches the runner.** See [Where Estia is today](#where-estia-is-today).
  L3 changes that for runners that declare `structured`.

## Sources

llama.cpp:

- Releases: <https://github.com/ggml-org/llama.cpp/releases/tag/b11193>,
  <https://github.com/ggml-org/llama.cpp/releases/tag/v0.5.0>, and the API
  listing at `https://api.github.com/repos/ggml-org/llama.cpp/releases`
- Release workflow: <https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/release.yml>
- Licence: <https://github.com/ggml-org/llama.cpp/blob/master/LICENSE>
- Server documentation: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Server source: <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>
  (`server-http.cpp`, `server-queue.cpp`, `server-models.cpp`,
  `server-common.cpp`, `server-schema.cpp`, `server-task.cpp`)
- Gemma 4 chat format detection: <https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp>
- Multimodal models: <https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md>
- Converter (`--sentence-transformers-dense-modules`): <https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py>

Other runtimes:

- Ollama's `llama-server` wrapper: <https://github.com/ollama/ollama/blob/main/llm/llama_server.go>
- LM Studio runtimes: <https://lmstudio.ai/docs/cli/runtime/runtime>,
  <https://lmstudio.ai/docs/app>, <https://github.com/BootBlock/Deguffer/issues/64>
- `llama-cpp-2`: <https://crates.io/crates/llama-cpp-2>,
  <https://github.com/utilityai/llama-cpp-rs>, <https://docs.rs/llama-cpp-2/latest/llama_cpp_2/>
- `llama-cpp-python`: <https://pypi.org/project/llama-cpp-python/>,
  <https://github.com/abetlen/llama-cpp-python/releases>
- macOS first-run delay: <https://github.com/openclaw/openclaw/issues/138672>
- ROCm device permissions: <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html>

Models:

- Gemma 4 licence: <https://ai.google.dev/gemma/docs/gemma_4_license>
- <https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf>,
  <https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf>,
  <https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf>
- <https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF>,
  <https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF>,
  <https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF>,
  <https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF>,
  <https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF>
- <https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF>,
  <https://huggingface.co/ggml-org/embeddinggemma-300M-qat-q4_0-GGUF>,
  <https://huggingface.co/google/embeddinggemma-300m>,
  <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF>
- Test models: <https://huggingface.co/ggml-org/tinygemma3-GGUF>,
  <https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF>
- Sizes, revisions, hashes, licences and parameter counts came from the
  Hugging Face API (`/api/models/<repo>`, `/tree/main`, `?expand[]=gguf`).

Benchmarks:

- <https://github.com/ryanssenn/gemma4.c> (CPU, E2B)
- <https://alfonsofortunato.com/blog/gemma-4-e4b-vs-26b-local-benchmarks/> (RTX 4070 Ti, E4B)
- <https://www.birjob.com/blog/gemma-4-apple-silicon-mlx-vs-llama-cpp> (Apple Silicon, E4B, MLX and llama.cpp)
