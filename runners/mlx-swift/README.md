# Swift MLX runner (`estia-mlx-swift`)

A compiled Swift binary that runs MLX models on Apple silicon. It is for a
host that must ship only signed code and cannot download a Python interpreter
at runtime, such as a sandboxed Mac app. The Python path
(`../mlx-python/`) needs a downloaded interpreter and pip packages; this
runner is compiled once and signed into the host's bundle instead. Model
weights are still downloaded at runtime. They are data, not code.

## Protocol

One-shot: the runner reads one JSON request from stdin until EOF, writes one
JSON object to stdout, and exits. It speaks the same requests as
`../mlx-python/oneshot-runner.py`, minus streaming:

| Request | Response |
|---|---|
| `{"type":"health"}` | `{"ok":true,"version":"mlx-swift","mlx_available":true,"detail":"…"}` |
| `{"type":"generate","model_path":…,"prompt":…,"max_tokens":512,"temperature":0.2}` | `{"ok":true,"text":"…"}` |
| `{"type":"embed","model_path":…,"input":…}` | `{"ok":true,"embedding":[…]}` |
| `{"type":"apple_health"}` | same shape as `health`; `mlx_available` means "the Apple model is usable" |
| `{"type":"apple_generate","prompt":…,"temperature":0.2}` | `{"ok":true,"text":"…"}` |

Any failure is `{"ok":false,"error":"…"}`. `max_tokens` defaults to 512 and
`temperature` to 0.0 (0.2 for `apple_generate`).

It does not speak protocol v2 (`hello`, `load`, `chat`, `cancel`, …), does
not stream, and loads the model from disk on every call. The Apple calls need
macOS 26 with Apple Intelligence available; elsewhere `apple_health` reports
the model as unavailable.

## Build

Requirements: Apple silicon, macOS 14 or later, full Xcode (the Command Line
Tools are not enough), and the Metal Toolchain component:

```bash
xcodebuild -downloadComponent MetalToolchain
```

`swift build` compiles the binary but cannot compile mlx-swift's Metal
shaders. The result answers `health` and then fails every `generate` and
`embed` with "Failed to load the default metallib". Use it only as a compile
check:

```bash
cd runners/mlx-swift
swift build -c release
```

The runnable build goes through `xcodebuild`:

```bash
cd runners/mlx-swift
xcodebuild build -scheme estia-mlx-swift -configuration Release \
  -destination 'generic/platform=macOS' -derivedDataPath .xcode-build \
  -skipMacroValidation
```

`-skipMacroValidation` is needed because xcodebuild refuses the
`MLXHuggingFace` package macro unless it has been enabled by hand. The output
is in `.xcode-build/Build/Products/Release/`: the `estia-mlx-swift` binary and
several `*.bundle` directories. `mlx-swift_Cmlx.bundle` holds
`default.metallib`. The binary needs these bundles in the same directory.

Smoke test:

```bash
cd .xcode-build/Build/Products/Release
echo '{"type":"health"}' | ./estia-mlx-swift
echo '{"type":"generate","model_path":"/path/to/gemma4-e4b-it-4bit-mlx","prompt":"Say hello."}' | ./estia-mlx-swift
```

`.build/`, `.xcode-build/` and a staged copy of the binary or bundles in this
directory are git-ignored.

## Using it from a host

1. Copy `estia-mlx-swift` and every `*.bundle` from the products directory
   into one directory inside the host's bundle.
2. Sign the binary with the host's identity and entitlements, like any other
   nested executable.
3. Build `estia-engine` without the `python-mlx` feature (it is off by
   default), so the Python runtime installer is not compiled in.
4. Drive the runner with the engine's one-shot client:
   `OneShot::new(Launch::new(path_to_binary), OneShotConfig::default(), observer)`,
   then `call` with `estia_proto::Request::Health`, `Generate`, `Embed`,
   `AppleHealth` or `AppleGenerate`. See `engine/src/oneshot.rs`.

Models: the loader picks the model factory from the `model_type` in the
model's `config.json`, so Gemma 4 bundles (`gemma4`) load through the VLM
factory. For embeddings, `MLXEmbedders` loads EmbeddingGemma (the default
embedding model) but not ModernBERT (see `loads_in_mlx_swift` in
`engine/src/models/embed.rs`).

The same embedding model gives different vectors in this runner and in the
Python runner, so an embedding fingerprint (`EmbedModel::fingerprint_for`)
includes the backend, `mlx-swift` or `mlx-python`. Do not compare vectors
across the two.

## Status

- Builds against mlx-swift-lm 3.31.x. `Package.resolved` pins 3.31.4 and
  swift-transformers 1.3.3.
- 3.31.4 is a floor: on 3.31.3, loading EmbeddingGemma crashes the process
  (see the comment in `Package.swift`).
- Run on-device: `health`, Gemma 4 model loading and generation from an
  `xcodebuild` build, and EmbeddingGemma embedding on 3.31.4.
- Generation formats the prompt with Gemma 4's turn tokens by hand instead of
  the processor's chat template. The comment in `runGenerate` explains why.
- Not implemented: protocol v2, streaming, cancel, and keeping a model loaded
  between calls.
