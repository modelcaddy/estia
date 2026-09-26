// swift-tools-version: 5.9
import PackageDescription

// Compiled MLX runner, for hosts that must ship only signed code.
//
// Produces a single executable `estia-mlx-swift` that speaks the same one-shot
// JSON-over-stdin/stdout protocol as ../mlx-python/oneshot-runner.py, so the
// engine's `OneShot` client drives it with no protocol changes. Unlike the
// Python path, no interpreter or package is downloaded at runtime: the
// inference code is compiled here and signed by the host along with the rest
// of its bundle. Model weights are still downloaded at runtime (data, not code).
//
// Compile check (does not build the Metal shaders, so the result cannot
// generate or embed):
//   cd runners/mlx-swift
//   swift build -c release
//
// Runnable build (needs full Xcode plus the Metal Toolchain component,
// `xcodebuild -downloadComponent MetalToolchain`):
//   cd runners/mlx-swift
//   xcodebuild build -scheme estia-mlx-swift -configuration Release \
//     -destination 'generic/platform=macOS' -derivedDataPath .xcode-build \
//     -skipMacroValidation
// (-skipMacroValidation is needed because xcodebuild refuses the
// MLXHuggingFace package macro unless it has been enabled by hand.)
//
// The binary and the *.bundle directories beside it in
// .xcode-build/Build/Products/Release/ ship together; see README.md.
let package = Package(
    name: "estia-mlx-swift",
    platforms: [
        // mlx-swift requires macOS 14+ on Apple silicon. A host that supports
        // older macOS versions must not offer this runner there.
        .macOS(.v14)
    ],
    dependencies: [
        // mlx-swift-lm is the active successor to mlx-swift-examples (the
        // reusable LLM/VLM/Embedder libraries were moved here in the 3.x split).
        // Crucially, it adds the `gemma4` / `gemma4_text` loaders that the old
        // package lacks, and Gemma 4 is the generation family in Estia's model
        // registry. Same module names (MLXLLM/MLXVLM/MLXLMCommon/MLXEmbedders),
        // new package identity.
        // 3.31.4 is a floor, not a preference: it carries "Fix EmbeddingGemma
        // init-order crash + dense head hidden size" (#223). On 3.31.3 the default
        // embed model (EmbeddingGemma-300M) does not merely fail to load — it trips
        // a Swift `fatalError` in MLXNN and takes the runner process with it, so
        // this runner could not embed at all. Verified by running the staged
        // runner against the model on both versions.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            from: "3.31.4"
        ),
        // 3.x decouples tokenization behind a `TokenizerLoader` protocol. We use
        // the official Hugging Face integration: MLXHuggingFace provides the
        // `#huggingFaceTokenizerLoader()` macro, whose expansion references the
        // `Tokenizers` module from swift-transformers (imported in Runner.swift).
        // No downloader package needed — weights are already on disk.
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMajor(from: "1.3.0")
        )
    ],
    targets: [
        .executableTarget(
            name: "estia-mlx-swift",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        )
    ]
)
