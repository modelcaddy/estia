// Estia compiled MLX runner (Swift, one-shot).
//
// For hosts that must ship only signed code and cannot download a Python
// interpreter. Speaks the one-shot JSON protocol of the Python runner
// (runners/mlx-python/oneshot-runner.py) for health, generate and embed (no
// streaming), so the engine's `OneShot` client (engine/src/oneshot.rs) drives
// both the same way. It also answers apple_health / apple_generate (Apple
// Foundation Models; see the end of this file).
//
//   Request  (one JSON object on stdin):
//     {"type":"health"}
//     {"type":"generate","model_path":"...","prompt":"...","max_tokens":512,"temperature":0.2}
//     {"type":"embed","model_path":"...","input":"..."}
//
//   Response (one JSON object on stdout):
//     health  : {"ok":true,"version":"...","mlx_available":true,"detail":"..."}
//     generate: {"ok":true,"text":"..."}            | {"ok":false,"error":"..."}
//     embed   : {"ok":true,"embedding":[...]}        | {"ok":false,"error":"..."}
//
// ─────────────────────────────────────────────────────────────────────────────
// STATUS: compiles against mlx-swift-lm 3.31.x (see Package.resolved).
// `health` and model loading (Gemma 4, model_type "gemma4") are verified on-
// device. Live token generation additionally requires the Xcode **Metal
// Toolchain** component (so the build can produce mlx-swift's default.metallib):
//   xcodebuild -downloadComponent MetalToolchain
// and a build through `xcodebuild`, since `swift build` cannot compile the
// Metal shaders (see Package.swift). Without the metallib, generate/embed fail
// at GPU init with "Failed to load the default metallib". This is a build
// requirement, not a code issue.
// ─────────────────────────────────────────────────────────────────────────────

import Foundation
import MLX
import MLXLMCommon
import MLXLLM      // registers text model types (incl. gemma4_text) with the factory
import MLXVLM      // registers multimodal model types (incl. gemma4) with the factory
import MLXEmbedders
import MLXHuggingFace  // #huggingFaceTokenizerLoader() macro (3.x tokenizer integration)
import Tokenizers      // swift-transformers; referenced by the macro's expansion

// Apple on-device foundation model (no model download needed). Present only in
// the macOS 26+ SDK; guarded so the runner still builds/runs on older systems.
#if canImport(FoundationModels)
import FoundationModels
#endif

// MARK: - Protocol types

struct RunnerRequest: Decodable {
    let type: String
    let model_path: String?
    let prompt: String?
    let max_tokens: Int?
    let temperature: Float?
    let input: String?
}

/// Emit one JSON object and flush. Matches the Python runner's single-write
/// behaviour so the Rust side reads exactly one envelope.
func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: []) else {
        FileHandle.standardOutput.write(Data(#"{"ok":false,"error":"failed to serialize response"}"#.utf8))
        return
    }
    FileHandle.standardOutput.write(data)
}

func fail(_ message: String) {
    emit(["ok": false, "error": message])
}

// MARK: - Entry point

@main
struct Runner {
    static func main() async {
        let raw = FileHandle.standardInput.readDataToEndOfFile()
        guard !raw.isEmpty else { fail("empty request"); return }
        guard let req = try? JSONDecoder().decode(RunnerRequest.self, from: raw) else {
            fail("invalid JSON request"); return
        }

        do {
            switch req.type {
            case "health":
                health()
            case "generate":
                try await generate(req)
            case "embed":
                try await embed(req)
            case "apple_health":
                appleHealth()
            case "apple_generate":
                await appleGenerate(req)
            default:
                fail("unknown request type: \(req.type)")
            }
        } catch {
            fail("\(error)")
        }
    }
}

// MARK: - health

/// Cheap, non-crashing readiness probe. We do not touch Metal here (mirrors the
/// Python runner, which avoids importing MLX during health to keep it safe on
/// machines without a usable GPU).
func health() {
    emit([
        "ok": true,
        "version": "mlx-swift",
        "mlx_available": true,
        "detail": "Bundled MLX (mlx-swift) runner.",
    ])
}

// MARK: - generate

func generate(_ req: RunnerRequest) async throws {
    guard let modelPath = req.model_path else { fail("generate: missing model_path"); return }
    guard let prompt = req.prompt else { fail("generate: missing prompt"); return }
    let maxTokens = req.max_tokens ?? 512
    let temperature = req.temperature ?? 0.0

    do {
        let text = try await runGenerate(
            modelDirectory: modelPath,
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature
        )
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.isEmpty {
            fail("MLX generated no text.")
        } else {
            emit(["ok": true, "text": cleaned])
        }
    } catch {
        fail("MLX generation failed: \(error)")
    }
}

/// Generation path (mlx-swift-lm 3.31.3). The top-level `loadModelContainer`
/// auto-routes Gemma 4 (multimodal `model_type: "gemma4"`) to the VLM factory;
/// text-only since no image is supplied. The prompt is formatted by hand
/// below and decoded with `MLXLMCommon.generate`.
func runGenerate(
    modelDirectory: String,
    prompt: String,
    maxTokens: Int,
    temperature: Float
) async throws -> String {
    // The top-level loader auto-routes by the model's config.json `model_type`
    // to the right factory (gemma4 → VLM, etc.), so we don't pick a factory by
    // hand. Loads weights + tokenizer straight from the on-disk directory.
    let url = URL(fileURLWithPath: modelDirectory)
    // `#huggingFaceTokenizerLoader()` (MLXHuggingFace macro) supplies the TokenizerLoader.
    // The top-level loader auto-routes by config.json model_type.
    let container = try await MLXLMCommon.loadModelContainer(
        from: url, using: #huggingFaceTokenizerLoader()
    )

    var parameters = GenerateParameters(temperature: temperature)
    parameters.maxTokens = maxTokens

    // This Gemma 4 MLX bundle uses unusual turn tokens (<|channel>, <turn|>),
    // and the VLM processor's `prepare` always applies a chat template — which
    // threw missingChatTemplate, because the downloader used to drop the repo's
    // standalone `chat_template.jinja`. So we bypass `prepare` entirely: format
    // by hand (the structure the Python runner proved works) and tokenize
    // straight into an LMInput.
    //
    // The downloader now keeps that .jinja, so `prepare` would no longer throw
    // for freshly-downloaded bundles — but bundles installed earlier still lack
    // it, and this hand-rolled path measured identical in quality to the real
    // template (see the note in oneshot-runner.py). Switching to
    // `prepare` is therefore a deliberate, separately-tested change, not a
    // consequence of the download fix. Left as-is on purpose.
    let formatted = "<|channel>user\n\(prompt)<turn|><|channel>model\n"

    let result = try await container.perform { (context: ModelContext) -> GenerateResult in
        let tokens = context.tokenizer.encode(text: formatted, addSpecialTokens: true)
        // Model expects a batched [1, seqLen] shape; a flat [seqLen] crashes MLX.
        let lmInput = LMInput(tokens: MLXArray(tokens).reshaped([1, tokens.count]))
        return try MLXLMCommon.generate(
            input: lmInput,
            parameters: parameters,
            context: context,
            didGenerate: { _ in .more }
        )
    }

    // Clean up this bundle's channel/turn markup. The model may emit hidden
    // reasoning channels and a "<|channel>final\n…" wrapper before the answer,
    // and a "<turn|>" / "<eos>" at the end.
    var text = result.output
    for stop in ["<turn|>", "<|turn>", "<eos>"] {
        if let r = text.range(of: stop) { text = String(text[..<r.lowerBound]) }
    }
    // Keep only what follows the LAST channel role marker (drops hidden
    // thought/analysis channels; keeps the final answer).
    if let r = text.range(of: "<|channel>", options: .backwards) {
        var tail = String(text[r.upperBound...])
        // Drop a leading role word + newline, e.g. "final\n" / "model\n".
        if let nl = tail.firstIndex(of: "\n") { tail = String(tail[tail.index(after: nl)...]) }
        text = tail
    }
    return text.trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - embed

func embed(_ req: RunnerRequest) async throws {
    guard let modelPath = req.model_path else { fail("embed: missing model_path"); return }
    guard let text = req.input else { fail("embed: missing input"); return }

    do {
        let vector = try await runEmbed(modelDirectory: modelPath, text: text)
        emit(["ok": true, "embedding": vector])
    } catch {
        fail("MLX embedding failed: \(error)")
    }
}

/// Embedding path (mlx-swift-lm 3.31.3 — mirrors MLXEmbedders/README.md,
/// reduced to a single input).
func runEmbed(modelDirectory: String, text: String) async throws -> [Float] {
    let url = URL(fileURLWithPath: modelDirectory)
    let container = try await EmbedderModelFactory.shared.loadContainer(
        from: url, using: #huggingFaceTokenizerLoader()
    )

    let vector: [Float] = await container.perform { (ctx: EmbedderModelContext) -> [Float] in
        let tokens = ctx.tokenizer.encode(text: text, addSpecialTokens: true)
        // Batch of one: shape [1, seqLen].
        let input = stacked([MLXArray(tokens)])
        let mask = (input .!= (ctx.tokenizer.eosTokenId ?? 0))
        let tokenTypes = MLXArray.zeros(like: input)
        // `.eval()` returns Void in this version (it forces compute in place),
        // so don't chain it — `asArray` below forces evaluation anyway.
        let pooled = ctx.pooling(
            ctx.model(input, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
            normalize: true, applyLayerNorm: true
        )
        // pooled is [1, hidden]; take the single row.
        return pooled.map { $0.asArray(Float.self) }[0]
    }
    return vector
}

// MARK: - Apple Foundation Models (macOS 26+)

/// Health probe for the Apple on-device model. Reuses the `health` envelope
/// (`estia_proto::HealthResp`; `mlx_available` doubles as "is this backend
/// usable"), so no new wire type is needed. Cheap and non-throwing.
func appleHealth() {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        switch SystemLanguageModel.default.availability {
        case .available:
            emit([
                "ok": true, "mlx_available": true, "version": "apple-foundation",
                "detail": "Apple on-device model ready.",
            ])
        case .unavailable(let reason):
            emit([
                "ok": true, "mlx_available": false, "version": "apple-foundation",
                "detail": "Apple Intelligence unavailable (\(reason)).",
            ])
        @unknown default:
            emit([
                "ok": true, "mlx_available": false, "version": "apple-foundation",
                "detail": "Apple Intelligence unavailable.",
            ])
        }
        return
    }
    #endif
    emit([
        "ok": true, "mlx_available": false, "version": "apple-foundation",
        "detail": "Apple on-device model requires macOS 26 or later.",
    ])
}

/// Generate with the Apple on-device model. Same `{ok, text}` / `{ok, error}`
/// envelope as `generate`, so the Rust client path is identical.
func appleGenerate(_ req: RunnerRequest) async {
    guard let prompt = req.prompt else { fail("apple_generate: missing prompt"); return }
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let session = LanguageModelSession()
        do {
            let options = GenerationOptions(temperature: Double(req.temperature ?? 0.2))
            let response = try await session.respond(to: prompt, options: options)
            let text = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty {
                fail("Apple model returned no text.")
            } else {
                emit(["ok": true, "text": text])
            }
        } catch {
            fail("Apple generation failed: \(error)")
        }
        return
    }
    #endif
    fail("Apple on-device model requires macOS 26 or later.")
}
