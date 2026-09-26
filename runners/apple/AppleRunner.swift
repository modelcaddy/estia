// Estia Apple Foundation Models runner (one-shot).
//
// A FoundationModels-only sibling of the Swift MLX runner
// (../mlx-swift/Sources/estia-mlx-swift/Runner.swift). The Python MLX runners
// cannot call Apple Foundation Models, so a host that runs MLX through Python
// but also wants the Apple on-device model uses this binary. It implements
// just the `apple_*` half of the one-shot JSON protocol and compiles with
// plain `swiftc` from the Command Line Tools (no Xcode, Metal toolchain or
// SwiftPM dependencies).
//
// Build (from runners/apple/):
//   swiftc -O -parse-as-library AppleRunner.swift -o estia-apple-runner
//   echo '{"type":"apple_health"}' | ./estia-apple-runner
//
// Calling Foundation Models needs the macOS 26 SDK at build time and macOS 26
// with Apple Intelligence available at run time. Built with an older SDK, or
// run on an older macOS, it still works as a process: apple_health reports
// the model as unavailable and apple_generate fails with an error.
//
//   Request  (one JSON object on stdin):
//     {"type":"apple_health"}
//     {"type":"apple_generate","prompt":"...","temperature":0.2}
//
//   Response (one JSON object on stdout):
//     apple_health  : {"ok":true,"mlx_available":true,"version":"apple-foundation","detail":"..."}
//     apple_generate: {"ok":true,"text":"..."} | {"ok":false,"error":"..."}
//
// The `apple_health` / `apple_generate` implementations are kept byte-
// compatible with the full runner's — if you change one, change both.

import Foundation

// Present only in the macOS 26+ SDK; guarded so this still builds on older SDKs
// (where apple_health then reports unavailable instead of failing to compile).
#if canImport(FoundationModels)
import FoundationModels
#endif

// MARK: - Protocol types

struct RunnerRequest: Decodable {
    let type: String
    let prompt: String?
    let temperature: Float?
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
struct AppleRunner {
    static func main() async {
        let raw = FileHandle.standardInput.readDataToEndOfFile()
        guard !raw.isEmpty else { fail("empty request"); return }
        guard let req = try? JSONDecoder().decode(RunnerRequest.self, from: raw) else {
            fail("invalid JSON request"); return
        }

        switch req.type {
        case "apple_health":
            appleHealth()
        case "apple_generate":
            await appleGenerate(req)
        default:
            fail("unknown request type: \(req.type) (the lite Apple runner only handles apple_*)")
        }
    }
}

// MARK: - Apple Foundation Models

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
/// envelope as the MLX `generate`, so the Rust client path is identical.
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
