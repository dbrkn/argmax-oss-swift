//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgumentParser
import CoreML
import Foundation
import TTSKit
import TTSKitMLX

/// Head-to-head encoder benchmark: TTSKit's CoreML voice-clone encoders vs the
/// MLX encoders in this package, on the same reference audio at several
/// durations. Complements the measured production-impact table in
/// `docs/voice-clone-design.md` with numbers from the actual Swift code paths.
struct BenchCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Benchmark CoreML vs MLX voice-clone encode latency"
    )

    @Option(help: "Reference audio (headerless float32 @24kHz mono with --raw-float32, else any readable file)")
    var refAudio: String

    @Flag(help: "Treat --ref-audio as headerless little-endian float32 samples at 24 kHz mono")
    var rawFloat32: Bool = false

    @Option(help: "Model folder with qwen3_tts/<component>/<version-dir>/<variant> layout for the CoreML encoders")
    var coremlModelsDir: String

    @Option(help: "Version directory for the CoreML encoders")
    var versionDir: String = "12hz-0.6b-base"

    @Option(help: "CoreML encoder variant (window)")
    var coremlVariant: String = "W16A16-10s"

    @Option(help: "Qwen3-TTS Base-family MLX checkpoint snapshot directory (default: cached HF snapshot)")
    var modelDir: String?

    @Option(help: "Comma-separated clip durations in seconds to benchmark")
    var durations: String = "3,8,15,32"

    @Option(help: "Timed iterations per duration (best-of reported)")
    var iterations: Int = 3

    mutating func run() async throws {
        let audioURL = URL(fileURLWithPath: refAudio)
        let base: [Float]
        if rawFloat32 {
            base = try Data(contentsOf: audioURL).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        } else {
            base = try AudioInput.loadMono(url: audioURL, sampleRate: 24000)
        }
        let secondsList = durations.split(separator: ",").compactMap { Double($0) }
        let clips: [(name: String, samples: [Float])] = secondsList.map { seconds in
            let target = Int(seconds * 24000)
            var clip = [Float]()
            while clip.count < target { clip.append(contentsOf: base.prefix(target - clip.count)) }
            return (String(format: "%.0fs", seconds), clip)
        }

        // --- CoreML encoders (TTSKit production path) ---
        let root = URL(fileURLWithPath: coremlModelsDir)
        func assetURL(_ component: String, _ name: String) -> URL {
            root.appending(path: "qwen3_tts/\(component)/\(versionDir)/\(coremlVariant)/\(name).mlmodelc")
        }
        let speaker = Qwen3SpeakerEncoder()
        let speech = Qwen3SpeechEncoder()
        let rvq = Qwen3SpeechEncoderRVQ()
        var loadStart = Date()
        try await speaker.loadModel(at: assetURL("speaker_encoder", "SpeakerEncoder"), computeUnits: .cpuAndNeuralEngine)
        try await speech.loadModel(at: assetURL("speech_encoder", "SpeechEncoder"), computeUnits: .cpuAndNeuralEngine)
        try await rvq.loadModel(at: assetURL("speech_encoder_rvq", "SpeechEncoderRVQ"), computeUnits: .cpuAndNeuralEngine)
        let coreml = VoiceCloneEncoder(speakerEncoder: speaker, speechEncoder: speech, rvqEncoder: rvq)
        let coremlLoad = Date().timeIntervalSince(loadStart)
        _ = try await coreml.encode(clips[0].samples, includeReferenceCodes: true)  // warmup

        // --- MLX encoders (this package) ---
        loadStart = Date()
        let mlx = try MlxVoiceCloneEncoder(
            modelDirectory: modelDir.map { URL(fileURLWithPath: $0) },
            maxReferenceSeconds: (secondsList.max() ?? 120) + 1
        )
        let mlxLoad = Date().timeIntervalSince(loadStart)
        _ = try mlx.encode(clips[0].samples, includeReferenceCodes: true, referenceText: nil)  // warmup

        print(String(format: "load: coreml %.2fs | mlx %.2fs", coremlLoad, mlxLoad))
        print("clip | coreml (ANE, fixed \(coremlVariant) window) | mlx (GPU, variable)")
        let windowSamples = speech.audioLength
        for clip in clips {
            var coremlBest = Double.infinity
            for _ in 0..<iterations {
                let t0 = Date()
                _ = try await coreml.encode(clip.samples, includeReferenceCodes: true)
                coremlBest = min(coremlBest, Date().timeIntervalSince(t0))
            }
            var mlxBest = Double.infinity
            for _ in 0..<iterations {
                let t0 = Date()
                _ = try mlx.encode(clip.samples, includeReferenceCodes: true, referenceText: nil)
                mlxBest = min(mlxBest, Date().timeIntervalSince(t0))
            }
            let truncated = clip.samples.count > windowSamples ? "  [coreml TRUNCATES]" : ""
            print(String(format: "%4@ | %7.1f ms | %7.1f ms%@", clip.name, coremlBest * 1000, mlxBest * 1000, truncated))
        }
    }
}
