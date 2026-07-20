//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgumentParser
import Foundation
import TTSKit
import TTSKitMLX

/// Full voice clone with the MLX encoders + MLX talker: the reference clip is
/// encoded by the variable-length MLX encoders, the CodeDecoder is the MLX
/// talker (`MlxCodeDecoder`, batched prefill), and the remaining components
/// (embedders, MultiCodeDecoder, SpeechDecoder) stay on CoreML.
struct TTSCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tts",
        abstract: "Voice-clone TTS with MLX encoders + MLX talker + CoreML rest"
    )

    @Option(name: .long, help: "Text to synthesize")
    var text: String

    @Option(name: .long, help: "Reference audio clip to clone the voice from (any readable audio format)")
    var refAudio: String

    @Option(name: .long, help: "Transcript of the reference clip (required unless --x-vector-only)")
    var refText: String?

    @Flag(name: .long, help: "Clone with the speaker x-vector only, skipping reference RVQ encoding")
    var xVectorOnly: Bool = false

    @Flag(name: .long, help: "Treat --ref-audio as headerless little-endian float32 samples at 24 kHz mono")
    var rawFloat32: Bool = false

    @Option(name: .long, help: "Output audio file path (wav)")
    var output: String = "mlx_voice_clone.wav"

    // MARK: Models

    @Option(name: .long, help: "Model folder with the qwen3_tts/<component>/<version-dir>/<variant> layout for the CoreML components")
    var coremlModelsDir: String

    @Option(name: .long, help: "Version directory for the CoreML components")
    var versionDir: String = "12hz-0.6b-base"

    @Option(name: .long, help: "CoreML CodeDecoder variant. The MLX talker replaces it at runtime, but TTSKit still resolves the asset path")
    var codeDecoderVariant: String = "W8A16-kv_len_256"

    @Option(name: .long, help: "CoreML MultiCodeDecoder variant")
    var multiCodeDecoderVariant: String = "W8A16-kv_len_16"

    @Option(name: .long, help: "CoreML CodeEmbedder variant")
    var codeEmbedderVariant: String = "W16A16"

    @Option(name: .long, help: "CoreML MultiCodeEmbedder variant")
    var multiCodeEmbedderVariant: String = "W16A16"

    @Option(name: .long, help: "CoreML TextProjector variant")
    var textProjectorVariant: String = "W16A16"

    @Option(name: .long, help: "CoreML SpeechDecoder variant")
    var speechDecoderVariant: String = "W8A16-kv_len_256-context_1-n_codes_4-cat"

    @Option(name: .long, help: "Qwen3-TTS MLX checkpoint snapshot directory for the encoders + talker (default: cached HF snapshot of \(ModelDirectory.defaultRepoID))")
    var modelDir: String?

    @Option(name: .long, help: "MLX talker KV budget in positions (prompt + generated frames)")
    var maxSequenceLength: Int = 1024

    @Option(name: .long, help: "Reference-length cap in seconds for the MLX encoders")
    var maxReferenceSeconds: Double = 120

    // MARK: Sampling

    @Option(name: .long, help: "Sampling temperature (0.0 for greedy)")
    var temperature: Float = GenerationOptions.defaultTemperature

    @Option(name: .long, help: "Top-k sampling (0 to disable)")
    var topK: Int = GenerationOptions.defaultTopK

    @Option(name: .long, help: "Max RVQ frames to generate")
    var maxNewTokens: Int = GenerationOptions.defaultMaxNewTokens

    @Option(name: .long, help: "Random seed for reproducible output")
    var seed: UInt64?

    @Flag(name: .long, help: "Enable verbose output")
    var verbose: Bool = false

    mutating func run() async throws {
        if !xVectorOnly, refText == nil {
            throw ValidationError("ICL voice cloning requires --ref-text (or pass --x-vector-only).")
        }

        // 1. Encode the reference with the variable-length MLX encoders.
        let audioURL = URL(fileURLWithPath: refAudio)
        let waveform: [Float]
        if rawFloat32 {
            waveform = try Data(contentsOf: audioURL).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        } else {
            waveform = try AudioInput.loadMono(url: audioURL, sampleRate: Double(MlxVoiceCloneEncoder.sampleRate))
        }
        print(String(
            format: "Reference: %@ (%.2f s)", audioURL.lastPathComponent,
            Double(waveform.count) / Double(MlxVoiceCloneEncoder.sampleRate)
        ))

        let mlxDirectory = modelDir.map { URL(fileURLWithPath: $0) }
        let encodeStart = Date()
        let encoder = try MlxVoiceCloneEncoder(
            modelDirectory: mlxDirectory, maxReferenceSeconds: maxReferenceSeconds
        )
        let prompt = try encoder.encode(waveform, includeReferenceCodes: !xVectorOnly, referenceText: refText)
        print(String(format: "MLX reference encode: %.2f s (codes: %d frames)",
                     Date().timeIntervalSince(encodeStart), prompt.referenceCodeFrames))

        // 2. TTSKit with the MLX talker swapped in for the CoreML CodeDecoder.
        let config = TTSKitConfig(
            model: .qwen3TTS_0_6b_base,
            modelFolder: URL(fileURLWithPath: coremlModelsDir),
            versionDir: versionDir,
            codeDecoderVariant: codeDecoderVariant,
            multiCodeDecoderVariant: multiCodeDecoderVariant,
            codeEmbedderVariant: codeEmbedderVariant,
            multiCodeEmbedderVariant: multiCodeEmbedderVariant,
            textProjectorVariant: textProjectorVariant,
            speechDecoderVariant: speechDecoderVariant,
            speechDecoderMode: .singleFunction,
            verbose: verbose,
            logLevel: verbose ? .debug : .info,
            seed: seed
        )
        config.codeDecoder = try makeMlxCodeDecoder(
            modelDirectory: mlxDirectory, maxSequenceLength: maxSequenceLength
        )

        let loadStart = Date()
        let tts = try await TTSKit(config)
        print(String(format: "Models loaded: %.2f s", Date().timeIntervalSince(loadStart)))

        // 3. Generate.
        var options = GenerationOptions(
            temperature: temperature, topK: topK, maxNewTokens: maxNewTokens
        )
        options.voiceClone = prompt
        options.chunkingStrategy = TextChunkingStrategy.none

        let result = try await tts.generate(text: text, options: options)

        // 4. Report + write.
        let timings = result.timings
        let prefillTokS = timings.prefill > 0 ? timings.prefillTokens / timings.prefill : 0
        let steps = Int(timings.totalDecodingLoops)
        let msPerStep = steps > 0 ? timings.decodingLoop * 1000 / Double(steps) : 0
        let duration = Double(result.audio.count) / Double(result.sampleRate)
        let rms = (result.audio.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(result.audio.count, 1))).squareRoot()

        print(String(format: "Prefill: %.0f tokens in %.0f ms (%.1f tok/s)",
                     timings.prefillTokens, timings.prefill * 1000, prefillTokS))
        print(String(format: "Decode: %d steps, %.1f ms/step (%.1f steps/s)",
                     steps, msPerStep, msPerStep > 0 ? 1000 / msPerStep : 0))
        print(String(format: "Audio: %.2f s, RMS %.4f (full pipeline %.2f s)",
                     duration, rms, timings.fullPipeline))

        let outputURL = URL(fileURLWithPath: output)
        let saved = try await AudioOutput.saveAudio(
            result.audio,
            toFolder: outputURL.deletingLastPathComponent(),
            filename: outputURL.deletingPathExtension().lastPathComponent,
            sampleRate: result.sampleRate,
            format: .wav
        )
        print("Wrote \(saved.path)")
    }
}
