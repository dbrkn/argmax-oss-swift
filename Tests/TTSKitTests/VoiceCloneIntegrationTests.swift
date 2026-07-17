//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
@testable import TTSKit
import XCTest

/// End-to-end ICL voice-clone test against real CoreML assets.
///
/// Gated on `TTSKIT_VC_MODELS_DIR` pointing at a model folder with the
/// `qwen3_tts/<component>/12hz-0.6b-base/<variant>/` layout including the
/// three voice-clone encoder assets (e.g. a local `argmaxinc/ttskit-internal`
/// snapshot). Skipped otherwise, so CI without assets stays green.
///
/// The reference clip and texts come from the bundled Python goldens, so the
/// generated clone is directly comparable to the Python `tts-cli` output for
/// the same sample.
final class VoiceCloneIntegrationTests: XCTestCase {
    /// Variant names as exported in the internal research asset repo. These
    /// differ from the public `ttskit-coreml` conventions (single-function
    /// SpeechDecoder, non-stateful CodeDecoder).
    static let internalVariants = (
        codeDecoder: "W8A16-kv_len_256",
        multiCodeDecoder: "W8A16-kv_len_16",
        codeEmbedder: "W16A16",
        multiCodeEmbedder: "W16A16",
        textProjector: "W16A16",
        speechDecoder: "W8A16-kv_len_256-context_1-n_codes_4-cat"
    )

    func testEndToEndICLVoiceClone() async throws {
        guard let modelsDir = ProcessInfo.processInfo.environment["TTSKIT_VC_MODELS_DIR"] else {
            throw XCTSkip("Set TTSKIT_VC_MODELS_DIR to a 12hz-0.6b-base model folder to run")
        }

        // Reference audio + texts from the bundled goldens.
        guard
            let goldensURL = Bundle.module.url(
                forResource: "goldens", withExtension: "json",
                subdirectory: "Resources/VoiceCloneGoldens"
            ),
            let refAudioURL = Bundle.module.url(
                forResource: "ref_audio", withExtension: "bin",
                subdirectory: "Resources/VoiceCloneGoldens"
            )
        else {
            throw XCTSkip("Voice-clone goldens not bundled")
        }
        struct Goldens: Decodable {
            let referenceText: String
            let synthesisText: String
        }
        let goldens = try JSONDecoder().decode(Goldens.self, from: Data(contentsOf: goldensURL))
        let refWaveform: [Float] = try Data(contentsOf: refAudioURL).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }

        let config = TTSKitConfig(
            model: .qwen3TTS_0_6b_base,
            modelFolder: URL(fileURLWithPath: modelsDir),
            codeDecoderVariant: Self.internalVariants.codeDecoder,
            multiCodeDecoderVariant: Self.internalVariants.multiCodeDecoder,
            codeEmbedderVariant: Self.internalVariants.codeEmbedder,
            multiCodeEmbedderVariant: Self.internalVariants.multiCodeEmbedder,
            textProjectorVariant: Self.internalVariants.textProjector,
            speechDecoderVariant: Self.internalVariants.speechDecoder,
            speechDecoderMode: .singleFunction,
            verbose: true,
            logLevel: .debug,
            seed: 42
        )
        let tts = try await TTSKit(config)
        try await tts.loadModels()
        try await tts.loadVoiceCloneModels()

        guard let encoder = tts.voiceCloneEncoder else {
            return XCTFail("voiceCloneEncoder not available after loadVoiceCloneModels")
        }

        // Encode the reference and check shape expectations from the goldens.
        let encoded = try await encoder.encode(refWaveform, includeReferenceCodes: true)
        XCTAssertEqual(encoded.speakerEmbedding.count, 1024)
        XCTAssertEqual(encoded.referenceCodeFrames, 101, "valid-frame trim mismatch vs goldens")

        let prompt = VoiceClonePrompt(
            speakerEmbedding: encoded.speakerEmbedding,
            referenceCodes: encoded.referenceCodes,
            referenceCodeFrames: encoded.referenceCodeFrames,
            referenceText: goldens.referenceText
        )

        var options = GenerationOptions()
        options.voiceClone = prompt
        options.chunkingStrategy = TextChunkingStrategy.none

        let result = try await tts.generate(
            text: goldens.synthesisText,
            options: options
        )

        let duration = Double(result.audio.count) / Double(result.sampleRate)
        let rms = sqrt(result.audio.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(result.audio.count, 1)))
        Logging.info(String(format: "Voice clone: %.2fs of audio, RMS %.4f", duration, rms))

        XCTAssertGreaterThan(duration, 2.0, "Suspiciously short clone")
        XCTAssertLessThan(duration, 30.0, "Suspiciously long clone (runaway generation)")
        XCTAssertGreaterThan(rms, 0.005, "Clone is near-silent")

        // Persist for listening-based A/B against the Python tts-cli output.
        let outURL = try await AudioOutput.saveAudio(
            result.audio,
            toFolder: FileManager.default.temporaryDirectory,
            filename: "ttskit_voice_clone_test",
            sampleRate: result.sampleRate,
            format: .wav
        )
        print("Voice-clone test audio written to \(outURL.path)")
    }
}
