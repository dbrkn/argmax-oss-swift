//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

#if canImport(MLX)

import Foundation
@testable import TTSKitMLX
import XCTest

/// Parity tests against the Python MLX reference (mlx-audio encoders driven by
/// `argmax_prototypes.pipeline.tts.qwen3_tts.mlx_voice_clone`).
///
/// Goldens are exported by `scripts/export_voice_clone_goldens.py` into
/// `Tests/TTSKitTests/Resources/VoiceCloneGoldens/` (shared with the CoreML
/// voice-clone goldens), referenced via a `#filePath`-relative path. Two
/// reference lengths are covered — 8 s and its 4× concatenation (32 s) — so
/// shape assertions double as proof that the encoders are genuinely
/// variable-length (the CoreML window is 10/15 s).
final class EncoderParityTests: XCTestCase {
    struct EncoderGoldens: Decodable {
        let samples: Int
        let xvectorDim: Int
        let codesShape: [Int]
        let xvectorFirst8: [Float]
        let codesFirstFrame: [Int32]
        let codesLastFrame: [Int32]
    }

    // MARK: - Fixtures

    static let goldensDirectory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // TTSKitMLXTests
        .deletingLastPathComponent() // Tests
        .appendingPathComponent("TTSKitTests/Resources/VoiceCloneGoldens")

    // Serial XCTest execution guards this cache; Swift 6 strict concurrency
    // can't see that, hence the explicit opt-out.
    nonisolated(unsafe) static var encoder: MlxVoiceCloneEncoder?

    private func loadEncoder() throws -> MlxVoiceCloneEncoder {
        if let encoder = Self.encoder { return encoder }
        let encoder: MlxVoiceCloneEncoder
        do {
            encoder = try MlxVoiceCloneEncoder()
        } catch {
            throw XCTSkip("MLX checkpoint snapshot unavailable: \(error.localizedDescription)")
        }
        Self.encoder = encoder
        return encoder
    }

    private func goldenURL(_ name: String) throws -> URL {
        let url = Self.goldensDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Golden \(name) not found (run scripts/export_voice_clone_goldens.py)")
        }
        return url
    }

    private func loadFloats(_ name: String) throws -> [Float] {
        try Data(contentsOf: goldenURL(name)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func loadInts(_ name: String) throws -> [Int32] {
        try Data(contentsOf: goldenURL(name)).withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }

    private func loadGoldens(_ key: String) throws -> EncoderGoldens {
        let data = try Data(contentsOf: goldenURL("mlx_encoder_goldens.json"))
        let all = try JSONDecoder().decode([String: EncoderGoldens].self, from: data)
        return try XCTUnwrap(all[key], "\(key) missing from mlx_encoder_goldens.json")
    }

    private static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
        precondition(a.count == b.count)
        var dot = 0.0, normA = 0.0, normB = 0.0
        for (x, y) in zip(a, b) {
            dot += Double(x) * Double(y)
            normA += Double(x) * Double(x)
            normB += Double(y) * Double(y)
        }
        return dot / (normA.squareRoot() * normB.squareRoot())
    }

    // MARK: - Parity assertions

    private func assertParity(goldensKey: String, waveformFile: String, tag: String) throws {
        let encoder = try loadEncoder()
        let goldens = try loadGoldens(goldensKey)
        let waveform = try loadFloats(waveformFile)
        XCTAssertEqual(waveform.count, goldens.samples, "\(tag): reference waveform length")

        // x-vector: fp accumulation-order differences are allowed, direction
        // must match.
        let xvector = encoder.encodeSpeaker(waveform)
        let goldenXvector = try loadFloats("\(goldensKey)_xvector.bin")
        XCTAssertEqual(xvector.count, goldens.xvectorDim, "\(tag): x-vector dim")
        let cosine = Self.cosineSimilarity(xvector, goldenXvector)
        XCTAssertGreaterThan(cosine, 0.999, "\(tag): x-vector cosine similarity vs Python")

        // RVQ codes: shape must be exact (the variable-length proof), the
        // first frame bit-exact, and ties near codebook boundaries may flip a
        // small number of individual codes (<1%).
        let (codes, frames) = encoder.encodeAudioCodes(waveform)
        let quantizers = SpeechTokenizerEncoder.validNumQuantizers
        XCTAssertEqual([quantizers, frames], goldens.codesShape, "\(tag): codes shape")

        let goldenCodes = try loadInts("\(goldensKey)_codes.bin")
        XCTAssertEqual(codes.count, goldenCodes.count, "\(tag): codes element count")

        let firstFrame = (0..<quantizers).map { codes[$0 * frames] }
        XCTAssertEqual(firstFrame, goldens.codesFirstFrame, "\(tag): first code frame")
        let lastFrame = (0..<quantizers).map { codes[$0 * frames + frames - 1] }
        XCTAssertEqual(lastFrame, goldens.codesLastFrame, "\(tag): last code frame")

        let matches = zip(codes, goldenCodes).filter(==).count
        let matchRate = Double(matches) / Double(goldenCodes.count)
        XCTAssertGreaterThanOrEqual(
            matchRate, 0.99,
            "\(tag): RVQ code match rate \(matchRate) (\(matches)/\(goldenCodes.count))"
        )
        print(String(format: "[%@] x-vector cosine: %.6f, code match: %.2f%%", tag, cosine, matchRate * 100))
    }

    func testParity8sReference() throws {
        try assertParity(goldensKey: "mlx_ref8s", waveformFile: "ref_audio_24k.bin", tag: "8s")
    }

    func testParity32sReference() throws {
        try assertParity(goldensKey: "mlx_ref32s", waveformFile: "ref_audio_24k_32s.bin", tag: "32s")
    }

    // MARK: - Prompt assembly

    func testEncodeProducesPrompt() throws {
        let encoder = try loadEncoder()
        let waveform = try loadFloats("ref_audio_24k.bin")
        let prompt = try encoder.encode(waveform, includeReferenceCodes: true, referenceText: "hello")
        XCTAssertEqual(prompt.speakerEmbedding.count, 1024)
        XCTAssertEqual(prompt.referenceText, "hello")
        let codes = try XCTUnwrap(prompt.referenceCodes)
        XCTAssertEqual(codes.count, SpeechTokenizerEncoder.validNumQuantizers * prompt.referenceCodeFrames)
    }

    func testReferenceLengthCap() throws {
        let encoder = try loadEncoder()
        let capped = MlxVoiceCloneEncoder(
            speakerEncoder: encoder.speakerEncoder,
            speechTokenizerEncoder: encoder.speechTokenizerEncoder,
            maxReferenceSeconds: 1
        )
        let waveform = [Float](repeating: 0.01, count: 2 * MlxVoiceCloneEncoder.sampleRate)
        XCTAssertThrowsError(try capped.encode(waveform, includeReferenceCodes: false)) { error in
            guard case TTSKitMLXError.referenceTooLong = error else {
                return XCTFail("Expected referenceTooLong, got \(error)")
            }
        }
    }
}

#endif // canImport(MLX)
