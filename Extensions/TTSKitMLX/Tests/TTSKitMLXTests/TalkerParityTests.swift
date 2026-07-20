//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
import MLX
import TTSKit
@testable import TTSKitMLX
import XCTest

/// Parity tests for the MLX-Swift talker (`MlxCodeDecoder`) against the
/// Python MLX reference (mlx-audio `Qwen3TTSTalkerForConditionalGeneration`).
///
/// Goldens are exported by `scripts/export_talker_goldens.py` into
/// `Tests/TTSKitTests/Resources/VoiceCloneGoldens/` (same convention as
/// `EncoderParityTests`): a realistic 47-position prompt prefix (control block
/// + audio codes summed with the projected text-PAD embedding, fp16), the
/// batched-prefill last-position logits/hidden, and 20 greedy decode tokens
/// where each step feeds back `codec_embedding(token) + textPadEmbed` — the
/// same tables both sides read from the checkpoint, so the decode loop is
/// bit-comparable end to end.
final class TalkerParityTests: XCTestCase {
    struct TalkerGoldens: Decodable {
        let prefixIds: [Int]
        let prefixLength: Int
        let hiddenSize: Int
        let vocabSize: Int
        let greedyTokens: [Int]
        let prefillArgmax: Int
    }

    // MARK: - Fixtures

    static let goldensDirectory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // TTSKitMLXTests
        .deletingLastPathComponent() // Tests
        .deletingLastPathComponent() // TTSKitMLX
        .deletingLastPathComponent() // Extensions
        .deletingLastPathComponent() // repository root
        .appendingPathComponent("Tests/TTSKitTests/Resources/VoiceCloneGoldens")

    static var talker: Talker?
    static var codecEmbedding: MLXArray?

    private func loadTalker() throws -> Talker {
        if let talker = Self.talker { return talker }
        let talker: Talker
        do {
            talker = try Talker(modelDirectory: try ModelDirectory.defaultSnapshot())
        } catch {
            throw XCTSkip("MLX checkpoint snapshot unavailable: \(error.localizedDescription)")
        }
        Self.talker = talker
        return talker
    }

    /// The talker's own codec embedding table `(vocabSize, hiddenSize)` —
    /// used only by tests to reproduce the reference's feedback path.
    private func loadCodecEmbedding() throws -> MLXArray {
        if let table = Self.codecEmbedding { return table }
        let directory = try ModelDirectory.defaultSnapshot()
        let weights = try ModelDirectory.loadWeights(directory: directory, glob: "model")
        let table = try XCTUnwrap(weights["talker.model.codec_embedding.weight"])
        Self.codecEmbedding = table
        return table
    }

    private func goldenURL(_ name: String) throws -> URL {
        let url = Self.goldensDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Golden \(name) not found (run scripts/export_talker_goldens.py)")
        }
        return url
    }

    private func loadFloats(_ name: String) throws -> [Float] {
        try Data(contentsOf: goldenURL(name)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func loadGoldens() throws -> TalkerGoldens {
        try JSONDecoder().decode(TalkerGoldens.self, from: Data(contentsOf: goldenURL("mlx_talker_goldens.json")))
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

    /// Prefix embeddings as an fp16 `(1, L, hidden)` MLXArray.
    private func loadPrefix(_ goldens: TalkerGoldens) throws -> MLXArray {
        let values = try loadFloats("mlx_talker_prefix_embeds.bin")
        XCTAssertEqual(values.count, goldens.prefixLength * goldens.hiddenSize)
        return MLXArray(values, [1, goldens.prefixLength, goldens.hiddenSize]).asType(.float16)
    }

    /// `codec_embedding(token) + textPadEmbed` cast fp16 — the reference
    /// decode loop's feedback embedding.
    private func feedbackEmbed(token: Int, table: MLXArray, textPad: MLXArray) -> MLXArray {
        (table[token] + textPad).asType(.float16).reshaped(1, 1, -1)
    }

    // MARK: - Python parity (talker level)

    func testPrefillAndGreedyDecodeParity() throws {
        let talker = try loadTalker()
        let goldens = try loadGoldens()
        let prefix = try loadPrefix(goldens)
        let textPad = MLXArray(try loadFloats("mlx_talker_feedback_text_embed.bin"))
        let table = try loadCodecEmbedding()

        // Batched prefill over the whole prefix in one forward.
        let cache = talker.makeCache()
        var (logits, hidden) = talker(prefix, cache: cache)
        eval(logits, hidden)
        XCTAssertEqual(cache[0].offset, goldens.prefixLength)

        let logitsF = logits.asType(.float32).asArray(Float.self)
        let hiddenF = hidden.asType(.float32).asArray(Float.self)
        XCTAssertEqual(logitsF.count, goldens.vocabSize)

        let goldenLogits = try loadFloats("mlx_talker_prefill_logits.bin")
        let goldenHidden = try loadFloats("mlx_talker_prefill_hidden.bin")
        let logitsCosine = Self.cosineSimilarity(logitsF, goldenLogits)
        let hiddenCosine = Self.cosineSimilarity(hiddenF, goldenHidden)
        XCTAssertGreaterThan(logitsCosine, 0.999, "prefill last-position logits vs Python")
        XCTAssertGreaterThan(hiddenCosine, 0.999, "prefill last-position hidden vs Python")

        // Greedy decode: token ids must match the Python reference exactly.
        var tokens = [Int]()
        var current = argMax(logits[0, -1]).item(Int.self)
        for _ in 0..<goldens.greedyTokens.count {
            tokens.append(current)
            (logits, hidden) = talker(feedbackEmbed(token: current, table: table, textPad: textPad), cache: cache)
            eval(logits)
            current = argMax(logits[0, -1]).item(Int.self)
        }
        XCTAssertEqual(tokens, goldens.greedyTokens, "greedy decode diverged from the Python MLX talker")

        print(String(format: "[talker] prefill logits cosine: %.6f, hidden cosine: %.6f, greedy %d/%d tokens exact",
                     logitsCosine, hiddenCosine, tokens.count, goldens.greedyTokens.count))
    }

    /// Batched prefill must be numerically consistent with feeding the same
    /// prefix one position at a time (no Python involved).
    func testBatchedPrefillMatchesSequential() throws {
        let talker = try loadTalker()
        let goldens = try loadGoldens()
        let prefix = try loadPrefix(goldens)

        let batchedCache = talker.makeCache()
        let (batchedLogits, _) = talker(prefix, cache: batchedCache)

        let sequentialCache = talker.makeCache()
        var sequentialLogits = MLXArray.zeros([1])
        for position in 0..<goldens.prefixLength {
            (sequentialLogits, _) = talker(prefix[0..., position ..< position + 1, 0...], cache: sequentialCache)
        }

        let batched = batchedLogits.asType(.float32).asArray(Float.self)
        let sequential = sequentialLogits.asType(.float32).asArray(Float.self)
        let cosine = Self.cosineSimilarity(batched, sequential)
        XCTAssertGreaterThan(cosine, 0.9999, "batched vs sequential prefill")
        XCTAssertEqual(
            argMax(batchedLogits[0, -1]).item(Int.self),
            argMax(sequentialLogits[0, -1]).item(Int.self)
        )
    }

    // MARK: - CodeDecoding interface

    /// Drive the same golden sequence through the `CodeDecoding` /
    /// `BatchPrefillCapable` surface the generate task uses, checking the
    /// external KVCache bookkeeping (`cacheLength`, rewind semantics) and the
    /// MLTensor output conversion.
    @available(macOS 15.0, *)
    func testDecoderInterfaceParityAndBookkeeping() async throws {
        _ = try loadTalker()
        let goldens = try loadGoldens()
        let decoder: MlxCodeDecoder
        do {
            decoder = try MlxCodeDecoder()
        } catch {
            throw XCTSkip("MLX checkpoint snapshot unavailable: \(error.localizedDescription)")
        }
        let table = try loadCodecEmbedding()
        let textPad = MLXArray(try loadFloats("mlx_talker_feedback_text_embed.bin"))

        let prefixValues = try loadFloats("mlx_talker_prefix_embeds.bin")
        let dim = goldens.hiddenSize
        let embeds: [[FloatType]] = (0..<goldens.prefixLength).map { position in
            prefixValues[position * dim ..< (position + 1) * dim].map { FloatType($0) }
        }

        let cache = try KVCache(
            cacheDim: decoder.kvCacheEmbedDim,
            maxSeqLength: decoder.kvCacheMaxSequenceLength,
            isStateful: decoder.isStateful
        )
        var output = try await decoder.prefill(embeds: embeds, cache: cache, state: nil)
        XCTAssertEqual(Int(cache.cacheLength), goldens.prefixLength, "prefill must advance the external cache")

        func greedyToken(_ output: CodeDecoderOutput) async throws -> Int {
            let tensor = try XCTUnwrap(output.logits as? MLTensor)
            let logits = await tensor.cast(to: Float.self).toFloatArray()
            return logits.indices.max(by: { logits[$0] < logits[$1] })!
        }

        var tokens = [Int]()
        var current = try await greedyToken(output)
        for _ in 0..<goldens.greedyTokens.count {
            tokens.append(current)
            let embed = feedbackEmbed(token: current, table: table, textPad: textPad)
                .asType(.float32).asArray(Float.self).map { FloatType($0) }
            output = try await decoder.decode(inputEmbeds: embed.asMLTensor(), cache: cache, state: nil)
            current = try await greedyToken(output)
        }
        XCTAssertEqual(tokens, goldens.greedyTokens, "interface-level greedy decode diverged")
        XCTAssertEqual(Int(cache.cacheLength), goldens.prefixLength + goldens.greedyTokens.count)

        // Hidden-state output layout: (1, hidden, 1, 1) fp16, like CoreML.
        let hiddenTensor = try XCTUnwrap(output.hiddenStates as? MLTensor)
        XCTAssertEqual(hiddenTensor.shape, [1, goldens.hiddenSize, 1, 1])

        // A fresh external cache (cacheLength 0) must reset the internal one.
        let freshCache = try KVCache(
            cacheDim: decoder.kvCacheEmbedDim,
            maxSeqLength: decoder.kvCacheMaxSequenceLength,
            isStateful: decoder.isStateful
        )
        let fresh = try await decoder.prefill(embeds: embeds, cache: freshCache, state: nil)
        let freshToken = try await greedyToken(fresh)
        XCTAssertEqual(freshToken, goldens.prefillArgmax)
    }
}
