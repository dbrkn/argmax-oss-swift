//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
@testable import TTSKit
import XCTest

/// KV-overflow behavior of `SpeechStreamWriter`: generations longer than the
/// SpeechDecoder cache window must re-prime a fresh cache (with recent-frame
/// context) instead of clamping writes and silently degrading the audio tail.
final class SpeechStreamWriterOverflowTests: XCTestCase {
    /// Like the main tests' mock, but advances the cache write position on
    /// every call the way the real decoder does — so `isFull` actually trips.
    private final class CacheAdvancingMockDecoder: SpeechDecoding, @unchecked Sendable {
        let sampleRate = 24_000
        let samplesPerFrame = 4
        let minimumBufferDuration: TimeInterval = 0.08
        let kvCacheEmbedDim = Qwen3TTSConstants.sdCacheDim
        let kvCacheMaxSequenceLength = Qwen3TTSConstants.sdMaxSeq
        let hiddenDim = Qwen3TTSConstants.sdHiddenDim
        let hiddenContextLen = Qwen3TTSConstants.sdHiddenContextLen
        let codesPerStep = 4
        var model: MLModel? { nil }

        private let lock = NSLock()
        private var _decodedFrameCount = 0
        var decodedFrameCount: Int { lock.withLock { _decodedFrameCount } }

        func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool) async throws {}
        func unloadModel() {}

        func decodeFrame(codes: [[Int32]], cache: SpeechDecoderCache) async throws -> [Float] {
            try await decodeFrameAsync(codes: codes, cache: cache).samples
        }

        func decodeFrameAsync(codes: [[Int32]], cache: SpeechDecoderCache) async throws -> SpeechDecoderTimedResult {
            lock.withLock { _decodedFrameCount += codes.count }
            // Advance the cache write position without materializing K/V data,
            // mirroring the real decoder's post-prediction cache update.
            cache.update()
            var samples: [Float] = []
            for frame in codes {
                samples.append(contentsOf: repeatElement(Float(frame.first ?? -1), count: samplesPerFrame))
            }
            return SpeechDecoderTimedResult(samples: samples, timings: SpeechTimings())
        }
    }

    func testLongGenerationReprimesInsteadOfOverflowing() async throws {
        let decoder = CacheAdvancingMockDecoder()
        let cache = try SpeechDecoderCache(codesPerStep: decoder.codesPerStep)
        let writer = SpeechStreamWriter(
            speechDecoder: decoder,
            sdCache: cache,
            callback: nil,
            pipelineStart: CFAbsoluteTimeGetCurrent(),
            baseTimings: SpeechTimings(),
            padFrame: [Int32](repeating: Qwen3TTSConstants.codecPAD, count: 16)
        )

        // 300 frames > the 256-slot cache: must trigger at least one re-prime.
        let totalFrames = 300
        var timings = SpeechTimings()
        for code0 in 0..<totalFrames {
            let frame = [Int32(code0)] + [Int32](repeating: 0, count: 15)
            let ok = try await writer.append(frame, stepStart: CFAbsoluteTimeGetCurrent(), loopTimings: &timings)
            XCTAssertTrue(ok)
        }
        try await writer.finish(loopTimings: &timings)

        XCTAssertGreaterThanOrEqual(writer.cacheReprimes, 1, "300 frames must not fit a 256-slot cache")

        // Emitted audio must be exactly every submitted frame once, in order —
        // re-prime context decodes are internal and their audio discarded.
        XCTAssertEqual(writer.collectedAudio.count, totalFrames * decoder.samplesPerFrame)
        for code0 in 0..<totalFrames {
            XCTAssertEqual(
                writer.collectedAudio[code0 * decoder.samplesPerFrame], Float(code0),
                "audio for frame \(code0) missing or out of order"
            )
        }

        // The decoder saw extra (context) frames beyond the emitted ones.
        XCTAssertGreaterThan(decoder.decodedFrameCount, totalFrames)

        // And the cache was never left in an overflowing state.
        XCTAssertLessThan(Int(cache.cacheLength), decoder.kvCacheMaxSequenceLength)
    }
}
