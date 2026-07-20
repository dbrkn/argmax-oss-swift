//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - SpeechStreamWriter

/// Buffers decoded RVQ frames and streams them out as audio, owning the
/// overlapped-decode / drain / partial-flush logic and callback emission.
///
/// Single-use and not thread-safe: create one per generation and drive it serially
/// from one task.
final class SpeechStreamWriter<Decoder: SpeechDecoding & Sendable> {
    private let speechDecoder: Decoder
    private let sdCache: SpeechDecoderCache
    private let callback: SpeechCallback
    private let pipelineStart: CFAbsoluteTime
    private let baseTimings: SpeechTimings
    private let padFrame: [Int32]
    /// RVQ frames consumed per decode call; read from `speechDecoder`.
    private let codesPerStep: Int
    /// PCM samples produced per RVQ frame; read from `speechDecoder`.
    private let samplesPerFrame: Int

    /// Audio accumulated across every emitted buffer, in emission order.
    private(set) var collectedAudio: [Float] = []

    /// Whether the first buffer has been emitted. Drives the synchronous-first-buffer
    /// path (for minimum TTFB) and the time-to-first-buffer timing.
    private(set) var hasEmittedFirstBuffer = false

    /// RVQ frames accumulated since the last flush (length `0..<codesPerStep`).
    private var rvqBuffer: [[Int32]] = []

    /// Number of recent frames re-decoded to prime a fresh cache when the
    /// SpeechDecoder KV cache fills (rounded down to a multiple of
    /// `codesPerStep`). Mirrors the Python reference's windowed decode
    /// (`left_context=25`): without this, generations longer than the cache
    /// window silently drop KV positions and degrade the audio tail.
    private let reprimeContextFrames: Int
    /// Sliding history of the most recently *submitted* frames, ≥ `reprimeContextFrames`.
    private var recentFrames: [[Int32]] = []
    /// Number of cache re-primes performed (long generations only); surfaced in logs.
    private(set) var cacheReprimes = 0

    /// In-flight overlapped SpeechDecoder decode awaiting its drain on the next flush.
    private var pendingDecode: (
        task: Task<SpeechDecoderTimedResult, Error>,
        padCount: Int,
        isFirstBuffer: Bool,
        stepStart: CFAbsoluteTime
    )?

    init(
        speechDecoder: Decoder,
        sdCache: SpeechDecoderCache,
        callback: SpeechCallback,
        pipelineStart: CFAbsoluteTime,
        baseTimings: SpeechTimings,
        padFrame: [Int32]
    ) {
        self.speechDecoder = speechDecoder
        self.sdCache = sdCache
        self.callback = callback
        self.pipelineStart = pipelineStart
        self.baseTimings = baseTimings
        self.padFrame = padFrame
        // Cache geometry comes straight from the decoder — no need to thread it in.
        self.codesPerStep = speechDecoder.codesPerStep
        self.samplesPerFrame = speechDecoder.samplesPerFrame
        self.reprimeContextFrames = max(speechDecoder.codesPerStep, 24 / speechDecoder.codesPerStep * speechDecoder.codesPerStep)
        rvqBuffer.reserveCapacity(speechDecoder.codesPerStep)
    }

    // MARK: - Driving the stream

    /// Append one decoded RVQ frame. Once `codesPerStep` frames have accumulated,
    /// drains the previous in-flight decode and submits a new one.
    ///
    /// - Returns: `false` when a callback (the drained buffer's, or the first
    ///   buffer's) asked generation to stop; `true` to continue.
    func append(
        _ frame: [Int32],
        stepStart: CFAbsoluteTime,
        loopTimings: inout SpeechTimings
    ) async throws -> Bool {
        rvqBuffer.append(frame)
        guard rvqBuffer.count == codesPerStep else { return true }
        // Drain the previous in-flight SD (if any) before kicking off a new one —
        // its execution overlapped with this step's CD/MCD work.
        guard try await drainPendingDecode(loopTimings: &loopTimings) else { return false }
        try await reprimeCacheIfNeeded(loopTimings: &loopTimings)
        return try await submitDecode(stepStart: stepStart, loopTimings: &loopTimings)
    }

    /// Drain any in-flight decode, then flush the remaining partial buffer (padded to
    /// `codesPerStep`, trimming the pad-derived trailing samples).
    ///
    /// - Returns: `false` when the drain's callback asked to stop. The final partial
    ///   flush's callback result is intentionally ignored — there is nothing left to
    ///   stop — matching the original loop behavior.
    @discardableResult
    func finish(loopTimings: inout SpeechTimings) async throws -> Bool {
        guard try await drainPendingDecode(loopTimings: &loopTimings) else { return false }
        guard !rvqBuffer.isEmpty else { return true }
        try await reprimeCacheIfNeeded(loopTimings: &loopTimings)

        let padCount = max(0, codesPerStep - rvqBuffer.count)
        var toSubmit = rvqBuffer
        for _ in 0..<padCount { toSubmit.append(padFrame) }
        let result = try await speechDecoder.decodeFrameAsync(codes: toSubmit, cache: sdCache)
        _ = emitDecodedBuffer(
            samples: result.samples,
            padCount: padCount,
            predictionTime: result.timings.speechDecoderPredictions,
            isFirstBuffer: !hasEmittedFirstBuffer,
            stepStart: CFAbsoluteTimeGetCurrent(),
            loopTimings: &loopTimings
        )
        rvqBuffer.removeAll(keepingCapacity: true)
        return true
    }

    /// Cancel any in-flight decode so it cannot outlive the loop. Called on every
    /// exit path (the overlapped `Task` does not inherit the loop's cancellation).
    func cancelPendingDecode() {
        pendingDecode?.task.cancel()
        pendingDecode = nil
    }

    // MARK: - Decode submission / drain

    /// Keep a bounded history of submitted frames for cache re-priming.
    private func recordSubmittedFrames(_ frames: [[Int32]]) {
        recentFrames.append(contentsOf: frames)
        if recentFrames.count > reprimeContextFrames {
            recentFrames.removeFirst(recentFrames.count - reprimeContextFrames)
        }
    }

    /// When the next decode would overflow the SpeechDecoder KV cache, reset it
    /// and re-decode the last `reprimeContextFrames` frames to rebuild real
    /// context (their audio is discarded — it was already emitted). The window
    /// boundary matches the Python reference's chunked decode; the ~`context /
    /// window` extra decode cost only applies to generations longer than the
    /// cache window (~21 s for the kv_len_256 asset).
    ///
    /// Callers must have drained `pendingDecode` first (the re-prime mutates
    /// `sdCache`).
    private func reprimeCacheIfNeeded(loopTimings: inout SpeechTimings) async throws {
        guard sdCache.isFull else { return }
        let context = recentFrames.suffix(reprimeContextFrames)
        sdCache.reset()
        cacheReprimes += 1
        Logging.info(
            "SpeechDecoder KV cache full: re-priming a fresh cache with \(context.count) context frames "
                + "(re-prime #\(cacheReprimes))"
        )
        var group: [[Int32]] = []
        group.reserveCapacity(codesPerStep)
        for frame in context {
            group.append(frame)
            guard group.count == codesPerStep else { continue }
            let result = try await speechDecoder.decodeFrameAsync(codes: group, cache: sdCache)
            loopTimings.speechDecoderPredictions += result.timings.speechDecoderPredictions
            loopTimings.speechDecoder += result.timings.speechDecoderPredictions
            group.removeAll(keepingCapacity: true)
        }
        // `reprimeContextFrames` is a multiple of `codesPerStep`, so `group` is
        // empty here unless the history is still shorter than one step group.
    }

    /// Await the in-flight decode (if any) and emit its buffer.
    /// Required before submitting a new decode so `sdCache` mutations stay ordered.
    private func drainPendingDecode(loopTimings: inout SpeechTimings) async throws -> Bool {
        guard let pending = pendingDecode else { return true }
        pendingDecode = nil
        let result = try await pending.task.value
        return emitDecodedBuffer(
            samples: result.samples,
            padCount: pending.padCount,
            predictionTime: result.timings.speechDecoderPredictions,
            isFirstBuffer: pending.isFirstBuffer,
            stepStart: pending.stepStart,
            loopTimings: &loopTimings
        )
    }

    /// Snapshot the current buffer and either decode synchronously (first buffer, for
    /// minimum TTFB) or kick off an overlapped `Task` drained on the next flush.
    /// Callers must drain any previous `pendingDecode` first.
    ///
    /// - Returns: `false` when the first-buffer callback asked to stop; otherwise `true`.
    private func submitDecode(stepStart: CFAbsoluteTime, loopTimings: inout SpeechTimings) async throws -> Bool {
        let snapshot = rvqBuffer
        rvqBuffer.removeAll(keepingCapacity: true)
        recordSubmittedFrames(snapshot)

        if !hasEmittedFirstBuffer {
            let result = try await speechDecoder.decodeFrameAsync(codes: snapshot, cache: sdCache)
            return emitDecodedBuffer(
                samples: result.samples,
                padCount: 0,
                predictionTime: result.timings.speechDecoderPredictions,
                isFirstBuffer: true,
                stepStart: stepStart,
                loopTimings: &loopTimings
            )
        } else {
            let decoder = speechDecoder
            let cache = sdCache
            pendingDecode = (
                task: Task { try await decoder.decodeFrameAsync(codes: snapshot, cache: cache) },
                padCount: 0,
                isFirstBuffer: false,
                stepStart: stepStart
            )
            return true
        }
    }

    /// Trim trailing pad-derived audio samples, append to ``collectedAudio``, fold the
    /// decode's timing into `loopTimings`, build a `SpeechProgress`, emit the callback,
    /// and bookkeep first-buffer / TTFB state.
    ///
    /// - Returns: `false` when the callback asked generation to stop; `true` otherwise.
    private func emitDecodedBuffer(
        samples sourceSamples: [Float],
        padCount: Int,
        predictionTime: TimeInterval,
        isFirstBuffer: Bool,
        stepStart: CFAbsoluteTime,
        loopTimings: inout SpeechTimings
    ) -> Bool {
        loopTimings.speechDecoderPredictions += predictionTime
        loopTimings.speechDecoder += predictionTime
        loopTimings.totalSpeechDecoderInvocations += 1

        var samples = sourceSamples
        if padCount > 0 {
            let padSamples = padCount * samplesPerFrame
            if samples.count > padSamples {
                samples = Array(samples.prefix(samples.count - padSamples))
            } else {
                samples.removeAll()
            }
        }
        collectedAudio.append(contentsOf: samples)

        let now = CFAbsoluteTimeGetCurrent()
        if isFirstBuffer {
            loopTimings.timeToFirstBuffer = now - pipelineStart
            hasEmittedFirstBuffer = true
        }

        let progress: SpeechProgress
        if isFirstBuffer {
            progress = SpeechProgress(audio: samples, timings: loopTimings, stepTime: now - stepStart)
        } else {
            var merged = baseTimings
            merged.merge(loopTimings)
            progress = SpeechProgress(audio: samples, timings: merged, stepTime: nil)
        }
        return callback?(progress) != false
    }
}
