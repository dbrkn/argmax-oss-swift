//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - Internal phase-result types

/// Output of the tokenize phase - fed into prefill and the generation loop.
struct TokenizeResult {
    let textTokenIds: [Int32]
    let trailingTextTokens: [Int32]
    let firstTextEmbed: [FloatType]
    let variableEmbed: [FloatType]
    let textPadEmbed: [FloatType]
    let timings: SpeechTimings
}

/// Output of the prefill phase - fed into the generation loop.
struct PrefillResult {
    let cdCache: KVCache
    let lastCdOutput: CodeDecoderOutput
    let timings: SpeechTimings
}

/// Output of the autoregressive generation loop.
struct GenerationLoopResult {
    let audio: [Float]
    let steps: Int
    let timings: SpeechTimings
}

// MARK: - Qwen3GenerateTask

/// Qwen3 TTS single-chunk generation task.
///
/// The core building block for Qwen3 TTS generation, analogous to `TranscribeTask`
/// in WhisperKit. Each task creates its own KV caches, MLState, and sampler, then
/// runs a complete prefill + autoregressive decode cycle.
///
/// Conforms to `SpeechGenerating` so it can be returned by
/// `TTSKit.setupGenerateTask(...)` and consumed by `TTSKit`'s generic
/// orchestration layer.
///
/// Thread safety: all stored properties are `let` (immutable after init). Each task
/// owns its own sampler (derived seed) so concurrent tasks don't share RNG state.
/// Model components are shared read-only references - `MLModel.prediction()` is
/// thread-safe. The class is `@unchecked Sendable` to permit `open` subclassing.
open class Qwen3GenerateTask: @unchecked Sendable, SpeechGenerating {
    /// Model components - concrete Qwen3 types for correct async method dispatch.
    /// Using `any Protocol` existentials would cause async extension methods to dispatch
    /// to the protocol default (sync path) instead of the Qwen3-specific MLTensor path.
    ///
    /// `codeDecoder` is the exception: everything the task calls on it is a
    /// `CodeDecoding` protocol requirement (no extension methods), so it stays
    /// protocol-typed — this is the seam that lets `Extensions/TTSKitMLX`
    /// substitute an MLX-backed talker via `TTSKitConfig.codeDecoder`.
    public let textProjector: Qwen3TextProjector
    public let codeEmbedder: Qwen3CodeEmbedder
    public let multiCodeEmbedder: Qwen3MultiCodeEmbedder
    public let codeDecoder: any CodeDecoding
    public let multiCodeDecoder: Qwen3MultiCodeDecoder
    public let speechDecoder: Qwen3SpeechDecoder
    public let sampler: any TokenSampling
    public let tokenizer: any TTSTokenizer
    public let suppressTokenIds: Set<Int>

    /// Timings captured at model-load time (modelLoading + tokenizerLoading populated).
    public let loadTimings: SpeechTimings

    /// Progress object for tracking generation. `totalUnitCount` is set to
    /// `maxNewTokens` at the start of generation; `completedUnitCount` is
    /// updated after each decoding step.
    public let progress: Progress

    // MARK: - Initialization

    public init(
        textProjector: Qwen3TextProjector,
        codeEmbedder: Qwen3CodeEmbedder,
        multiCodeEmbedder: Qwen3MultiCodeEmbedder,
        codeDecoder: any CodeDecoding,
        multiCodeDecoder: Qwen3MultiCodeDecoder,
        speechDecoder: Qwen3SpeechDecoder,
        sampler: any TokenSampling,
        tokenizer: any TTSTokenizer,
        suppressTokenIds: Set<Int>,
        loadTimings: SpeechTimings = SpeechTimings(),
        progress: Progress? = nil
    ) {
        self.textProjector = textProjector
        self.codeEmbedder = codeEmbedder
        self.multiCodeEmbedder = multiCodeEmbedder
        self.codeDecoder = codeDecoder
        self.multiCodeDecoder = multiCodeDecoder
        self.speechDecoder = speechDecoder
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.suppressTokenIds = suppressTokenIds
        self.loadTimings = loadTimings
        self.progress = progress ?? Progress()
    }

    // MARK: - SpeechGenerating defaults

    /// Default voice for Qwen3 TTS. Matches `Qwen3Speaker.ryan`.
    public var defaultVoice: String { Qwen3Speaker.ryan.rawValue }

    /// Default language for Qwen3 TTS. Matches `Qwen3Language.english`.
    public var defaultLanguage: String { Qwen3Language.english.rawValue }

    // MARK: - Audio format (forwarded from speechDecoder)

    public var sampleRate: Int { speechDecoder.sampleRate }
    public var samplesPerFrame: Int { speechDecoder.samplesPerFrame }
    public var minimumBufferDuration: TimeInterval { speechDecoder.minimumBufferDuration }

    // MARK: - Run

    /// Generate speech for a single text segment.
    ///
    /// Creates fresh KV caches, runs prefill, then autoregressive generation with
    /// interleaved SpeechDecoder audio output. Safe to call concurrently from
    /// multiple tasks against the same model instances.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Raw string matching `Qwen3Speaker.rawValue`; falls back to `.ryan`.
    ///   - language: Raw string matching `Qwen3Language.rawValue`; falls back to `.english`.
    ///   - options: Generation options (temperature, top-k, etc.)
    ///   - callback: Per-step callback receiving decoded audio and running timings.
    ///               `TTSProgress.stepTime` is non-nil only on the first step.
    ///               Return `false` to cancel; `nil` or `true` to continue.
    ///   - prefixCache: Optional cached prefix state to skip invariant prefill tokens.
    /// - Returns: A `SpeechResult` containing the complete audio and timings for this chunk.
    /// - Throws: `TTSError` on generation failure or task cancellation.
    open func run(
        text: String,
        voice: String,
        language: String,
        options: GenerationOptions,
        callback: SpeechCallback,
        prefixCache: TTSPromptCache? = nil
    ) async throws -> SpeechResult {
        let qwen3Speaker = Qwen3Speaker(rawValue: voice) ?? .ryan
        let lang = Qwen3Language(rawValue: language) ?? .english

        var timings = loadTimings
        let pipelineStart = CFAbsoluteTimeGetCurrent()

        progress.totalUnitCount = Int64(options.maxNewTokens)
        progress.completedUnitCount = 0

        // Create task-local MLState for stateful decoders (nil for non-stateful)
        let cdState = codeDecoder.makeState()

        // Phase 1: Tokenize text and build initial embeddings
        var tokenizeResult = try await tokenizeAndBuildEmbeds(text: text)
        if options.voiceClone?.referenceCodes != nil {
            // ICL voice clone bakes the full synthesis text into the prefix;
            // the generation loop sees text PAD on every step.
            tokenizeResult = TokenizeResult(
                textTokenIds: tokenizeResult.textTokenIds,
                trailingTextTokens: [],
                firstTextEmbed: tokenizeResult.firstTextEmbed,
                variableEmbed: tokenizeResult.variableEmbed,
                textPadEmbed: tokenizeResult.textPadEmbed,
                timings: tokenizeResult.timings
            )
        }
        timings.merge(tokenizeResult.timings)

        // Phase 2: Prefill the CodeDecoder with the prompt prefix
        let prefillResult = try await prefillCodeDecoder(
            tokenizeResult: tokenizeResult,
            speaker: qwen3Speaker, lang: lang,
            options: options,
            prefixCache: prefixCache,
            voice: voice, language: language,
            cdState: cdState
        )
        timings.merge(prefillResult.timings)

        // Phase 3: Autoregressive RVQ generation with interleaved audio decode
        let loopResult = try await runGenerationLoop(
            tokenizeResult: tokenizeResult,
            prefillResult: prefillResult,
            cdState: cdState,
            options: options,
            pipelineStart: pipelineStart,
            callback: callback,
            baseTimings: timings
        )
        timings.merge(loopResult.timings)
        timings.timeToFirstBuffer = loopResult.timings.timeToFirstBuffer

        timings.fullPipeline = CFAbsoluteTimeGetCurrent() - pipelineStart
        timings.inputAudioSeconds = Double(loopResult.audio.count) / Double(speechDecoder.sampleRate)

        progress.completedUnitCount = progress.totalUnitCount

        let genMs = timings.decodingLoop * 1000
        let avgMs = loopResult.steps > 0 ? genMs / Double(loopResult.steps) : 0
        let stepsPerSec = loopResult.steps > 0 ? Double(loopResult.steps) / timings.decodingLoop : 0
        Logging.info(
            String(
                format: "Generation: %d frames in %.1fms (%.1fms/step, %.1f frames/s)",
                loopResult.steps, genMs, avgMs, stepsPerSec
            ))

        return SpeechResult(audio: loopResult.audio, timings: timings, sampleRate: speechDecoder.sampleRate)
    }

    // MARK: - Phase 1: Tokenize

    /// Tokenize `text` and pre-compute the initial embeddings needed for prefill and decoding.
    private func tokenizeAndBuildEmbeds(
        text: String
    ) async throws -> TokenizeResult {
        let start = CFAbsoluteTimeGetCurrent()

        let textTokenIds = tokenizer.encode(text: text).map { Int32($0) }
        guard !textTokenIds.isEmpty else { throw TTSError.emptyText }

        let firstTextEmbed = try await textProjector.project(tokenId: textTokenIds[0])
        let codecBOSEmbed = try await codeEmbedder.embed(tokenId: Qwen3TTSConstants.codecBOS)
        let variableEmbed = EmbedUtilities.addEmbeddings(firstTextEmbed, codecBOSEmbed)
        let textPadEmbed = try await textProjector.project(tokenId: Qwen3TTSConstants.textPAD)

        var phaseTimings = SpeechTimings()
        phaseTimings.tokenize = CFAbsoluteTimeGetCurrent() - start

        return TokenizeResult(
            textTokenIds: textTokenIds,
            trailingTextTokens: Array(textTokenIds.dropFirst()),
            firstTextEmbed: firstTextEmbed,
            variableEmbed: variableEmbed,
            textPadEmbed: textPadEmbed,
            timings: phaseTimings
        )
    }

    // MARK: - Phase 2: Prefill

    /// Prefill the CodeDecoder KV cache with the invariant prompt prefix.
    ///
    /// If `prefixCache` matches the current voice/language/instruction, restores the
    /// cached state and only decodes the variable token. Otherwise runs a full prefill.
    private func prefillCodeDecoder(
        tokenizeResult: TokenizeResult,
        speaker: Qwen3Speaker,
        lang: Qwen3Language,
        options: GenerationOptions,
        prefixCache: TTSPromptCache?,
        voice: String,
        language: String,
        cdState: Any?
    ) async throws -> PrefillResult {
        let start = CFAbsoluteTimeGetCurrent()

        let cdCache = try KVCache(
            cacheDim: codeDecoder.kvCacheEmbedDim,
            maxSeqLength: codeDecoder.kvCacheMaxSequenceLength,
            isStateful: codeDecoder.isStateful
        )

        // Voice cloning never uses the speaker-keyed prompt cache: the ICL
        // prefix embeds the per-utterance text, and the x-vector prefix is
        // keyed by reference clip, not by `voice`.
        let usedCache = options.voiceClone == nil
            && prefixCache?.matches(voice: voice, language: language, instruction: options.instruction) == true
        var totalPrefillTokens: Int
        var lastCdOutput: CodeDecoderOutput?

        if usedCache, let prefixCache {
            cdCache.restore(from: prefixCache.kvSnapshot)
            if let stateData = prefixCache.stateData {
                if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = cdState as? MLState {
                    mlState.restore(from: stateData)
                }
            }
            totalPrefillTokens = prefixCache.prefixLength + 1

            // TODO: Remove forking logic with package with min os version upgrade
            if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), !options.forceLegacyEmbedPath {
                lastCdOutput = try await codeDecoder.decode(
                    inputEmbeds: tokenizeResult.variableEmbed.asMLTensor(), cache: cdCache, state: cdState
                )
            } else {
                let embedArr = try EmbedUtilities.createEmbedMLArray(tokenizeResult.variableEmbed)
                lastCdOutput = try await codeDecoder.decode(inputEmbeds: embedArr, cache: cdCache, state: cdState)
            }
        } else {
            let embedDim = codeDecoder.embedSize
            let combinedEmbeds: [[FloatType]]
            if let clone = options.voiceClone, clone.referenceCodes != nil {
                let icl = try await buildVoiceCloneICLEmbeddings(
                    prompt: clone,
                    referenceText: clone.referenceText ?? "",
                    lang: lang,
                    instruction: options.instruction,
                    textTokenIds: tokenizeResult.textTokenIds,
                    embedDim: embedDim
                )
                combinedEmbeds = icl.embeds
                // Arm the guardrail anchor probe over the synthesis-text KV span
                // (observe-only; no effect on audio unless the executor is later
                // enabled). No-op for decoders that aren't GuardrailObservable.
                if let g = options.guardrails, g.enabled, let obs = codeDecoder as? GuardrailObservable {
                    obs.beginGuardrailObservation(
                        anchorLayer: g.anchorLayer, anchorHead: g.anchorHead,
                        textStart: icl.mainTextRange.lowerBound, textEnd: icl.mainTextRange.upperBound,
                        recordTrajectory: g.recordTrajectory)
                    Logging.info("Guardrails armed: anchor L\(g.anchorLayer)H\(g.anchorHead), "
                        + "text KV span [\(icl.mainTextRange.lowerBound),\(icl.mainTextRange.upperBound))")
                }
            } else {
                combinedEmbeds = try await buildCombinedEmbeddings(
                    speaker: speaker, lang: lang,
                    instruction: options.instruction,
                    firstTextEmbed: tokenizeResult.firstTextEmbed,
                    embedDim: embedDim,
                    speakerEmbeddingOverride: options.voiceClone.map { clone in
                        clone.speakerEmbedding.map { FloatType($0) }
                    }
                )
            }
            totalPrefillTokens = combinedEmbeds.count

            // TODO: Remove forking logic with package with min os version upgrade
            if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), !options.forceLegacyEmbedPath {
                if let batchDecoder = codeDecoder as? BatchPrefillCapable {
                    // Batched prefill: one forward pass over the whole prefix.
                    // The sequential loop below is dominated by per-call
                    // dispatch overhead on decoders that support batching.
                    lastCdOutput = try await batchDecoder.prefill(embeds: combinedEmbeds, cache: cdCache, state: cdState)
                } else {
                    for embed in combinedEmbeds {
                        lastCdOutput = try await codeDecoder.decode(inputEmbeds: embed.asMLTensor(), cache: cdCache, state: cdState)
                    }
                }
            } else {
                for (embedIndex, embed) in combinedEmbeds.enumerated() {
                    if embedIndex > 0, let keyUpdates = lastCdOutput?.keyCacheUpdates, let valueUpdates = lastCdOutput?.valueCacheUpdates {
                        cdCache.update(keyCacheUpdates: keyUpdates, valueCacheUpdates: valueUpdates)
                    }
                    let embedArr = try EmbedUtilities.createEmbedMLArray(embed)
                    lastCdOutput = try await codeDecoder.decode(inputEmbeds: embedArr, cache: cdCache, state: cdState)
                }
            }
        }

        guard let resolvedLastCdOutput = lastCdOutput else {
            throw TTSError.generationFailed("Prefill produced no decoder output")
        }

        var phaseTimings = SpeechTimings()
        phaseTimings.prefill = CFAbsoluteTimeGetCurrent() - start
        phaseTimings.prefillTokens = Double(totalPrefillTokens)

        let prefillMs = phaseTimings.prefill * 1000
        let prefillTokS = phaseTimings.prefill > 0 ? Double(totalPrefillTokens) / phaseTimings.prefill : 0
        let cacheTag = usedCache ? " (cache hit, restored \(prefixCache?.prefixLength ?? 0) tokens)" : ""
        Logging.info(
            String(
                format: "Prefill: %.1fms (%d tokens, %.1f tok/s)%@",
                prefillMs, totalPrefillTokens, prefillTokS, cacheTag
            ))

        return PrefillResult(cdCache: cdCache, lastCdOutput: resolvedLastCdOutput, timings: phaseTimings)
    }

    // MARK: - Phase 3: Generation loop

    /// Run the autoregressive RVQ generation loop, delivering audio frames via `callback`.
    ///
    /// `baseTimings` carries the tokenize + prefill phase timings and is used to build
    /// accurate cumulative `SpeechProgress` values for callbacks.
    /// Returns the assembled audio, the number of steps completed, and the loop-phase timings.
    private func runGenerationLoop(
        tokenizeResult: TokenizeResult,
        prefillResult: PrefillResult,
        cdState: Any?,
        options: GenerationOptions,
        pipelineStart: CFAbsoluteTime,
        callback: SpeechCallback,
        baseTimings: SpeechTimings
    ) async throws -> GenerationLoopResult {
        let cdCache = prefillResult.cdCache
        var lastCdOutput = prefillResult.lastCdOutput
        var timings = baseTimings

        let roleTokenIds = tokenizer.encode(text: "<|im_start|>assistant\n").map { Int32($0) }
        let maxStepsByPrefill = 8 * (roleTokenIds.count + tokenizeResult.textTokenIds.count)

        let codesPerStep = speechDecoder.codesPerStep
        let sdCache = try SpeechDecoderCache(
            cacheDim: speechDecoder.kvCacheEmbedDim,
            maxSeqLength: speechDecoder.kvCacheMaxSequenceLength,
            hiddenDim: speechDecoder.hiddenDim,
            hiddenContextLen: speechDecoder.hiddenContextLen,
            codesPerStep: codesPerStep
        )

        var generatedTokens: [Int32] = []
        var code0 = await sampler.sampleCodec0(
            logits: lastCdOutput.logits,
            temperature: options.temperature, topK: options.topK,
            generatedTokens: generatedTokens,
            repetitionPenalty: options.repetitionPenalty,
            suppressTokenIds: suppressTokenIds
        )
        generatedTokens.append(code0)

        var stepIndex = 0
        var stopRequested = false

        let padFrame: [Int32] = [Int32](repeating: Qwen3TTSConstants.codecPAD, count: 16)

        // Owns RVQ-frame buffering, the overlapped SpeechDecoder decode/drain, pad
        // trimming, audio accumulation, and callback emission. Extracted so this loop
        // stays flat; SpeechDecoder timings are folded into `timings` (passed `inout`).
        let writer = SpeechStreamWriter(
            speechDecoder: speechDecoder,
            sdCache: sdCache,
            callback: callback,
            pipelineStart: pipelineStart,
            baseTimings: baseTimings,
            padFrame: padFrame
        )

        // Tear down the writer's overlapped decode on every exit path — it is an
        // unstructured Task that does not inherit this loop's cancellation.
        defer { writer.cancelPendingDecode() }

        // Guardrail detection (RD-655, Stage 1 observe-only). Active only when
        // enabled AND the decoder exposes the anchor observable (MLX talker).
        let gConfig = options.guardrails
        let guardObs: GuardrailObservable? =
            (gConfig?.enabled == true) ? (codeDecoder as? GuardrailObservable) : nil
        let guardMonitor: CoverageMonitor? = guardObs == nil ? nil
            : CoverageMonitor(ntok: max(1, tokenizeResult.textTokenIds.count), config: gConfig!.monitor)
        var guardStats = GuardrailStats()
        if let g = gConfig, guardObs != nil {
            guardStats.active = true
            guardStats.observeOnly = g.observeOnly
            guardStats.anchor = [g.anchorLayer, g.anchorHead]
        }
        // Executor (Stage 2) is active only when guardrails are enabled, the
        // decoder is observable, AND not observe-only. In that mode the loop
        // decouples vocoding from decode (accumulates RVQ frames, vocodes once
        // after the loop) so a rollback can trim the frame list, KV, monitor and
        // token history without unwinding the streaming SpeechDecoder — matching
        // the DESIGN.md decision that rollback is incompatible with live streaming.
        let executorActive = (guardObs != nil) && (gConfig?.observeOnly == false)
        // Step the monitor with the anchor fraction the just-completed decode
        // recorded; returns the confirmed failure (if any) so the executor can act.
        // Observe-only ignores the return value (Stage 1: count fires, never roll back).
        func guardStep() -> GuardrailFailure? {
            guard let obs = guardObs, let f = obs.lastAnchorFraction else { return nil }
            guard let fire = guardMonitor?.step(f) else { return nil }
            guardStats.record(fire)
            return fire
        }

        // Executor (Stage 2) state — used only when `executorActive`. `prefillLen`
        // is the KV position right after prefill; a rollback to decode step `k`
        // resets the external cache to `prefillLen + k` (the MLX decoder trims its
        // internal cache to match on the next forward). `frames`/`hiddenHistory`
        // are the rewindable per-step records; `hiddenHistory[i]` is step i's
        // decoder hidden state, needed to re-drive step (i+1)'s multi-code decode
        // after a rollback. In full mode the loop appends to `frames` instead of
        // streaming through the writer, then vocodes `frames` once after the loop.
        let prefillLen = Int(cdCache.cacheLength)
        var frames: [[Int32]] = []
        var hiddenHistory: [any EmbedTensorType] = []
        var rollbackCount = 0
        var executorGaveUp = false
        let rollbackDeadline = CFAbsoluteTimeGetCurrent() + (gConfig?.maxRollbackSeconds ?? 120)
        if executorActive { frames.reserveCapacity(options.maxNewTokens) }

        // TODO: Remove forking logic with package with min os version upgrade
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), !options.forceLegacyEmbedPath {
            let textPadEmbedTensor: MLTensor = try await textProjector.project(tokenId: Qwen3TTSConstants.textPAD)

            while code0 != Qwen3TTSConstants.codecEOS
                && !cdCache.isFull
                && stepIndex < options.maxNewTokens
                && stepIndex < maxStepsByPrefill
            {
                try Task.checkCancellation()
                let stepStart = CFAbsoluteTimeGetCurrent()

                let codeEmbedStart = CFAbsoluteTimeGetCurrent()
                let code0EmbedTensor: MLTensor = try await codeEmbedder.embed(tokenId: code0)
                timings.codeEmbed += CFAbsoluteTimeGetCurrent() - codeEmbedStart

                let mcdStart = CFAbsoluteTimeGetCurrent()
                guard let hiddenStatesTensor = lastCdOutput.hiddenStates as? MLTensor else {
                    throw TTSError.generationFailed("Expected MLTensor hidden states on async path")
                }
                let mcdResult = try await multiCodeDecoder.generateMultiCodes(
                    hiddenStatesTensor: hiddenStatesTensor,
                    code0EmbedTensor: code0EmbedTensor,
                    multiCodeEmbedder: multiCodeEmbedder,
                    sampler: sampler, options: options
                )
                timings.multiCodeDecoder += CFAbsoluteTimeGetCurrent() - mcdStart
                timings.multiCodeDecoderPredictions += mcdResult.timings.multiCodeDecoderPredictions
                timings.multiCodeDecoderSampling += mcdResult.timings.multiCodeDecoderSampling
                timings.multiCodeDecoderEmbedding += mcdResult.timings.multiCodeDecoderEmbedding
                timings.decodingKvCaching += mcdResult.timings.decodingKvCaching
                timings.totalMultiCodeDecoderPredictions += mcdResult.timings.totalMultiCodeDecoderPredictions

                // Captured before `code0` is resampled below; ingested into the writer
                // after sampling (nothing between here and then reads the buffer).
                let rvqFrame = [code0] + mcdResult.codes

                let codecHiddenStart = CFAbsoluteTimeGetCurrent()
                guard let lastMcdCode = mcdResult.codes.last else {
                    throw TTSError.generationFailed("Multi-code generation result has no codes")
                }
                let code15OffsetId = lastMcdCode + Int32(multiCodeDecoder.codecVocabSize * 14)
                let code15EmbedTensor: MLTensor = try await multiCodeEmbedder.embed(tokenId: code15OffsetId)
                var allCodeEmbedTensors: [MLTensor] = [code0EmbedTensor]
                if let tensorEmbeds = mcdResult.offsetCodeEmbedTensors {
                    allCodeEmbedTensors += tensorEmbeds
                } else {
                    allCodeEmbedTensors += mcdResult.offsetCodeEmbeds.map { $0.asMLTensor() }
                }
                allCodeEmbedTensors.append(code15EmbedTensor)
                let codecHiddenTensor = EmbedUtilities.sumEmbeddings(allCodeEmbedTensors)
                timings.codecHidden += CFAbsoluteTimeGetCurrent() - codecHiddenStart

                let textProjStart = CFAbsoluteTimeGetCurrent()
                let textEmbedTensor: MLTensor =
                    stepIndex < tokenizeResult.trailingTextTokens.count
                    ? try await textProjector.project(tokenId: tokenizeResult.trailingTextTokens[stepIndex])
                    : textPadEmbedTensor
                let combinedTensor = EmbedUtilities.addEmbeddings(codecHiddenTensor, textEmbedTensor)
                timings.textProjection += CFAbsoluteTimeGetCurrent() - textProjStart

                let decodingStart = CFAbsoluteTimeGetCurrent()
                lastCdOutput = try await codeDecoder.decode(inputEmbeds: combinedTensor, cache: cdCache, state: cdState)
                timings.decodingPredictions += CFAbsoluteTimeGetCurrent() - decodingStart - lastCdOutput.internalCacheUpdateTime
                timings.kvCacheUpdate += lastCdOutput.internalCacheUpdateTime

                // Record this step's hidden state (full mode) BEFORE stepping the
                // monitor: a rollback to step k re-drives step k's multi-code decode
                // from step (k−1)'s hidden, so the history must include this step.
                if executorActive { hiddenHistory.append(lastCdOutput.hiddenStates) }
                if let fire = guardStep(), executorActive, !executorGaveUp {
                    if rollbackCount >= (gConfig?.maxRetries ?? 10) || CFAbsoluteTimeGetCurrent() > rollbackDeadline {
                        // Budget exhausted: stop rolling back and keep decoding to the
                        // end from here (degrade to baseline — never hang).
                        executorGaveUp = true
                        guardStats.gaveUp = true
                        Logging.info("Guardrail executor gave up after \(rollbackCount) rollbacks "
                            + "(\(fire.failure.rawValue) at step \(stepIndex)); continuing without rollback")
                    } else {
                        // Rewind to decode step `target` (≤ stepIndex). Step `stepIndex`
                        // was decoded but not yet committed (no frame appended, no next
                        // token sampled), so discarding steps [target, stepIndex] leaves
                        // no half-applied state.
                        let target = max(0, min(fire.rollbackStep, stepIndex))
                        cdCache.cacheLength = Int32(prefillLen + target)   // MLX trims internal cache on next forward
                        if frames.count > target { frames.removeLast(frames.count - target) }
                        let resumeHidden: any EmbedTensorType =
                            target > 0 ? hiddenHistory[target - 1] : prefillResult.lastCdOutput.hiddenStates
                        if hiddenHistory.count > target { hiddenHistory.removeLast(hiddenHistory.count - target) }
                        lastCdOutput = CodeDecoderOutput(
                            logits: lastCdOutput.logits,        // placeholder; overwritten by step `target`'s decode before any read
                            hiddenStates: resumeHidden,
                            keyCacheUpdates: nil, valueCacheUpdates: nil)
                        // Token history holds [seed, out(0)…out(stepIndex−1)] = stepIndex+1
                        // entries; keep [0, target] so generatedTokens[target] is the
                        // input token to step `target`.
                        if generatedTokens.count > target + 1 {
                            generatedTokens.removeLast(generatedTokens.count - (target + 1))
                        }
                        code0 = generatedTokens[target]
                        guardMonitor?.rollback(to: target)
                        guardObs?.truncateGuardrailTrajectory(to: target)
                        // Stochastic sampling (temp>0) — reseeding varies the retry so a
                        // deterministic stall is escaped. Deterministic per run.
                        sampler.reseed(UInt64(0xA5A5A5 &+ UInt64(rollbackCount) &* 2_654_435_761 &+ UInt64(target)))
                        guardStats.rollbacks += 1
                        guardStats.rewoundAudioSeconds += Double(stepIndex - target) / 12.5
                        guardStats.addedWallSeconds += CFAbsoluteTimeGetCurrent() - stepStart
                        rollbackCount += 1
                        Logging.info("Guardrail rollback #\(rollbackCount): \(fire.failure.rawValue) at step "
                            + "\(stepIndex) → rewind to \(target) (−\(stepIndex - target) steps)")
                        stepIndex = target
                        continue                                // abort this step; re-decode from `target`
                    }
                }

                let samplingStart = CFAbsoluteTimeGetCurrent()
                code0 = await sampler.sampleCodec0(
                    logits: lastCdOutput.logits,
                    temperature: options.temperature, topK: options.topK,
                    generatedTokens: generatedTokens,
                    repetitionPenalty: options.repetitionPenalty,
                    suppressTokenIds: suppressTokenIds
                )
                generatedTokens.append(code0)
                timings.decodingSampling += CFAbsoluteTimeGetCurrent() - samplingStart

                // Full mode (executor) accumulates frames for a single post-loop
                // vocode pass; observe/off mode streams through the writer as before
                // (unchanged, bit-identical audio — the Stage-1 fidelity invariant).
                if executorActive {
                    frames.append(rvqFrame)
                } else if !(try await writer.append(rvqFrame, stepStart: stepStart, loopTimings: &timings)) {
                    stopRequested = true
                    break
                }

                timings.decodingLoop += CFAbsoluteTimeGetCurrent() - stepStart
                stepIndex += 1
                progress.completedUnitCount = Int64(stepIndex)

                if stepIndex == 1 || stepIndex % 10 == 0 {
                    let stepMs = (CFAbsoluteTimeGetCurrent() - stepStart) * 1000
                    Logging.debug(
                        String(
                            format: "  Step %d: %.1fms (avg %.1fms/step)",
                            stepIndex, stepMs, timings.decodingLoop * 1000 / Double(stepIndex)))
                }
            }
        } else {
            // Legacy embed path (forced via options.forceLegacyEmbedPath, or pre-macOS-15
            // OSes — note the multifunction SpeechDecoder asset still requires iOS 18+,
            // so SD calls will throw at runtime on older OSes).
            while code0 != Qwen3TTSConstants.codecEOS
                && !cdCache.isFull
                && stepIndex < options.maxNewTokens
                && stepIndex < maxStepsByPrefill
            {
                try Task.checkCancellation()
                let stepStart = CFAbsoluteTimeGetCurrent()

                let cacheUpdateStart = CFAbsoluteTimeGetCurrent()
                if let keyUpdates = lastCdOutput.keyCacheUpdates, let valueUpdates = lastCdOutput.valueCacheUpdates {
                    cdCache.update(keyCacheUpdates: keyUpdates, valueCacheUpdates: valueUpdates)
                }
                timings.kvCacheUpdate += CFAbsoluteTimeGetCurrent() - cacheUpdateStart

                let codeEmbedStart = CFAbsoluteTimeGetCurrent()
                let code0Embed = try await codeEmbedder.embed(tokenId: code0)
                timings.codeEmbed += CFAbsoluteTimeGetCurrent() - codeEmbedStart

                let mcdStart = CFAbsoluteTimeGetCurrent()
                guard let hiddenStates = lastCdOutput.hiddenStates as? [FloatType] else {
                    throw TTSError.generationFailed("Expected [FloatType] hidden states on legacy path")
                }
                let mcdResult = try await multiCodeDecoder.generateMultiCodes(
                    hiddenStates: hiddenStates, code0Embed: code0Embed,
                    multiCodeEmbedder: multiCodeEmbedder, sampler: sampler, options: options
                )
                timings.multiCodeDecoder += CFAbsoluteTimeGetCurrent() - mcdStart
                timings.multiCodeDecoderPredictions += mcdResult.timings.multiCodeDecoderPredictions
                timings.multiCodeDecoderSampling += mcdResult.timings.multiCodeDecoderSampling
                timings.multiCodeDecoderEmbedding += mcdResult.timings.multiCodeDecoderEmbedding
                timings.decodingKvCaching += mcdResult.timings.decodingKvCaching
                timings.totalMultiCodeDecoderPredictions += mcdResult.timings.totalMultiCodeDecoderPredictions

                // Captured before `code0` is resampled below; ingested into the writer
                // after sampling (nothing between here and then reads the buffer).
                let rvqFrame = [code0] + mcdResult.codes

                let codecHiddenStart = CFAbsoluteTimeGetCurrent()
                guard let lastMcdCode = mcdResult.codes.last else {
                    throw TTSError.generationFailed("Multi-code generation result has no codes")
                }
                let code15OffsetId = lastMcdCode + Int32(multiCodeDecoder.codecVocabSize * 14)
                var allCodeEmbeds: [[FloatType]] = [code0Embed]
                allCodeEmbeds += mcdResult.offsetCodeEmbeds
                try await allCodeEmbeds.append(multiCodeEmbedder.embed(tokenId: code15OffsetId))
                let codecHidden = EmbedUtilities.sumEmbeddings(allCodeEmbeds)
                timings.codecHidden += CFAbsoluteTimeGetCurrent() - codecHiddenStart

                let textProjStart = CFAbsoluteTimeGetCurrent()
                let textTokenEmbed: [FloatType] =
                    stepIndex < tokenizeResult.trailingTextTokens.count
                    ? try await textProjector.project(tokenId: tokenizeResult.trailingTextTokens[stepIndex])
                    : tokenizeResult.textPadEmbed
                let combinedArr = try EmbedUtilities.createEmbedMLArray(EmbedUtilities.addEmbeddings(codecHidden, textTokenEmbed))
                timings.textProjection += CFAbsoluteTimeGetCurrent() - textProjStart

                let decodingStart = CFAbsoluteTimeGetCurrent()
                lastCdOutput = try await codeDecoder.decode(inputEmbeds: combinedArr, cache: cdCache, state: cdState)
                timings.decodingPredictions += CFAbsoluteTimeGetCurrent() - decodingStart
                // Legacy path is observe-only (the executor runs on the MLX async
                // path); the CoreML sync decoder is not GuardrailObservable anyway.
                _ = guardStep()

                let samplingStart = CFAbsoluteTimeGetCurrent()
                code0 = await sampler.sampleCodec0(
                    logits: lastCdOutput.logits,
                    temperature: options.temperature, topK: options.topK,
                    generatedTokens: generatedTokens,
                    repetitionPenalty: options.repetitionPenalty,
                    suppressTokenIds: suppressTokenIds
                )
                generatedTokens.append(code0)
                timings.decodingSampling += CFAbsoluteTimeGetCurrent() - samplingStart

                if !(try await writer.append(rvqFrame, stepStart: stepStart, loopTimings: &timings)) {
                    stopRequested = true
                    break
                }

                timings.decodingLoop += CFAbsoluteTimeGetCurrent() - stepStart
                stepIndex += 1
                progress.completedUnitCount = Int64(stepIndex)

                if stepIndex == 1 || stepIndex % 10 == 0 {
                    let stepMs = (CFAbsoluteTimeGetCurrent() - stepStart) * 1000
                    Logging.debug(
                        String(
                            format: "  Step %d: %.1fms (avg %.1fms/step)",
                            stepIndex, stepMs, timings.decodingLoop * 1000 / Double(stepIndex)))
                }
            }
        }

        // Full mode decoupled vocoding from the loop so rollbacks could trim the
        // frame list; the final (post-rollback) frames are vocoded here in one
        // pass through the writer. Guardrails-full is not a live-streaming mode
        // (see DESIGN.md), so emitting all audio at the end is expected.
        if executorActive {
            for frame in frames {
                _ = try await writer.append(frame, stepStart: CFAbsoluteTimeGetCurrent(), loopTimings: &timings)
            }
            Logging.info("Guardrail executor: \(guardStats.rollbacks) rollback(s), "
                + "\(String(format: "%.1f", guardStats.rewoundAudioSeconds))s audio rewound, "
                + "\(frames.count) frames vocoded\(guardStats.gaveUp ? " (gave up: budget)" : "")")
        }

        // Drain the in-flight decode from the last completed flush, then flush any
        // remaining partial buffer. Skipped entirely when the callback already asked
        // to stop in the loop, so we don't emit one more `SpeechProgress` after that.
        if !stopRequested {
            try await writer.finish(loopTimings: &timings)
        }

        let stopReason: String
        if code0 == Qwen3TTSConstants.codecEOS {
            stopReason = "EOS token"
        } else if cdCache.isFull {
            stopReason = "KV cache full (\(cdCache.cacheLength)/\(cdCache.maxSeqLength))"
        } else if stepIndex >= maxStepsByPrefill {
            stopReason = "Audio token ratio limit (\(stepIndex)/\(maxStepsByPrefill) steps)"
        } else {
            stopReason = "maxNewTokens limit (\(options.maxNewTokens))"
        }
        Logging.info("Loop stopped: \(stopReason) after \(stepIndex) steps")

        // Guardrail finalization: log the run's fire stats and, for anchor
        // validation, dump the f(t) trajectory when GUARDRAIL_TRAJECTORY_OUT is set.
        if let obs = guardObs {
            if gConfig?.recordTrajectory == true { guardStats.fTrajectory = obs.guardrailTrajectory() }
            Logging.info(guardStats.summary)
            if let out = ProcessInfo.processInfo.environment["GUARDRAIL_TRAJECTORY_OUT"] {
                let traj = obs.guardrailTrajectory()
                let fires = guardStats.events.map {
                    "{\"type\":\"\($0.failure.rawValue)\",\"fire\":\($0.fireStep),\"rollback\":\($0.rollbackStep)}"
                }.joined(separator: ",")
                let fstr = traj.map { String($0) }.joined(separator: ",")
                let diag = obs.guardrailDiagnostics()
                let gaStr = diag.globalArgmax.map { String($0) }.joined(separator: ",")
                let tmStr = diag.textMass.map { String($0) }.joined(separator: ",")
                let json = "{\"anchor\":[\(gConfig!.anchorLayer),\(gConfig!.anchorHead)],"
                    + "\"ntok\":\(tokenizeResult.textTokenIds.count),\"steps\":\(stepIndex),"
                    + "\"textStart\":\(diag.textStart),\"textEnd\":\(diag.textEnd),"
                    + "\"fires\":[\(fires)],\"f\":[\(fstr)],"
                    + "\"rollbacks\":\(guardStats.rollbacks),\"gaveUp\":\(guardStats.gaveUp),"
                    + "\"rewoundAudioSeconds\":\(guardStats.rewoundAudioSeconds),"
                    + "\"globalArgmax\":[\(gaStr)],\"textMass\":[\(tmStr)]}"
                try? json.write(toFile: out, atomically: true, encoding: .utf8)
                Logging.info("Guardrail trajectory (\(traj.count) steps, \(guardStats.events.count) fires) -> \(out)")
            }
            obs.endGuardrailObservation()
        }

        timings.totalDecodingLoops = Double(stepIndex)
        return GenerationLoopResult(audio: writer.collectedAudio, steps: stepIndex, timings: timings)
    }

    // MARK: - Embedding Helpers

    /// Build the full combined embedding sequence (text track + codec track) for prefill.
    /// The returned array includes both the invariant prefix and the variable last token.
    func buildCombinedEmbeddings(
        speaker: Qwen3Speaker,
        lang: Qwen3Language,
        instruction: String?,
        firstTextEmbed: [FloatType],
        embedDim: Int,
        speakerEmbeddingOverride: [FloatType]? = nil
    ) async throws -> [[FloatType]] {
        let zeroCodecEmbed = EmbedUtilities.zeroEmbed(dim: embedDim)

        var instructTextEmbeds: [[FloatType]] = []
        var instructCodecEmbeds: [[FloatType]] = []
        if let instruction, !instruction.isEmpty {
            let instructPrompt = "<|im_start|>user\n\(instruction)<|im_end|>\n"
            let instructTokenIds = tokenizer.encode(text: instructPrompt).map { Int32($0) }
            for tokenId in instructTokenIds {
                try await instructTextEmbeds.append(textProjector.project(tokenId: tokenId))
                instructCodecEmbeds.append(zeroCodecEmbed)
            }
            Logging.debug("Instruction: \(instructTokenIds.count) tokens")
        }

        let rolePrefix = "<|im_start|>assistant\n"
        let roleTokenIds = tokenizer.encode(text: rolePrefix).map { Int32($0) }

        var textTrackEmbeds: [[FloatType]] = instructTextEmbeds
        for tokenId in roleTokenIds {
            try await textTrackEmbeds.append(textProjector.project(tokenId: tokenId))
        }
        let textPadEmbed = try await textProjector.project(tokenId: Qwen3TTSConstants.textPAD)
        let textBosEmbed = try await textProjector.project(tokenId: Qwen3TTSConstants.textBOS)

        let codecIds: [Int32] = [
            Qwen3TTSConstants.codecThink,
            Qwen3TTSConstants.codecThinkBos,
            lang.tokenID,
            Qwen3TTSConstants.codecThinkEos,
            speaker.tokenID,
            Qwen3TTSConstants.codecPAD,
            Qwen3TTSConstants.codecBOS
        ]
        var codecTrackEmbeds: [[FloatType]] = []
        for (slot, codecId) in codecIds.enumerated() {
            // Slot 4 is the speaker slot; x-vector-only voice cloning replaces
            // the speaker token's embedding with the reference clip's x-vector.
            if slot == 4, let speakerEmbeddingOverride {
                guard speakerEmbeddingOverride.count == embedDim else {
                    throw TTSError.invalidConfiguration(
                        "Speaker embedding dim \(speakerEmbeddingOverride.count) != decoder embed dim \(embedDim)"
                    )
                }
                codecTrackEmbeds.append(speakerEmbeddingOverride)
            } else {
                try await codecTrackEmbeds.append(codeEmbedder.embed(tokenId: codecId))
            }
        }

        let numPads = codecIds.count - 2
        for _ in 0..<numPads {
            textTrackEmbeds.append(textPadEmbed)
        }
        textTrackEmbeds.append(textBosEmbed)
        textTrackEmbeds.append(firstTextEmbed)

        var fullCodecTrackEmbeds: [[FloatType]] = instructCodecEmbeds
        fullCodecTrackEmbeds.append(contentsOf: Array(repeating: zeroCodecEmbed, count: roleTokenIds.count))
        fullCodecTrackEmbeds.append(contentsOf: codecTrackEmbeds)

        assert(
            textTrackEmbeds.count == fullCodecTrackEmbeds.count,
            "Track alignment mismatch: text=\(textTrackEmbeds.count) codec=\(fullCodecTrackEmbeds.count)")

        return zip(textTrackEmbeds, fullCodecTrackEmbeds).map { EmbedUtilities.addEmbeddings($0.0, $0.1) }
    }

    // MARK: - Prompt Cache Building

    /// Build a prompt cache by prefilling the invariant prefix tokens through the CodeDecoder.
    ///
    /// The invariant prefix includes: optional instruction tokens, role prefix,
    /// speaker/language control tokens, and BOS. Only the last token (first text token
    /// + codecBOS) varies per utterance and is excluded from the cache.
    ///
    /// Returns a snapshot of the KV cache state after prefilling; on cache hit the
    /// generation task restores this state and only decodes the variable token.
    open func buildPromptCache(
        voice: String,
        language: String,
        instruction: String?
    ) async throws -> TTSPromptCache {
        let qwen3Speaker = Qwen3Speaker(rawValue: voice) ?? .ryan
        let lang = Qwen3Language(rawValue: language) ?? .english
        let embedDim = codeDecoder.embedSize

        // Build invariant embeddings (everything except the last variable token).
        // Use a dummy firstTextEmbed since we drop the last element.
        let dummyFirstTextEmbed = EmbedUtilities.zeroEmbed(dim: embedDim)
        let allEmbeds = try await buildCombinedEmbeddings(
            speaker: qwen3Speaker,
            lang: lang,
            instruction: instruction,
            firstTextEmbed: dummyFirstTextEmbed,
            embedDim: embedDim
        )
        let invariantEmbeds = Array(allEmbeds.dropLast())

        // Pre-initialize the MultiCodeDecoder ANE pipeline concurrently with the
        // CodeDecoder prefill loop below. Cache build is the right place for this
        // one-time cost: it absorbs the ~150ms without affecting TTFB, and the
        // warmed pipeline persists for all subsequent generation calls.
        let mcdWarmupTask: Task<Void, Never>?
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            let mcd = multiCodeDecoder
            mcdWarmupTask = Task { try? await mcd.prewarmInference() }
        } else {
            mcdWarmupTask = nil
        }

        let cdState = codeDecoder.makeState()
        let cdCache = try KVCache(
            cacheDim: codeDecoder.kvCacheEmbedDim,
            maxSeqLength: codeDecoder.kvCacheMaxSequenceLength,
            isStateful: codeDecoder.isStateful
        )

        var lastCdOutput: CodeDecoderOutput?
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *) {
            // Async path: decoder updates cache internally
            for embed in invariantEmbeds {
                lastCdOutput = try await codeDecoder.decode(inputEmbeds: embed.asMLTensor(), cache: cdCache, state: cdState)
            }
        } else {
            for (embedIndex, embed) in invariantEmbeds.enumerated() {
                if embedIndex > 0, let keyUpdates = lastCdOutput?.keyCacheUpdates, let valueUpdates = lastCdOutput?.valueCacheUpdates {
                    cdCache.update(keyCacheUpdates: keyUpdates, valueCacheUpdates: valueUpdates)
                }
                let embedArr = try EmbedUtilities.createEmbedMLArray(embed)
                lastCdOutput = try await codeDecoder.decode(inputEmbeds: embedArr, cache: cdCache, state: cdState)
            }
            // Commit the last pending KV update so the snapshot is fully self-contained
            if let keyUpdates = lastCdOutput?.keyCacheUpdates, let valueUpdates = lastCdOutput?.valueCacheUpdates {
                cdCache.update(keyCacheUpdates: keyUpdates, valueCacheUpdates: valueUpdates)
            }
        }

        // Snapshot MLState for stateful models
        var stateData: KVStateData?
        if #available(macOS 15.0, iOS 18.0, watchOS 11.0, visionOS 2.0, *), let mlState = cdState as? MLState {
            stateData = mlState.snapshot()
        }

        // Ensure warmup is done before returning - by this point the CodeDecoder
        // loop has run (~2700ms), so this await is a no-op in practice.
        await mcdWarmupTask?.value

        Logging.info("Built prompt cache: \(invariantEmbeds.count) invariant tokens, isStateful=\(codeDecoder.isStateful) for \(voice)/\(language)")

        return TTSPromptCache(
            voice: voice,
            language: language,
            instruction: instruction,
            prefixLength: invariantEmbeds.count,
            kvSnapshot: cdCache.snapshot(),
            stateData: stateData
        )
    }
}
