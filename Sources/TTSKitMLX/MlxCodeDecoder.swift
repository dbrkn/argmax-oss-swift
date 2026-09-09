//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

#if canImport(MLX)

import CoreML
import Foundation
import MLX
import TTSKit

/// MLX-backed drop-in replacement for TTSKit's CoreML `Qwen3CodeDecoder`.
///
/// Conforms to `CodeDecoding`, so it slots into the unchanged
/// `Qwen3GenerateTask` loop via `TTSKitConfig.codeDecoder`, and to
/// `BatchPrefillCapable`, so the task prefills the whole prompt prefix in one
/// batched forward pass instead of one CoreML call per position — the CoreML
/// talker prefills sequentially at ~30 tok/s, which costs 5–8 s of
/// time-to-first-audio on 150–300-token ICL voice-clone prefixes.
///
/// Cache design: the real KV state lives in MLX arrays owned by this instance
/// (`TalkerKVCacheLayer` per layer). The external TTSKit `KVCache` still
/// drives the generation loop's stop conditions (`cacheLength` / `isFull`), so
/// every forward mirrors its position bookkeeping into it via `cache.update()`
/// — but the external MLMultiArray K/V never hold data, and
/// `kvCacheEmbedDim == 1` keeps those placeholder buffers at a few bytes.
/// The internal cache re-synchronizes from `cache.cacheLength` on every call:
/// `0` resets it (start of a generation), a smaller value rewinds it (prompt
/// cache restore), a larger value throws.
///
/// Thread safety: unlike the stateless CoreML decoder, the internal MLX cache
/// is per-instance state — one instance supports one generation at a time
/// (matching the Python `MlxCodeDecoder` adapter). Voice-clone generation is
/// single-chunk, so this does not constrain the intended use.
public final class MlxCodeDecoder: CodeDecoding, BatchPrefillCapable, @unchecked Sendable {
    /// Always `nil`: the talker runs on MLX, not CoreML.
    public private(set) var model: MLModel?

    /// Placeholder K/V channel count for the external `KVCache` — see the
    /// cache-design note above. The real per-layer MLX cache geometry is
    /// derived from `TalkerConfig`.
    public let kvCacheEmbedDim = 1
    /// KV budget for the generation loop's `isFull` stop condition and the
    /// ICL prompt-fit validation. Unlike the CoreML asset (compile-time
    /// `kv_len`, typically 256), the MLX cache grows dynamically, so this is
    /// a configurable logical cap rather than an allocation size.
    public let kvCacheMaxSequenceLength: Int
    public var embedSize: Int { talker?.config.hiddenSize ?? TalkerConfig().hiddenSize }
    public var isStateful: Bool { false }

    private let modelDirectory: URL
    private var talker: Talker?
    private var internalCache: [TalkerKVCacheLayer] = []

    /// Load the talker from a Qwen3-TTS MLX checkpoint snapshot directory
    /// (see ``ModelDirectory``). Pass `nil` to resolve the default repo's
    /// cached Hugging Face snapshot.
    ///
    /// - Parameter maxSequenceLength: logical KV budget (prompt + generated
    ///   frames). 1024 comfortably covers ICL voice-clone prompts that the
    ///   256-slot CoreML variant rejects or truncates generation for.
    public init(modelDirectory: URL? = nil, maxSequenceLength: Int = 1024) throws {
        self.modelDirectory = try modelDirectory ?? ModelDirectory.defaultSnapshot()
        self.kvCacheMaxSequenceLength = maxSequenceLength
        self.talker = try Talker(modelDirectory: self.modelDirectory)
    }

    // MARK: - MLModelLoading

    /// The MLX weights are loaded from `modelDirectory` (at init or here after
    /// an `unloadModel()`); the CoreML asset URL TTSKit resolves for the
    /// `code_decoder` component is deliberately ignored.
    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        guard !prewarmMode else { return }
        if talker == nil {
            talker = try Talker(modelDirectory: modelDirectory)
        }
    }

    public func unloadModel() {
        talker = nil
        internalCache = []
    }

    // MARK: - CodeDecoding

    /// No CoreML `MLState`: KV state is managed internally (see cache design).
    public func makeState() -> Any? { nil }

    public func decode(inputEmbeds: any EmbedInputType, cache: KVCache, state: Any? = nil) async throws -> CodeDecoderOutput {
        if #available(macOS 15.0, *), let tensor = inputEmbeds as? MLTensor {
            let embeds = await tensor.cast(to: Float.self).toFloatArray().map { FloatType($0) }
            let (logits, hidden) = try forward(embeds: [embeds], cache: cache)
            return tensorOutput(logits: logits, hidden: hidden)
        }
        guard let array = inputEmbeds as? MLMultiArray else {
            throw TTSError.generationFailed("MlxCodeDecoder: unsupported embed input type \(type(of: inputEmbeds))")
        }
        let (logits, hidden) = try forward(embeds: [EmbedUtilities.extractEmbed(from: array)], cache: cache)
        return try legacyOutput(logits: logits, hidden: hidden)
    }

    // MARK: - BatchPrefillCapable

    public func prefill(embeds: [[FloatType]], cache: KVCache, state: Any? = nil) async throws -> CodeDecoderOutput {
        guard !embeds.isEmpty else {
            throw TTSError.generationFailed("MlxCodeDecoder: empty prefill prefix")
        }
        // The causal mask of a batched forward covers only the new positions,
        // so batching is valid only from an empty cache — the one way the
        // generate task calls it. Anything else falls back to per-position.
        let batches = cache.cacheLength == 0 ? [embeds] : embeds.map { [$0] }
        let forwardStart = CFAbsoluteTimeGetCurrent()
        var last: (logits: [Float], hidden: [FloatType])?
        for batch in batches {
            last = try forward(embeds: batch, cache: cache)
        }
        guard let last else {
            throw TTSError.generationFailed("MlxCodeDecoder: prefill produced no output")
        }
        let forwardMs = (CFAbsoluteTimeGetCurrent() - forwardStart) * 1000
        Logging.debug(String(
            format: "MlxCodeDecoder batched prefill: %d tokens in %.1fms (%.1f tok/s talker-only)",
            embeds.count, forwardMs, Double(embeds.count) / (forwardMs / 1000)
        ))
        if #available(macOS 15.0, *) {
            return tensorOutput(logits: last.logits, hidden: last.hidden)
        }
        return try legacyOutput(logits: last.logits, hidden: last.hidden)
    }

    // MARK: - Forward

    /// Run one talker forward over `embeds.count` positions, keeping the real
    /// KV in the internal MLX cache and mirroring position bookkeeping into
    /// the external `cache`. Returns last-position logits and hidden state.
    private func forward(embeds: [[FloatType]], cache: KVCache) throws -> (logits: [Float], hidden: [FloatType]) {
        guard let talker else {
            throw TTSError.generationFailed("MlxCodeDecoder model not loaded")
        }
        try syncInternalCache(to: Int(cache.cacheLength))

        let dim = talker.config.hiddenSize
        var flat = [Float]()
        flat.reserveCapacity(embeds.count * dim)
        for embed in embeds {
            guard embed.count == dim else {
                throw TTSError.generationFailed("MlxCodeDecoder: embed dim \(embed.count) != \(dim)")
            }
            for value in embed { flat.append(Float(value)) }
        }
        // fp16 activations entering the transformer, matching the fp16
        // embeddings of the CoreML pipeline and the Python MLX adapter.
        let x = MLXArray(flat, [1, embeds.count, dim]).asType(.float16)

        let (logitsArray, hiddenArray) = talker(x, cache: internalCache)
        eval(logitsArray, hiddenArray)

        // Mirror the consumed positions into the external cache so the
        // generation loop's cacheLength / isFull bookkeeping stays correct.
        for _ in 0..<embeds.count {
            cache.update()
        }

        let logits = logitsArray.asType(.float32).asArray(Float.self)
        let hidden = hiddenArray.asType(.float32).asArray(Float.self).map { FloatType($0) }
        return (logits, hidden)
    }

    /// Re-synchronize the internal MLX cache with the external cache position.
    private func syncInternalCache(to position: Int) throws {
        guard let talker else { return }
        if position == 0 {
            internalCache = talker.makeCache()
            return
        }
        let offset = internalCache.first?.offset ?? 0
        if position < offset {
            for layer in internalCache { layer.trim(to: position) }
        } else if position > offset {
            throw TTSError.generationFailed(
                "MlxCodeDecoder: external cache position \(position) is ahead of the internal MLX cache "
                    + "(\(offset)). Start a new generation (cacheLength 0); KV snapshots from other "
                    + "decoder instances cannot be restored into the MLX cache."
            )
        }
    }

    // MARK: - Output conversion

    /// MLTensor outputs for the async generation path: logits `(1, 1, vocab)`
    /// float32, hidden `(1, hidden, 1, 1)` fp16 (the CoreML output layouts).
    @available(macOS 15.0, *)
    private func tensorOutput(logits: [Float], hidden: [FloatType]) -> CodeDecoderOutput {
        CodeDecoderOutput(
            logits: MLTensor(shape: [1, 1, logits.count], scalars: logits, scalarType: Float.self),
            hiddenStates: hidden.asMLTensor(),
            keyCacheUpdates: nil,
            valueCacheUpdates: nil
        )
    }

    /// MLMultiArray/`[FloatType]` outputs for the legacy generation path.
    /// `keyCacheUpdates`/`valueCacheUpdates` stay `nil`: the loop skips its
    /// external KV writes and this decoder advances the cache itself.
    private func legacyOutput(logits: [Float], hidden: [FloatType]) throws -> CodeDecoderOutput {
        let logitsArray = try MLMultiArray(shape: [1, 1, NSNumber(value: logits.count)], dataType: .float16)
        let ptr = logitsArray.dataPointer.bindMemory(to: FloatType.self, capacity: logits.count)
        for (i, value) in logits.enumerated() { ptr[i] = FloatType(value) }
        return CodeDecoderOutput(
            logits: logitsArray,
            hiddenStates: hidden,
            keyCacheUpdates: nil,
            valueCacheUpdates: nil
        )
    }
}

/// Convenience factory: an MLX talker ready to assign to
/// `TTSKitConfig.codeDecoder` (or `TTSKit.codeDecoder` before `loadModels()`).
public func makeMlxCodeDecoder(modelDirectory: URL? = nil, maxSequenceLength: Int = 1024) throws -> any CodeDecoding {
    try MlxCodeDecoder(modelDirectory: modelDirectory, maxSequenceLength: maxSequenceLength)
}

#endif // canImport(MLX)
