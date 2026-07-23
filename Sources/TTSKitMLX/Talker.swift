//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

#if canImport(MLX)

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

/// Configuration for the Qwen3-TTS talker transformer (`Qwen3TTSTalkerConfig`
/// in the Python reference). Defaults match the released 12hz-0.6b checkpoints;
/// the checkpoint's `config.json` `talker_config` overrides individual fields.
public struct TalkerConfig: Decodable, Sendable {
    public var vocabSize = 3072
    public var hiddenSize = 1024
    public var intermediateSize = 3072
    public var numHiddenLayers = 28
    public var numAttentionHeads = 16
    public var numKeyValueHeads = 8
    public var headDim = 128
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 1_000_000

    public init() {}

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = TalkerConfig()
        vocabSize = c.value(Int.self, forKey: .vocabSize, default: d.vocabSize)
        hiddenSize = c.value(Int.self, forKey: .hiddenSize, default: d.hiddenSize)
        intermediateSize = c.value(Int.self, forKey: .intermediateSize, default: d.intermediateSize)
        numHiddenLayers = c.value(Int.self, forKey: .numHiddenLayers, default: d.numHiddenLayers)
        numAttentionHeads = c.value(Int.self, forKey: .numAttentionHeads, default: d.numAttentionHeads)
        numKeyValueHeads = c.value(Int.self, forKey: .numKeyValueHeads, default: d.numKeyValueHeads)
        headDim = c.value(Int.self, forKey: .headDim, default: d.headDim)
        rmsNormEps = c.value(Float.self, forKey: .rmsNormEps, default: d.rmsNormEps)
        ropeTheta = c.value(Float.self, forKey: .ropeTheta, default: d.ropeTheta)
    }
}

/// Root-level `quantization` dict of an mlx-community checkpoint
/// (e.g. `{"group_size": 64, "bits": 8, "mode": "affine"}`).
public struct QuantizationSpec: Decodable, Sendable {
    public var groupSize = 64
    public var bits = 8

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = QuantizationSpec()
        groupSize = c.value(Int.self, forKey: .groupSize, default: d.groupSize)
        bits = c.value(Int.self, forKey: .bits, default: d.bits)
    }
}

// MARK: - Linear layers

/// Linear projection backed by either an affine-quantized weight triple
/// (`weight`/`scales`/`biases`, as in the mlx-community 8-bit repos) or a
/// plain floating-point weight. Bias-free — the talker uses
/// `attention_bias=false` throughout.
final class TalkerLinear {
    let weight: MLXArray
    let scales: MLXArray?
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int

    init(checkpoint: Checkpoint, prefix: String, quantization: QuantizationSpec?) throws {
        self.weight = try checkpoint.tensor("\(prefix).weight")
        if let quantization, checkpoint.weights["\(prefix).scales"] != nil {
            self.scales = try checkpoint.tensor("\(prefix).scales")
            self.biases = try checkpoint.tensor("\(prefix).biases")
            self.groupSize = quantization.groupSize
            self.bits = quantization.bits
        } else {
            self.scales = nil
            self.biases = nil
            self.groupSize = 0
            self.bits = 0
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let scales, let biases else { return matmul(x, weight.T) }
        return quantizedMM(
            x, weight, scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits
        )
    }
}

// MARK: - KV cache

/// Per-layer growable KV cache, mirroring `mlx_lm.models.cache.KVCache`:
/// buffers grow in `step`-sized chunks; `update` appends the new positions and
/// returns views trimmed to the valid length. `trim(to:)` rewinds the logical
/// offset without touching the buffers (stale tail data is never read because
/// every fetch slices `..<offset`).
final class TalkerKVCacheLayer {
    private static let step = 256

    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    private(set) var offset = 0

    /// Append `newKeys`/`newValues` of shape `(B, kvHeads, L, headDim)` and
    /// return the full valid cache `(B, kvHeads, offset + L, headDim)`.
    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let previous = offset
        let added = newKeys.dim(2)

        if keys == nil || previous + added > keys!.dim(2) {
            let (batch, kvHeads, _, headDim) = (newKeys.dim(0), newKeys.dim(1), 0, newKeys.dim(3))
            let capacity = ((previous + added + Self.step - 1) / Self.step) * Self.step
            let grownKeys = MLXArray.zeros([batch, kvHeads, capacity, headDim], dtype: newKeys.dtype)
            let grownValues = MLXArray.zeros([batch, kvHeads, capacity, headDim], dtype: newValues.dtype)
            if let keys, let values, previous > 0 {
                grownKeys[0..., 0..., 0 ..< previous, 0...] = keys[0..., 0..., 0 ..< previous, 0...]
                grownValues[0..., 0..., 0 ..< previous, 0...] = values[0..., 0..., 0 ..< previous, 0...]
            }
            keys = grownKeys
            values = grownValues
        }

        keys![0..., 0..., previous ..< previous + added, 0...] = newKeys
        values![0..., 0..., previous ..< previous + added, 0...] = newValues
        offset = previous + added
        return (
            keys![0..., 0..., 0 ..< offset, 0...],
            values![0..., 0..., 0 ..< offset, 0...]
        )
    }

    /// Rewind the logical position (e.g. restoring a shorter prefix). No-op
    /// when `position >= offset`.
    func trim(to position: Int) {
        offset = min(offset, max(position, 0))
    }
}

// MARK: - Transformer layers

/// Grouped-query attention with QK RMSNorm and rotary embeddings.
///
/// RoPE note: the Python reference uses interleaved multimodal RoPE
/// (`mrope_section [24, 20, 20]`), but the TTS pipeline always passes
/// identical positions on all three T/H/W axes, which makes the interleaved
/// combination an exact identity — `where(mask, f, f) == f`. Plain 1-D RoPE
/// (rotate-half convention, cos/sin computed in float32 and cast to the
/// activation dtype) therefore reproduces the reference bit-for-bit.
final class TalkerAttention {
    let qProj: TalkerLinear
    let kProj: TalkerLinear
    let vProj: TalkerLinear
    let oProj: TalkerLinear
    let qNormWeight: MLXArray
    let kNormWeight: MLXArray

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let rmsNormEps: Float

    init(config: TalkerConfig, checkpoint: Checkpoint, prefix: String, quantization: QuantizationSpec?) throws {
        self.qProj = try TalkerLinear(checkpoint: checkpoint, prefix: "\(prefix).q_proj", quantization: quantization)
        self.kProj = try TalkerLinear(checkpoint: checkpoint, prefix: "\(prefix).k_proj", quantization: quantization)
        self.vProj = try TalkerLinear(checkpoint: checkpoint, prefix: "\(prefix).v_proj", quantization: quantization)
        self.oProj = try TalkerLinear(checkpoint: checkpoint, prefix: "\(prefix).o_proj", quantization: quantization)
        self.qNormWeight = try checkpoint.tensor("\(prefix).q_norm.weight")
        self.kNormWeight = try checkpoint.tensor("\(prefix).k_norm.weight")
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.rmsNormEps = config.rmsNormEps
    }

    /// Rotate-half RoPE application: `x * cos + rotateHalf(x) * sin`.
    private static func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return x * cos + rotated * sin
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?, cache: TalkerKVCacheLayer,
        layerIndex: Int = -1, probe: AnchorProbe? = nil
    ) -> MLXArray {
        let (batch, seqLen) = (x.dim(0), x.dim(1))

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim)
        var k = kProj(x).reshaped(batch, seqLen, numKVHeads, headDim)
        var v = vProj(x).reshaped(batch, seqLen, numKVHeads, headDim)

        q = MLXFast.rmsNorm(q, weight: qNormWeight, eps: rmsNormEps)
        k = MLXFast.rmsNorm(k, weight: kNormWeight, eps: rmsNormEps)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        q = Self.applyRotary(q, cos: cos, sin: sin)
        k = Self.applyRotary(k, cos: cos, sin: sin)

        let (cachedK, cachedV) = cache.update(keys: k, values: v)

        // Observe-only text-anchor readout on the anchor layer during decode.
        // A separate score computation; the SDPA output below is unchanged.
        if let probe, seqLen == 1, layerIndex == probe.anchorLayer {
            probe.observe(q: q, cachedK: cachedK, scale: pow(Float(headDim), -0.5),
                          grp: numHeads / numKVHeads)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: cachedK, values: cachedV,
            scale: pow(Float(headDim), -0.5), mask: mask
        )
        return oProj(out.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim))
    }
}

/// Pre-norm decoder layer: RMSNorm → attention → residual, RMSNorm → SwiGLU
/// MLP → residual.
final class TalkerLayer {
    let attention: TalkerAttention
    let gateProj: TalkerLinear
    let upProj: TalkerLinear
    let downProj: TalkerLinear
    let inputNormWeight: MLXArray
    let postAttentionNormWeight: MLXArray
    let rmsNormEps: Float

    init(config: TalkerConfig, checkpoint: Checkpoint, prefix: String, quantization: QuantizationSpec?) throws {
        self.attention = try TalkerAttention(
            config: config, checkpoint: checkpoint, prefix: "\(prefix).self_attn", quantization: quantization
        )
        self.gateProj = try TalkerLinear(checkpoint: checkpoint, prefix: "\(prefix).mlp.gate_proj", quantization: quantization)
        self.upProj = try TalkerLinear(checkpoint: checkpoint, prefix: "\(prefix).mlp.up_proj", quantization: quantization)
        self.downProj = try TalkerLinear(checkpoint: checkpoint, prefix: "\(prefix).mlp.down_proj", quantization: quantization)
        self.inputNormWeight = try checkpoint.tensor("\(prefix).input_layernorm.weight")
        self.postAttentionNormWeight = try checkpoint.tensor("\(prefix).post_attention_layernorm.weight")
        self.rmsNormEps = config.rmsNormEps
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?, cache: TalkerKVCacheLayer,
        layerIndex: Int = -1, probe: AnchorProbe? = nil
    ) -> MLXArray {
        var h = MLXFast.rmsNorm(x, weight: inputNormWeight, eps: rmsNormEps)
        h = x + attention(h, cos: cos, sin: sin, mask: mask, cache: cache, layerIndex: layerIndex, probe: probe)
        var m = MLXFast.rmsNorm(h, weight: postAttentionNormWeight, eps: rmsNormEps)
        m = downProj(silu(gateProj(m)) * upProj(m))
        return h + m
    }
}

// MARK: - Talker

/// Qwen3-TTS talker: the Qwen3 0.6B transformer backbone plus the codec-0
/// head (`Qwen3TTSTalkerForConditionalGeneration` minus the code predictor and
/// the input embedding tables — TTSKit supplies pre-summed input embeddings).
///
/// Port of the Python reference (mlx-audio `talker.py`); weights come from the
/// `talker.model.layers.*` / `talker.model.norm` / `talker.codec_head` keys of
/// a Qwen3-TTS MLX checkpoint. The transformer weights are 8-bit affine
/// quantized in the mlx-community repos (`config.json`'s `quantization` dict);
/// unquantized checkpoints load through the same path.
final class Talker {
    let config: TalkerConfig
    let layers: [TalkerLayer]
    let normWeight: MLXArray
    let codecHead: TalkerLinear
    private let invFreq: MLXArray // (headDim / 2,) float32

    convenience init(modelDirectory: URL) throws {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        struct RootConfig: Decodable {
            let talkerConfig: TalkerConfig?
            let quantization: QuantizationSpec?
            enum CodingKeys: String, CodingKey {
                case talkerConfig = "talker_config"
                case quantization
            }
        }
        let root = try JSONDecoder().decode(RootConfig.self, from: Data(contentsOf: configURL))

        // sanitize(): keep the transformer + codec-head slices of the talker.*
        // keys, strip the prefix. The embedding tables, text projection, and
        // code predictor stay on their existing TTSKit CoreML components.
        let prefix = "talker."
        let raw = try ModelDirectory.loadWeights(directory: modelDirectory, glob: "model")
        var weights = [String: MLXArray]()
        for (key, value) in raw where key.hasPrefix(prefix) {
            let stripped = String(key.dropFirst(prefix.count))
            guard stripped.hasPrefix("model.layers.")
                || stripped.hasPrefix("model.norm.")
                || stripped.hasPrefix("codec_head.")
            else { continue }
            weights[stripped] = value
        }
        guard !weights.isEmpty else {
            throw TTSKitMLXError.invalidCheckpoint(
                "No talker weights in \(modelDirectory.path). Expected a Qwen3-TTS MLX "
                    + "checkpoint such as \(ModelDirectory.defaultRepoID)."
            )
        }
        try self.init(
            config: root.talkerConfig ?? TalkerConfig(),
            checkpoint: Checkpoint(weights: weights, source: modelDirectory),
            quantization: root.quantization
        )
    }

    init(config: TalkerConfig, checkpoint: Checkpoint, quantization: QuantizationSpec?) throws {
        self.config = config
        self.layers = try (0..<config.numHiddenLayers).map { i in
            try TalkerLayer(
                config: config, checkpoint: checkpoint,
                prefix: "model.layers.\(i)", quantization: quantization
            )
        }
        self.normWeight = try checkpoint.tensor("model.norm.weight")
        self.codecHead = try TalkerLinear(checkpoint: checkpoint, prefix: "codec_head", quantization: quantization)

        let dims = MLXArray(stride(from: 0, to: config.headDim, by: 2).map { Float($0) })
        self.invFreq = 1.0 / pow(MLXArray(config.ropeTheta), dims / Float(config.headDim))
    }

    func makeCache() -> [TalkerKVCacheLayer] {
        (0..<layers.count).map { _ in TalkerKVCacheLayer() }
    }

    /// Forward pass over `inputsEmbeds` `(1, L, hiddenSize)`, appending `L`
    /// positions to `cache`.
    ///
    /// Returns last-position outputs only — `logits (1, 1, vocabSize)` and
    /// `hidden (1, 1, hiddenSize)` — since autoregressive generation never
    /// consumes intermediate-position outputs (the codec head over all `L`
    /// prefill positions would be wasted work).
    func callAsFunction(
        _ inputsEmbeds: MLXArray, cache: [TalkerKVCacheLayer], probe: AnchorProbe? = nil
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let seqLen = inputsEmbeds.dim(1)
        let offset = cache.first?.offset ?? 0

        // Rotary cos/sin for positions offset ..< offset + L, computed in
        // float32 then cast to the activation dtype (the reference's
        // TalkerRotaryEmbedding with identical positions on all MRoPE axes).
        let positions = MLXArray(Int32(offset) ..< Int32(offset + seqLen)).asType(.float32)
        let freqs = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        let emb = concatenated([freqs, freqs], axis: -1) // (L, headDim)
        let cos = MLX.cos(emb).asType(inputsEmbeds.dtype).reshaped(1, 1, seqLen, config.headDim)
        let sin = MLX.sin(emb).asType(inputsEmbeds.dtype).reshaped(1, 1, seqLen, config.headDim)

        // Additive causal mask for multi-position (prefill) forwards. Matches
        // the reference: batched forwards only occur with an empty cache, so
        // the mask covers exactly the new positions.
        // Built in float32 (0 * -1e9 == 0) before the cast to the activation
        // dtype — multiplying after the fp16 cast would produce 0 * -inf = NaN.
        var mask: MLXArray?
        if seqLen > 1 {
            let indices = MLXArray(Int32(0) ..< Int32(seqLen))
            let future = indices.expandedDimensions(axis: 1) .< indices.expandedDimensions(axis: 0)
            mask = (future.asType(.float32) * Float(-1e9)).asType(inputsEmbeds.dtype)
        }

        var x = inputsEmbeds
        for (i, (layer, layerCache)) in zip(layers, cache).enumerated() {
            x = layer(x, cos: cos, sin: sin, mask: mask, cache: layerCache, layerIndex: i, probe: probe)
        }
        let hidden = MLXFast.rmsNorm(
            x[0..., (seqLen - 1)..., 0...], weight: normWeight, eps: config.rmsNormEps
        )
        return (codecHead(hidden), hidden)
    }
}

#endif // canImport(MLX)
