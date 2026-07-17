//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

/// Configuration for the Mimi-style speech-tokenizer encoder
/// (`Qwen3TTSTokenizerEncoderConfig` in the Python reference). Decoded from
/// `speech_tokenizer/config.json`'s `encoder_config`; missing keys fall back
/// to the reference dataclass defaults.
public struct SpeechTokenizerEncoderConfig: Decodable, Sendable {
    public var frameRate = 12.5
    public var audioChannels = 1
    public var codebookDim = 256
    public var codebookSize = 2048
    public var compress = 2
    public var dilationGrowthRate = 2
    public var headDim = 64
    public var hiddenSize = 512
    public var intermediateSize = 2048
    public var kernelSize = 7
    public var lastKernelSize = 3
    public var normEps = 1e-5
    public var numAttentionHeads = 8
    public var numFilters = 64
    public var numHiddenLayers = 8
    public var numQuantizers = 32
    public var numResidualLayers = 1
    public var residualKernelSize = 3
    public var ropeTheta = 10000.0
    public var samplingRate = 24000
    public var upsamplingRatios = [8, 6, 5, 4]
    public var useCausalConv = true

    public init() {}

    enum CodingKeys: String, CodingKey {
        case frameRate = "frame_rate"
        case audioChannels = "audio_channels"
        case codebookDim = "codebook_dim"
        case codebookSize = "codebook_size"
        case compress
        case dilationGrowthRate = "dilation_growth_rate"
        case headDim = "head_dim"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case kernelSize = "kernel_size"
        case lastKernelSize = "last_kernel_size"
        case normEps = "norm_eps"
        case numAttentionHeads = "num_attention_heads"
        case numFilters = "num_filters"
        case numHiddenLayers = "num_hidden_layers"
        case numQuantizers = "num_quantizers"
        case numResidualLayers = "num_residual_layers"
        case residualKernelSize = "residual_kernel_size"
        case ropeTheta = "rope_theta"
        case samplingRate = "sampling_rate"
        case upsamplingRatios = "upsampling_ratios"
        case useCausalConv = "use_causal_conv"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = SpeechTokenizerEncoderConfig()
        frameRate = c.value(Double.self, forKey: .frameRate, default: d.frameRate)
        audioChannels = c.value(Int.self, forKey: .audioChannels, default: d.audioChannels)
        codebookDim = c.value(Int.self, forKey: .codebookDim, default: d.codebookDim)
        codebookSize = c.value(Int.self, forKey: .codebookSize, default: d.codebookSize)
        compress = c.value(Int.self, forKey: .compress, default: d.compress)
        dilationGrowthRate = c.value(Int.self, forKey: .dilationGrowthRate, default: d.dilationGrowthRate)
        headDim = c.value(Int.self, forKey: .headDim, default: d.headDim)
        hiddenSize = c.value(Int.self, forKey: .hiddenSize, default: d.hiddenSize)
        intermediateSize = c.value(Int.self, forKey: .intermediateSize, default: d.intermediateSize)
        kernelSize = c.value(Int.self, forKey: .kernelSize, default: d.kernelSize)
        lastKernelSize = c.value(Int.self, forKey: .lastKernelSize, default: d.lastKernelSize)
        normEps = c.value(Double.self, forKey: .normEps, default: d.normEps)
        numAttentionHeads = c.value(Int.self, forKey: .numAttentionHeads, default: d.numAttentionHeads)
        numFilters = c.value(Int.self, forKey: .numFilters, default: d.numFilters)
        numHiddenLayers = c.value(Int.self, forKey: .numHiddenLayers, default: d.numHiddenLayers)
        numQuantizers = c.value(Int.self, forKey: .numQuantizers, default: d.numQuantizers)
        numResidualLayers = c.value(Int.self, forKey: .numResidualLayers, default: d.numResidualLayers)
        residualKernelSize = c.value(Int.self, forKey: .residualKernelSize, default: d.residualKernelSize)
        ropeTheta = c.value(Double.self, forKey: .ropeTheta, default: d.ropeTheta)
        samplingRate = c.value(Int.self, forKey: .samplingRate, default: d.samplingRate)
        upsamplingRatios = c.value([Int].self, forKey: .upsamplingRatios, default: d.upsamplingRatios)
        useCausalConv = c.value(Bool.self, forKey: .useCausalConv, default: d.useCausalConv)
    }
}

// MARK: - SeaNet convolutions

/// Causal streamable 1-D convolution (`StreamableConv1d` in the Python
/// reference, single-shot path). Left-pads by the effective receptive field
/// minus stride, plus right "extra" padding so the final partial frame is
/// still produced. NLC layout `[batch, time, channels]`.
final class CausalStreamConv1d {
    let weight: MLXArray // (out, kernel, in) — transposed from PyTorch on load
    let bias: MLXArray?
    let stride: Int
    let dilation: Int
    let kernelSize: Int
    let padMode: PadMode

    init(
        checkpoint: Checkpoint, prefix: String,
        stride: Int = 1, dilation: Int = 1, bias: Bool = true, padMode: PadMode = .constant
    ) throws {
        self.weight = try checkpoint.tensor("\(prefix).weight").transposed(0, 2, 1)
        self.bias = bias ? try checkpoint.tensor("\(prefix).bias") : nil
        self.stride = stride
        self.dilation = dilation
        self.kernelSize = weight.dim(1)
        self.padMode = padMode
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let effectiveKernel = (kernelSize - 1) * dilation + 1
        let padTotal = effectiveKernel - stride
        // Right padding so a trailing partial frame is not dropped
        // (`get_extra_padding_for_conv1d` in the Python reference).
        let length = x.dim(1)
        let nFrames = Double(max(length + padTotal - effectiveKernel, 0)) / Double(stride) + 1.0
        let idealLength = (Int(nFrames.rounded(.up)) - 1) * stride + effectiveKernel - padTotal
        let extraPadding = max(0, idealLength - length)
        let padded = MLX.padded(
            x, widths: [IntOrPair(0), IntOrPair((padTotal, extraPadding)), IntOrPair(0)], mode: padMode
        )
        var y = conv1d(padded, weight, stride: stride, dilation: dilation)
        if let bias { y = y + bias }
        return y
    }
}

/// SeaNet residual block: two ELU→conv stages with an identity skip
/// (`true_skip`; the released checkpoint has no shortcut conv).
final class SeanetResnetBlock {
    let convs: [CausalStreamConv1d]

    init(checkpoint: Checkpoint, prefix: String, dilation: Int) throws {
        // Raw checkpoint indices: block.1 = dilated kx1 conv, block.3 = 1x1 conv
        // (block.0/block.2 are the parameter-free ELUs in the PyTorch ModuleList).
        self.convs = try [
            CausalStreamConv1d(checkpoint: checkpoint, prefix: "\(prefix).block.1.conv", dilation: dilation),
            CausalStreamConv1d(checkpoint: checkpoint, prefix: "\(prefix).block.3.conv"),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for conv in convs {
            h = conv(elu(h, alpha: 1.0))
        }
        return h + x
    }
}

/// SeaNet encoder: init conv → 4 × (residual blocks + strided downsample) →
/// final conv. Downsamples 24 kHz audio by 4·5·6·8 = 960× to 25 Hz frames.
final class SeanetEncoder {
    let initConv: CausalStreamConv1d
    let residuals: [[SeanetResnetBlock]]
    let downsamples: [CausalStreamConv1d]
    let finalConv: CausalStreamConv1d

    init(config: SpeechTokenizerEncoderConfig, checkpoint: Checkpoint) throws {
        // Raw checkpoint layout (flat PyTorch ModuleList under encoder.encoder.layers):
        // 0: init conv, then per downsampling stage i: 1+3i residual block,
        // 3+3i strided conv (parameter-free ELUs occupy the gaps), 14: final conv.
        self.initConv = try CausalStreamConv1d(
            checkpoint: checkpoint, prefix: "encoder.encoder.layers.0.conv"
        )
        var residuals = [[SeanetResnetBlock]]()
        var downsamples = [CausalStreamConv1d]()
        // Strides run over the reversed upsampling ratios: [4, 5, 6, 8].
        for (i, ratio) in config.upsamplingRatios.reversed().enumerated() {
            var blocks = [SeanetResnetBlock]()
            var dilation = 1
            for _ in 0..<config.numResidualLayers {
                blocks.append(
                    try SeanetResnetBlock(
                        checkpoint: checkpoint,
                        prefix: "encoder.encoder.layers.\(1 + 3 * i)",
                        dilation: dilation
                    )
                )
                dilation *= config.dilationGrowthRate
            }
            residuals.append(blocks)
            downsamples.append(
                try CausalStreamConv1d(
                    checkpoint: checkpoint,
                    prefix: "encoder.encoder.layers.\(3 + 3 * i).conv",
                    stride: ratio
                )
            )
        }
        self.residuals = residuals
        self.downsamples = downsamples
        self.finalConv = try CausalStreamConv1d(
            checkpoint: checkpoint, prefix: "encoder.encoder.layers.14.conv"
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = initConv(x)
        for (blocks, downsample) in zip(residuals, downsamples) {
            for block in blocks {
                h = block(h)
            }
            h = downsample(elu(h, alpha: 1.0))
        }
        return finalConv(elu(h, alpha: 1.0))
    }
}

// MARK: - Transformer

/// Pre-norm transformer layer with RoPE attention, LayerScale residuals, and
/// a GELU MLP (the Python reference's `TransformerLayer` with `gating=False`).
final class EncoderTransformerLayer {
    let norm1Weight: MLXArray
    let norm1Bias: MLXArray
    let norm2Weight: MLXArray
    let norm2Bias: MLXArray
    let qWeight: MLXArray
    let kWeight: MLXArray
    let vWeight: MLXArray
    let oWeight: MLXArray
    let fc1Weight: MLXArray
    let fc2Weight: MLXArray
    let layerScale1: MLXArray
    let layerScale2: MLXArray

    let numHeads: Int
    let headDim: Int
    let ropeBase: Float
    let normEps: Float

    init(config: SpeechTokenizerEncoderConfig, checkpoint: Checkpoint, prefix: String) throws {
        self.norm1Weight = try checkpoint.tensor("\(prefix).input_layernorm.weight")
        self.norm1Bias = try checkpoint.tensor("\(prefix).input_layernorm.bias")
        self.norm2Weight = try checkpoint.tensor("\(prefix).post_attention_layernorm.weight")
        self.norm2Bias = try checkpoint.tensor("\(prefix).post_attention_layernorm.bias")
        self.qWeight = try checkpoint.tensor("\(prefix).self_attn.q_proj.weight")
        self.kWeight = try checkpoint.tensor("\(prefix).self_attn.k_proj.weight")
        self.vWeight = try checkpoint.tensor("\(prefix).self_attn.v_proj.weight")
        self.oWeight = try checkpoint.tensor("\(prefix).self_attn.o_proj.weight")
        self.fc1Weight = try checkpoint.tensor("\(prefix).mlp.fc1.weight")
        self.fc2Weight = try checkpoint.tensor("\(prefix).mlp.fc2.weight")
        self.layerScale1 = try checkpoint.tensor("\(prefix).self_attn_layer_scale.scale")
        self.layerScale2 = try checkpoint.tensor("\(prefix).mlp_layer_scale.scale")
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.ropeBase = Float(config.ropeTheta)
        // The Python reference hardcodes LayerNorm eps 1e-5 for norm="layer_norm".
        self.normEps = 1e-5
    }

    private func attention(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let (batch, time, _) = (x.dim(0), x.dim(1), x.dim(2))
        // The reference's fused in_proj is the row-concatenation [q; k; v];
        // separate projections are arithmetically identical.
        var q = matmul(x, qWeight.T).reshaped(batch, time, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = matmul(x, kWeight.T).reshaped(batch, time, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = matmul(x, vWeight.T).reshaped(batch, time, numHeads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: ropeBase, scale: 1.0, offset: 0)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: ropeBase, scale: 1.0, offset: 0)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: pow(Float(headDim), -0.5), mask: .array(mask)
        )
        return matmul(out.transposed(0, 2, 1, 3).reshaped(batch, time, numHeads * headDim), oWeight.T)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = MLXFast.layerNorm(x, weight: norm1Weight, bias: norm1Bias, eps: normEps)
        h = x + layerScale1 * attention(h, mask: mask)
        var m = MLXFast.layerNorm(h, weight: norm2Weight, bias: norm2Bias, eps: normEps)
        m = matmul(geluApproximate(matmul(m, fc1Weight.T)), fc2Weight.T)
        return h + layerScale2 * m
    }
}

// MARK: - Residual vector quantization

/// Euclidean codebook (encode path). The lookup table is derived from the
/// stored EMA statistics: `embedding = embedding_sum / max(cluster_usage, eps)`
/// — the Python reference's `EuclideanCodebook.update_in_place()`.
final class EuclideanCodebook {
    static let eps: Float = 1e-5

    let embedding: MLXArray // (codebookSize, dim), fp32
    let halfSquaredNorms: MLXArray // (codebookSize,) = ||e||² / 2, the reference's `_c2`

    init(checkpoint: Checkpoint, prefix: String) throws {
        let embeddingSum = try checkpoint.tensor("\(prefix).embed_sum").asType(.float32)
        let clusterUsage = try checkpoint.tensor("\(prefix).cluster_usage").asType(.float32)
        self.embedding = embeddingSum / maximum(clusterUsage, Self.eps).expandedDimensions(axis: -1)
        self.halfSquaredNorms = embedding.square().sum(axis: -1) / 2
    }

    /// Nearest-codeword indices for `x` of shape `(..., dim)`:
    /// `argmin ||x − e||²` computed as `argmin (||e||²/2 − x·e)`.
    func encode(_ x: MLXArray) -> MLXArray {
        let targetShape = Array(x.shape.dropLast())
        let flat = x.reshaped(-1, x.dim(-1)).asType(.float32)
        let dotProducts = matmul(flat, embedding.T)
        return argMin(halfSquaredNorms - dotProducts, axis: -1).reshaped(targetShape)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        take(embedding, codes.flattened(), axis: 0).reshaped(codes.shape + [embedding.dim(-1)])
    }
}

/// One branch of the split RVQ: a 1×1 input projection followed by
/// `codebooks.count` residual quantization stages.
final class ResidualVectorQuantizerBranch {
    let inputProjWeight: MLXArray // (dim, 1, hidden) MLX conv layout
    let codebooks: [EuclideanCodebook]

    init(checkpoint: Checkpoint, prefix: String, quantizers: Int) throws {
        self.inputProjWeight = try checkpoint.tensor("\(prefix).input_proj.weight").transposed(0, 2, 1)
        self.codebooks = try (0..<quantizers).map { i in
            try EuclideanCodebook(checkpoint: checkpoint, prefix: "\(prefix).layers.\(i).codebook")
        }
    }

    /// Encode `(batch, time, hidden)` features into `(batch, quantizers, time)` codes.
    func encode(_ x: MLXArray) -> MLXArray {
        var residual = conv1d(x, inputProjWeight).asType(.float32)
        var codes = [MLXArray]()
        for codebook in codebooks {
            let indices = codebook.encode(residual) // (B, T)
            codes.append(indices)
            if codes.count < codebooks.count {
                residual = residual - codebook.decode(indices)
            }
        }
        return stacked(codes, axis: 1)
    }
}

// MARK: - Encoder

/// Mimi-style speech-tokenizer encoder: SeaNet conv encoder → causal RoPE
/// transformer → 2× conv downsample (25 Hz → 12.5 Hz) → split residual vector
/// quantizer. Produces the 16-codebook RVQ codes that the ICL voice-clone
/// prefix embeds (~12.5 frames per second of reference audio).
///
/// Port of the Python reference `Qwen3TTSSpeechTokenizerEncoder` (mlx-audio),
/// encode path only; the vocoder decoder half of the checkpoint is not loaded.
public final class SpeechTokenizerEncoder {
    /// Only the first 16 quantizers feed the ICL prompt (1 semantic + 15
    /// acoustic); the released checkpoint stores 32. Later residual stages
    /// cannot affect earlier codes, so the rest are neither loaded nor run.
    public static let validNumQuantizers = 16

    public let config: SpeechTokenizerEncoderConfig
    /// Waveform samples per output code frame (1920 for the released checkpoint).
    public let downsampleRate: Int

    private let seanet: SeanetEncoder
    private let transformerLayers: [EncoderTransformerLayer]
    private let downsample: CausalStreamConv1d
    private let semanticQuantizer: ResidualVectorQuantizerBranch
    private let acousticQuantizer: ResidualVectorQuantizerBranch

    /// Load from a checkpoint snapshot's `speech_tokenizer/` directory
    /// (`config.json` + `*.safetensors`).
    public convenience init(modelDirectory: URL) throws {
        let tokenizerDir = modelDirectory.appendingPathComponent("speech_tokenizer")
        struct TokenizerConfig: Decodable {
            let encoderConfig: SpeechTokenizerEncoderConfig?
            enum CodingKeys: String, CodingKey {
                case encoderConfig = "encoder_config"
            }
        }
        let configURL = tokenizerDir.appendingPathComponent("config.json")
        let tokenizerConfig = try JSONDecoder().decode(TokenizerConfig.self, from: Data(contentsOf: configURL))
        guard let encoderConfig = tokenizerConfig.encoderConfig else {
            throw TTSKitMLXError.invalidCheckpoint(
                "\(configURL.path) has no encoder_config — this checkpoint's speech tokenizer "
                    + "is decoder-only and cannot encode reference audio into RVQ codes."
            )
        }
        let weights = try ModelDirectory.loadWeights(directory: tokenizerDir, glob: "model")
        try self.init(config: encoderConfig, checkpoint: Checkpoint(weights: weights, source: tokenizerDir))
    }

    init(config: SpeechTokenizerEncoderConfig, checkpoint: Checkpoint) throws {
        self.config = config
        self.seanet = try SeanetEncoder(config: config, checkpoint: checkpoint)
        self.transformerLayers = try (0..<config.numHiddenLayers).map { i in
            try EncoderTransformerLayer(
                config: config, checkpoint: checkpoint,
                prefix: "encoder.encoder_transformer.layers.\(i)"
            )
        }
        // 25 Hz SeaNet frames → the tokenizer's 12.5 Hz code frame rate.
        let seanetDownsample = config.upsamplingRatios.reduce(1, *)
        let encoderFrameRate = Double(config.samplingRate) / Double(seanetDownsample)
        let downsampleStride = Int(encoderFrameRate / config.frameRate)
        self.downsampleRate = seanetDownsample * downsampleStride
        self.downsample = try CausalStreamConv1d(
            checkpoint: checkpoint, prefix: "encoder.downsample.conv",
            stride: downsampleStride, bias: false, padMode: .edge
        )
        self.semanticQuantizer = try ResidualVectorQuantizerBranch(
            checkpoint: checkpoint,
            prefix: "encoder.quantizer.semantic_residual_vector_quantizer",
            quantizers: 1
        )
        self.acousticQuantizer = try ResidualVectorQuantizerBranch(
            checkpoint: checkpoint,
            prefix: "encoder.quantizer.acoustic_residual_vector_quantizer",
            quantizers: Self.validNumQuantizers - 1
        )
    }

    /// Encode a waveform into RVQ codes.
    ///
    /// - Parameter audio: waveform, shape `(batch, samples)`, 24 kHz mono in `[-1, 1]`.
    /// - Returns: codes, shape `(batch, 16, codeFrames)` int, one frame per 80 ms.
    public func encode(_ audio: MLXArray) -> MLXArray {
        var h = seanet(audio.expandedDimensions(axis: -1)) // (B, T, hidden)

        // Full causal attention mask (the model was trained with causal
        // attention; the single-shot encode path does not window it).
        let time = h.dim(1)
        var mask = full([time, time], values: MLXArray(-Float.infinity), type: Float.self)
        mask = triu(mask, k: 1).asType(h.dtype)

        for layer in transformerLayers {
            h = layer(h, mask: mask)
        }
        h = downsample(h)

        let semantic = semanticQuantizer.encode(h) // (B, 1, T')
        let acoustic = acousticQuantizer.encode(h) // (B, 15, T')
        return concatenated([semantic, acoustic], axis: 1)
    }
}
