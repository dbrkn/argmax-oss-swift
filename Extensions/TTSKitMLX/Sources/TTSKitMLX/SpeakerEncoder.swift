//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import MLX
import MLXNN

// MARK: - Configuration

/// Configuration for the ECAPA-TDNN speaker encoder (`Qwen3TTSSpeakerEncoderConfig`
/// in the Python reference). Defaults match the released 12hz-0.6b-base checkpoint;
/// the checkpoint's `config.json` only overrides `enc_dim` / `sample_rate`.
public struct SpeakerEncoderConfig: Decodable, Sendable {
    public var melDim = 128
    public var encDim = 1024
    public var encChannels = [512, 512, 512, 512, 1536]
    public var encKernelSizes = [5, 3, 3, 3, 1]
    public var encDilations = [1, 2, 3, 4, 1]
    public var encAttentionChannels = 128
    public var encRes2NetScale = 8
    public var encSEChannels = 128
    public var sampleRate = 24000

    public init() {}

    enum CodingKeys: String, CodingKey {
        case melDim = "mel_dim"
        case encDim = "enc_dim"
        case encChannels = "enc_channels"
        case encKernelSizes = "enc_kernel_sizes"
        case encDilations = "enc_dilations"
        case encAttentionChannels = "enc_attention_channels"
        case encRes2NetScale = "enc_res2net_scale"
        case encSEChannels = "enc_se_channels"
        case sampleRate = "sample_rate"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = SpeakerEncoderConfig()
        melDim = c.value(Int.self, forKey: .melDim, default: d.melDim)
        encDim = c.value(Int.self, forKey: .encDim, default: d.encDim)
        encChannels = c.value([Int].self, forKey: .encChannels, default: d.encChannels)
        encKernelSizes = c.value([Int].self, forKey: .encKernelSizes, default: d.encKernelSizes)
        encDilations = c.value([Int].self, forKey: .encDilations, default: d.encDilations)
        encAttentionChannels = c.value(Int.self, forKey: .encAttentionChannels, default: d.encAttentionChannels)
        encRes2NetScale = c.value(Int.self, forKey: .encRes2NetScale, default: d.encRes2NetScale)
        encSEChannels = c.value(Int.self, forKey: .encSEChannels, default: d.encSEChannels)
        sampleRate = c.value(Int.self, forKey: .sampleRate, default: d.sampleRate)
    }
}

// MARK: - Layers

/// Reflect-pad the time axis (axis 1, NLC layout) by `pad` samples on each
/// side, mirroring without repeating the boundary element (PyTorch
/// `mode="reflect"`). Implemented as a single gather along the time axis.
func reflectPadTime(_ x: MLXArray, pad: Int) -> MLXArray {
    guard pad > 0 else { return x }
    let t = x.dim(1)
    var indices = [Int32]()
    indices.reserveCapacity(t + 2 * pad)
    for i in stride(from: pad, through: 1, by: -1) { indices.append(Int32(i)) }
    indices.append(contentsOf: (0..<t).map(Int32.init))
    for i in stride(from: t - 2, through: t - 1 - pad, by: -1) { indices.append(Int32(i)) }
    return take(x, MLXArray(indices), axis: 1)
}

/// TDNN block: reflect-padded ("same") dilated 1-D convolution + ReLU.
///
/// Layout note: the Python reference keeps activations in NCL and transposes
/// around every conv; this port keeps NLC (`[batch, time, channels]`, MLX's
/// native conv layout) throughout — the arithmetic is identical.
final class TimeDelayNetBlock {
    let weight: MLXArray // (out, kernel, in)
    let bias: MLXArray
    let dilation: Int
    let pad: Int

    init(checkpoint: Checkpoint, prefix: String, kernelSize: Int, dilation: Int) throws {
        self.weight = asMLXConvWeight(try checkpoint.tensor("\(prefix).conv.weight")).asType(.float32)
        self.bias = try checkpoint.tensor("\(prefix).conv.bias").asType(.float32)
        self.dilation = dilation
        self.pad = (kernelSize - 1) * dilation / 2
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = reflectPadTime(x, pad: pad)
        return relu(conv1d(padded, weight, stride: 1, padding: 0, dilation: dilation) + bias)
    }
}

/// Res2Net block: the channel axis is split into `scale` chunks; chunk 0
/// passes through, each later chunk is TDNN-processed with a running sum.
final class Res2NetBlock {
    let blocks: [TimeDelayNetBlock]
    let scale: Int

    init(checkpoint: Checkpoint, prefix: String, scale: Int, kernelSize: Int, dilation: Int) throws {
        self.scale = scale
        self.blocks = try (0..<(scale - 1)).map { i in
            try TimeDelayNetBlock(
                checkpoint: checkpoint, prefix: "\(prefix).blocks.\(i)",
                kernelSize: kernelSize, dilation: dilation
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let chunks = split(x, parts: scale, axis: -1)
        var outputs = [MLXArray]()
        var part = chunks[0]
        outputs.append(part)
        for i in 1..<scale {
            part = i == 1 ? blocks[0](chunks[1]) : blocks[i - 1](chunks[i] + part)
            outputs.append(part)
        }
        return concatenated(outputs, axis: -1)
    }
}

/// Squeeze-and-excitation channel attention over the time-pooled activation.
final class SqueezeExcitationBlock {
    let weight1: MLXArray
    let bias1: MLXArray
    let weight2: MLXArray
    let bias2: MLXArray

    init(checkpoint: Checkpoint, prefix: String) throws {
        self.weight1 = asMLXConvWeight(try checkpoint.tensor("\(prefix).conv1.weight")).asType(.float32)
        self.bias1 = try checkpoint.tensor("\(prefix).conv1.bias").asType(.float32)
        self.weight2 = asMLXConvWeight(try checkpoint.tensor("\(prefix).conv2.weight")).asType(.float32)
        self.bias2 = try checkpoint.tensor("\(prefix).conv2.bias").asType(.float32)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pooled = mean(x, axis: 1, keepDims: true) // (B, 1, C)
        let se = sigmoid(conv1d(relu(conv1d(pooled, weight1) + bias1), weight2) + bias2)
        return x * se
    }
}

/// SE-Res2Net block: TDNN → Res2Net → TDNN → SE, with a residual connection.
final class SqueezeExcitationRes2NetBlock {
    let tdnn1: TimeDelayNetBlock
    let res2netBlock: Res2NetBlock
    let tdnn2: TimeDelayNetBlock
    let seBlock: SqueezeExcitationBlock

    init(checkpoint: Checkpoint, prefix: String, config: SpeakerEncoderConfig, kernelSize: Int, dilation: Int) throws {
        self.tdnn1 = try TimeDelayNetBlock(
            checkpoint: checkpoint, prefix: "\(prefix).tdnn1", kernelSize: 1, dilation: 1
        )
        self.res2netBlock = try Res2NetBlock(
            checkpoint: checkpoint, prefix: "\(prefix).res2net_block",
            scale: config.encRes2NetScale, kernelSize: kernelSize, dilation: dilation
        )
        self.tdnn2 = try TimeDelayNetBlock(
            checkpoint: checkpoint, prefix: "\(prefix).tdnn2", kernelSize: 1, dilation: 1
        )
        self.seBlock = try SqueezeExcitationBlock(checkpoint: checkpoint, prefix: "\(prefix).se_block")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = tdnn1(x)
        h = res2netBlock(h)
        h = tdnn2(h)
        h = seBlock(h)
        return h + residual
    }
}

/// Attentive statistics pooling: attention-weighted mean and standard
/// deviation over time, conditioned on the global statistics.
final class AttentiveStatisticsPooling {
    static let eps: Float = 1e-12

    let tdnn: TimeDelayNetBlock
    let convWeight: MLXArray
    let convBias: MLXArray

    init(checkpoint: Checkpoint, prefix: String) throws {
        self.tdnn = try TimeDelayNetBlock(
            checkpoint: checkpoint, prefix: "\(prefix).tdnn", kernelSize: 1, dilation: 1
        )
        self.convWeight = asMLXConvWeight(try checkpoint.tensor("\(prefix).conv.weight")).asType(.float32)
        self.convBias = try checkpoint.tensor("\(prefix).conv.bias").asType(.float32)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Global statistics, broadcast over time. (B, T, C) layout.
        let globalMean = mean(x, axis: 1, keepDims: true)
        let globalStd = sqrt(variance(x, axis: 1, keepDims: true) + Self.eps)
        let features = concatenated(
            [
                x,
                broadcast(globalMean, to: x.shape),
                broadcast(globalStd, to: x.shape),
            ],
            axis: -1
        )

        var attention = tanh(tdnn(features))
        attention = conv1d(attention, convWeight) + convBias
        attention = softmax(attention, axis: 1) // over time

        let weightedMean = sum(attention * x, axis: 1, keepDims: true)
        let weightedVar = sum(attention * square(x - weightedMean), axis: 1, keepDims: true)
        let weightedStd = sqrt(maximum(weightedVar, Self.eps))
        return concatenated([weightedMean, weightedStd], axis: -1) // (B, 1, 2C)
    }
}

// MARK: - Encoder

/// ECAPA-TDNN speaker encoder for Qwen3-TTS voice cloning.
///
/// Variable-length by construction: statistics pooling aggregates over however
/// many mel frames it is given, so — unlike the fixed-window CoreML asset —
/// references need no tiling, padding, or truncation.
///
/// Port of the Python reference `Qwen3TTSSpeakerEncoder` (mlx-audio); weights
/// come from the `speaker_encoder.*` keys of a Base-family checkpoint's root
/// safetensors and are unquantized (bf16, upcast to fp32 on load — MLX would
/// promote them against the fp32 mel anyway).
public final class SpeakerEncoder {
    public let config: SpeakerEncoderConfig

    private let initialBlock: TimeDelayNetBlock
    private let seRes2NetBlocks: [SqueezeExcitationRes2NetBlock]
    private let mfa: TimeDelayNetBlock
    private let asp: AttentiveStatisticsPooling
    private let fcWeight: MLXArray
    private let fcBias: MLXArray

    /// Load from a checkpoint snapshot directory (root `config.json` +
    /// `model*.safetensors` with `speaker_encoder.`-prefixed keys).
    public convenience init(modelDirectory: URL) throws {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        struct RootConfig: Decodable {
            let speakerEncoderConfig: SpeakerEncoderConfig?
            enum CodingKeys: String, CodingKey {
                case speakerEncoderConfig = "speaker_encoder_config"
            }
        }
        let root = try JSONDecoder().decode(RootConfig.self, from: Data(contentsOf: configURL))

        // sanitize(): keep speaker_encoder.* keys, strip the prefix. Conv
        // layout normalization happens per-tensor at layer init.
        let prefix = "speaker_encoder."
        let raw = try ModelDirectory.loadWeights(directory: modelDirectory, glob: "model")
        var weights = [String: MLXArray]()
        for (key, value) in raw where key.hasPrefix(prefix) {
            weights[String(key.dropFirst(prefix.count))] = value
        }
        guard !weights.isEmpty else {
            throw TTSKitMLXError.invalidCheckpoint(
                "No speaker_encoder weights in \(modelDirectory.path). Only Base-family "
                    + "Qwen3-TTS checkpoints include the ECAPA speaker encoder — use a repo "
                    + "like \(ModelDirectory.defaultRepoID)."
            )
        }
        try self.init(
            config: root.speakerEncoderConfig ?? SpeakerEncoderConfig(),
            checkpoint: Checkpoint(weights: weights, source: modelDirectory)
        )
    }

    init(config: SpeakerEncoderConfig, checkpoint: Checkpoint) throws {
        self.config = config
        self.initialBlock = try TimeDelayNetBlock(
            checkpoint: checkpoint, prefix: "blocks.0",
            kernelSize: config.encKernelSizes[0], dilation: config.encDilations[0]
        )
        self.seRes2NetBlocks = try (1..<(config.encChannels.count - 1)).map { i in
            try SqueezeExcitationRes2NetBlock(
                checkpoint: checkpoint, prefix: "blocks.\(i)", config: config,
                kernelSize: config.encKernelSizes[i], dilation: config.encDilations[i]
            )
        }
        self.mfa = try TimeDelayNetBlock(
            checkpoint: checkpoint, prefix: "mfa",
            kernelSize: config.encKernelSizes.last!, dilation: config.encDilations.last!
        )
        self.asp = try AttentiveStatisticsPooling(checkpoint: checkpoint, prefix: "asp")
        self.fcWeight = asMLXConvWeight(try checkpoint.tensor("fc.weight")).asType(.float32)
        self.fcBias = try checkpoint.tensor("fc.bias").asType(.float32)
        eval(fcWeight)
    }

    /// Encode a mel spectrogram into a speaker embedding.
    ///
    /// - Parameter mel: log-mel spectrogram, shape `(batch, timeFrames, melDim)`.
    /// - Returns: speaker embedding, shape `(batch, encDim)`.
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = initialBlock(mel)
        var seOutputs = [MLXArray]()
        for block in seRes2NetBlocks {
            x = block(x)
            seOutputs.append(x)
        }
        // Multi-layer feature aggregation over the SE-Res2Net outputs only.
        x = mfa(concatenated(seOutputs, axis: -1))
        x = asp(x) // (B, 1, 2C)
        x = conv1d(x, fcWeight) + fcBias // (B, 1, encDim)
        return x.squeezed(axis: 1)
    }
}
