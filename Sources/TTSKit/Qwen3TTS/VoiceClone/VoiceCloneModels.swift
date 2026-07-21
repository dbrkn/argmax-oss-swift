//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

// MARK: - SpeakerEncoder

/// ECAPA-TDNN speaker encoder (x-vector) backed by a CoreML model.
///
/// Input `mel_spectrogram (1, melDim, 1, melLength)` float16; output
/// `speaker_embedding (1, embeddingDim)`. The mel window is a compile-time
/// constant of the asset (published variant: 960 frames ≈ 10.24 s @ 24 kHz).
///
/// Thread safety: dimensions are set once in `loadModel()` and read-only
/// thereafter; `MLModel.prediction()` is thread-safe.
public class Qwen3SpeakerEncoder: MLModelLoading, @unchecked Sendable {
    @Protected public var model: MLModel?

    public private(set) var melDim: Int = 128
    public private(set) var melLength: Int = 960
    public private(set) var embeddingDim: Int = 1024

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        guard !prewarmMode else { return }

        if let constraint = loaded.modelDescription.inputDescriptionsByName["mel_spectrogram"]?.multiArrayConstraint {
            let shape = constraint.shape.map(\.intValue)
            guard shape.count == 4 else {
                throw TTSError.modelLoadingFailed("SpeakerEncoder: unexpected mel_spectrogram shape \(shape)")
            }
            melDim = shape[1]
            melLength = shape[3]
        }
        self.model = loaded
    }

    /// - Parameter mel: row-major `(melDim, melLength)` log-mel matrix.
    /// - Returns: the x-vector `(embeddingDim,)`.
    public func encode(mel: [Float]) async throws -> [Float] {
        guard let model else { throw TTSError.generationFailed("SpeakerEncoder model not loaded") }
        guard mel.count == melDim * melLength else {
            throw TTSError.generationFailed(
                "SpeakerEncoder: expected \(melDim * melLength) mel values, got \(mel.count)"
            )
        }
        let input = try VoiceCloneArrays.makeFloatArray(
            mel, shape: [1, NSNumber(value: melDim), 1, NSNumber(value: melLength)]
        )
        let provider = try MLDictionaryFeatureProvider(
            dictionary: ["mel_spectrogram": MLFeatureValue(multiArray: input)]
        )
        let output = try await model.asyncPrediction(from: provider)
        guard let embedding = output.featureValue(for: "speaker_embedding")?.multiArrayValue else {
            throw TTSError.generationFailed("SpeakerEncoder: missing speaker_embedding output")
        }
        embeddingDim = embedding.count
        return VoiceCloneArrays.toFloats(embedding)
    }

    public func unloadModel() {
        model = nil
    }
}

// MARK: - SpeechEncoder

/// Mimi continuous speech encoder backed by a CoreML model.
///
/// Input `audio_waveform (1, 1, 1, audioLength)` float16 (fixed window,
/// published variant: 240000 samples = 10 s @ 24 kHz); output
/// `projected_embeddings (1, 2 * vqDim, 1, numCodes)` — channels `[..<vqDim]`
/// are the semantic projection, `[vqDim...]` the acoustic projection.
public class Qwen3SpeechEncoder: MLModelLoading, @unchecked Sendable {
    @Protected public var model: MLModel?

    public private(set) var audioLength: Int = 240_000
    public private(set) var vqDim: Int = 256
    public private(set) var numCodes: Int = 125

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        guard !prewarmMode else { return }

        if let constraint = loaded.modelDescription.inputDescriptionsByName["audio_waveform"]?.multiArrayConstraint {
            let shape = constraint.shape.map(\.intValue)
            guard shape.count == 4 else {
                throw TTSError.modelLoadingFailed("SpeechEncoder: unexpected audio_waveform shape \(shape)")
            }
            audioLength = shape[3]
        }
        if let constraint = loaded.modelDescription.outputDescriptionsByName["projected_embeddings"]?.multiArrayConstraint {
            let shape = constraint.shape.map(\.intValue)
            if shape.count == 4 {
                vqDim = shape[1] / 2
                numCodes = shape[3]
            }
        }
        self.model = loaded
    }

    /// Projected embeddings for a full fixed-size window.
    ///
    /// - Parameter waveform: exactly `audioLength` samples (pre-padded).
    /// - Returns: row-major `(2 * vqDim, numCodes)` matrix.
    public func encode(waveform: [Float]) async throws -> [Float] {
        guard let model else { throw TTSError.generationFailed("SpeechEncoder model not loaded") }
        guard waveform.count == audioLength else {
            throw TTSError.generationFailed(
                "SpeechEncoder: expected \(audioLength) samples, got \(waveform.count)"
            )
        }
        let input = try VoiceCloneArrays.makeFloatArray(
            waveform, shape: [1, 1, 1, NSNumber(value: audioLength)]
        )
        let provider = try MLDictionaryFeatureProvider(
            dictionary: ["audio_waveform": MLFeatureValue(multiArray: input)]
        )
        let output = try await model.asyncPrediction(from: provider)
        guard let projected = output.featureValue(for: "projected_embeddings")?.multiArrayValue else {
            throw TTSError.generationFailed("SpeechEncoder: missing projected_embeddings output")
        }
        return VoiceCloneArrays.toFloats(projected)
    }

    public func unloadModel() {
        model = nil
    }
}

// MARK: - SpeechEncoderRVQ

/// Per-codebook residual vector quantization step backed by a CoreML model.
///
/// Inputs `residual (1, vqDim, 1, numCodes)` + `codebook_idx int32 (1,)`;
/// outputs `new_residual` (same shape) and `code (1, numCodes)`.
///
/// The full RVQ encode runs two branches over the SpeechEncoder projections:
/// the semantic branch (codebook 0) on channels `[..<vqDim]`, then the
/// acoustic branch **resets** the residual to channels `[vqDim...]` and runs
/// codebooks 1..<16.
public class Qwen3SpeechEncoderRVQ: MLModelLoading, @unchecked Sendable {
    @Protected public var model: MLModel?

    /// Number of semantic quantizers (codebook 0).
    public static let numSemanticQuantizers = 1
    public private(set) var numQuantizers: Int = 16

    public init() {}

    public func loadModel(at url: URL, computeUnits: MLComputeUnits, prewarmMode: Bool = false) async throws {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        let loaded = try await MLModel.load(contentsOf: url, configuration: modelConfig)

        guard !prewarmMode else { return }

        self.model = loaded
    }

    /// Encode SpeechEncoder projections into RVQ codes.
    ///
    /// - Parameter projected: row-major `(2 * vqDim, numCodes)` from
    ///   ``Qwen3SpeechEncoder/encode(waveform:)``.
    /// - Returns: `(codes, quantizers, frames)` with `codes` row-major
    ///   `(quantizers, frames)`.
    public func encode(projected: [Float]) async throws -> (codes: [Int32], quantizers: Int, frames: Int) {
        guard let model else { throw TTSError.generationFailed("SpeechEncoderRVQ model not loaded") }
        guard
            let constraint = model.modelDescription.inputDescriptionsByName["residual"]?.multiArrayConstraint
        else {
            throw TTSError.modelLoadingFailed("SpeechEncoderRVQ: missing residual input constraint")
        }
        let shape = constraint.shape.map(\.intValue)
        let vqDim = shape[1]
        let numCodes = shape[3]
        guard projected.count == 2 * vqDim * numCodes else {
            throw TTSError.generationFailed(
                "SpeechEncoderRVQ: expected \(2 * vqDim * numCodes) projected values, got \(projected.count)"
            )
        }

        let semantic = Array(projected[0..<(vqDim * numCodes)])
        let acoustic = Array(projected[(vqDim * numCodes)...])

        var codes = [Int32](repeating: 0, count: numQuantizers * numCodes)

        // Semantic branch: codebook 0.
        var residual = semantic
        for q in 0..<Self.numSemanticQuantizers {
            let (newResidual, stepCodes) = try await step(
                residual: residual, codebookIdx: Int32(q), vqDim: vqDim, numCodes: numCodes, model: model
            )
            residual = newResidual
            for t in 0..<numCodes { codes[q * numCodes + t] = stepCodes[t] }
        }

        // Acoustic branch: reset residual, codebooks 1..<numQuantizers.
        residual = acoustic
        for q in Self.numSemanticQuantizers..<numQuantizers {
            let (newResidual, stepCodes) = try await step(
                residual: residual, codebookIdx: Int32(q), vqDim: vqDim, numCodes: numCodes, model: model
            )
            residual = newResidual
            for t in 0..<numCodes { codes[q * numCodes + t] = stepCodes[t] }
        }

        return (codes, numQuantizers, numCodes)
    }

    private func step(
        residual: [Float], codebookIdx: Int32, vqDim: Int, numCodes: Int, model: MLModel
    ) async throws -> ([Float], [Int32]) {
        let residualArray = try VoiceCloneArrays.makeFloatArray(
            residual, shape: [1, NSNumber(value: vqDim), 1, NSNumber(value: numCodes)]
        )
        let idxArray = try EmbedUtilities.makeInt32Array([codebookIdx])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "residual": MLFeatureValue(multiArray: residualArray),
            "codebook_idx": MLFeatureValue(multiArray: idxArray),
        ])
        let output = try await model.asyncPrediction(from: provider)
        guard
            let newResidual = output.featureValue(for: "new_residual")?.multiArrayValue,
            let code = output.featureValue(for: "code")?.multiArrayValue
        else {
            throw TTSError.generationFailed("SpeechEncoderRVQ: missing outputs")
        }
        let codeFloats = VoiceCloneArrays.toFloats(code)
        return (VoiceCloneArrays.toFloats(newResidual), codeFloats.map { Int32($0.rounded()) })
    }

    public func unloadModel() {
        model = nil
    }
}

// MARK: - Array helpers

enum VoiceCloneArrays {
    static func makeFloatArray(_ values: [Float], shape: [NSNumber]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        values.withUnsafeBufferPointer { src in
            array.withUnsafeMutableBytes { dst, _ in
                dst.bindMemory(to: Float.self).baseAddress!.update(from: src.baseAddress!, count: values.count)
            }
        }
        return array
    }

    static func toFloats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var out = [Float](repeating: 0, count: count)
        switch array.dataType {
        case .float32:
            array.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float.self)
                out = Array(src.prefix(count))
            }
        case .float16:
            array.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float16.self)
                for i in 0..<count { out[i] = Float(src[i]) }
            }
        default:
            for i in 0..<count { out[i] = array[i].floatValue }
        }
        return out
    }
}
