//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
import CoreML
import Foundation

// MARK: - Prompt value type

/// Everything derived from a reference clip that the generation prefix needs.
///
/// Produced by ``VoiceCloneEncoder/encode(_:includeReferenceCodes:)``, or
/// constructed directly from precomputed values (e.g. encoded server-side, or
/// exported from the Python reference implementation for parity testing).
public struct VoiceClonePrompt: Sendable {
    /// ECAPA-TDNN x-vector, substituted into the speaker slot of the codec
    /// track (shape `(embeddingDim,)`).
    public let speakerEmbedding: [Float]
    /// Reference RVQ codes for ICL mode, `numQuantizers` rows by
    /// `frames` columns, row-major. `nil` for x-vector-only cloning.
    public let referenceCodes: [Int32]?
    public let referenceCodeFrames: Int
    /// Transcript of the reference clip. Required when `referenceCodes` is set.
    public let referenceText: String?

    public init(
        speakerEmbedding: [Float],
        referenceCodes: [Int32]? = nil,
        referenceCodeFrames: Int = 0,
        referenceText: String? = nil
    ) {
        self.speakerEmbedding = speakerEmbedding
        self.referenceCodes = referenceCodes
        self.referenceCodeFrames = referenceCodeFrames
        self.referenceText = referenceText
    }
}

// MARK: - Encoder

/// Encodes a reference clip into a ``VoiceClonePrompt`` using the CoreML
/// voice-clone assets (SpeakerEncoder, SpeechEncoder, SpeechEncoderRVQ).
///
/// The assets have fixed compile-time windows; the input shaping mirrors the
/// Python reference exactly:
/// - SpeakerEncoder: short references are **tiled** (not zero-padded) to the
///   mel window so ECAPA pools statistics over real speech, then the mel is
///   right-padded/trimmed to the fixed frame count.
/// - SpeechEncoder: the waveform is right zero-padded/trimmed to the fixed
///   sample window; after RVQ encoding, code frames corresponding to the
///   zero-padded tail are trimmed off (skipping this poisons the ICL prefix
///   with silence codes).
public final class VoiceCloneEncoder {
    public let speakerEncoder: Qwen3SpeakerEncoder
    public let speechEncoder: Qwen3SpeechEncoder?
    public let rvqEncoder: Qwen3SpeechEncoderRVQ?
    public let melSpectrogram: MelSpectrogram

    public static let sampleRate = 24000

    public init(
        speakerEncoder: Qwen3SpeakerEncoder,
        speechEncoder: Qwen3SpeechEncoder?,
        rvqEncoder: Qwen3SpeechEncoderRVQ?,
        melConfig: MelSpectrogramConfig = MelSpectrogramConfig()
    ) {
        self.speakerEncoder = speakerEncoder
        self.speechEncoder = speechEncoder
        self.rvqEncoder = rvqEncoder
        self.melSpectrogram = MelSpectrogram(config: melConfig)
    }

    /// Encode a 24 kHz mono reference waveform.
    ///
    /// - Parameters:
    ///   - waveform: reference samples in `[-1, 1]` at 24 kHz.
    ///   - includeReferenceCodes: `true` for ICL cloning (requires the
    ///     SpeechEncoder + RVQ assets), `false` for x-vector-only.
    public func encode(
        _ waveform: [Float],
        includeReferenceCodes: Bool
    ) async throws -> VoiceClonePrompt {
        guard !waveform.isEmpty else {
            throw TTSError.generationFailed("Reference waveform is empty")
        }

        // --- x-vector ---
        let melWindow = speakerEncoder.melLength * melSpectrogram.config.hopSize
        let tiled = Self.tileToWindow(waveform, targetSamples: melWindow)
        let (melValues, melFrames) = melSpectrogram.process(tiled)
        let melInput = Self.padOrTrimFrames(
            melValues, frames: melFrames,
            numMels: melSpectrogram.config.numMels,
            targetFrames: speakerEncoder.melLength
        )
        let speakerEmbedding = try await speakerEncoder.encode(mel: melInput)

        guard includeReferenceCodes else {
            return VoiceClonePrompt(speakerEmbedding: speakerEmbedding)
        }
        guard let speechEncoder, let rvqEncoder else {
            throw TTSError.invalidConfiguration(
                "ICL voice cloning requires the SpeechEncoder and SpeechEncoderRVQ assets"
            )
        }

        // --- reference RVQ codes ---
        let window = Self.padOrTrim(waveform, targetLength: speechEncoder.audioLength)
        let projected = try await speechEncoder.encode(waveform: window)
        let codes = try await rvqEncoder.encode(projected: projected)

        // Trim code frames that correspond to the zero-padded tail.
        let valid = Self.validFrameCount(
            realSamples: waveform.count,
            windowSamples: speechEncoder.audioLength,
            numCodes: codes.frames
        )
        let trimmed = Self.trimFrames(codes.codes, quantizers: codes.quantizers, frames: codes.frames, keep: valid)

        return VoiceClonePrompt(
            speakerEmbedding: speakerEmbedding,
            referenceCodes: trimmed,
            referenceCodeFrames: valid
        )
    }

    // MARK: - Window shaping (unit-tested against Python goldens)

    /// Tile a short waveform to fill `targetSamples` (then trim); pass-through
    /// when already long enough. Mirrors `np.tile(wav, reps)[:target]`.
    static func tileToWindow(_ waveform: [Float], targetSamples: Int) -> [Float] {
        guard !waveform.isEmpty, waveform.count < targetSamples else { return waveform }
        var out = [Float]()
        out.reserveCapacity(targetSamples)
        while out.count < targetSamples {
            out.append(contentsOf: waveform.prefix(targetSamples - out.count))
        }
        return out
    }

    /// Right zero-pad or right-trim to `targetLength`.
    static func padOrTrim(_ x: [Float], targetLength: Int) -> [Float] {
        if x.count == targetLength { return x }
        if x.count > targetLength { return Array(x.prefix(targetLength)) }
        return x + [Float](repeating: 0, count: targetLength - x.count)
    }

    /// Right zero-pad or trim a row-major `(numMels, frames)` matrix along the
    /// frame axis to `targetFrames`.
    static func padOrTrimFrames(
        _ mel: [Float], frames: Int, numMels: Int, targetFrames: Int
    ) -> [Float] {
        if frames == targetFrames { return mel }
        var out = [Float](repeating: 0, count: numMels * targetFrames)
        let copyFrames = min(frames, targetFrames)
        for m in 0..<numMels {
            for t in 0..<copyFrames {
                out[m * targetFrames + t] = mel[m * frames + t]
            }
        }
        return out
    }

    /// Number of RVQ code frames that correspond to real (non-padded) audio.
    static func validFrameCount(realSamples: Int, windowSamples: Int, numCodes: Int) -> Int {
        let framesPerSample = Double(numCodes) / Double(windowSamples)
        return min(numCodes, Int((Double(realSamples) * framesPerSample).rounded()))
    }

    /// Keep the first `keep` frames of a row-major `(quantizers, frames)` code matrix.
    static func trimFrames(_ codes: [Int32], quantizers: Int, frames: Int, keep: Int) -> [Int32] {
        guard keep < frames else { return codes }
        var out = [Int32](repeating: 0, count: quantizers * keep)
        for q in 0..<quantizers {
            for t in 0..<keep {
                out[q * keep + t] = codes[q * frames + t]
            }
        }
        return out
    }
}
