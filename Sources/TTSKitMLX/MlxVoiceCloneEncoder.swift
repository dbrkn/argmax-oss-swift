//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

#if canImport(MLX)

import Foundation
import MLX
import TTSKit

/// Encodes a reference clip into a ``TTSKit/VoiceClonePrompt`` using the
/// MLX-Swift ports of the Qwen3-TTS voice-clone encoders.
///
/// Unlike TTSKit's CoreML `VoiceCloneEncoder`, both encoders are
/// variable-length: the ECAPA speaker encoder pools statistics over the
/// full-length mel (no tiling/window), and the Mimi encoder consumes the raw
/// waveform (no zero-padded window, so there is no padded-tail code trim
/// either). Any reference duration up to ``maxReferenceSeconds`` works.
///
/// The produced prompt is byte-compatible with the CoreML path — feed its
/// JSON to `argmax-cli tts --voice-clone-prompt` on any platform.
public final class MlxVoiceCloneEncoder {
    public let speakerEncoder: SpeakerEncoder
    public let speechTokenizerEncoder: SpeechTokenizerEncoder
    public let melSpectrogram: MelSpectrogram
    /// Reference-length cap in seconds. MLX encode peak Metal memory scales
    /// ~90 MB per reference second (measured: 0.7 GB @ 3 s → 2.9 GB @ 32 s),
    /// so an unbounded reference would exhaust GPU memory (~27 GB for a
    /// 5-minute clip). References over the cap throw — never silently truncate.
    public let maxReferenceSeconds: Double

    public static let sampleRate = 24000

    public init(
        speakerEncoder: SpeakerEncoder,
        speechTokenizerEncoder: SpeechTokenizerEncoder,
        melConfig: MelSpectrogramConfig = MelSpectrogramConfig(),
        maxReferenceSeconds: Double = 120
    ) {
        self.speakerEncoder = speakerEncoder
        self.speechTokenizerEncoder = speechTokenizerEncoder
        self.melSpectrogram = MelSpectrogram(config: melConfig)
        self.maxReferenceSeconds = maxReferenceSeconds
    }

    /// Load both encoders from a Base-family Qwen3-TTS MLX checkpoint
    /// snapshot directory (see ``ModelDirectory``). Pass `nil` to resolve the
    /// default repo's cached Hugging Face snapshot.
    public convenience init(
        modelDirectory: URL? = nil,
        melConfig: MelSpectrogramConfig = MelSpectrogramConfig(),
        maxReferenceSeconds: Double = 120
    ) throws {
        let directory = try modelDirectory ?? ModelDirectory.defaultSnapshot()
        self.init(
            speakerEncoder: try SpeakerEncoder(modelDirectory: directory),
            speechTokenizerEncoder: try SpeechTokenizerEncoder(modelDirectory: directory),
            melConfig: melConfig,
            maxReferenceSeconds: maxReferenceSeconds
        )
    }

    /// Encode a 24 kHz mono reference waveform.
    ///
    /// - Parameters:
    ///   - waveform: reference samples in `[-1, 1]` at 24 kHz.
    ///   - includeReferenceCodes: `true` for ICL cloning, `false` for
    ///     x-vector-only.
    ///   - referenceText: transcript of the reference clip, stored in the
    ///     prompt for ICL prefix assembly.
    public func encode(
        _ waveform: [Float],
        includeReferenceCodes: Bool,
        referenceText: String? = nil
    ) throws -> VoiceClonePrompt {
        guard !waveform.isEmpty else {
            throw TTSKitMLXError.invalidInput("Reference waveform is empty")
        }
        let seconds = Double(waveform.count) / Double(Self.sampleRate)
        guard seconds <= maxReferenceSeconds else {
            throw TTSKitMLXError.referenceTooLong(
                String(
                    format: "Reference is %.1f s but maxReferenceSeconds is %.0f s. "
                        + "MLX encode peak Metal memory scales ~90 MB per reference second "
                        + "(0.7 GB @ 3 s → 2.9 GB @ 32 s), so longer references risk "
                        + "out-of-memory; trim the reference or raise the cap explicitly.",
                    seconds, maxReferenceSeconds
                )
            )
        }

        let speakerEmbedding = encodeSpeaker(waveform)
        guard includeReferenceCodes else {
            return VoiceClonePrompt(speakerEmbedding: speakerEmbedding, referenceText: referenceText)
        }

        let (codes, frames) = encodeAudioCodes(waveform)
        // Release the encode's Metal buffer pool before generation: MLX caches
        // freed buffers, so a long-reference encode (~90 MB/s peak, ~19 GB at
        // 210 s) otherwise stays resident through the talker loop and can OOM
        // the process on 32–36 GB machines — especially with guardrail
        // restarts re-decoding repeatedly.
        GPU.clearCache()
        return VoiceClonePrompt(
            speakerEmbedding: speakerEmbedding,
            referenceCodes: codes,
            referenceCodeFrames: frames,
            referenceText: referenceText
        )
    }

    /// ECAPA x-vector for a 24 kHz mono waveform, shape `(encDim,)`.
    ///
    /// The full-length mel is encoded directly — no tiling of short
    /// references, no truncation of long ones.
    public func encodeSpeaker(_ waveform: [Float]) -> [Float] {
        let (melValues, melFrames) = melSpectrogram.process(waveform)
        // Row-major (numMels, T) → the encoder's (1, T, numMels) layout.
        let mel = MLXArray(melValues, [1, melSpectrogram.config.numMels, melFrames])
            .transposed(0, 2, 1)
        let embedding = speakerEncoder(mel).asType(.float32)
        eval(embedding)
        return embedding.asArray(Float.self)
    }

    /// RVQ codes for a 24 kHz mono waveform: row-major `(16, frames)` int32,
    /// one frame per 80 ms of reference audio.
    public func encodeAudioCodes(_ waveform: [Float]) -> (codes: [Int32], frames: Int) {
        let audio = MLXArray(waveform, [1, waveform.count])
        let codes = speechTokenizerEncoder.encode(audio).squeezed(axis: 0).asType(.int32)
        eval(codes)
        return (codes.asArray(Int32.self), codes.dim(-1))
    }
}

#endif // canImport(MLX)
