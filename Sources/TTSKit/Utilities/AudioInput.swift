//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

// @preconcurrency: AVAudioConverterInputBlock is @Sendable but AVAudioPCMBuffer
// is not marked Sendable; the block only runs serially inside `convert(to:...)`.
@preconcurrency import AVFoundation
import Foundation
import os

// MARK: - Audio Input

/// Decodes audio files into mono float32 sample arrays for model input.
///
/// Any container/codec readable by `AVAudioFile` is supported (wav, m4a, mp3,
/// caf, ...). Multi-channel input is downmixed to mono and resampled to the
/// requested rate via `AVAudioConverter`. Voice cloning consumes 24 kHz mono
/// (`VoiceCloneEncoder.sampleRate`).
///
/// `AudioOutput` is playback/export-only; this is its decode-side complement.
/// Reference clips are conventionally seconds long, so the whole file is read
/// into memory in one pass.
public enum AudioInput {
    /// Load `url` as mono float32 samples at `sampleRate` Hz.
    ///
    /// - Parameters:
    ///   - url: Audio file to decode (any `AVAudioFile`-readable format).
    ///   - sampleRate: Target sample rate in Hz (e.g. 24000 for voice cloning).
    /// - Returns: Mono samples in `[-1, 1]` at the target rate.
    /// - Throws: `TTSError.generationFailed` if the file cannot be read or converted.
    public static func loadMono(url: URL, sampleRate: Double) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw TTSError.generationFailed(
                "Cannot read audio file at \(url.path): \(error.localizedDescription)"
            )
        }

        let sourceFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0 else {
            throw TTSError.generationFailed("Audio file is empty: \(url.path)")
        }
        guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else {
            throw TTSError.generationFailed("Cannot allocate decode buffer for \(url.path)")
        }
        try file.read(into: sourceBuffer)

        // Fast path: already mono float32 at the target rate.
        if sourceFormat.sampleRate == sampleRate,
            sourceFormat.channelCount == 1,
            sourceFormat.commonFormat == .pcmFormatFloat32,
            let channelData = sourceBuffer.floatChannelData
        {
            return Array(UnsafeBufferPointer(start: channelData[0], count: Int(sourceBuffer.frameLength)))
        }

        guard
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false
            ),
            let converter = AVAudioConverter(from: sourceFormat, to: targetFormat)
        else {
            throw TTSError.generationFailed(
                "Cannot convert \(url.lastPathComponent) (\(sourceFormat)) to \(Int(sampleRate)) Hz mono"
            )
        }

        // Single-shot input block: hand the converter the whole file, then EOS.
        // The output loop keeps draining until the converter reports end-of-stream
        // so the resampler flushes its internal tail. The lock satisfies the
        // block's @Sendable requirement (it is only ever called serially here).
        let inputConsumed = OSAllocatedUnfairLock(initialState: false)
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            let alreadyConsumed = inputConsumed.withLock { consumed -> Bool in
                defer { consumed = true }
                return consumed
            }
            if alreadyConsumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            outStatus.pointee = .haveData
            return sourceBuffer
        }

        var samples: [Float] = []
        samples.reserveCapacity(Int((Double(frameCount) * sampleRate / sourceFormat.sampleRate).rounded(.up)))

        let chunkCapacity: AVAudioFrameCount = 8192
        while true {
            guard let chunk = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: chunkCapacity) else {
                throw TTSError.generationFailed("Cannot allocate conversion buffer for \(url.path)")
            }
            var conversionError: NSError?
            let status = converter.convert(to: chunk, error: &conversionError, withInputFrom: inputBlock)
            if let conversionError {
                throw TTSError.generationFailed(
                    "Audio conversion failed for \(url.path): \(conversionError.localizedDescription)"
                )
            }
            if chunk.frameLength > 0, let channelData = chunk.floatChannelData {
                samples.append(contentsOf: UnsafeBufferPointer(start: channelData[0], count: Int(chunk.frameLength)))
            }
            if status == .endOfStream || status == .error || chunk.frameLength == 0 {
                break
            }
        }

        guard !samples.isEmpty else {
            throw TTSError.generationFailed("Audio conversion produced no samples for \(url.path)")
        }
        return samples
    }
}
