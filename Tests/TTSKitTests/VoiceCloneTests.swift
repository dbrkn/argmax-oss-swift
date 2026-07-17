//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
@testable import TTSKit
import XCTest

/// Golden-value tests for the voice-clone preprocessing front-end.
///
/// Goldens are exported from the Python reference implementation by
/// `scripts/export_voice_clone_goldens.py` (see that script's header for the
/// invocation) and cover the numerically bug-prone pieces: the mel
/// spectrogram DSP, the SpeakerEncoder tiling rule, and the RVQ
/// padded-tail trim bookkeeping.
final class VoiceCloneGoldenTests: XCTestCase {
    struct Goldens: Decodable {
        struct MelParams: Decodable {
            let nFFT: Int
            let numMels: Int
            let hopSize: Int
            let winSize: Int
            let fmin: Float
            let fmax: Float
            let samplingRate: Int
        }

        let melParams: MelParams
        let refSamples: Int
        let melShape: [Int]
        let melChecksumRowMeans: [Float]
        let melFirstFrame: [Float]
        let melLastFrame: [Float]
        let speakerMelShape: [Int]
        let speakerTiles: Int
        let speechWindowSamples: Int
        let numCodes: Int
        let validFrames: Int
    }

    private func goldenURL(_ name: String, _ ext: String) throws -> URL {
        guard
            let url = Bundle.module.url(
                forResource: name, withExtension: ext,
                subdirectory: "Resources/VoiceCloneGoldens"
            )
        else {
            throw XCTSkip("Voice-clone goldens not bundled (run scripts/export_voice_clone_goldens.py)")
        }
        return url
    }

    private func loadFloats(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: goldenURL(name, "bin"))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func loadGoldens() throws -> Goldens {
        let data = try Data(contentsOf: goldenURL("goldens", "json"))
        return try JSONDecoder().decode(Goldens.self, from: data)
    }

    func testMelSpectrogramMatchesPythonReference() throws {
        let goldens = try loadGoldens()
        let waveform = try loadFloats("ref_audio")
        XCTAssertEqual(waveform.count, goldens.refSamples)

        var config = MelSpectrogramConfig()
        config.nFFT = goldens.melParams.nFFT
        config.numMels = goldens.melParams.numMels
        config.hopSize = goldens.melParams.hopSize
        config.winSize = goldens.melParams.winSize
        config.fmin = goldens.melParams.fmin
        config.fmax = goldens.melParams.fmax
        config.samplingRate = goldens.melParams.samplingRate

        let mel = MelSpectrogram(config: config)
        let (values, frames) = mel.process(waveform)

        XCTAssertEqual(goldens.melShape, [config.numMels, frames])

        let reference = try loadFloats("mel")
        XCTAssertEqual(reference.count, values.count)

        // Element-wise comparison in log-mel space. The Python reference runs
        // torch float32 STFT; vDSP differs only by accumulation order.
        var maxAbsDiff: Float = 0
        for i in 0..<values.count {
            maxAbsDiff = max(maxAbsDiff, abs(values[i] - reference[i]))
        }
        XCTAssertLessThan(maxAbsDiff, 5e-3, "log-mel max abs diff too large")

        // Spot-check the exported fingerprints too, so a stale .bin is caught.
        for (m, expected) in goldens.melFirstFrame.enumerated() {
            XCTAssertEqual(values[m * frames], expected, accuracy: 5e-3)
        }
        for (m, expected) in goldens.melLastFrame.enumerated() {
            XCTAssertEqual(values[m * frames + frames - 1], expected, accuracy: 5e-3)
        }
    }

    func testSpeakerWindowTilingMatchesPythonReference() throws {
        let goldens = try loadGoldens()
        let waveform = try loadFloats("ref_audio")

        let config = MelSpectrogramConfig()
        let melFrames = goldens.speakerMelShape[1]
        let tiled = VoiceCloneEncoder.tileToWindow(
            waveform, targetSamples: melFrames * config.hopSize
        )
        XCTAssertEqual(tiled.count, melFrames * config.hopSize)

        let mel = MelSpectrogram(config: config)
        let (values, frames) = mel.process(tiled)
        let padded = VoiceCloneEncoder.padOrTrimFrames(
            values, frames: frames, numMels: config.numMels, targetFrames: melFrames
        )

        let reference = try loadFloats("speaker_mel_input")
        XCTAssertEqual(reference.count, padded.count)
        var maxAbsDiff: Float = 0
        for i in 0..<padded.count {
            maxAbsDiff = max(maxAbsDiff, abs(padded[i] - reference[i]))
        }
        XCTAssertLessThan(maxAbsDiff, 5e-3)
    }

    func testSpeechWindowPaddingMatchesPythonReference() throws {
        let goldens = try loadGoldens()
        let waveform = try loadFloats("ref_audio")

        let window = VoiceCloneEncoder.padOrTrim(waveform, targetLength: goldens.speechWindowSamples)
        let reference = try loadFloats("speech_window")
        XCTAssertEqual(window, reference)
    }

    func testValidFrameTrimMatchesPythonReference() throws {
        let goldens = try loadGoldens()
        let valid = VoiceCloneEncoder.validFrameCount(
            realSamples: goldens.refSamples,
            windowSamples: goldens.speechWindowSamples,
            numCodes: goldens.numCodes
        )
        XCTAssertEqual(valid, goldens.validFrames)
    }
}
