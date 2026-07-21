//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
@testable import TTSKit
import XCTest

/// Golden-value tests for the voice-clone preprocessing front-end.
///
/// The mel DSP is asserted against fingerprints exported from the Python
/// reference implementation (bundled in `goldens.json` alongside the raw
/// reference samples); the window-shaping helpers are pure logic and are
/// tested synthetically.
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
            throw XCTSkip("Voice-clone goldens not bundled")
        }
        return url
    }

    private func loadGoldens() throws -> Goldens {
        try JSONDecoder().decode(Goldens.self, from: Data(contentsOf: goldenURL("goldens", "json")))
    }

    private func loadReference() throws -> [Float] {
        let data = try Data(contentsOf: goldenURL("ref_audio", "bin"))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    // MARK: - Mel DSP vs Python reference fingerprints

    func testMelSpectrogramMatchesPythonReference() throws {
        let goldens = try loadGoldens()
        let waveform = try loadReference()
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

        // First and last frames of the leading mel rows, element-wise.
        for (m, expected) in goldens.melFirstFrame.enumerated() {
            XCTAssertEqual(values[m * frames], expected, accuracy: 5e-3)
        }
        for (m, expected) in goldens.melLastFrame.enumerated() {
            XCTAssertEqual(values[m * frames + frames - 1], expected, accuracy: 5e-3)
        }
        // Whole-row means catch drift anywhere along the time axis.
        for (m, expected) in goldens.melChecksumRowMeans.enumerated() {
            var sum: Float = 0
            for t in 0..<frames { sum += values[m * frames + t] }
            XCTAssertEqual(sum / Float(frames), expected, accuracy: 5e-3, "row \(m) mean drift")
        }
    }

    // MARK: - Window shaping (pure logic, synthetic inputs)

    func testTileToWindowRepeatsRealAudio() {
        let wave: [Float] = [1, 2, 3]
        let tiled = VoiceCloneEncoder.tileToWindow(wave, targetSamples: 8)
        XCTAssertEqual(tiled, [1, 2, 3, 1, 2, 3, 1, 2])
        // Long-enough input passes through untouched.
        XCTAssertEqual(VoiceCloneEncoder.tileToWindow(wave, targetSamples: 3), wave)
        XCTAssertEqual(VoiceCloneEncoder.tileToWindow(wave, targetSamples: 2), wave)
    }

    func testTileCountMatchesPythonReference() throws {
        let goldens = try loadGoldens()
        // The golden's tile count was computed for the bundled reference against
        // the speaker mel window: ceil(window / refSamples).
        let windowSamples = goldens.speakerMelShape[1] * goldens.melParams.hopSize
        let expectedTiles = Int(ceil(Double(windowSamples) / Double(goldens.refSamples)))
        XCTAssertEqual(expectedTiles, goldens.speakerTiles)
        let tiled = VoiceCloneEncoder.tileToWindow(
            [Float](repeating: 0.5, count: goldens.refSamples), targetSamples: windowSamples
        )
        XCTAssertEqual(tiled.count, windowSamples)
    }

    func testPadOrTrim() {
        XCTAssertEqual(VoiceCloneEncoder.padOrTrim([1, 2, 3], targetLength: 5), [1, 2, 3, 0, 0])
        XCTAssertEqual(VoiceCloneEncoder.padOrTrim([1, 2, 3], targetLength: 2), [1, 2])
        XCTAssertEqual(VoiceCloneEncoder.padOrTrim([1, 2, 3], targetLength: 3), [1, 2, 3])
    }

    func testPadOrTrimFrames() {
        // (2 mels, 3 frames) -> (2 mels, 5 frames), right-padded per row.
        let mel: [Float] = [1, 2, 3, 4, 5, 6]
        let padded = VoiceCloneEncoder.padOrTrimFrames(mel, frames: 3, numMels: 2, targetFrames: 5)
        XCTAssertEqual(padded, [1, 2, 3, 0, 0, 4, 5, 6, 0, 0])
        let trimmed = VoiceCloneEncoder.padOrTrimFrames(mel, frames: 3, numMels: 2, targetFrames: 2)
        XCTAssertEqual(trimmed, [1, 2, 4, 5])
    }

    func testValidFrameTrimMatchesPythonReference() throws {
        let goldens = try loadGoldens()
        let valid = VoiceCloneEncoder.validFrameCount(
            realSamples: goldens.refSamples,
            windowSamples: goldens.speechWindowSamples,
            numCodes: goldens.numCodes
        )
        XCTAssertEqual(valid, goldens.validFrames)

        // Full-window and over-window inputs keep every code frame.
        XCTAssertEqual(VoiceCloneEncoder.validFrameCount(realSamples: 240_000, windowSamples: 240_000, numCodes: 125), 125)
        XCTAssertEqual(VoiceCloneEncoder.validFrameCount(realSamples: 500_000, windowSamples: 240_000, numCodes: 125), 125)
    }

    func testTrimFrames() {
        // (2 quantizers, 3 frames), keep 2.
        let codes: [Int32] = [1, 2, 3, 4, 5, 6]
        XCTAssertEqual(VoiceCloneEncoder.trimFrames(codes, quantizers: 2, frames: 3, keep: 2), [1, 2, 4, 5])
        XCTAssertEqual(VoiceCloneEncoder.trimFrames(codes, quantizers: 2, frames: 3, keep: 3), codes)
    }
}
