//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Accelerate
import Foundation

// MARK: - Configuration

/// Mel-spectrogram parameters for the Qwen3-TTS voice-clone SpeakerEncoder.
///
/// Defaults are bit-compatible with the upstream Qwen3-TTS reference
/// (`torch.stft(center=False)` over a reflect-padded signal, periodic Hann
/// window, slaney-normalized librosa mel filterbank, `log(max(mel, 1e-5))`
/// dynamic-range compression).
public struct MelSpectrogramConfig: Sendable {
    public var nFFT: Int = 1024
    public var numMels: Int = 128
    public var samplingRate: Int = 24000
    public var hopSize: Int = 256
    public var winSize: Int = 1024
    public var fmin: Float = 0
    public var fmax: Float = 12000
    public var logClipValue: Float = 1e-5

    public init() {}
}

// MARK: - Implementation

/// Accelerate-backed mel-spectrogram front-end for voice cloning.
///
/// Layout of the result matches the Python reference: `numMels` rows by
/// `T` frames, row-major, where `T = (paddedLength - nFFT) / hopSize + 1`
/// with reflective padding of `(nFFT - hopSize) / 2` samples on each side.
public final class MelSpectrogram {
    public let config: MelSpectrogramConfig

    private let window: [Float]
    /// Slaney mel filterbank, `numMels x (nFFT/2 + 1)`, row-major.
    private let filterbank: [Float]
    private let fftSetup: FFTSetup
    private let log2n: vDSP_Length

    public init(config: MelSpectrogramConfig = MelSpectrogramConfig()) {
        precondition(config.nFFT.nonzeroBitCount == 1, "nFFT must be a power of two")
        self.config = config
        // Periodic Hann (`torch.hann_window` default): 0.5 * (1 - cos(2πn / N)).
        self.window = (0..<config.winSize).map { n in
            0.5 * (1 - cos(2 * Float.pi * Float(n) / Float(config.winSize)))
        }
        self.filterbank = Self.slaneyMelFilterbank(config: config)
        self.log2n = vDSP_Length(log2(Double(config.nFFT)))
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create FFT setup for nFFT=\(config.nFFT)")
        }
        self.fftSetup = setup
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    /// Compute the log-mel spectrogram of a mono waveform.
    ///
    /// - Parameter waveform: mono samples at `config.samplingRate`.
    /// - Returns: `(mel, frames)` where `mel` is `numMels * frames` floats,
    ///   row-major (`mel[m * frames + t]`).
    public func process(_ waveform: [Float]) -> (mel: [Float], frames: Int) {
        let nFFT = config.nFFT
        let hop = config.hopSize
        let numBins = nFFT / 2 + 1
        let numMels = config.numMels

        // Reflective padding of (nFFT - hop) / 2 on each side, matching
        // `torch.nn.functional.pad(mode="reflect")`.
        let pad = (nFFT - hop) / 2
        let padded = Self.reflectPad(waveform, pad: pad)

        let frames = padded.count >= nFFT ? (padded.count - nFFT) / hop + 1 : 0
        guard frames > 0 else { return ([], 0) }

        // Power spectrum per frame -> magnitude with the reference's +1e-9
        // epsilon inside the sqrt.
        var magnitudes = [Float](repeating: 0, count: frames * numBins)
        var windowed = [Float](repeating: 0, count: nFFT)
        var realPart = [Float](repeating: 0, count: nFFT / 2)
        var imagPart = [Float](repeating: 0, count: nFFT / 2)

        for t in 0..<frames {
            let start = t * hop
            padded.withUnsafeBufferPointer { src in
                vDSP_vmul(src.baseAddress! + start, 1, window, 1, &windowed, 1, vDSP_Length(nFFT))
            }
            realPart.withUnsafeMutableBufferPointer { realBuf in
                imagPart.withUnsafeMutableBufferPointer { imagBuf in
                    var split = DSPSplitComplex(realp: realBuf.baseAddress!, imagp: imagBuf.baseAddress!)
                    windowed.withUnsafeBufferPointer { w in
                        w.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFFT / 2) {
                            vDSP_ctoz($0, 2, &split, 1, vDSP_Length(nFFT / 2))
                        }
                    }
                    vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                    // vDSP_fft_zrip scales all outputs by 2; undo it, then unpack
                    // the DC/Nyquist packing into `numBins` magnitudes.
                    let re = realBuf.baseAddress!
                    let im = imagBuf.baseAddress!
                    let base = t * numBins
                    let dc = re[0] * 0.5
                    let nyquist = im[0] * 0.5
                    magnitudes[base] = sqrt(dc * dc + 1e-9)
                    magnitudes[base + numBins - 1] = sqrt(nyquist * nyquist + 1e-9)
                    for k in 1..<(nFFT / 2) {
                        let reK = re[k] * 0.5
                        let imK = im[k] * 0.5
                        magnitudes[base + k] = sqrt(reK * reK + imK * imK + 1e-9)
                    }
                }
            }
        }

        // mel = filterbank (numMels x numBins) @ magnitudes^T (numBins x frames)
        // Compute as (numMels x numBins) * (frames x numBins)^T via vDSP_mmul on
        // the transposed magnitude layout.
        var mel = [Float](repeating: 0, count: numMels * frames)
        var magT = [Float](repeating: 0, count: numBins * frames)
        vDSP_mtrans(magnitudes, 1, &magT, 1, vDSP_Length(numBins), vDSP_Length(frames))
        vDSP_mmul(
            filterbank, 1, magT, 1, &mel, 1,
            vDSP_Length(numMels), vDSP_Length(frames), vDSP_Length(numBins)
        )

        // log(max(mel, clip))
        var clip = config.logClipValue
        vDSP_vthr(mel, 1, &clip, &mel, 1, vDSP_Length(mel.count))
        var count = Int32(mel.count)
        vvlogf(&mel, mel, &count)

        return (mel, frames)
    }

    // MARK: - Helpers

    static func reflectPad(_ x: [Float], pad: Int) -> [Float] {
        precondition(x.count > pad, "Signal must be longer than the reflect pad")
        var out = [Float]()
        out.reserveCapacity(x.count + 2 * pad)
        for i in stride(from: pad, to: 0, by: -1) { out.append(x[i]) }
        out.append(contentsOf: x)
        for i in stride(from: x.count - 2, through: x.count - 1 - pad, by: -1) { out.append(x[i]) }
        return out
    }

    /// librosa-compatible slaney mel filterbank (`norm="slaney"`, `htk=False`).
    static func slaneyMelFilterbank(config: MelSpectrogramConfig) -> [Float] {
        let numBins = config.nFFT / 2 + 1
        let numMels = config.numMels

        func hzToMel(_ hz: Float) -> Float {
            let fSp: Float = 200.0 / 3
            if hz < 1000 { return hz / fSp }
            let minLogHz: Float = 1000
            let minLogMel = minLogHz / fSp
            let logstep = log(6.4) / 27.0
            return minLogMel + log(Double(hz) / Double(minLogHz)).asFloat / Float(logstep)
        }

        func melToHz(_ mel: Float) -> Float {
            let fSp: Float = 200.0 / 3
            let minLogMel: Float = 1000 / fSp
            if mel < minLogMel { return mel * fSp }
            let logstep = Float(log(6.4) / 27.0)
            return 1000 * exp(logstep * (mel - minLogMel))
        }

        let melMin = hzToMel(config.fmin)
        let melMax = hzToMel(config.fmax)
        let melPoints = (0..<(numMels + 2)).map { i in
            melToHz(melMin + (melMax - melMin) * Float(i) / Float(numMels + 1))
        }

        let fftFreqs = (0..<numBins).map { k in
            Float(k) * Float(config.samplingRate) / 2 / Float(numBins - 1)
        }

        var weights = [Float](repeating: 0, count: numMels * numBins)
        for m in 0..<numMels {
            let fLeft = melPoints[m]
            let fCenter = melPoints[m + 1]
            let fRight = melPoints[m + 2]
            // Slaney normalization: 2 / bandwidth.
            let enorm = 2 / (fRight - fLeft)
            for k in 0..<numBins {
                let f = fftFreqs[k]
                let up = (f - fLeft) / (fCenter - fLeft)
                let down = (fRight - f) / (fRight - fCenter)
                let w = max(0, min(up, down))
                if w > 0 { weights[m * numBins + k] = w * enorm }
            }
        }
        return weights
    }
}

private extension Double {
    var asFloat: Float { Float(self) }
}
