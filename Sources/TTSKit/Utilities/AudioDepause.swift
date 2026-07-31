//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// Reference de-pausing (RD-691 long-form ICL stability study).
///
/// Long silences in a voice-clone reference destabilize the decoder's
/// reference-audio↔reference-text alignment during prefill; the main-text
/// generation then inherits that bad starting point (measured: de-pausing
/// alone took the long-reference failure set from 3/7 to 1/7 baseline
/// failures). Algorithm per the study: 10 ms units; a unit is silent when its
/// peak is more than `floorDb` below the loudest unit; delete silent runs
/// longer than `minPauseSeconds`.
public enum AudioDepause {
    public static func depause(
        _ waveform: [Float], sampleRate: Int,
        floorDb: Float = 35, minPauseSeconds: Double = 0.4
    ) -> [Float] {
        let unit = max(1, sampleRate / 100)                     // 10 ms
        let nUnits = waveform.count / unit
        guard nUnits > 2 else { return waveform }
        var peaks = [Float](repeating: 0, count: nUnits)
        var maxPeak: Float = 0
        for u in 0..<nUnits {
            var p: Float = 0
            for i in (u * unit)..<((u + 1) * unit) { p = max(p, abs(waveform[i])) }
            peaks[u] = p
            maxPeak = max(maxPeak, p)
        }
        guard maxPeak > 0 else { return waveform }
        let thr = maxPeak * pow(10, -floorDb / 20)
        let minRun = Int(minPauseSeconds * 100)                 // in 10 ms units
        var keep = [Bool](repeating: true, count: nUnits)
        var runStart = -1
        for u in 0...nUnits {
            let silent = u < nUnits && peaks[u] < thr
            if silent && runStart < 0 { runStart = u }
            if !silent && runStart >= 0 {
                if u - runStart > minRun {
                    for k in runStart..<u { keep[k] = false }
                }
                runStart = -1
            }
        }
        var out = [Float]()
        out.reserveCapacity(waveform.count)
        for u in 0..<nUnits where keep[u] {
            out.append(contentsOf: waveform[(u * unit)..<((u + 1) * unit)])
        }
        out.append(contentsOf: waveform[(nUnits * unit)...])
        return out.isEmpty ? waveform : out
    }
}
