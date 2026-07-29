//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// Prompt-binding failure detector (guardrails v2) for unchunked voice-clone
/// generation — replaces the RD-655 coverage monitor's dwell statistics with
/// the empirically dominant failure signature.
///
/// Catastrophic unchunked failures (WER≈1, SIM≈0.07: the generation never
/// binds to the reference prompt and babbles to the step cap) show a sustained
/// **flip-flop** in the text-anchor fraction from the first steps: `f`
/// oscillates between ~0.02 and a fixed mid value on nearly every step instead
/// of ramping. Healthy generations jitter briefly during warm-up, then ramp.
/// Measured on clean observe-mode trajectories (34 gens, 9 catastrophic):
///   - sustained alternation (both 60-step windows) > 0.65 → 89% recall, 0% FPR
///   - OR text-attention-mass collapse (mass<0.3 on >30% of steps) → **100%
///     recall, 0% FPR** at a single decision ~130 steps (~10.4 s of audio)
/// vs the RD-655 coverage monitor's 53% recall @ 33% FPR on true labels.
///
/// Pure logic (no MLX): fed per decode step with the anchor fraction and text
/// mass the probe already computes — no new models, O(window) state.
public struct BindingMonitorConfig: Codable, Sendable, Equatable {
    /// Steps ignored at the start (warm-up jitter affects everyone).
    public var warmupSteps: Int = 10
    /// Each persistence window length; decision fires after two full windows.
    public var windowSteps: Int = 60
    /// Sign-alternation rate above which a window counts as flip-flopping.
    /// CALIBRATION (RD-655 corpora texts × eval-disjoint speakers, 13 runs):
    /// alternation does NOT separate — healthy long generations reach 0.75 —
    /// so the default disables it (>1 never fires). Text-mass collapse is the
    /// signal that generalizes: the catastrophic failure sat at 0.96 low-mass
    /// fraction vs ≤0.71 for every healthy run → threshold 0.8. Partial
    /// degradations (WER 0.4–0.6) are NOT separable from these signals — a
    /// known limitation of the attention-side-only detector.
    public var alternationThreshold: Double = 1.1
    /// Fraction of steps with textMass below `textMassFloor` that flags collapse.
    public var textMassLowFraction: Double = 0.8
    public var textMassFloor: Float = 0.3
    public init() {}
}

public final class BindingMonitor {
    public enum Verdict: Sendable, Equatable {
        /// Generation is bound to the prompt (or too short to judge yet).
        case ok
        /// Prompt-binding failure confirmed at `step` — restart the generation.
        case bindingFailure(step: Int)
    }

    private let cfg: BindingMonitorConfig
    private var fHist: [Float] = []
    private var tmLow: [Bool] = []
    private var nextDecision: Int
    private var firstDecisionDone = false
    private var highWater: Float = 0

    /// Step index of the FIRST decision; later decisions repeat every
    /// `windowSteps` on the trailing two windows, so a generation that binds
    /// initially and degrades later still fires.
    public var decisionStep: Int { cfg.warmupSteps + 2 * cfg.windowSteps }

    public init(config: BindingMonitorConfig = BindingMonitorConfig()) {
        self.cfg = config
        self.nextDecision = config.warmupSteps + 2 * config.windowSteps
    }

    /// Reset for a fresh attempt (after a restart).
    public func reset() {
        fHist.removeAll(keepingCapacity: true)
        tmLow.removeAll(keepingCapacity: true)
        nextDecision = decisionStep
        firstDecisionDone = false
        highWater = 0
    }

    /// Feed one decode step. At the first decision step and every
    /// `windowSteps` thereafter, evaluates the TRAILING two windows and returns
    /// `.bindingFailure` when the flip-flop / mass-collapse signature is present.
    @discardableResult
    public func step(f: Float, textMass: Float?) -> Verdict {
        fHist.append(f)
        tmLow.append((textMass ?? 1.0) < cfg.textMassFloor)
        highWater = max(highWater, f)
        let n = fHist.count
        guard n >= nextDecision else { return .ok }
        nextDecision = n + cfg.windowSteps
        let wasFirst = !firstDecisionDone
        firstDecisionDone = true

        // Late decisions are gated on the anchor never having reached the text
        // end: near end-of-text, f legitimately jitters while remaining speech
        // catches up to where the anchor has already read (measured: healthy
        // gens fire spuriously there at hw≈0.99, with real speech still coming).
        if !wasFirst && highWater >= 0.97 { return .ok }

        let w = cfg.windowSteps
        let w1 = alternationRate(Array(fHist[(n - 2 * w) ..< (n - w)]))
        let w2 = alternationRate(Array(fHist[(n - w) ..< n]))
        let sustainedFlipFlop = min(w1, w2) > cfg.alternationThreshold
        let recentLow = tmLow[(n - 2 * w) ..< n]
        let massCollapse = Double(recentLow.filter { $0 }.count) / Double(2 * w) > cfg.textMassLowFraction
        if sustainedFlipFlop || massCollapse {
            return .bindingFailure(step: n)
        }
        return .ok
    }

    /// Sign-alternation rate of successive `f` deltas (flip-flop ≈ 1, ramp ≈ 0).
    static func alternationRate(_ seg: [Float]) -> Double {
        var signs: [Int] = []
        for i in 1..<seg.count {
            let d = seg[i] - seg[i - 1]
            if d != 0 { signs.append(d > 0 ? 1 : -1) }
        }
        guard signs.count > 4 else { return 0 }
        var flips = 0
        for i in 1..<signs.count where signs[i] != signs[i - 1] { flips += 1 }
        return Double(flips) / Double(signs.count - 1)
    }

    private func alternationRate(_ seg: [Float]) -> Double { Self.alternationRate(seg) }
}
