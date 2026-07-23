//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// Online coverage monitor — the deployed skip + hallucination detector.
///
/// Faithful Swift port of RD-655 `coverage_monitor.py`. Both failure types are
/// coverage anomalies on one dwell histogram (`count[]`) against a P10 "settle
/// frontier":
///   - **skip**          = coverage *deficit* — a settled bin was rushed past
///                         (`count < commitFrac · fair`)
///   - **hallucination** = coverage *stall*   — the settle high-water fails to
///                         advance for `> stallFrac · fair` steps
///
/// Signal: `f(t) ∈ [0,1]`, the anchor head's normalized text-argmax per decode
/// step (see ``TalkerAnchorProbe``). Pure logic, bounded state (`O(nBins +
/// nSteps)`); no MLX dependency so it unit-tests without the model. The
/// operating point and every branch mirror the Python; parity is locked by
/// `CoverageMonitorParityTests`.
public struct CoverageMonitorConfig: Sendable, Equatable {
    public var tokPerBin: Int = 10
    public var commitFrac: Double = 0.3
    public var win: Int = 40
    public var settlePctl: Double = 10
    public var fairWin: Int = 15
    public var fairInit: Double = 40
    public var overFrac: Double = 2.5
    public var stallFrac: Double = 3.0
    /// Steps before a skip fire searched for the last-good bin's dwell.
    public var coverageLookback: Int = 150
    /// Cap on rollback depth in steps (≈128 s at 12.5 Hz).
    public var maxRollback: Int = 1600
    public var adaptive: Bool = true

    public init() {}
}

/// One confirmed failure, with the step to resume decoding from.
public struct GuardrailFailure: Sendable, Equatable {
    public enum Kind: String, Sendable { case skip, hallucination }
    public let failure: Kind
    /// Decode step at which the failure was confirmed.
    public let fireStep: Int
    /// Decode step to resume from; `latency = (fireStep − rollbackStep) / 12.5`.
    public let rollbackStep: Int
    /// The failed text bins (skipped bins, or the stalled high-water bin).
    public let bins: [Int]
    public let fairCount: Double
    public let nBins: Int
}

public final class CoverageMonitor {
    public let nBins: Int
    private let cfg: CoverageMonitorConfig

    // f fed so far (kept across rollbacks; truncated by rollbackTo).
    private var fHist: [Float] = []
    // Derived per-step state (cleared on reset / replay; NOT fHist).
    private var rawHist: [Int] = []
    private var count: [Double]
    private var skipped: [Bool]
    private var window: [Float] = []               // trailing fHist for the settle frontier (last `win`)
    private var completedDwells: [Double] = []
    private var fairCount: Double
    private var settledBin = 0
    private var maxCommitted = -1
    private var hwStep = 0
    private var t = -1

    public init(ntok: Int, config: CoverageMonitorConfig = CoverageMonitorConfig()) {
        self.cfg = config
        self.nBins = max(8, Int((Double(ntok) / Double(config.tokPerBin)).rounded()))
        self.count = [Double](repeating: 0, count: nBins)
        self.skipped = [Bool](repeating: false, count: nBins)
        self.fairCount = config.fairInit
    }

    /// Number of steps currently held = the live decode position.
    public var nSteps: Int { fHist.count }

    /// Advance one decode step with this step's `f`. Returns a failure on the
    /// firing step, else `nil`.
    @discardableResult
    public func step(_ fT: Float) -> GuardrailFailure? {
        fHist.append(fT)
        return update(fT)
    }

    /// Rewind to `n` steps (after the executor rewinds the decode): keep
    /// `fHist[:n]` and replay to rebuild state — exact prefix state, no drift.
    public func rollback(to n: Int) {
        fHist = Array(fHist.prefix(n))
        reset()
        for f in fHist { _ = update(f) }
    }

    private func reset() {
        rawHist.removeAll(keepingCapacity: true)
        count = [Double](repeating: 0, count: nBins)
        skipped = [Bool](repeating: false, count: nBins)
        window.removeAll(keepingCapacity: true)
        completedDwells.removeAll(keepingCapacity: true)
        fairCount = cfg.fairInit
        settledBin = 0
        maxCommitted = -1
        hwStep = 0
        t = -1
    }

    private func update(_ fT: Float) -> GuardrailFailure? {
        t += 1
        let nb = nBins
        let b = min(Int(fT * Float(nb)), nb - 1)
        rawHist.append(b)
        count[b] += 1
        window.append(fT)
        if window.count > cfg.win { window.removeFirst(window.count - cfg.win) }

        // --- settle: lock verdicts for bins the argmax provably moved past ---
        let settleTo = min(Int(Self.percentile(window, cfg.settlePctl) * Double(nb)), nb - 1)
        let fairAtSettle = fairCount
        if settledBin < settleTo {
            for pb in settledBin..<settleTo {
                if count[pb] < cfg.commitFrac * fairAtSettle {
                    skipped[pb] = true                                   // under-coverage → skip
                } else {
                    maxCommitted = pb                                    // covered → frontier moves
                    if count[pb] <= cfg.overFrac * fairAtSettle {        // normal → a rate sample
                        completedDwells.append(count[pb])
                        if cfg.adaptive, completedDwells.count >= cfg.fairWin {
                            fairCount = Self.median(completedDwells.suffix(cfg.fairWin))
                        }
                    }
                }
            }
        }
        let advanced = settleTo > settledBin
        settledBin = max(settledBin, settleTo)
        if advanced || settledBin == 0 { hwStep = t }

        // --- SKIP: a locked-skipped bin sits behind the committed frontier ---
        var committedFrontier = maxCommitted
        if settledBin > maxCommitted, count[settledBin] >= cfg.commitFrac * fairCount {
            committedFrontier = settledBin                              // look-ahead: fire this step
        }
        if committedFrontier > 0 {
            var skippedBins: [Int] = []
            for i in 0..<committedFrontier where skipped[i] { skippedBins.append(i) }
            if let first = skippedBins.first {
                let rb = skipRollback(fireStep: t, firstSkipped: first)
                return GuardrailFailure(failure: .skip, fireStep: t, rollbackStep: rb,
                                        bins: skippedBins, fairCount: fairCount, nBins: nb)
            }
        }

        // --- HALLUCINATION: frontier stall ---
        if settledBin > 0, Double(t - hwStep) > cfg.stallFrac * fairCount {
            let rb = max(hwStep, t - cfg.maxRollback)
            return GuardrailFailure(failure: .hallucination, fireStep: t, rollbackStep: rb,
                                    bins: [settledBin], fairCount: fairCount, nBins: nb)
        }
        return nil
    }

    /// Resume step for a fired skip: dwell of the last-good bin (`firstSkipped−1`)
    /// just before the gap — median of its raw-argmax steps within `lookback`,
    /// snapped to an actual visit. Causal; capped. Mirrors `_skip_rollback`.
    private func skipRollback(fireStep: Int, firstSkipped: Int) -> Int {
        let floorStep = max(0, fireStep - cfg.maxRollback)
        let lastGood = firstSkipped - 1
        if lastGood < 0 { return floorStep }
        var steps: [Int] = []
        for (i, bin) in rawHist.enumerated() where i <= fireStep && bin == lastGood { steps.append(i) }
        if steps.isEmpty { return floorStep }
        let recent = steps.filter { $0 >= fireStep - cfg.coverageLookback }
        let target: Int
        if !recent.isEmpty {
            let med = Self.median(recent.map(Double.init))
            // Snap to the actual visit nearest the median (a jitter spike can put
            // the median index in a hole between visits).
            target = recent.min(by: { abs(Double($0) - med) < abs(Double($1) - med) })!
        } else {
            target = steps.last!
        }
        return max(target, floorStep)
    }

    // MARK: - numpy-matching reductions

    /// `np.median` (even length → mean of the two middle elements).
    static func median<S: Sequence>(_ xs: S) -> Double where S.Element == Double {
        let a = xs.sorted()
        let n = a.count
        guard n > 0 else { return 0 }
        return n % 2 == 1 ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2
    }

    /// `np.percentile` with linear interpolation, matching numpy's default.
    static func percentile(_ xs: [Float], _ p: Double) -> Double {
        guard !xs.isEmpty else { return 0 }
        let a = xs.map(Double.init).sorted()
        if a.count == 1 { return a[0] }
        let rank = (p / 100) * Double(a.count - 1)
        let lo = Int(rank.rounded(.down))
        let hi = Int(rank.rounded(.up))
        if lo == hi { return a[lo] }
        return a[lo] + (rank - Double(lo)) * (a[hi] - a[lo])
    }
}

/// Offline one-shot wrapper over ``CoverageMonitor`` — feed a whole trajectory,
/// return the first confirmed failure (or `nil`). Being a wrapper it can never
/// disagree with the online form, so the parity tests gate both.
public func detectFailure(_ f: [Float], ntok: Int,
                          config: CoverageMonitorConfig = CoverageMonitorConfig()) -> GuardrailFailure? {
    let mon = CoverageMonitor(ntok: ntok, config: config)
    for ft in f {
        if let res = mon.step(ft) { return res }
    }
    return nil
}
