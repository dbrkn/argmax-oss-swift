//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// ACI hard-CMask alignment (RD-691) — faithful port of rongxiang's
/// `aci_align.ACIAlign`, the intervention sibling to ``SoftAlign``.
///
/// Where SoftAlign steers with a smooth Huber bias, ACI clamps each configured
/// head's attention to a hard window around a monotone text center: for head
/// `h` only positions `[center-(rho-1), center+(rho-1)]` inside the ICL text
/// region stay unmasked; the rest of the text region is driven to `-1e30`.
/// The center is an ONLINE MONOTONE-DP track over that head's own text
/// attention (stay / advance-1 recurrence), per `(layer, head)`; heads may
/// span multiple layers, each with its own DP center and per-head radius rho.
///
/// RD-691 additions ported here:
///  - **prefill DP (pDP)**: ``seedFromPrefill`` advances every head's DP over
///    the REFERENCE codec frames' attention rows during prefill, so the
///    main-generation track continues from the reference alignment instead of
///    cold-starting (the fix that took loc-ACI-deep to 0/7 failures on the
///    long-ICL failure set).
///  - loc-ACI-deep: `window` extends the armed mask past the fire step and
///    `extraRollback` (executor-side) deepens the rollback.
///
/// Pure Swift/CPU (`[Float]` rows handed in by the MLX hook); the hook builds
/// the additive MLX mask from the returned band.
public struct ACIConfig: Sendable {
    /// `{layer: [head, ...]}` cross-layer CMask head set.
    public var groups: [Int: [Int]]
    /// `{(layer, head) -> rho}` per-head window radius (flattened key `layer*1000+head`).
    public var rho: [Int: Int]
    /// Mask every step (pair with maxRetries=0) instead of arming on rollback.
    public var alwaysOn: Bool = false
    /// loc-ACI-deep mask-hold: armed window extends `window` steps past the fire.
    /// TODO(verify vs PR27 final): study value not in the fetched revision.
    public var window: Int = 40
    /// Center from monotone DP ("dp") or raw argmax ("argmax").
    public var centerMode: String = "dp"
    /// Mask duty cycle: apply `applyN` of every `stride` armed steps ((1,1) = continuous, shipped loc-ACI).
    public var stride: Int = 1
    public var applyN: Int = 1

    public static func key(_ layer: Int, _ head: Int) -> Int { layer * 1000 + head }

    /// RD-691 calibrated presets (rho = round(8·C_E)+1 from teacher-forced attention entropy).
    public static let preset06b = ACIConfig(
        groups: [5: [13], 6: [0, 1, 4, 5]],
        rho: [key(5, 13): 6, key(6, 0): 10, key(6, 1): 5, key(6, 4): 10, key(6, 5): 6])
    public static let preset17b = ACIConfig(
        groups: [3: [0, 1, 12, 13]],
        rho: [key(3, 0): 4, key(3, 1): 4, key(3, 13): 8, key(3, 12): 16])

    public init(groups: [Int: [Int]], rho: [Int: Int]) {
        self.groups = groups
        self.rho = rho
    }
}

public final class ACIAlign {
    private let cfg: ACIConfig
    public private(set) var armed = false
    private var armStep: Int?
    private var stepIdx = 0
    /// Running monotone-DP cost per (layer,head), over the ICL text span (ntf).
    private var dp: [Int: [Float]] = [:]
    /// Monotone non-decreasing center per (layer,head).
    private var ctr: [Int: Int] = [:]
    /// Rollback snapshots keyed by decode step: step -> (dp, ctr) copies.
    private var snaps: [Int: ([Int: [Float]], [Int: Int])] = [:]

    public init(config: ACIConfig) { self.cfg = config }

    public var layers: Set<Int> { Set(cfg.groups.keys) }
    public func heads(for layer: Int) -> [Int] { cfg.groups[layer] ?? [] }

    public func reset() {
        armed = false; armStep = nil; stepIdx = 0
        dp.removeAll(); ctr.removeAll(); snaps.removeAll()
    }

    /// Arm the localized mask on a rollback fire (`skipStep` = monitor fire step).
    public func arm(skipStep: Int) {
        armed = true
        armStep = skipStep + cfg.window
    }

    public func setStep(_ pos: Int) { stepIdx = pos }

    public func shouldApply() -> Bool {
        if cfg.alwaysOn { return true }
        guard armed, let armStep else { return false }
        return stepIdx <= armStep && stepIdx % cfg.stride < cfg.applyN
    }

    /// Snapshot the DP state at decode step `step` (called by the executor's
    /// per-step checkpointing) and restore on rollback — the Swift equivalent
    /// of the python cache-length-keyed snapshots.
    public func snapshot(step: Int) {
        snaps[step] = (dp, ctr)
        if snaps.count > 1700, let oldest = snaps.keys.min() { snaps.removeValue(forKey: oldest) }
    }

    public func rollback(to step: Int) {
        for k in snaps.keys where k > step { snaps.removeValue(forKey: k) }
        if let s = snaps[step] { dp = s.0; ctr = s.1 } else { dp.removeAll(); ctr.removeAll() }
    }

    /// Advance one head's monotone DP with this step's text-region attention
    /// distribution `seg` (softmax over the FULL row, sliced to the text span;
    /// `navail` = positions of the span actually in cache). Returns the
    /// PREVIOUS monotone center the mask window centers on (python parity).
    @discardableResult
    private func advance(layer: Int, head: Int, seg: [Float], ntf: Int, navail: Int) -> Int {
        let key = ACIConfig.key(layer, head)
        var cost = [Float](repeating: .infinity, count: ntf)
        for i in 0..<min(navail, ntf, seg.count) { cost[i] = -log(seg[i] + 1e-9) }
        if var prev = dp[key], prev.count == ntf {
            // stay(dp[i]) vs advance-1(dp[i-1]) -> monotone track
            var next = [Float](repeating: 0, count: ntf)
            var shifted: Float = .infinity
            for i in 0..<ntf {
                next[i] = cost[i] + min(prev[i], shifted)
                shifted = prev[i]
            }
            _ = prev  // keep prev alive until loop done (value semantics)
            dp[key] = next
        } else {
            dp[key] = cost
        }
        var cNew: Int
        if cfg.centerMode == "dp" {
            var best = 0
            var bestV = Float.infinity
            for (i, v) in dp[key]!.enumerated() where v < bestV { best = i; bestV = v }
            cNew = best
        } else {
            var best = 0
            var bestV = -Float.infinity
            for i in 0..<min(navail, seg.count) where seg[i] > bestV { best = i; bestV = seg[i] }
            cNew = navail > 0 ? best : 0
        }
        let cPrev = min(ctr[key] ?? cNew, max(0, navail - 1))
        ctr[key] = max(cNew, cPrev)               // monotone non-decreasing
        return cPrev
    }

    /// One decode step for `layer`: advance all its heads' DP tracks and, when
    /// `apply`, return the additive band per head: `[head: [Float](T)]` with 0
    /// inside the window, -1e30 over the rest of the text region `[pf, tt)`,
    /// 0 outside (prefix/audio history unmasked).
    public func layerCMask(layer: Int, segs: [Int: [Float]], pf: Int, tt: Int,
                           T: Int, ntf: Int, navail: Int, apply: Bool) -> [Int: [Float]]? {
        var out: [Int: [Float]]? = apply ? [:] : nil
        for h in heads(for: layer) {
            guard let seg = segs[h] else { continue }
            let cPrev = advance(layer: layer, head: h, seg: seg, ntf: ntf, navail: navail)
            if apply {
                let r = cfg.rho[ACIConfig.key(layer, h)] ?? 8
                let lo = pf + max(0, cPrev - (r - 1))
                let hi = pf + min(navail - 1, cPrev + (r - 1))
                var band = [Float](repeating: 0, count: T)
                for i in pf..<min(tt, T) { band[i] = -1e30 }
                if hi >= lo { for i in lo...min(hi, T - 1) { band[i] = 0 } }
                out![h] = band
            }
        }
        return out
    }

    /// prefill DP (pDP, RD-691): advance a head's DP over the REFERENCE codec
    /// frames' attention rows captured during prefill, one row per frame in
    /// generation order. After this the decode-time DP continues from the
    /// reference alignment trajectory instead of a cold start.
    public func seedFromPrefill(layer: Int, head: Int, rows: [[Float]], ntf: Int) {
        for row in rows {
            advance(layer: layer, head: head, seg: row, ntf: ntf, navail: min(ntf, row.count))
        }
    }
}
