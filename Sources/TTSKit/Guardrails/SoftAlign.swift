//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// Soft-align bias state machine (RD-655 `align.py`) — the directed intervention
/// armed on a coverage rollback. It holds the config + pace integrator; the
/// MLX hook reads ``center``/``lambda``/``delta``/``biasHeads`` to build the
/// Huber penalty tensor added to the anchor-layer heads' attention scores over
/// the rolled-back span. This type is pure logic (no MLX) so it unit-tests
/// against the Python reference; the tensor construction lives in TTSKitMLX.
///
/// `pen(pos) = −lambda · huber(|pos − center|; delta)` pulls the head back
/// toward the on-pace `center` whether it runs ahead (skip) or behind
/// (lag/re-read). `center` is an integrator: a robust bounded rate propagated
/// from its own prior value, re-seeded after each rollback — never a raw
/// position read-out (a skip/stall can't yank it).
public struct SoftAlignConfig: Codable, Sendable, Equatable {
    public var lambda: Double = 0.2
    public var delta: Double = 10
    public var stride: Int = 2
    public var applyN: Int = 1
    public var pace: Double = 0.254      // fixed-pace fallback rate (tokens/step)
    public var floor: Double = 0.2
    public var cap: Double = 0.40
    public init() {}
}

public final class SoftAlign {
    // Pace-integrator internals (validated; scale-free — align.py §6).
    static let paceWin = 300
    static let paceStride = 50
    static let paceMinHist = 150
    static let paceAnchor = 64

    public let lambda: Double
    public let delta: Double
    let stride: Int
    let applyN: Int
    let pace: Double
    let floor: Double
    let cap: Double
    /// Head indices (on the anchor/bias layer) the bias is applied to.
    public let biasHeads: Set<Int>

    public private(set) var armed = false
    /// Integrated pace reference τ (nil → fixed-pace fallback / not yet seeded).
    public private(set) var center: Double?
    private var centerAt = -1
    private var stepIdx = 0
    private var armStep: Int?

    public init(config: SoftAlignConfig = SoftAlignConfig(), biasHeads: [Int]) {
        self.lambda = config.lambda
        self.delta = config.delta
        self.stride = max(1, config.stride)
        self.applyN = max(1, min(config.applyN, max(1, config.stride)))
        self.pace = config.pace
        self.floor = config.floor
        self.cap = config.cap
        self.biasHeads = Set(biasHeads)
    }

    /// New utterance: disarm + clear the integrator.
    public func reset() {
        armed = false; center = nil; centerAt = -1; stepIdx = 0; armStep = nil
    }

    /// Arm the bias and anchor the localized span at the fire step (upper bound
    /// of the rolled-back span). Full strength while `stepIdx <= skipStep`, then off.
    public func arm(skipStep: Int) { armed = true; armStep = skipStep }

    public func setStep(_ pos: Int) { stepIdx = pos }

    /// Apply the bias this step? Only armed, within the rolled-back span, on
    /// `applyN` of every `stride`-cycle. Unarmed short-circuits before `armStep`.
    public func shouldApply() -> Bool {
        guard armed, let armStep else { return false }
        return stepIdx <= armStep && stepIdx % stride < applyN
    }

    /// The pace center for this step (integrator τ, or the fixed-pace fallback).
    public func centerFor(textStart: Int) -> Double {
        center ?? (Double(textStart) + pace * Double(stepIdx))
    }

    /// Advance the integrator from the f-history (the monitor's; it rewinds on
    /// rollback so `cur` drops and τ re-seeds). Call each step before the
    /// forward, only while armed. `textStart`/`textEnd` are the synthesis-text
    /// KV span (align.py's `prefill`/`t_text`).
    public func updateCenter(fHist: [Float], textStart: Int, textEnd: Int) {
        guard armed, lambda > 0 else { return }
        let n = fHist.count
        if n < Self.paceMinHist { center = nil; return }        // fixed-pace fallback
        let span = Double(max(1, textEnd - textStart))
        let a = fHist.map { Double(textStart) + Double($0) * span }   // attended position a(t)
        let cur = n
        let a0 = max(8, cur - Self.paceWin)
        // robust local pace = median strided slope over [a0, cur)
        var slopes: [Double] = []
        var i = a0 + Self.paceStride
        while i < cur { slopes.append((a[i] - a[i - Self.paceStride]) / Double(Self.paceStride)); i += 1 }
        var rate = slopes.isEmpty ? pace : CoverageMonitor.median(slopes)
        rate = min(max(rate, floor), cap)
        if center == nil || cur <= centerAt {                    // seed (start / post-rollback)
            let anchorEnd = min(a0 + Self.paceAnchor, a.count)
            let seed = CoverageMonitor.median(Array(a[a0..<anchorEnd]))
            center = seed + rate * Double(cur - a0)
        } else {                                                 // propagate own trace
            center = center! + rate * Double(cur - centerAt)
        }
        center = min(center!, Double(textEnd))
        centerAt = cur
    }
}
