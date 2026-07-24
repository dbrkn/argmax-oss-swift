//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// Configuration for the MLX-talker decoding guardrails (RD-655 port).
///
/// **Model-agnostic vs model-specific.** The detector (``CoverageMonitorConfig``)
/// is fully model-agnostic — its thresholds are in *token* units, validated to
/// port across sizes unchanged. What differs per model is the **text-anchor
/// head** the observable reads `f(t)` from, and the heads the soft-align bias
/// acts on. Upstream re-identified these per checkpoint:
///
/// | model | anchor (layer,head) | bias heads (layer) | source |
/// |---|---|---|---|
/// | 0.6B-Base | L6H0 | L6{0,1} | RD-655 primary calibration |
/// | 1.7B-Base | L3H0 | L3{0,1} | RD-655 §Portability (all-head sweep) |
///
/// Same head *indices* (0/1), different *layer* — so only `anchorLayer` /
/// `biasLayer` change with size. ``resolve(hiddenSize:layerCount:versionDir:)``
/// picks the preset; an explicit override always wins. The anchor coordinate is
/// validated against the loaded model's dims at install (loud error if out of
/// range), because a wrong head silently produces a garbage signal.
public struct GuardrailConfig: Codable, Sendable, Equatable {
    /// Master switch. Off by default — guardrails ship dark until validated on
    /// our own eval set, then the default flips.
    public var enabled: Bool = false

    /// Observe + detect + emit telemetry, but never roll back (Stage 1). The
    /// audio is bit-identical to guardrails-off; use this to *measure* fire rate
    /// and latency on a workload before enabling the fix. Also the only safe
    /// mode under real-time streaming playback (rollback can't un-emit audio).
    public var observeOnly: Bool = false

    // MARK: Model-specific anchor (the only size-dependent knobs)

    /// Layer whose ``anchorHead`` supplies `f(t)`.
    public var anchorLayer: Int
    /// Head index on ``anchorLayer`` read for the text-argmax fraction.
    public var anchorHead: Int
    /// Layer the soft-align bias acts on (normally == ``anchorLayer``).
    public var biasLayer: Int
    /// Head indices on ``biasLayer`` the bias is applied to.
    public var biasHeads: [Int]

    // MARK: Model-agnostic detector + executor

    public var monitor = CoverageMonitorConfig()
    /// Huber bias peak strength; 0 disables the bias (reseed-only recovery).
    public var lambda: Double = 0.2
    /// Huber vertex — quadratic within ±delta of the on-pace center, linear beyond.
    public var delta: Double = 10
    public var stride: Int = 2
    /// Hard cap on rollbacks per generation (upstream default).
    public var maxRetries: Int = 10
    /// Wall-clock budget (s) for all guardrail rollback work in one generation;
    /// exceeding it abandons further rollbacks and returns the current decode
    /// (degrade to baseline — never hang).
    public var maxRollbackSeconds: Double = 120
    /// Read `f(t)` back to host every `observeStride` steps (1 = every step).
    /// The escape valve if a platform shows readback overhead over budget.
    public var observeStride: Int = 1
    /// Record the full per-step `f(t)` trajectory into telemetry (offline
    /// analysis / anchor re-validation). Small but opt-in.
    public var recordTrajectory: Bool = false

    public init(
        anchorLayer: Int = 6, anchorHead: Int = 0,
        biasLayer: Int = 6, biasHeads: [Int] = [0, 1]
    ) {
        self.anchorLayer = anchorLayer
        self.anchorHead = anchorHead
        self.biasLayer = biasLayer
        self.biasHeads = biasHeads
    }

    /// Validated presets. `default06bBase` (L6H0) and `default17bBase` (L3H0).
    ///
    /// **`monitor.fairInit` is lowered from RD-655's 40 to 15 for our regime.**
    /// RD-655 calibrated 40 on long single-shot texts that dwell ~40 decode
    /// steps per coverage bin. Our voice-clone synthesis is short (≈31 text
    /// tokens over ≈100 decode steps → ≈13 steps/bin), so at 40 the commit
    /// threshold `commit_frac·fair`=12 marks nearly every genuinely-covered bin
    /// as under-dwelt → a false-positive skip flood. Measured on a captured
    /// 100%-monotone (healthy) `f(t)`: fairInit=40 → 60 fires, fairInit≤20 → 0.
    /// 15 leaves margin below the flood knee while keeping the stall path (which
    /// keys off frozen coverage, not dwell) intact. The pure
    /// `CoverageMonitorConfig` default stays 40 so RD-655 parity tests are
    /// unaffected; only the deployed guardrail presets carry the retune.
    /// `maxRetries` is lowered from RD-655's 10 to **3** for the chunked
    /// production path. Calibration on the RD-655 corpora through the chunked
    /// pipeline showed the online detector's per-chunk precision is ~5% (the
    /// chunked regime rarely exhibits coverage failures, and f(t) does not
    /// separate the residual bad chunks), so most rollbacks are blind
    /// resampling: the first retries carry that benefit, while retries 4–10
    /// were pure thrash (51/99 eval samples hit the 10-retry cap, +40% mean
    /// generation latency).
    public static let default06bBase: GuardrailConfig = {
        var c = GuardrailConfig(anchorLayer: 6, anchorHead: 0, biasLayer: 6, biasHeads: [0, 1])
        c.monitor.fairInit = 15
        c.maxRetries = 3
        return c
    }()
    public static let default17bBase: GuardrailConfig = {
        var c = GuardrailConfig(anchorLayer: 3, anchorHead: 0, biasLayer: 3, biasHeads: [0, 1])
        c.monitor.fairInit = 15
        c.maxRetries = 3
        return c
    }()

    /// Pick the anchor preset for the loaded model. The layer count is the
    /// robust discriminator (0.6B talker has more layers than the 1.7B's
    /// consolidated stack); `versionDir` is a fallback hint. Detector/executor
    /// knobs are model-agnostic and carried unchanged.
    public static func resolve(hiddenSize: Int? = nil, layerCount: Int? = nil, versionDir: String? = nil) -> GuardrailConfig {
        // 1.7B-Base: the anchor family consolidated into L3 (RD-655 §Portability).
        if let v = versionDir?.lowercased(), v.contains("1.7b") { return default17bBase }
        if let h = hiddenSize, h >= 1536 { return default17bBase }   // 1.7B hidden 2048 vs 0.6B 1024
        return default06bBase
    }

    /// Validate the anchor/bias coordinates against the loaded model's geometry.
    /// Throws with a precise message rather than silently reading a bad head.
    public func validate(layerCount: Int, headsPerLayer: Int) throws {
        func check(_ layer: Int, _ head: Int, _ what: String) throws {
            guard layer >= 0, layer < layerCount else {
                throw GuardrailError.invalidAnchor("\(what) layer \(layer) out of range [0,\(layerCount))")
            }
            guard head >= 0, head < headsPerLayer else {
                throw GuardrailError.invalidAnchor("\(what) head \(head) out of range [0,\(headsPerLayer))")
            }
        }
        try check(anchorLayer, anchorHead, "anchor")
        for h in biasHeads { try check(biasLayer, h, "bias") }
    }
}

public enum GuardrailError: Error, CustomStringConvertible {
    case invalidAnchor(String)
    public var description: String {
        switch self {
        case .invalidAnchor(let m): return "Guardrail anchor invalid: \(m)"
        }
    }
}
