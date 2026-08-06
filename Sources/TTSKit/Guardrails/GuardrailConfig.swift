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

    /// Intervention alignment: "soft" = SoftAlign Huber bias (RD-655),
    /// "aci" = RD-691 hard-CMask (monotone-DP windows + prefill-DP seeding +
    /// de-paused references). ACI presets resolve with the anchor preset.
    public var alignment: String = "soft"
    /// loc-ACI-deep: deepen every rollback by this many extra steps.
    /// PR27 final ships 0 (plain rollback; prefill-DP carries the fix).
    public var extraRollback: Int = 0
    /// Coverage-complete EOS-force (RD-655 extension to loc-ACI): once the
    /// coverage monitor's FINAL text bin reaches its fair dwell, force EOS
    /// `eosGrace` steps later if the model has not terminated on its own —
    /// stops non-terminating (trailing-hallucination) runaways. Off by
    /// default; the CLI enables it for `--guardrails aci`.
    public var eosPromote: Bool = false
    /// Never-bound rescue (streaming-safe: acts only on unreleased audio).
    /// Runs the v2 BindingMonitor alongside the coverage monitor in the aci
    /// arm; a binding fire restarts the CHUNK from the prefill with a fresh
    /// seed (rollback cannot fix a decode whose entire prefix is garbage).
    public var bindingRescue: Bool = true
    /// Whole-chunk restart budget (binding fires + rejected acceptance),
    /// separate from the located-rollback budget `maxRetries`.
    public var maxRestarts: Int = 2
    /// End-of-chunk acceptance: the chunk is rejected (and restarted) when
    /// BOTH the committed-coverage fraction and the anchor high-water are
    /// below these floors — the never-bound signature at chunk end.
    public var acceptMinCoverage: Double = 0.5
    public var acceptMinHighWater: Float = 0.8
    /// Steps after coverage-complete before forcing EOS (~1 s @ 12.5 Hz):
    /// long enough to finish the last word, short enough to stop before
    /// trailing hallucination.
    public var eosGrace: Int = 12

    /// Guardrails v2: replace the RD-655 coverage monitor with the
    /// ``BindingMonitor`` (sustained flip-flop + text-mass collapse; validated
    /// 100% recall / 0% FPR on clean unchunked trajectories) and replace
    /// mid-stream rollback with full restart-from-prefill on a confirmed
    /// binding failure. Unchunked generation only.
    public var v2: Bool = false
    public var binding = BindingMonitorConfig()

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
    /// UNCHUNKED-PRODUCTION BRANCH: presets carry RD-655's NATIVE operating
    /// point (fairInit 40, maxRetries 10) — the values rongxiang validated on
    /// unchunked long-form generation, which is this branch's regime. The
    /// chunked-production branch carries different overrides (fairInit 15,
    /// maxRetries 3); do not cross-port them. Re-calibration on unchunked
    /// voice-clone data may adjust these (and the anchor head) empirically.
    public static let default06bBase = GuardrailConfig(anchorLayer: 6, anchorHead: 0, biasLayer: 6, biasHeads: [0, 1])
    public static let default17bBase = GuardrailConfig(anchorLayer: 3, anchorHead: 0, biasLayer: 3, biasHeads: [0, 1])

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
