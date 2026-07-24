//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

#if canImport(MLX)

import Foundation
import MLX

/// Per-generation text-anchor observable for the MLX talker (RD-655 `f(t)`).
///
/// Created by the orchestrator for one guarded generation and passed into the
/// talker forward — **never** stored on the shared talker module (the MLX
/// talker runs serialized, but we keep the state per-task regardless). When the
/// forward reaches ``anchorLayer`` on a single-step decode, it computes a
/// *separate* softmax of the anchor head's attention over the main-text KV span
/// `[textStart, textEnd)` and records the normalized argmax fraction
/// `f(t) = (argmax − textStart) / (textEnd − textStart) ∈ [0,1]`.
///
/// **This side computation never touches the real SDPA output or the sampler
/// RNG** — audio is bit-identical to guardrails-off (fidelity invariant).
///
/// Unlike RD-655's custom-voice baseline (which derives the text region from
/// `prefill`/`T_text`), the orchestrator supplies the **absolute KV positions**
/// of the synthesis text directly, because our ICL voice-clone prefix
/// interleaves the text track with the codec track — only the builder knows
/// where the main text sits.
public final class AnchorProbe {
    public let anchorLayer: Int
    public let anchorHead: Int
    /// Absolute KV position of the first main-text token (inclusive).
    public let textStart: Int
    /// One past the last main-text token.
    public let textEnd: Int

    /// Most recent decode step's `f(t)` (nil before the first decode step).
    public private(set) var lastF: Float?
    /// Full trajectory when recording is on (offline anchor validation).
    public private(set) var trajectory: [Float] = []
    /// Diagnostics (recorded alongside `trajectory` when recording): per step the
    /// GLOBAL argmax over the whole KV (absolute position) and the fraction of
    /// attention mass inside the text span. A dip in `f` with low `textMass` /
    /// a `globalArgmax` outside `[textStart,textEnd)` means the anchor is looking
    /// elsewhere (control/reference tokens), i.e. the span/head needs correcting.
    public private(set) var globalArgmax: [Int] = []
    public private(set) var textMass: [Float] = []
    private let record: Bool

    public init(anchorLayer: Int, anchorHead: Int, textStart: Int, textEnd: Int, recordTrajectory: Bool = false) {
        self.anchorLayer = anchorLayer
        self.anchorHead = anchorHead
        self.textStart = max(0, textStart)
        self.textEnd = max(textStart + 1, textEnd)
        self.record = recordTrajectory
    }

    /// Reset for a new generation (or a rollback replay truncation handled by
    /// the caller trimming `trajectory`).
    public func reset() {
        lastF = nil
        trajectory.removeAll(keepingCapacity: true)
        globalArgmax.removeAll(keepingCapacity: true)
        textMass.removeAll(keepingCapacity: true)
    }

    /// Truncate the recorded trajectory to `n` steps (mirrors a monitor rollback).
    public func truncateTrajectory(to n: Int) {
        guard record else { return }
        if trajectory.count > n { trajectory.removeLast(trajectory.count - n) }
        if globalArgmax.count > n { globalArgmax.removeLast(globalArgmax.count - n) }
        if textMass.count > n { textMass.removeLast(textMass.count - n) }
    }

    /// Compute and record `f(t)` from the anchor head's scores against the
    /// cached keys. Called by ``TalkerAttention`` on the anchor layer only.
    ///
    /// - Parameters:
    ///   - q: post-RoPE queries `(1, numHeads, 1, headDim)` for this step.
    ///   - cachedK: post-RoPE keys `(1, numKVHeads, T, headDim)`.
    ///   - scale: attention scale.
    ///   - grp: GQA group size (`numHeads / numKVHeads`).
    func observe(q: MLXArray, cachedK: MLXArray, scale: Float, grp: Int) {
        let t = cachedK.dim(2)
        let end = min(textEnd, t)
        guard end > textStart else { return }               // text region not in cache yet
        let kvHead = anchorHead / grp
        // (headDim) · (headDim, T) -> (T); a separate score, not the SDPA path.
        let qh = q[0, anchorHead, 0, 0...]                  // (headDim)
        let kh = cachedK[0, kvHead]                          // (T, headDim)
        let scores = (kh.matmul(qh) * scale)                // (T)
        let seg = scores[textStart ..< end]
        let am = textStart + argMax(seg).item(Int.self)     // absolute attended text position
        let f = Float(am - textStart) / Float(max(1, textEnd - textStart))
        lastF = f
        if record {
            trajectory.append(f)
            // Diagnostics: where does the head actually look, and how much mass
            // is inside the text span? (softmax over the full KV row.)
            let probs = softmax(scores, axis: -1)
            globalArgmax.append(argMax(scores).item(Int.self))
            textMass.append(probs[textStart ..< end].sum().item(Float.self))
        }
    }
}

#endif // canImport(MLX)
