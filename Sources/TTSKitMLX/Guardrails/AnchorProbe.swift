//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

#if canImport(MLX)

import Foundation
import MLX
import TTSKit

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
    /// Most recent step's attention mass inside the text span (softmax over the
    /// full KV row). Computed every step — the v2 BindingMonitor consumes it
    /// live (mass collapse is half of the binding-failure signature).
    public private(set) var lastTextMass: Float?
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

    // MARK: Head scan (anchor re-calibration, observe-only tuning runs)
    //
    // When `headScan` is on, every decode step records the text-span argmax of
    // EVERY head on EVERY layer (not just the anchor head), so an offline sweep
    // can rank alternative anchor heads for a new regime (e.g. chunked
    // voice-clone). The per-layer scores stay lazy on-device (`scanStepBuf`)
    // and are read back once per step via ``flushScanStep()``; observe-only,
    // never active alongside rollbacks.
    public let headScan: Bool
    public private(set) var scanLayers = 0
    public private(set) var scanHeads = 0
    /// Per step: layer-major flattened `[scanLayers × scanHeads]` absolute
    /// argmax positions within the text span.
    public private(set) var scanSteps: [[Int32]] = []
    private var scanStepBuf: [Int: MLXArray] = [:]

    // MARK: ACI hard-CMask (RD-691)
    //
    // When `aci` is attached, the forward calls ``aciDecodeMask`` on every
    // group layer each decode step: the probe computes each configured head's
    // text-region attention distribution (softmax over the full row, sliced to
    // the ICL text span — python parity), hands it to the pure-Swift
    // ``ACIAlign`` DP, and converts the returned hard band into the additive
    // SDPA mask. During prefill, ``aciPrefillSeed`` captures the REFERENCE
    // codec frames' rows and advances the DP (pDP) so the decode continues
    // from the reference alignment.
    public var aci: ACIAlign?
    /// Full ICL text span (reference + main text) — the ACI DP domain.
    public var aciTextStart: Int = -1
    public var aciTextEnd: Int = -1
    /// Reference codec-frame span (pDP rows).
    public var refCodecStart: Int = -1
    public var refCodecEnd: Int = -1

    /// The anchor head's band from the last masked step (python parity: the
    /// monitor's f is read POST-CMask, so it judges the corrected trajectory).
    private var lastAnchorBand: [Float]?

    /// Decode-step ACI for one group layer. Returns the additive mask
    /// `(numHeads, T)` flattened head-major, or nil when not masking this step.
    func aciDecodeMask(layer: Int, q: MLXArray, cachedK: MLXArray, scale: Float, numHeads: Int) -> MLXArray? {
        guard let aci, aci.layers.contains(layer), aciTextStart >= 0 else { return nil }
        let t = cachedK.dim(2)
        let tt = min(aciTextEnd, t)
        guard tt > aciTextStart else { return nil }
        let ntf = aciTextEnd - aciTextStart
        let navail = tt - aciTextStart
        let numKV = cachedK.dim(1)
        let grp = numHeads / numKV
        var segs: [Int: [Float]] = [:]
        for h in aci.heads(for: layer) {
            let qh = q[0, h, 0, 0...]
            let kh = cachedK[0, h / grp]
            let scores = (kh.matmul(qh) * scale)
            let probs = softmax(scores, axis: -1)              // full row, then slice (parity)
            segs[h] = probs[aciTextStart ..< tt].asArray(Float.self)
        }
        let apply = aci.shouldApply()
        let bands = aci.layerCMask(layer: layer, segs: segs, pf: aciTextStart, tt: tt,
                                   T: t, ntf: ntf, navail: navail, apply: apply)
        // Python parity: the monitor's f is read POST-CMask on the anchor head,
        // so it judges the corrected trajectory (else it never sees the mask
        // working and keeps re-firing).
        if layer == anchorLayer {
            lastAnchorBand = apply ? bands?[anchorHead] : nil
        }
        guard apply, let bands, !bands.isEmpty else { return nil }
        var flat = [Float](repeating: 0, count: numHeads * t)
        for (h, band) in bands {
            for i in 0..<t { flat[h * t + i] = band[i] }
        }
        return MLXArray(flat, [1, numHeads, 1, t])
    }

    /// Prefill pDP capture for one group layer: causal-masked full-row softmax
    /// of the reference codec frames' rows, sliced to the ICL text span, fed
    /// row-by-row into the DP.
    func aciPrefillSeed(layer: Int, q: MLXArray, cachedK: MLXArray, scale: Float, numHeads: Int) {
        guard let aci, aci.layers.contains(layer), refCodecStart >= 0, aciTextStart >= 0 else { return }
        let t = cachedK.dim(2)
        let rs = min(refCodecStart, t), re = min(refCodecEnd, t)
        let tt = min(aciTextEnd, t)
        guard re > rs, tt > aciTextStart else { return }
        let ntf = aciTextEnd - aciTextStart
        let numKV = cachedK.dim(1)
        let grp = numHeads / numKV
        // causal mask for the row block: row (absolute pos p) may attend cols <= p
        let rows = MLXArray(Array(Int32(rs)..<Int32(re)), [re - rs, 1])
        let cols = MLXArray(Array(Int32(0)..<Int32(t)), [1, t])
        let causal = (cols .> rows).asType(.float32) * Float(-1e30)
        for h in aci.heads(for: layer) {
            let qh = q[0, h, rs..<re, 0...]                      // (R, hd)
            let kh = cachedK[0, h / grp]                          // (T, hd)
            let scores = qh.matmul(kh.transposed(1, 0)) * scale + causal   // (R, T)
            let probs = softmax(scores, axis: -1)[0..., aciTextStart ..< tt]
            let flat = probs.asArray(Float.self)                  // R × navail
            let navail = tt - aciTextStart
            var rowsArr: [[Float]] = []
            rowsArr.reserveCapacity(re - rs)
            for r in 0..<(re - rs) {
                rowsArr.append(Array(flat[(r * navail)..<((r + 1) * navail)]))
            }
            aci.seedFromPrefill(layer: layer, head: h, rows: rowsArr, ntf: ntf)
        }
    }

    // MARK: Soft-align bias (RD-655 Stage-2 recovery)
    //
    // Set by the orchestrator each decode step while a rollback recovery is
    // armed; read by `TalkerAttention` on `biasLayer`. When `biasActive`, an
    // additive Huber penalty `−lambda·huber(|pos − biasCenter|; biasDelta)` over
    // the text span `[textStart, textEnd)` is added to the attention scores of
    // `biasHeads`, pulling those heads back onto the on-pace text position. This
    // *does* change the SDPA output (and thus the audio) — it is the directed
    // intervention, active only during a bounded biased-retry window.
    public var biasLayer: Int = -1
    public var biasHeads: Set<Int> = []
    public var biasActive = false
    public var biasCenter: Double = 0
    public var biasLambda: Double = 0
    public var biasDelta: Double = 10

    public init(anchorLayer: Int, anchorHead: Int, textStart: Int, textEnd: Int,
                recordTrajectory: Bool = false, headScan: Bool = false) {
        self.anchorLayer = anchorLayer
        self.anchorHead = anchorHead
        self.textStart = max(0, textStart)
        self.textEnd = max(textStart + 1, textEnd)
        self.record = recordTrajectory
        self.headScan = headScan
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

    /// Head scan: record the text-span argmax of every head on `layer` for this
    /// decode step. Lazy (no host sync here) — ``flushScanStep()`` reads the
    /// whole step's layers back in one transfer. Scale is omitted: argmax is
    /// invariant to a positive scalar.
    func observeScan(layer: Int, q: MLXArray, cachedK: MLXArray) {
        let t = cachedK.dim(2)
        let end = min(textEnd, t)
        guard end > textStart else { return }
        let numHeads = q.dim(1)
        let numKV = cachedK.dim(1)
        let grp = numHeads / numKV
        let qg = q[0, 0..., 0, 0...].reshaped(numKV, grp, q.dim(3))     // (numKV, grp, hd)
        let kAll = cachedK[0]                                            // (numKV, T, hd)
        let scores = qg.matmul(kAll.transposed(0, 2, 1))                 // (numKV, grp, T)
        let seg = scores[0..., 0..., textStart ..< end]
        scanStepBuf[layer] = argMax(seg, axis: -1).reshaped(numHeads) + textStart
        if layer >= scanLayers { scanLayers = layer + 1 }
        scanHeads = numHeads
    }

    /// Read this step's buffered per-layer scans back to host (one transfer).
    /// Called by the decoder once per forward, after its own eval.
    public func flushScanStep() {
        guard headScan, !scanStepBuf.isEmpty else { return }
        let layers = scanStepBuf.keys.sorted()
        let stacked = concatenated(layers.compactMap { scanStepBuf[$0] }, axis: 0)
        scanSteps.append(stacked.asType(.int32).asArray(Int32.self))
        scanStepBuf.removeAll(keepingCapacity: true)
    }

    /// Additive attention-score bias for the bias layer during a biased-retry
    /// step, or `nil` when inactive. Shape `(1, numHeads, 1, T)`, broadcasting
    /// over the single decode query: `biasHeads` rows carry the Huber penalty
    /// over `[textStart, min(textEnd, T))`, all other rows/positions are 0. Added
    /// to the SDPA mask, so 0 is a no-op for untouched heads/positions.
    func biasScores(t: Int, numHeads: Int) -> MLXArray? {
        guard biasActive, biasLambda > 0, !biasHeads.isEmpty else { return nil }
        let end = min(textEnd, t)
        guard end > textStart else { return nil }
        // Normalized Huber penalty over the text span (matches hook.py: quadratic
        // core 0.5·a²/δ within δ, unit-slope linear a−0.5δ beyond).
        var pen = [Float](repeating: 0, count: t)
        let d = biasDelta
        for pos in textStart..<end {
            let dist = abs(Double(pos) - biasCenter)
            let huber = dist < d ? 0.5 * dist * dist / d : dist - 0.5 * d
            pen[pos] = Float(-biasLambda * huber)
        }
        let penRow = MLXArray(pen, [1, 1, 1, t])                 // (1,1,1,T)
        // Per-head gate: 1 for bias heads, 0 otherwise → (1, numHeads, 1, 1).
        var gate = [Float](repeating: 0, count: numHeads)
        for h in biasHeads where h < numHeads { gate[h] = 1 }
        let gateCol = MLXArray(gate, [1, numHeads, 1, 1])
        return penRow * gateCol                                  // broadcast → (1, numHeads, 1, T)
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
        var scores = (kh.matmul(qh) * scale)                // (T)
        // Read f POST-bias when the anchor head is itself biased (matches
        // hook.py): the monitor must judge the *corrected* trajectory, else it
        // never sees the soft-align bias working and keeps re-firing. Adds the
        // same Huber penalty the SDPA path applies, over the text span.
        if biasActive, biasLambda > 0, biasHeads.contains(anchorHead) {
            var pen = [Float](repeating: 0, count: t)
            for pos in textStart..<end {
                let dist = abs(Double(pos) - biasCenter)
                let huber = dist <= biasDelta ? 0.5 * dist * dist / biasDelta : dist - 0.5 * biasDelta
                pen[pos] = Float(-biasLambda * huber)
            }
            scores = scores + MLXArray(pen, [t])
        }
        // ACI post-mask read: apply the anchor head's hard band from this
        // step's CMask so the monitor judges the corrected trajectory.
        if let band = lastAnchorBand, band.count == t {
            scores = scores + MLXArray(band, [t])
        }
        let seg = scores[textStart ..< end]
        let am = textStart + argMax(seg).item(Int.self)     // absolute attended text position
        let f = Float(am - textStart) / Float(max(1, textEnd - textStart))
        lastF = f
        // Text-span attention mass every step (softmax over the full KV row):
        // consumed live by the v2 BindingMonitor, recorded for diagnostics.
        let probs = softmax(scores, axis: -1)
        let mass = probs[textStart ..< end].sum().item(Float.self)
        lastTextMass = mass
        if record {
            trajectory.append(f)
            globalArgmax.append(argMax(scores).item(Int.self))
            textMass.append(mass)
        }
    }
}

#endif // canImport(MLX)
