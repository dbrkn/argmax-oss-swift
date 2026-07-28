//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// A code decoder that can expose the RD-655 text-anchor fraction `f(t)` for
/// online failure detection (Stage 1, observe-only).
///
/// Declared in TTSKit so the generation orchestrator can drive detection
/// without importing TTSKitMLX; the MLX talker conforms. A decoder that does
/// not conform (e.g. the CoreML `Qwen3CodeDecoder`, whose attention is sealed
/// in the compiled asset) simply makes guardrails a no-op.
public protocol GuardrailObservable: AnyObject {
    /// Arm the anchor probe for one generation. `anchorLayer`/`anchorHead` are
    /// the model-specific text-anchor head (``GuardrailConfig``); `textStart`/
    /// `textEnd` are the **absolute KV positions** of the synthesis text in the
    /// prefill (the orchestrator knows them from building the prefix).
    func beginGuardrailObservation(
        anchorLayer: Int, anchorHead: Int, textStart: Int, textEnd: Int, recordTrajectory: Bool)

    /// The most recent decode step's `f(t) ∈ [0,1]` (nil before the first step).
    var lastAnchorFraction: Float? { get }

    /// The most recent step's attention mass inside the text span (nil before
    /// the first step, or if the decoder doesn't compute it). Consumed by the
    /// v2 BindingMonitor. Default nil.
    var lastAnchorTextMass: Float? { get }

    /// The recorded `f(t)` trajectory (empty unless recording was requested).
    func guardrailTrajectory() -> [Float]

    /// Diagnostics for anchor validation: per-step global argmax (absolute KV
    /// position) + text-span attention mass, and the span itself. Empty/-1 when
    /// not recording.
    func guardrailDiagnostics() -> (globalArgmax: [Int], textMass: [Float], textStart: Int, textEnd: Int)

    /// Truncate the recorded trajectory to `n` steps (mirrors a rollback).
    func truncateGuardrailTrajectory(to n: Int)

    /// Disarm and clear the probe at end of generation.
    func endGuardrailObservation()

    /// Executor (Stage 2): set the soft-align bias for the next forward. While
    /// `active`, the decoder adds `−lambda·huber(|pos − center|; delta)` to the
    /// attention scores of `biasHeads` on `biasLayer` over the synthesis-text KV
    /// span, pulling those heads onto the on-pace text position. Called each step
    /// during a biased-retry window; `active: false` clears it (no-op forward).
    /// Default no-op for observe-only decoders.
    func setGuardrailBias(active: Bool, center: Double, lambda: Double, delta: Double, biasLayer: Int, biasHeads: [Int])

    /// Executor (Stage 2): rewind the decoder's internal KV state to
    /// `decodeStep` decode steps after the prefill, so the next forward
    /// re-decodes from there. O(1) trim (append-only cache). The orchestrator
    /// separately trims its own external cache + code buffer + monitor.
    func guardrailRollback(toDecodeStep decodeStep: Int)
}

extension GuardrailObservable {
    // Default no-op so observe-only decoders need not implement it.
    public func guardrailRollback(toDecodeStep decodeStep: Int) {}
    public var lastAnchorTextMass: Float? { nil }
    public func setGuardrailBias(active: Bool, center: Double, lambda: Double, delta: Double, biasLayer: Int, biasHeads: [Int]) {}
    public func guardrailDiagnostics() -> (globalArgmax: [Int], textMass: [Float], textStart: Int, textEnd: Int) {
        ([], [], -1, -1)
    }
}
