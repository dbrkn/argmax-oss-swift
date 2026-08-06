//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// Per-generation guardrail telemetry — the observability the run needs and
/// production monitoring wants. Attached to the speech result alongside timings
/// and surfaced through the CLI / OpenBench sink.
///
/// Metrics chosen to answer: *did it fire, how often, on what, how much extra
/// work did it cost, and did it give up?* — plus the raw trajectory for offline
/// anchor validation.
public struct GuardrailStats: Sendable, Equatable {
    /// One rollback event.
    public struct Event: Sendable, Equatable {
        public let failure: GuardrailFailure.Kind
        public let fireStep: Int
        public let rollbackStep: Int
        /// Audio span rewound and re-decoded, seconds (`(fire−rollback)/12.5`).
        public var rewoundSeconds: Double { Double(fireStep - rollbackStep) / 12.5 }
        public init(failure: GuardrailFailure.Kind, fireStep: Int, rollbackStep: Int) {
            self.failure = failure; self.fireStep = fireStep; self.rollbackStep = rollbackStep
        }
    }

    /// Guardrails ran at all (enabled + MLX talker + anchor valid).
    public var active = false
    /// Observe-only (detected + logged, no rollback applied).
    public var observeOnly = false
    /// Whether any failure fired this generation.
    public var fired: Bool { !events.isEmpty }
    /// Every fire, in order (includes re-fires of the same span).
    public var events: [Event] = []
    /// Rollbacks actually executed (== events.count in fix mode; 0 observe-only).
    public var rollbacks = 0
    /// Distinct fires by type.
    public var skipFires = 0
    public var hallucinationFires = 0
    /// Total audio-seconds rewound (Σ rewoundSeconds of executed rollbacks).
    public var rewoundAudioSeconds = 0.0
    /// Wall-clock the guardrail machinery added (observable readback + monitor +
    /// re-decode), seconds — the production latency cost, measured.
    public var addedWallSeconds = 0.0
    /// Per-step observable overhead, mean ms (readback + monitor step).
    public var meanObserveMs = 0.0
    /// Hit `maxRetries` or `maxRollbackSeconds` and returned the un-fixed decode.
    public var gaveUp = false
    /// EOS was forced by coverage-complete promotion (RD-655 `eosPromote`) —
    /// the model had spoken the full text but did not terminate on its own.
    public var eosPromoted = false
    /// Whole-chunk restarts executed (binding fires + rejected acceptance).
    public var restarts = 0
    /// One entry per restart: "reason@step#attempt" (e.g. "binding@190#1").
    public var restartLog: [String] = []
    /// End-of-chunk acceptance record (aci + bindingRescue only).
    public var acceptCoverage = 1.0
    public var acceptHighWater: Float = 1
    public var acceptAccepted = true
    /// Anchor coordinate actually used (for the run record; model-specific).
    public var anchor: [Int] = []
    /// Optional full `f(t)` trajectory (only when `recordTrajectory`).
    public var fTrajectory: [Float]?

    public init() {}

    mutating func record(_ f: GuardrailFailure) {
        events.append(Event(failure: f.failure, fireStep: f.fireStep, rollbackStep: f.rollbackStep))
        switch f.failure {
        case .skip: skipFires += 1
        case .hallucination: hallucinationFires += 1
        }
    }

    /// One-line summary for logs.
    public var summary: String {
        guard active else { return "guardrails: inactive" }
        if observeOnly {
            return String(format: "guardrails(observe): fires=%d (skip=%d hall=%d) overhead=%.2fms/step",
                          events.count, skipFires, hallucinationFires, meanObserveMs)
        }
        return String(format: "guardrails: rollbacks=%d (skip=%d hall=%d) rewound=%.1fs added=%.1fs%@",
                      rollbacks, skipFires, hallucinationFires, rewoundAudioSeconds, addedWallSeconds,
                      gaveUp ? " [gave-up]" : "")
    }
}
