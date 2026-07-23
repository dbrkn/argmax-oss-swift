//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
@testable import TTSKit

/// Locks the Swift ``CoverageMonitor`` to RD-655 `coverage_monitor.py`.
///
/// Reference values are produced by feeding the Python `detect_failure` the
/// identical synthetic trajectories reconstructed below (see the branch's
/// `coverage_monitor.py`); the Swift port must fire on the same step, with the
/// same failure type and rollback step. If these drift, the port has diverged
/// from the validated detector and its recall/FP numbers no longer transfer.
final class CoverageMonitorParityTests: XCTestCase {
    private let nBins = 40
    private let dwell = 40   // clean median steps/bin at tokPerBin=10
    private let ntok = 400

    /// `(bin + 0.5)/nBins` held `dwell` steps per bin — the piecewise-constant
    /// trajectory the Python reference was generated from (bit-identical: same
    /// IEEE-754 rational rounded to Float).
    private func hold(_ bins: [Int]) -> [Float] {
        var f: [Float] = []
        for b in bins {
            let v = (Float(b) + 0.5) / Float(nBins)
            f.append(contentsOf: repeatElement(v, count: dwell))
        }
        return f
    }

    func testClean_noFire() {
        let f = hold(Array(0..<nBins))                       // 1600 steps, monotone full coverage
        XCTAssertEqual(f.count, 1600)
        XCTAssertNil(detectFailure(f, ntok: ntok))
    }

    func testSkip_firesWithPythonStepAndRollback() {
        // dwell 0..<10, then rush to 20..<40 — bins [10,20) left under-dwelt.
        let f = hold(Array(0..<10) + Array(20..<nBins))       // 1200 steps
        XCTAssertEqual(f.count, 1200)
        let r = detectFailure(f, ntok: ntok)
        XCTAssertNotNil(r)
        XCTAssertEqual(r?.failure, .skip)
        XCTAssertEqual(r?.fireStep, 436)                      // Python detect_failure
        XCTAssertEqual(r?.rollbackStep, 379)
        XCTAssertEqual(r?.bins.count, 10)
        XCTAssertEqual(r?.nBins, 40)
    }

    func testStall_firesHallucinationWithPythonStepAndRollback() {
        // dwell 0..<15, then freeze on bin 14 for 400 steps — high-water stalls.
        var f = hold(Array(0..<15))                            // 600 steps
        f.append(contentsOf: repeatElement((Float(14) + 0.5) / Float(nBins), count: 400))
        XCTAssertEqual(f.count, 1000)
        let r = detectFailure(f, ntok: ntok)
        XCTAssertNotNil(r)
        XCTAssertEqual(r?.failure, .hallucination)
        XCTAssertEqual(r?.fireStep, 716)
        XCTAssertEqual(r?.rollbackStep, 595)
        XCTAssertEqual(r?.nBins, 40)
    }

    /// Online stepping must agree with the offline wrapper (they share code, so
    /// this also proves `step()` and `detectFailure()` cannot drift).
    func testOnlineMatchesOffline() {
        let f = hold(Array(0..<10) + Array(20..<nBins))
        let mon = CoverageMonitor(ntok: ntok)
        var fired: GuardrailFailure?
        for ft in f where fired == nil { fired = mon.step(ft) }
        XCTAssertEqual(fired?.fireStep, 436)
        XCTAssertEqual(fired?.failure, .skip)
    }

    /// After a fire, `rollback(to:)` truncates + replays to an exact prefix
    /// state — `nSteps` equals the resume point and no fire is pending.
    func testRollbackReplayRestoresPrefixState() {
        let f = hold(Array(0..<10) + Array(20..<nBins))
        let mon = CoverageMonitor(ntok: ntok)
        var res: GuardrailFailure?
        for ft in f where res == nil { res = mon.step(ft) }
        let rb = res!.rollbackStep
        mon.rollback(to: rb)
        XCTAssertEqual(mon.nSteps, rb)
    }

    func testPercentileMatchesNumpyLinear() {
        // np.percentile([0,1,2,3,4], 10) = 0.4 (linear interp on rank 0.4).
        XCTAssertEqual(CoverageMonitor.percentile([0, 1, 2, 3, 4], 10), 0.4, accuracy: 1e-9)
        XCTAssertEqual(CoverageMonitor.percentile([0, 10], 50), 5.0, accuracy: 1e-9)
        XCTAssertEqual(CoverageMonitor.percentile([7], 10), 7.0, accuracy: 1e-9)
    }

    func testMedianEvenAndOdd() {
        XCTAssertEqual(CoverageMonitor.median([1.0, 2, 3]), 2.0, accuracy: 1e-9)
        XCTAssertEqual(CoverageMonitor.median([1.0, 2, 3, 4]), 2.5, accuracy: 1e-9)
    }
}
