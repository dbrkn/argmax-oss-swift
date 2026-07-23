//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
@testable import TTSKit

/// Locks the Swift ``SoftAlign`` pace integrator to RD-655 `align.py`.
/// Reference values come from driving the Python `SoftAlign` with the identical
/// synthetic f-history (see the branch's align.py).
final class SoftAlignParityTests: XCTestCase {
    func testPaceIntegratorMatchesPython() {
        let sa = SoftAlign(config: SoftAlignConfig(), biasHeads: [0, 1])
        sa.arm(skipStep: 300)
        let textStart = 100, textEnd = 200
        let fHist: [Float] = (0..<400).map { Float(min(0.7, Double($0) / 400 * 0.9)) }

        let expected: [(n: Int, center: Double, apply: Bool)] = [
            (150, 140.8375, true), (200, 152.0875, true), (250, 163.3375, true),
            (300, 174.5875, true), (350, 185.8375, false), (400, 197.0875, false),
        ]
        for e in expected {
            sa.setStep(e.n)
            sa.updateCenter(fHist: Array(fHist.prefix(e.n)), textStart: textStart, textEnd: textEnd)
            XCTAssertEqual(sa.center ?? .nan, e.center, accuracy: 1e-3, "center@\(e.n)")
            XCTAssertEqual(sa.shouldApply(), e.apply, "shouldApply@\(e.n)")
        }
    }

    func testShouldApplyGating() {
        // stride=2, applyN=1, armed at 50: apply on even steps within [.., 50].
        let sa = SoftAlign(config: SoftAlignConfig(), biasHeads: [0, 1])
        sa.arm(skipStep: 50)
        let expect: [(Int, Bool)] = [(10, true), (11, false), (48, true), (49, false),
                                     (50, true), (51, false), (52, false)]
        for (s, want) in expect {
            sa.setStep(s)
            XCTAssertEqual(sa.shouldApply(), want, "gate@\(s)")
        }
    }

    func testDisarmedAndZeroLambdaAreInert() {
        let sa = SoftAlign(config: SoftAlignConfig(), biasHeads: [0, 1])
        XCTAssertFalse(sa.shouldApply())                       // unarmed
        var z = SoftAlignConfig(); z.lambda = 0
        let sz = SoftAlign(config: z, biasHeads: [0, 1])
        sz.arm(skipStep: 300); sz.setStep(200)
        sz.updateCenter(fHist: [Float](repeating: 0.5, count: 200), textStart: 100, textEnd: 200)
        XCTAssertNil(sz.center)                                // lambda=0 → integrator inert
    }
}
