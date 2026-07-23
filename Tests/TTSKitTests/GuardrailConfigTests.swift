//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
@testable import TTSKit

/// The detector is model-agnostic; the anchor head is not. These lock the
/// per-model anchor presets (0.6B → L6H0, 1.7B → L3H0) and the load-time
/// validation that refuses a head the model doesn't have.
final class GuardrailConfigTests: XCTestCase {
    func testResolvesAnchorByModelSize() {
        // 0.6B: hidden 1024 → L6H0.
        let c06 = GuardrailConfig.resolve(hiddenSize: 1024, versionDir: "12hz-0.6b-base")
        XCTAssertEqual([c06.anchorLayer, c06.anchorHead], [6, 0])
        XCTAssertEqual(c06.biasHeads, [0, 1])
        // 1.7B: hidden 2048 → L3H0 (same head indices, consolidated layer).
        let c17 = GuardrailConfig.resolve(hiddenSize: 2048, versionDir: "12hz-1.7b-base")
        XCTAssertEqual([c17.anchorLayer, c17.anchorHead], [3, 0])
        XCTAssertEqual(c17.biasLayer, 3)
        // versionDir hint alone is enough.
        XCTAssertEqual(GuardrailConfig.resolve(versionDir: "12hz-1.7b-base").anchorLayer, 3)
        XCTAssertEqual(GuardrailConfig.resolve(versionDir: "12hz-0.6b-base").anchorLayer, 6)
    }

    func testDetectorKnobsAreModelAgnostic() {
        // The monitor operating point does NOT change between sizes.
        XCTAssertEqual(GuardrailConfig.default06bBase.monitor, GuardrailConfig.default17bBase.monitor)
    }

    func testValidateRejectsOutOfRangeAnchor() {
        // 0.6B preset (L6) on a model with only 4 layers must throw.
        XCTAssertThrowsError(try GuardrailConfig.default06bBase.validate(layerCount: 4, headsPerLayer: 16))
        // In-range passes.
        XCTAssertNoThrow(try GuardrailConfig.default17bBase.validate(layerCount: 28, headsPerLayer: 16))
        // Bad head index throws.
        var bad = GuardrailConfig.default17bBase
        bad.biasHeads = [0, 99]
        XCTAssertThrowsError(try bad.validate(layerCount: 28, headsPerLayer: 16))
    }
}
