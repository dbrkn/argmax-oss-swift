// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription
import Foundation

let package = Package(
    name: "argmax-oss-swift",
    platforms: [
        .iOS(.v16),
        .macOS(.v14),
        .watchOS(.v10),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "ArgmaxOSS",
            targets: ["ArgmaxOSS"]
        ),
        .library(
            name: "WhisperKit",
            targets: ["WhisperKit"]
        ),
        .library(
            name: "TTSKit",
            targets: ["TTSKit"]
        ),
        .library(
            name: "SpeakerKit",
            targets: ["SpeakerKit"]
        ),
        .library(
            name: "TTSKitMLX",
            targets: ["TTSKitMLX"]
        ),
        .library(
            name: "ArgmaxOSSDynamic",
            type: .dynamic,
            targets: ["ArgmaxOSS"]
        ),
        .executable(
            name: "argmax-cli",
            targets: ["ArgmaxCLI"]
        ),
        .executable(
            name: "whisperkit-cli",
            targets: ["ArgmaxCLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.7.0"),
        // 0.31.5+ requires swift-tools 6.3; 0.31.4 is the newest release that
        // builds with the Xcode 26.0 toolchain (Swift 6.2).
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.4"),
    ] + (isServerEnabled() ? [
        .package(url: "https://github.com/vapor/vapor.git", from: "4.115.1"),
        .package(url: "https://github.com/apple/swift-openapi-generator", from: "1.10.2"),
        .package(url: "https://github.com/apple/swift-openapi-runtime", from: "1.8.2"),
        .package(url: "https://github.com/swift-server/swift-openapi-vapor", from: "1.0.1"),
    ] : []),
    targets: [
        .target(
            name: "ArgmaxOSS",
            dependencies: [
                "ArgmaxCore",
                "WhisperKit",
                "TTSKit",
                "SpeakerKit",
            ],
            swiftSettings: swiftSettings()
        ),
        .target(
            name: "ArgmaxCore",
            swiftSettings: swiftSettings()
        ),
        .target(
            name: "WhisperKit",
            dependencies: [
                "ArgmaxCore",
            ],
            swiftSettings: swiftSettings()
        ),
        .target(
            name: "TTSKit",
            dependencies: [
                "ArgmaxCore",
            ],
            swiftSettings: swiftSettings()
        ),
        .target(
            name: "SpeakerKit",
            dependencies: [
                "ArgmaxCore",
                "WhisperKit",
            ],
            swiftSettings: swiftSettings()
        ),
        // MLX talker backend for TTSKit voice cloning. The MLX products are
        // macOS-only (mlx-swift has no watchOS support); on other platforms
        // the target compiles empty behind `#if canImport(MLX)`. No library
        // evolution: mlx-swift modules are not resilient, so this target is
        // excluded from the binary-framework (xcframework) surface.
        .target(
            name: "TTSKitMLX",
            dependencies: [
                "TTSKit",
                .product(name: "MLX", package: "mlx-swift", condition: .when(platforms: [.macOS])),
                .product(name: "MLXNN", package: "mlx-swift", condition: .when(platforms: [.macOS])),
                .product(name: "MLXFast", package: "mlx-swift", condition: .when(platforms: [.macOS])),
            ],
            swiftSettings: swiftSettings(libraryEvolution: false)
        ),
        .testTarget(
            name: "ArgmaxCoreTests",
            dependencies: [
                "ArgmaxCore",
            ],
            resources: [
                .process("External/Resources"),
            ],
            swiftSettings: swiftSettings(libraryEvolution: false)
        ),
        .testTarget(
            name: "WhisperKitTests",
            dependencies: [
                "WhisperKit",
            ],
            exclude: ["UnitTestsPlan.xctestplan"],
            resources: [
                .process("Resources"),
            ],
            swiftSettings: swiftSettings(libraryEvolution: false)
        ),
        .testTarget(
            name: "TTSKitTests",
            dependencies: [
                "TTSKit"
            ],
            resources: [.copy("Resources")],
            swiftSettings: swiftSettings(libraryEvolution: false)
        ),
        .testTarget(
            name: "SpeakerKitTests",
            dependencies: [
                "SpeakerKit",
                "WhisperKit",
            ],
            resources: [
                .process("Resources"),
            ],
            swiftSettings: swiftSettings(libraryEvolution: false)
        ),
        .testTarget(
            name: "TTSKitMLXTests",
            dependencies: [
                .target(name: "TTSKitMLX", condition: .when(platforms: [.macOS])),
            ],
            swiftSettings: swiftSettings(libraryEvolution: false)
        ),
        .executableTarget(
            name: "ArgmaxCLI",
            dependencies: [
                "WhisperKit",
                "TTSKit",
                "SpeakerKit",
                .target(name: "TTSKitMLX", condition: .when(platforms: [.macOS])),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ] + (isServerEnabled() ? [
                .product(name: "Vapor", package: "vapor"),
                .product(name: "OpenAPIRuntime", package: "swift-openapi-runtime"),
                .product(name: "OpenAPIVapor", package: "swift-openapi-vapor"),
            ] : []),
            path: "Sources/ArgmaxCLI",
            exclude: (isServerEnabled() ? [] : ["Server"]),
            swiftSettings: swiftSettings(libraryEvolution: false) + (isServerEnabled() ? [.define("BUILD_SERVER_CLI")] : [])
        )
    ],
    swiftLanguageModes: [.v6]
)

func isServerEnabled() -> Bool {
    if let enabledValue = Context.environment["BUILD_ALL"] {
        return enabledValue.lowercased() == "true" || enabledValue == "1"
    }

    // Default disabled, change to true temporarily for local development
    return false
}

func swiftSettings(libraryEvolution: Bool = true) -> [SwiftSetting] {
    // Opt-in to Swift 6.2's "Approachable Concurrency" upcoming features.
    // These reduce false-positive concurrency diagnostics by making the
    // compiler infer isolation in places where it's almost always what the
    // developer intended:
    //   - InferIsolatedConformances (SE-0470): a protocol conformance on a
    //     globally-isolated type (e.g. @MainActor) is itself inferred to be
    //     isolated to that same actor, instead of forcing a `nonisolated`
    //     conformance that can't touch the type's state.
    //   - NonisolatedNonsendingByDefault (SE-0461): a `nonisolated` async
    //     function runs on the caller's actor by default rather than hopping
    //     to the generic executor, avoiding spurious Sendable errors on
    //     arguments and return values that never actually cross actors.
    let approachableConcurrencySettings: [SwiftSetting] = [
        .enableUpcomingFeature("InferIsolatedConformances"),
        .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    ]

    // Equivalent to Xcode's BUILD_LIBRARY_FOR_DISTRIBUTION setting: enables
    // library evolution and module stability so these targets can be linked
    // against prebuilt binary frameworks (e.g. an .xcframework) without
    // requiring the framework to be rebuilt for every Swift compiler version.
    let dynamicSettings: [SwiftSetting] = libraryEvolution ? [
        .unsafeFlags([
            "-enable-library-evolution",
            "-Xfrontend", "-alias-module-names-in-module-interface",
        ])
    ] : []

    return approachableConcurrencySettings + dynamicSettings
}
