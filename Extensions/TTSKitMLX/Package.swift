// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// Standalone Mac-only extension package. Deliberately NOT referenced by the
// root manifests: `mlx-swift` floors at macOS 14 / iOS 17 (no watchOS), so
// pulling it into `TTSKit` would force a package-wide platform bump. See
// docs/voice-clone-design.md, "Backend decision" row D.
let package = Package(
    name: "TTSKitMLX",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "TTSKitMLX",
            targets: ["TTSKitMLX"]
        ),
        .executable(
            name: "ttskit-mlx-cli",
            targets: ["TTSKitMLXCLI"]
        ),
    ],
    dependencies: [
        .package(path: "../.."),
        // 0.31.5+ requires swift-tools 6.3; 0.31.4 is the newest release that
        // builds with the Xcode 26.0 toolchain (Swift 6.2).
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.4"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "TTSKitMLX",
            dependencies: [
                .product(name: "TTSKit", package: "argmax-oss-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "TTSKitMLXCLI",
            dependencies: [
                "TTSKitMLX",
                .product(name: "TTSKit", package: "argmax-oss-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "TTSKitMLXTests",
            dependencies: ["TTSKitMLX"]
        ),
    ]
)
