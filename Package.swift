// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ArraySwift",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
        .visionOS(.v1),
        .watchOS(.v8),
        .tvOS(.v15)
    ],
    products: [
        .library(
            name: "ArraySwift",
            targets: ["ArraySwift"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "ArraySwift",
            path: "Sources/ArraySwift"
        ),
        .testTarget(
            name: "ArraySwiftTests",
            dependencies: ["ArraySwift"]
        ),
    ]
)
