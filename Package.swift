// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "ArraySwift",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "ArraySwift",
            targets: ["ArraySwift"]
        ),
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
