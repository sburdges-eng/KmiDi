// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CoreMLLMRunner",
    platforms: [
        .macOS(.v14),  // Runtime: stateful Core ML requires macOS 15+ / iOS 18+
    ],
    targets: [
        .executableTarget(
            name: "CoreMLLMRunner",
            path: "Sources/CoreMLLMRunner"
        ),
    ]
)
