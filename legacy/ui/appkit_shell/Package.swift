// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "KmiDiAppKitShell",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "KmiDiApp", targets: ["KmiDiApp"]),
    ],
    targets: [
        .executableTarget(
            name: "KmiDiApp",
            path: "Sources/KmiDiApp"
        ),
    ]
)
