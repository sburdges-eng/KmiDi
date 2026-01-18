import Foundation

/// Provides default paths for snapshot files.
/// Paths are configurable via UserDefaults.
struct SnapshotPaths {
    private enum UserDefaultsKeys {
        static let snapshotDirectory = "com.kmidi.snapshotDirectory"
        static let intentSchemaPath = "com.kmidi.intentSchemaPath"
    }

    /// Default snapshot directory (can be overridden)
    static var snapshotDirectory: String {
        get {
            if let path = UserDefaults.standard.string(forKey: UserDefaultsKeys.snapshotDirectory), !path.isEmpty {
                return path
            }
            // Default: ~/Library/Application Support/KmiDi/snapshots
            let home = NSHomeDirectory()
            return (home as NSString).appendingPathComponent("Library/Application Support/KmiDi/snapshots")
        }
        set {
            UserDefaults.standard.set(newValue, forKey: UserDefaultsKeys.snapshotDirectory)
        }
    }

    /// Emotion snapshot file path
    static var emotionSnapshot: String {
        (snapshotDirectory as NSString).appendingPathComponent("emotion_snapshot.json")
    }

    /// ML debug snapshot file path
    static var mlDebugSnapshot: String {
        (snapshotDirectory as NSString).appendingPathComponent("ml_debug_snapshot.json")
    }

    /// Intent schema file path (from Python)
    static var intentSchema: String {
        get {
            if let path = UserDefaults.standard.string(forKey: UserDefaultsKeys.intentSchemaPath), !path.isEmpty {
                return path
            }
            // Default: look in orchestrator output directory
            let home = NSHomeDirectory()
            return (home as NSString).appendingPathComponent("orchestrator_outputs/final_intent.json")
        }
        set {
            UserDefaults.standard.set(newValue, forKey: UserDefaultsKeys.intentSchemaPath)
        }
    }

    /// Trust state file path (for C++ engine)
    static var trustState: String {
        (snapshotDirectory as NSString).appendingPathComponent("trust_state.json")
    }

    /// Ensure snapshot directory exists
    static func ensureDirectoryExists() {
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: snapshotDirectory) {
            try? fileManager.createDirectory(atPath: snapshotDirectory, withIntermediateDirectories: true, attributes: nil)
        }
    }
}
