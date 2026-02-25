import Foundation

/// Manages AI trust control state.
/// Persists to UserDefaults and provides notifications for state changes.
class AITrustManager: ObservableObject {
    static let shared = AITrustManager()

    // UserDefaults keys
    private enum Keys {
        static let globalAIEnabled = "com.kmidi.trust.globalAIEnabled"
        static let emotionAnalysisEnabled = "com.kmidi.trust.emotionAnalysisEnabled"
        static let melodySuggestionsEnabled = "com.kmidi.trust.melodySuggestionsEnabled"
        static let harmonySuggestionsEnabled = "com.kmidi.trust.harmonySuggestionsEnabled"
        static let grooveSuggestionsEnabled = "com.kmidi.trust.grooveSuggestionsEnabled"
        static let teachingOverlaysEnabled = "com.kmidi.trust.teachingOverlaysEnabled"
    }

    // Notification name for trust state changes
    static let trustStateChangedNotification = Notification.Name("AITrustStateChanged")

    @Published var globalAIEnabled: Bool {
        didSet {
            UserDefaults.standard.set(globalAIEnabled, forKey: Keys.globalAIEnabled)
            postNotification()
        }
    }

    @Published var emotionAnalysisEnabled: Bool {
        didSet {
            UserDefaults.standard.set(emotionAnalysisEnabled, forKey: Keys.emotionAnalysisEnabled)
            postNotification()
        }
    }

    @Published var melodySuggestionsEnabled: Bool {
        didSet {
            UserDefaults.standard.set(melodySuggestionsEnabled, forKey: Keys.melodySuggestionsEnabled)
            postNotification()
        }
    }

    @Published var harmonySuggestionsEnabled: Bool {
        didSet {
            UserDefaults.standard.set(harmonySuggestionsEnabled, forKey: Keys.harmonySuggestionsEnabled)
            postNotification()
        }
    }

    @Published var grooveSuggestionsEnabled: Bool {
        didSet {
            UserDefaults.standard.set(grooveSuggestionsEnabled, forKey: Keys.grooveSuggestionsEnabled)
            postNotification()
        }
    }

    @Published var teachingOverlaysEnabled: Bool {
        didSet {
            UserDefaults.standard.set(teachingOverlaysEnabled, forKey: Keys.teachingOverlaysEnabled)
            postNotification()
        }
    }

    private init() {
        // Load from UserDefaults with sensible defaults
        globalAIEnabled = UserDefaults.standard.object(forKey: Keys.globalAIEnabled) as? Bool ?? true
        emotionAnalysisEnabled = UserDefaults.standard.object(forKey: Keys.emotionAnalysisEnabled) as? Bool ?? true
        melodySuggestionsEnabled = UserDefaults.standard.object(forKey: Keys.melodySuggestionsEnabled) as? Bool ?? true
        harmonySuggestionsEnabled = UserDefaults.standard.object(forKey: Keys.harmonySuggestionsEnabled) as? Bool ?? true
        grooveSuggestionsEnabled = UserDefaults.standard.object(forKey: Keys.grooveSuggestionsEnabled) as? Bool ?? true
        teachingOverlaysEnabled = UserDefaults.standard.object(forKey: Keys.teachingOverlaysEnabled) as? Bool ?? false
    }

    private func postNotification() {
        NotificationCenter.default.post(name: Self.trustStateChangedNotification, object: self)
    }

    /// Export trust state as JSON for C++ engine consumption.
    func exportTrustStateJSON() -> String? {
        let state: [String: Any] = [
            "global_ai_enabled": globalAIEnabled,
            "emotion_analysis_enabled": emotionAnalysisEnabled,
            "melody_suggestions_enabled": melodySuggestionsEnabled,
            "harmony_suggestions_enabled": harmonySuggestionsEnabled,
            "groove_suggestions_enabled": grooveSuggestionsEnabled,
            "teaching_overlays_enabled": teachingOverlaysEnabled
        ]

        guard let data = try? JSONSerialization.data(withJSONObject: state, options: .prettyPrinted),
              let jsonString = String(data: data, encoding: .utf8) else {
            return nil
        }

        return jsonString
    }

    /// Write trust state to file for C++ engine.
    func writeTrustStateToFile(at path: String) -> Bool {
        guard let json = exportTrustStateJSON() else {
            return false
        }

        do {
            try json.write(toFile: path, atomically: true, encoding: .utf8)
            return true
        } catch {
            print("Failed to write trust state: \(error)")
            return false
        }
    }
}
