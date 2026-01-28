import Foundation

/// Debug information snapshot from ML pipeline.
/// Decoded from JSON snapshot file.
struct MLDebugSnapshot: Codable {
    let version: Int
    let timestamp: Double
    let emotion: EmotionStateSnapshot.Emotion
    let labels: EmotionStateSnapshot.Labels?
    let mlDebug: MLDebugInfo

    struct MLDebugInfo: Codable {
        let emotionRecognizerEnabled: Bool
        let melodyTransformerEnabled: Bool
        let harmonyPredictorEnabled: Bool
        let dynamicsEngineEnabled: Bool
        let groovePredictorEnabled: Bool
        let inferenceTimeMs: Double
        let fallbackActive: Bool
        let noteProbSummary: [Float]
        let chordProbSummary: [Float]
        let grooveSummary: [Float]
        let dynamicsSummary: [Float]
    }

    /// Validate version compatibility.
    func isValidVersion() -> Bool {
        version == 1
    }

    /// Safe decoding with defaults for missing fields.
    static func decode(from data: Data) -> MLDebugSnapshot? {
        let decoder = JSONDecoder()
        do {
            return try decoder.decode(MLDebugSnapshot.self, from: data)
        } catch {
            print("Failed to decode MLDebugSnapshot: \(error)")
            return nil
        }
    }

    /// Load from file path.
    static func load(from filePath: String) -> MLDebugSnapshot? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else {
            return nil
        }
        return decode(from: data)
    }
}
