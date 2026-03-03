import Foundation

/// Immutable snapshot of EmotionState from C++ engine.
/// Decoded from JSON snapshot file.
struct EmotionStateSnapshot: Codable {
    let version: Int
    let timestamp: Double
    let emotion: Emotion
    let labels: Labels?

    struct Emotion: Codable {
        let valence: Float
        let arousal: Float
        let dominance: Float
        let complexity: Float
    }

    struct Labels: Codable {
        let primary: String?
        let secondary: String?
    }

    /// Validate version compatibility.
    func isValidVersion() -> Bool {
        version == 1
    }

    /// Safe decoding with defaults for missing fields.
    static func decode(from data: Data) -> EmotionStateSnapshot? {
        let decoder = JSONDecoder()
        do {
            return try decoder.decode(EmotionStateSnapshot.self, from: data)
        } catch {
            print("Failed to decode EmotionStateSnapshot: \(error)")
            return nil
        }
    }

    /// Load from file path.
    static func load(from filePath: String) -> EmotionStateSnapshot? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else {
            return nil
        }
        return decode(from: data)
    }
}
