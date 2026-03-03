import Foundation

/// Intent IR v1 - Swift model matching C IntentFrame struct.
/// Decoded from JSON snapshot file produced by Python or C++.
struct IntentFrame: Codable {
    let meta: IntentMeta
    let emotion: EmotionState
    let music: MusicalIntent
    let time: TimeScope
    let constraints: IntentConstraints
    let provenance: IntentProvenance

    struct IntentMeta: Codable {
        let irVersion: Int
        let intentId: UInt64
        let sessionId: UInt64

        enum CodingKeys: String, CodingKey {
            case irVersion = "ir_version"
            case intentId = "intent_id"
            case sessionId = "session_id"
        }
    }

    struct EmotionState: Codable {
        let valence: Float      // [-1.0, 1.0]
        let arousal: Float     // [0.0, 1.0]
        let dominance: Float   // [0.0, 1.0]
        let discreteId: Int16  // -1 if unused
        let intensity: Float   // [0.0, 1.0]
        let confidence: Float  // [0.0, 1.0]

        enum CodingKeys: String, CodingKey {
            case valence
            case arousal
            case dominance
            case discreteId = "discrete_id"
            case intensity
            case confidence
        }
    }

    struct MusicalIntent: Codable {
        let tempoBias: Float        // -1.0 → +1.0
        let rhythmicDensity: Float  // 0.0 → 1.0
        let grooveStrength: Float    // 0.0 → 1.0
        let harmonicTension: Float   // 0.0 → 1.0
        let harmonicMotion: Float    // 0.0 → 1.0
        let modePreference: Int8     // -1, 0, +1
        let melodicActivity: Float   // 0.0 → 1.0
        let contourVariance: Float   // 0.0 → 1.0
        let dynamicRange: Float      // 0.0 → 1.0
        let textureDensity: Float    // 0.0 → 1.0

        enum CodingKeys: String, CodingKey {
            case tempoBias = "tempo_bias"
            case rhythmicDensity = "rhythmic_density"
            case grooveStrength = "groove_strength"
            case harmonicTension = "harmonic_tension"
            case harmonicMotion = "harmonic_motion"
            case modePreference = "mode_preference"
            case melodicActivity = "melodic_activity"
            case contourVariance = "contour_variance"
            case dynamicRange = "dynamic_range"
            case textureDensity = "texture_density"
        }
    }

    struct TimeScope: Codable {
        let startBar: Int32    // -1 = immediate
        let endBar: Int32      // -1 = open-ended
        let fadeInBeats: Float
        let fadeOutBeats: Float

        enum CodingKeys: String, CodingKey {
            case startBar = "start_bar"
            case endBar = "end_bar"
            case fadeInBeats = "fade_in_beats"
            case fadeOutBeats = "fade_out_beats"
        }
    }

    struct IntentConstraints: Codable {
        let allowedEnginesMask: UInt32
        let forbiddenEnginesMask: UInt32
        let maxCpuCost: Float
        let maxEventRate: Float

        enum CodingKeys: String, CodingKey {
            case allowedEnginesMask = "allowed_engines_mask"
            case forbiddenEnginesMask = "forbidden_engines_mask"
            case maxCpuCost = "max_cpu_cost"
            case maxEventRate = "max_event_rate"
        }
    }

    struct IntentProvenance: Codable {
        let source: IntentSource
        let userOverrideWeight: Float  // 0.0 → 1.0

        enum CodingKeys: String, CodingKey {
            case source
            case userOverrideWeight = "user_override_weight"
        }
    }

    enum IntentSource: Int, Codable {
        case uiDirect = 0
        case uiEdit = 1
        case mlText = 2
        case mlAudio = 3
        case preset = 4
        case automation = 5
    }

    /// Validate version compatibility.
    func isValidVersion() -> Bool {
        meta.irVersion == 1
    }

    /// Safe decoding with defaults for missing fields.
    static func decode(from data: Data) -> IntentFrame? {
        let decoder = JSONDecoder()
        do {
            return try decoder.decode(IntentFrame.self, from: data)
        } catch {
            print("Failed to decode IntentFrame: \(error)")
            return nil
        }
    }

    /// Load from file path.
    static func load(from filePath: String) -> IntentFrame? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else {
            return nil
        }
        return decode(from: data)
    }

    /// Get emotion description for display
    func emotionDescription() -> String {
        let v = emotion.valence
        let a = emotion.arousal
        let d = emotion.dominance

        if v > 0.3 && a > 0.6 {
            return "Joyful/Excited"
        } else if v > 0.3 && a < 0.4 {
            return "Calm/Content"
        } else if v < -0.3 && a > 0.6 {
            return "Angry/Agitated"
        } else if v < -0.3 && a < 0.4 {
            return "Sad/Depressed"
        } else {
            return "Neutral"
        }
    }

    /// Get mode description
    func modeDescription() -> String {
        switch music.modePreference {
        case -1:
            return "Minor"
        case 1:
            return "Major"
        default:
            return "Neutral"
        }
    }

    /// Get tempo description
    func tempoDescription() -> String {
        if music.tempoBias > 0.3 {
            return "Fast"
        } else if music.tempoBias < -0.3 {
            return "Slow"
        } else {
            return "Moderate"
        }
    }
}
