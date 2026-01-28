import Foundation

/// Immutable snapshot of Intent Schema from Python.
/// Decoded from JSON file produced by Python orchestrator.
struct IntentSchemaSnapshot: Codable {
    // Phase 0: Core Wound/Desire
    let songRoot: SongRoot?
    // Phase 1: Emotional & Intent
    let songIntent: SongIntent?
    // Phase 2: Technical Constraints
    let technicalConstraints: TechnicalConstraints?
    // System
    let systemDirective: SystemDirective?

    // Meta
    let title: String?
    let created: String?

    // Optional reasoning engine outputs (for display context)
    let explanation: String?
    let ruleBreakingLogic: String?

    struct SongRoot: Codable {
        let coreEvent: String?
        let coreResistance: String?
        let coreLonging: String?
        let coreStakes: String?
        let coreTransformation: String?

        enum CodingKeys: String, CodingKey {
            case coreEvent = "core_event"
            case coreResistance = "core_resistance"
            case coreLonging = "core_longing"
            case coreStakes = "core_stakes"
            case coreTransformation = "core_transformation"
        }
    }

    struct SongIntent: Codable {
        let moodPrimary: String?
        let moodSecondaryTension: Double?
        let imageryTexture: String?
        let vulnerabilityScale: String?
        let narrativeArc: String?

        enum CodingKeys: String, CodingKey {
            case moodPrimary = "mood_primary"
            case moodSecondaryTension = "mood_secondary_tension"
            case imageryTexture = "imagery_texture"
            case vulnerabilityScale = "vulnerability_scale"
            case narrativeArc = "narrative_arc"
        }
    }

    struct TechnicalConstraints: Codable {
        let technicalGenre: String?
        let technicalTempoRange: [Int]?  // [min, max]
        let technicalKey: String?
        let technicalMode: String?
        let technicalGrooveFeel: String?
        let technicalRuleToBreak: String?
        let ruleBreakingJustification: String?

        enum CodingKeys: String, CodingKey {
            case technicalGenre = "technical_genre"
            case technicalTempoRange = "technical_tempo_range"
            case technicalKey = "technical_key"
            case technicalMode = "technical_mode"
            case technicalGrooveFeel = "technical_groove_feel"
            case technicalRuleToBreak = "technical_rule_to_break"
            case ruleBreakingJustification = "rule_breaking_justification"
        }
    }

    struct SystemDirective: Codable {
        let outputTarget: String?
        let outputFeedbackLoop: String?

        enum CodingKeys: String, CodingKey {
            case outputTarget = "output_target"
            case outputFeedbackLoop = "output_feedback_loop"
        }
    }

    enum CodingKeys: String, CodingKey {
        case songRoot = "song_root"
        case songIntent = "song_intent"
        case technicalConstraints = "technical_constraints"
        case systemDirective = "system_directive"
        case title
        case created
        case explanation
        case ruleBreakingLogic = "rule_breaking_logic"
    }

    /// Safe decoding with graceful handling of missing fields.
    static func decode(from data: Data) -> IntentSchemaSnapshot? {
        let decoder = JSONDecoder()
        do {
            return try decoder.decode(IntentSchemaSnapshot.self, from: data)
        } catch {
            print("Failed to decode IntentSchemaSnapshot: \(error)")
            return nil
        }
    }

    /// Load from file path.
    static func load(from filePath: String) -> IntentSchemaSnapshot? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else {
            return nil
        }
        return decode(from: data)
    }
}
