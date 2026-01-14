import Foundation
import Combine

/// View model for Emotion Inspector.
/// Observes snapshot file updates and provides emotion state to view.
class EmotionInspectorViewModel: ObservableObject {
    @Published private(set) var snapshot: EmotionStateSnapshot?
    @Published var showContextualNotes: Bool = false

    private let snapshotPath: String
    private var watcher: SnapshotWatcher?
    private var cancellables = Set<AnyCancellable>()

    init(snapshotPath: String) {
        self.snapshotPath = snapshotPath
        setupWatcher()
    }

    private func setupWatcher() {
        watcher = SnapshotWatcher(filePath: snapshotPath, throttleInterval: 0.1)

        watcher?.$lastUpdate
            .compactMap { [weak self] _ in
                self?.loadSnapshot()
            }
            .assign(to: &$snapshot)
    }

    private func loadSnapshot() -> EmotionStateSnapshot? {
        return EmotionStateSnapshot.load(from: snapshotPath)
    }

    /// Get primary emotion label.
    var primaryLabel: String {
        snapshot?.labels?.primary ?? "Neutral"
    }

    /// Get secondary emotion label.
    var secondaryLabel: String? {
        snapshot?.labels?.secondary
    }

    /// Get contextual note text (optional, non-prescriptive).
    var contextualNote: String? {
        guard let emotion = snapshot?.emotion else { return nil }

        // Low arousal with negative valence often reads as resignation
        if emotion.arousal < 0.3 && emotion.valence < -0.5 {
            return "Low arousal with negative valence often reads as resignation"
        }

        // High arousal with positive valence suggests excitement
        if emotion.arousal > 0.7 && emotion.valence > 0.5 {
            return "High arousal with positive valence suggests excitement"
        }

        // High complexity with low dominance can indicate uncertainty
        if emotion.complexity > 0.7 && emotion.dominance < 0.3 {
            return "High complexity with low dominance can indicate uncertainty"
        }

        return nil
    }
}
