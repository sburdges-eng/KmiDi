import Foundation
import Combine

/// View model for Intent Schema Inspector.
/// Observes Python-generated JSON file updates.
class IntentSchemaInspectorViewModel: ObservableObject {
    @Published private(set) var snapshot: IntentSchemaSnapshot?

    private let snapshotPath: String
    private var watcher: SnapshotWatcher?
    private var cancellables = Set<AnyCancellable>()

    init(snapshotPath: String) {
        self.snapshotPath = snapshotPath
        setupWatcher()
    }

    private func setupWatcher() {
        watcher = SnapshotWatcher(filePath: snapshotPath, throttleInterval: 0.2)

        watcher?.$lastUpdate
            .compactMap { [weak self] _ in
                self?.loadSnapshot()
            }
            .assign(to: &$snapshot)
    }

    private func loadSnapshot() -> IntentSchemaSnapshot? {
        return IntentSchemaSnapshot.load(from: snapshotPath)
    }
}
