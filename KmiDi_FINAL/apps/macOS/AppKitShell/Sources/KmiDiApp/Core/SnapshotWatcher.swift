import Foundation
import Combine

/// Watches a snapshot file for changes and publishes updates.
/// Throttles updates to avoid overwhelming the UI.
class SnapshotWatcher: ObservableObject {
    @Published private(set) var lastUpdate: Date?
    private var fileSource: DispatchSourceFileSystemObject?
    private let filePath: String
    private let throttleInterval: TimeInterval
    private var lastThrottledUpdate: Date = Date.distantPast

    init(filePath: String, throttleInterval: TimeInterval = 0.1) {
        self.filePath = filePath
        self.throttleInterval = throttleInterval
        startWatching()
    }

    deinit {
        stopWatching()
    }

    private func startWatching() {
        let fileURL = URL(fileURLWithPath: filePath)
        let fileDescriptor = open(filePath, O_EVTONLY)

        guard fileDescriptor >= 0 else {
            print("Failed to open file for watching: \(filePath)")
            return
        }

        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fileDescriptor,
            eventMask: .write,
            queue: DispatchQueue.global(qos: .utility)
        )

        source.setEventHandler { [weak self] in
            self?.handleFileChange()
        }

        source.setCancelHandler {
            close(fileDescriptor)
        }

        source.resume()
        self.fileSource = source

        // Initial load
        handleFileChange()
    }

    private func stopWatching() {
        fileSource?.cancel()
        fileSource = nil
    }

    private func handleFileChange() {
        let now = Date()

        // Throttle: only update if enough time has passed
        guard now.timeIntervalSince(lastThrottledUpdate) >= throttleInterval else {
            return
        }

        lastThrottledUpdate = now

        DispatchQueue.main.async { [weak self] in
            self?.lastUpdate = now
        }
    }

    /// Check if file exists.
    func fileExists() -> Bool {
        FileManager.default.fileExists(atPath: filePath)
    }

    /// Get file modification date.
    func fileModificationDate() -> Date? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: filePath),
              let modificationDate = attributes[.modificationDate] as? Date else {
            return nil
        }
        return modificationDate
    }
}
