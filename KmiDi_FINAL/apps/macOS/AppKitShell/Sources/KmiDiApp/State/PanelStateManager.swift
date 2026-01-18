import AppKit

/// Central authority for panel state persistence and lookup.
final class PanelStateManager {
    private let storageKey = "com.kmidi.ui.panels.state"
    private let userDefaults: UserDefaults
    private var stateByID: [String: PanelState] = [:]

    init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
        loadPersistedState()
    }

    // MARK: Registration

    /// Registers a panel and returns the merged (default + persisted) state.
    @discardableResult
    func register(_ provider: PanelStateProviding) -> PanelState {
        let defaults = provider.defaultPanelState
        let merged = defaults.mergingPersisted(persistedState(for: defaults.panelID))
        stateByID[defaults.panelID] = merged
        return merged
    }

    // MARK: Lookup

    func state(for panelID: String) -> PanelState? {
        stateByID[panelID]
    }

    // MARK: Mutations

    func updateVisibility(for panelID: String, isVisible: Bool) {
        guard var state = stateByID[panelID] else { return }
        state.isVisible = isVisible
        // Collapsed follows visibility for docked panels.
        state.isCollapsed = !isVisible ? true : state.isCollapsed
        stateByID[panelID] = state
        persistCurrentState()
    }

    func updateCollapsed(for panelID: String, isCollapsed: Bool) {
        guard var state = stateByID[panelID] else { return }
        state.isCollapsed = isCollapsed
        if isCollapsed { state.isVisible = false }
        stateByID[panelID] = state
        persistCurrentState()
    }

    func updateSize(for panelID: String, size: CGFloat) {
        guard var state = stateByID[panelID] else { return }
        state.storedSize = max(state.minimumSize, min(size, state.maximumSize))
        stateByID[panelID] = state
        persistCurrentState()
    }

    // MARK: Persistence

    private func loadPersistedState() {
        guard let data = userDefaults.data(forKey: storageKey) else { return }
        do {
            let decoded = try JSONDecoder().decode([PersistedPanelState].self, from: data)
            persistedStates = Dictionary(uniqueKeysWithValues: decoded.map { ($0.panelID, $0) })
        } catch {
            persistedStates = [:]
        }
    }

    private func persistCurrentState() {
        let toPersist = stateByID.values.map {
            PersistedPanelState(
                panelID: $0.panelID,
                visible: $0.isVisible,
                collapsed: $0.isCollapsed,
                size: $0.storedSize
            )
        }
        if let data = try? JSONEncoder().encode(toPersist) {
            userDefaults.set(data, forKey: storageKey)
        }
    }

    private func persistedState(for id: String) -> PersistedPanelState? {
        persistedStates[id]
    }

    private var persistedStates: [String: PersistedPanelState] = [:]
}
