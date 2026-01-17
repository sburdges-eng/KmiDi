import AppKit

/// Immutable contract each panel must provide.
struct PanelState: Codable, Equatable {
    let panelID: String
    let panelTitle: String
    let preferredSize: CGFloat
    let minimumSize: CGFloat
    let maximumSize: CGFloat
    var isVisible: Bool
    var isCollapsed: Bool
    var canDetach: Bool

    /// Last known size (width for vertical splits, height for horizontal).
    var storedSize: CGFloat

    init(
        panelID: String,
        panelTitle: String,
        preferredSize: CGFloat,
        minimumSize: CGFloat,
        maximumSize: CGFloat,
        isVisible: Bool = true,
        isCollapsed: Bool = false,
        canDetach: Bool = false,
        storedSize: CGFloat? = nil
    ) {
        self.panelID = panelID
        self.panelTitle = panelTitle
        self.preferredSize = preferredSize
        self.minimumSize = minimumSize
        self.maximumSize = maximumSize
        self.isVisible = isVisible
        self.isCollapsed = isCollapsed
        self.canDetach = canDetach
        self.storedSize = storedSize ?? preferredSize
    }

    func clampedSize() -> CGFloat {
        max(minimumSize, min(storedSize, maximumSize))
    }

    func mergingPersisted(_ persisted: PersistedPanelState?) -> PanelState {
        guard let persisted else { return self }
        return PanelState(
            panelID: panelID,
            panelTitle: panelTitle,
            preferredSize: preferredSize,
            minimumSize: minimumSize,
            maximumSize: maximumSize,
            isVisible: persisted.visible,
            isCollapsed: persisted.collapsed,
            canDetach: canDetach,
            storedSize: persisted.size ?? preferredSize
        )
    }
}

/// Lightweight persisted subset.
struct PersistedPanelState: Codable {
    let panelID: String
    let visible: Bool
    let collapsed: Bool
    let size: CGFloat?
}

/// Panels conform to this to provide their default contract.
protocol PanelStateProviding {
    var defaultPanelState: PanelState { get }
}
