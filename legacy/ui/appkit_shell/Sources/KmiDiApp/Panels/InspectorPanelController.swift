import AppKit

final class InspectorPanelController: NSViewController {
    private var tabView: NSTabView?

    override func loadView() {
        SnapshotPaths.ensureDirectoryExists()

        let container = PanelContainerView()
        let tabView = NSTabView()
        self.tabView = tabView

        // Emotion Inspector tab
        let emotionController = EmotionInspectorController(snapshotPath: SnapshotPaths.emotionSnapshot)
        let emotionTab = NSTabViewItem(viewController: emotionController)
        emotionTab.label = "Emotion"
        tabView.addTabViewItem(emotionTab)

        // Intent Schema Inspector tab
        let intentController = IntentSchemaInspectorController(snapshotPath: SnapshotPaths.intentSchema)
        let intentTab = NSTabViewItem(viewController: intentController)
        intentTab.label = "Intent"
        tabView.addTabViewItem(intentTab)

        #if DEBUG
        // ML Debug Panel (debug only)
        let debugController = NSHostingController(rootView: MLDebugPanelView(snapshotPath: SnapshotPaths.mlDebugSnapshot))
        let debugTab = NSTabViewItem(viewController: debugController)
        debugTab.label = "Debug"
        tabView.addTabViewItem(debugTab)
        #endif

        container.addSubview(tabView)
        tabView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            tabView.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            tabView.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            tabView.topAnchor.constraint(equalTo: container.topAnchor),
            tabView.bottomAnchor.constraint(equalTo: container.bottomAnchor)
        ])

        view = container
    }
}

// Registration defaults for persistence.
extension InspectorPanelController: PanelStateProviding {
    var defaultPanelState: PanelState {
        PanelState(
            panelID: "inspector",
            panelTitle: "Inspector",
            preferredSize: 280,
            minimumSize: 220,
            maximumSize: 420,
            isVisible: true,
            isCollapsed: false,
            canDetach: false
        )
    }
}
