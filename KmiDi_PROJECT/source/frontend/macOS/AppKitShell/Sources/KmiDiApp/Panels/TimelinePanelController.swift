import AppKit

final class TimelinePanelController: NSViewController {
    private var juceHost: JUCEHostView?

    override func loadView() {
        SnapshotPaths.ensureDirectoryExists()

        let container = PanelContainerView()

        let titleLabel = PanelUIFactory.makeTitleLabel("Timeline")
        let detailLabel = PanelUIFactory.makeDetailLabel(
            "Arranger/timeline surface (AppKit-backed). SwiftUI is not used here."
        )

        // Create JUCE host view - this embeds the TimelineComponent
        // The JUCE component is created in JUCEHostView.init and owned by the NSView
        // Lifecycle: Created on loadView, destroyed when view is deallocated
        let juceHost = JUCEHostView(frame: .zero)
        self.juceHost = juceHost
        juceHost.translatesAutoresizingMaskIntoConstraints = false
        juceHost.heightAnchor.constraint(greaterThanOrEqualToConstant: 360).isActive = true

        // Set emotion snapshot path for background tint
        // This allows the timeline to read emotion state and apply subtle background coloring
        juceHost.setEmotionSnapshotPath(SnapshotPaths.emotionSnapshot)

        let stack = PanelUIFactory.makeStack()
        stack.spacing = 10
        stack.addArrangedSubview(titleLabel)
        stack.addArrangedSubview(detailLabel)
        stack.addArrangedSubview(juceHost)

        container.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 12),
            stack.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -12),
            stack.topAnchor.constraint(equalTo: container.topAnchor, constant: 12),
            stack.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -12),
        ])

        view = container

        // Set up trust state file writing
        setupTrustStateWriting()
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        // JUCE component is already created in loadView via JUCEHostView.init
        // No additional setup needed here - resize handling is automatic via layout()
    }

    override func viewWillAppear() {
        super.viewWillAppear()
        // Ensure JUCE component is visible when panel appears
        // This is handled automatically by JUCEHostView, but we can add hooks here if needed
    }

    deinit {
        // JUCE component is automatically destroyed when JUCEHostView is deallocated
        // The unique_ptr in JUCEHostView ensures proper cleanup
        // No manual cleanup needed here
    }

    private func setupTrustStateWriting() {
        // Write initial trust state
        AITrustManager.shared.writeTrustStateToFile(at: SnapshotPaths.trustState)

        // Observe trust state changes and write to file
        NotificationCenter.default.addObserver(
            forName: AITrustManager.trustStateChangedNotification,
            object: nil,
            queue: .main
        ) { _ in
            AITrustManager.shared.writeTrustStateToFile(at: SnapshotPaths.trustState)
        }
    }
}

// Registration defaults for persistence.
extension TimelinePanelController: PanelStateProviding {
    var defaultPanelState: PanelState {
        PanelState(
            panelID: "timeline",
            panelTitle: "Timeline",
            preferredSize: 900,
            minimumSize: 640,
            maximumSize: 2600,
            isVisible: true,
            isCollapsed: false,
            canDetach: false
        )
    }
}
