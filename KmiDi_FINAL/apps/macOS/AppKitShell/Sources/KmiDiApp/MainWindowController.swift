import AppKit

final class MainWindowController: NSWindowController, NSToolbarDelegate {
    private let splitController = MainSplitViewController()

    init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1400, height: 900),
            styleMask: [
                .titled,
                .closable,
                .miniaturizable,
                .resizable,
                .fullSizeContentView,
            ],
            backing: .buffered,
            defer: false
        )

        window.title = "KmiDi"
        window.setFrameAutosaveName("KmiDiMainWindow")
        window.center()
        window.contentViewController = splitController
        window.toolbarStyle = .unified
        window.toolbar = buildToolbar()
        window.isReleasedWhenClosed = false
        window.makeFirstResponder(splitController.view) // Allow keyboard input

        super.init(window: window)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func windowDidLoad() {
        super.windowDidLoad()
        refreshToolbarToggleStates()
        setupKeyboardShortcuts()
    }

    @objc func toggleInspectorPanel(_ sender: Any?) {
        splitController.toggleInspectorPanel()
        refreshToolbarToggleStates()
    }

    @objc func toggleBrowserPanel(_ sender: Any?) {
        splitController.toggleBrowserPanel()
        refreshToolbarToggleStates()
    }

    @objc func showPreferences(_ sender: Any?) {
        let preferencesController = NSHostingController(rootView: AITrustPreferencesView())
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 500, height: 400),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.contentViewController = preferencesController
        window.title = "AI Preferences"
        window.center()
        window.makeKeyAndOrderFront(nil)
    }

    // MARK: - Keyboard Shortcuts

    private func setupKeyboardShortcuts() {
        // Keyboard shortcuts are handled via menu items and keyDown override
        // Cmd+Z, Cmd+Shift+Z are handled by Edit menu
        // Cmd+B, Cmd+I are handled by View menu
        // Space is handled in keyDown override
    }

    override func keyDown(with event: NSEvent) {
        // Handle Space key for transport (only when timeline has focus)
        if event.keyCode == 49 && event.modifierFlags.intersection([.command, .shift, .option, .control]).isEmpty {
            // Space key without modifiers - transport play/stop
            handleTransportToggle()
            return
        }

        // Let other keys fall through to default handling
        super.keyDown(with: event)
    }

    private func handleTransportToggle() {
        // TODO: Route to transport system
        // Placeholder: just log for now
        print("Transport toggle (Space key)")
    }

    // MARK: - History Actions

    @objc func handleUndo(_ sender: Any?) {
        // TODO: Route to history system
        // Placeholder: just log for now
        print("Undo requested")
    }

    @objc func handleRedo(_ sender: Any?) {
        // TODO: Route to history system
        // Placeholder: just log for now
        print("Redo requested")
    }

    // MARK: - Toolbar

    private func buildToolbar() -> NSToolbar {
        let toolbar = NSToolbar(identifier: NSToolbar.Identifier("KmiDiMainToolbar"))
        toolbar.delegate = self
        toolbar.allowsUserCustomization = true
        toolbar.displayMode = .iconAndLabel
        return toolbar
    }

    private func makeToggleItem(
        identifier: NSToolbarItem.Identifier,
        title: String,
        action: Selector
    ) -> NSToolbarItem {
        let button = NSButton(title: title, target: self, action: action)
        button.setButtonType(.toggle)
        button.bezelStyle = .texturedRounded
        button.state = toggleState(for: identifier)

        let item = NSToolbarItem(itemIdentifier: identifier)
        item.label = title
        item.paletteLabel = title
        item.toolTip = "Show/Hide \(title)"
        item.view = button
        return item
    }

    private func toggleState(for identifier: NSToolbarItem.Identifier) -> NSControl.StateValue {
        switch identifier {
        case .toggleInspectorItem:
            return splitController.isInspectorVisible ? .on : .off
        case .toggleBrowserItem:
            return splitController.isBrowserVisible ? .on : .off
        default:
            return .off
        }
    }

    private func refreshToolbarToggleStates() {
        guard let toolbar = window?.toolbar else { return }
        for item in toolbar.items {
            guard let button = item.view as? NSButton else { continue }
            button.state = toggleState(for: item.itemIdentifier)
        }
    }

    // MARK: - NSToolbarDelegate

    func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        [.toggleInspectorItem, .flexibleSpace, .toggleBrowserItem]
    }

    func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        [.toggleInspectorItem, .toggleBrowserItem, .flexibleSpace]
    }

    func toolbar(
        _ toolbar: NSToolbar,
        itemForItemIdentifier itemIdentifier: NSToolbarItem.Identifier,
        willBeInsertedIntoToolbar flag: Bool
    ) -> NSToolbarItem? {
        switch itemIdentifier {
        case .toggleInspectorItem:
            return makeToggleItem(
                identifier: itemIdentifier,
                title: "Inspector",
                action: #selector(toggleInspectorPanel(_:))
            )
        case .toggleBrowserItem:
            return makeToggleItem(
                identifier: itemIdentifier,
                title: "Browser",
                action: #selector(toggleBrowserPanel(_:))
            )
        default:
            return nil
        }
    }
}

private extension NSToolbarItem.Identifier {
    static let toggleInspectorItem = NSToolbarItem.Identifier("com.kmidi.toolbar.toggleInspector")
    static let toggleBrowserItem = NSToolbarItem.Identifier("com.kmidi.toolbar.toggleBrowser")
}
