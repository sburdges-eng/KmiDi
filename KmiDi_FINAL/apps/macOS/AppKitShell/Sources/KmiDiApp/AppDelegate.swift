import AppKit

@main
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var mainWindowController: MainWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMenus()
        let controller = MainWindowController()
        controller.showWindow(self)
        controller.window?.makeKeyAndOrderFront(self)
        mainWindowController = controller
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    // MARK: - Menu Setup

    private func setupMenus() {
        let mainMenu = NSMenu()

        // App menu (KmiDi)
        let appMenuItem = NSMenuItem()
        appMenuItem.submenu = buildAppMenu()
        mainMenu.addItem(appMenuItem)

        // File menu
        let fileMenuItem = NSMenuItem()
        fileMenuItem.submenu = buildFileMenu()
        mainMenu.addItem(fileMenuItem)

        // Edit menu
        let editMenuItem = NSMenuItem()
        editMenuItem.submenu = buildEditMenu()
        mainMenu.addItem(editMenuItem)

        // View menu
        let viewMenuItem = NSMenuItem()
        viewMenuItem.submenu = buildViewMenu()
        mainMenu.addItem(viewMenuItem)

        // Window menu
        let windowMenuItem = NSMenuItem()
        windowMenuItem.submenu = buildWindowMenu()
        mainMenu.addItem(windowMenuItem)

        // Help menu
        let helpMenuItem = NSMenuItem()
        helpMenuItem.submenu = buildHelpMenu()
        mainMenu.addItem(helpMenuItem)

        NSApp.mainMenu = mainMenu
    }

    private func buildAppMenu() -> NSMenu {
        let menu = NSMenu()

        menu.addItem(withTitle: "About KmiDi", action: #selector(showAbout(_:)), keyEquivalent: "")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Preferences...", action: #selector(showPreferences(_:)), keyEquivalent: ",")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Hide KmiDi", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        menu.addItem(withTitle: "Hide Others", action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
            .keyEquivalentModifierMask = [.command, .option]
        menu.addItem(withTitle: "Show All", action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Quit KmiDi", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        return menu
    }

    private func buildFileMenu() -> NSMenu {
        let menu = NSMenu()

        menu.addItem(withTitle: "New", action: #selector(newDocument(_:)), keyEquivalent: "n")
        menu.addItem(withTitle: "Open...", action: #selector(openDocument(_:)), keyEquivalent: "o")
        menu.addItem(withTitle: "Open Recent", action: nil, keyEquivalent: "")
            .submenu = NSMenu()
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Save", action: #selector(saveDocument(_:)), keyEquivalent: "s")
        menu.addItem(withTitle: "Save As...", action: #selector(saveDocumentAs(_:)), keyEquivalent: "S")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Export...", action: #selector(exportDocument(_:)), keyEquivalent: "e")

        return menu
    }

    private func buildEditMenu() -> NSMenu {
        let menu = NSMenu()

        menu.addItem(withTitle: "Undo", action: #selector(undo(_:)), keyEquivalent: "z")
        menu.addItem(withTitle: "Redo", action: #selector(redo(_:)), keyEquivalent: "Z")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        menu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        menu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Find", action: #selector(performFind(_:)), keyEquivalent: "f")

        return menu
    }

    private func buildViewMenu() -> NSMenu {
        let menu = NSMenu()

        menu.addItem(withTitle: "Toggle Inspector", action: #selector(toggleInspectorPanel(_:)), keyEquivalent: "i")
        menu.addItem(withTitle: "Toggle Browser", action: #selector(toggleBrowserPanel(_:)), keyEquivalent: "b")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Customize Toolbar...", action: #selector(NSWindow.toggleToolbarShown(_:)), keyEquivalent: "")

        return menu
    }

    private func buildWindowMenu() -> NSMenu {
        let menu = NSMenu()

        menu.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        menu.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Bring All to Front", action: #selector(NSApplication.arrangeInFront(_:)), keyEquivalent: "")

        return menu
    }

    private func buildHelpMenu() -> NSMenu {
        let menu = NSMenu()

        menu.addItem(withTitle: "KmiDi Help", action: #selector(showHelp(_:)), keyEquivalent: "")
        menu.addItem(withTitle: "Documentation", action: #selector(showDocumentation(_:)), keyEquivalent: "")

        return menu
    }

    // MARK: - Menu Actions (Stubs)

    @objc func showAbout(_ sender: Any?) {
        // TODO: Show about panel
    }

    @objc func newDocument(_ sender: Any?) {
        // TODO: Create new document
    }

    @objc func openDocument(_ sender: Any?) {
        // TODO: Open document dialog
    }

    @objc func saveDocument(_ sender: Any?) {
        // TODO: Save current document
    }

    @objc func saveDocumentAs(_ sender: Any?) {
        // TODO: Save document as dialog
    }

    @objc func exportDocument(_ sender: Any?) {
        // TODO: Export dialog
    }

    @objc func undo(_ sender: Any?) {
        // TODO: Route to history system
        mainWindowController?.handleUndo(sender)
    }

    @objc func redo(_ sender: Any?) {
        // TODO: Route to history system
        mainWindowController?.handleRedo(sender)
    }

    @objc func performFind(_ sender: Any?) {
        // TODO: Show find panel
    }

    @objc func showHelp(_ sender: Any?) {
        // TODO: Show help
    }

    @objc func showDocumentation(_ sender: Any?) {
        // TODO: Open documentation
    }

    @objc func toggleInspectorPanel(_ sender: Any?) {
        mainWindowController?.toggleInspectorPanel(sender)
    }

    @objc func toggleBrowserPanel(_ sender: Any?) {
        mainWindowController?.toggleBrowserPanel(sender)
    }

    @objc func showPreferences(_ sender: Any?) {
        mainWindowController?.showPreferences(sender)
    }
}
