import AppKit

final class MainSplitViewController: NSSplitViewController, NSSplitViewDelegate {
    private let panelStateManager = PanelStateManager()

    private let inspectorItem: NSSplitViewItem
    private let timelineItem: NSSplitViewItem
    private let browserItem: NSSplitViewItem

    private var inspectorState: PanelState
    private var timelineState: PanelState
    private var browserState: PanelState

    private var appliedInitialSizes = false

    init() {
        let inspector = InspectorPanelController()
        let timeline = TimelinePanelController()
        let browser = BrowserPanelController()

        // Register panel contracts and merge persisted state.
        // Add any future panels here by creating the controller and registering it with the manager.
        inspectorState = panelStateManager.register(inspector)
        timelineState = panelStateManager.register(timeline)
        browserState = panelStateManager.register(browser)

        inspectorItem = NSSplitViewItem(sidebarWithViewController: inspector)
        timelineItem = NSSplitViewItem(viewController: timeline)
        browserItem = NSSplitViewItem(sidebarWithViewController: browser)

        super.init(nibName: nil, bundle: nil)
        configureSplitViewItems()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        splitView.isVertical = true
        splitView.autosaveName = "com.kmidi.ui.split.main"
        splitView.delegate = self
        splitViewItems = [inspectorItem, timelineItem, browserItem]

        applyState(inspectorState, to: inspectorItem)
        applyState(timelineState, to: timelineItem)
        applyState(browserState, to: browserItem)
    }

    override func viewDidLayout() {
        super.viewDidLayout()
        if !appliedInitialSizes {
            appliedInitialSizes = applyInitialSizesIfNeeded()
        }
    }

    // MARK: - Panel Visibility

    var isInspectorVisible: Bool { !inspectorItem.isCollapsed }
    var isBrowserVisible: Bool { !browserItem.isCollapsed }

    func toggleInspectorPanel() {
        toggle(item: inspectorItem, state: &inspectorState)
    }

    func toggleBrowserPanel() {
        toggle(item: browserItem, state: &browserState)
    }

    private func toggle(item: NSSplitViewItem, state: inout PanelState) {
        let targetVisible = item.isCollapsed
        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.15
            item.animator().isCollapsed = !targetVisible
        }
        state.isVisible = targetVisible
        state.isCollapsed = !targetVisible
        panelStateManager.updateVisibility(for: state.panelID, isVisible: targetVisible)
        panelStateManager.updateCollapsed(for: state.panelID, isCollapsed: !targetVisible)
    }

    // MARK: - Split configuration

    private func configureSplitViewItems() {
        configure(item: inspectorItem, with: inspectorState)
        configure(item: timelineItem, with: timelineState)
        configure(item: browserItem, with: browserState)
    }

    private func configure(item: NSSplitViewItem, with state: PanelState) {
        let isTimeline = state.panelID == timelineState.panelID
        item.canCollapse = !isTimeline
        item.minimumThickness = state.minimumSize
        item.maximumThickness = state.maximumSize
        item.holdingPriority = isTimeline
            ? NSLayoutConstraint.Priority(200)
            : NSLayoutConstraint.Priority(260)
        item.collapseBehavior = .preferResizingSiblings
    }

    private func applyState(_ state: PanelState, to item: NSSplitViewItem) {
        item.isCollapsed = state.isCollapsed || !state.isVisible
    }

    private func applyInitialSizesIfNeeded() -> Bool {
        guard splitView.isVertical else { return true }
        guard splitView.subviews.count >= 3 else { return false }

        let divider = splitView.dividerThickness
        let totalWidth = splitView.bounds.width
        guard totalWidth > 0 else { return false }

        let inspectorSize = inspectorState.clampedSize()
        let browserSize = browserState.clampedSize()

        let minRemaining = timelineState.minimumSize
        let maxInspector = inspectorState.maximumSize
        let maxBrowser = browserState.maximumSize

        let clampedInspector = max(inspectorState.minimumSize, min(inspectorSize, maxInspector))
        let clampedBrowser = max(browserState.minimumSize, min(browserSize, maxBrowser))

        let remaining = totalWidth - clampedInspector - clampedBrowser - (2 * divider)
        let timelineSize = max(minRemaining, remaining)

        // Position dividers: first after inspector, second before browser.
        splitView.setPosition(clampedInspector, ofDividerAt: 0)
        let secondDividerPosition = clampedInspector + divider + timelineSize
        splitView.setPosition(secondDividerPosition, ofDividerAt: 1)
        return true
    }

    // MARK: - NSSplitViewDelegate

    func splitViewDidResizeSubviews(_ notification: Notification) {
        // Persist sizes only for visible, non-collapsed items.
        if !inspectorItem.isCollapsed {
            let width = inspectorItem.viewController.view.frame.width
            inspectorState.storedSize = width
            panelStateManager.updateSize(for: inspectorState.panelID, size: width)
        }
        if !timelineItem.isCollapsed {
            let width = timelineItem.viewController.view.frame.width
            timelineState.storedSize = width
            panelStateManager.updateSize(for: timelineState.panelID, size: width)
        }
        if !browserItem.isCollapsed {
            let width = browserItem.viewController.view.frame.width
            browserState.storedSize = width
            panelStateManager.updateSize(for: browserState.panelID, size: width)
        }
    }
}
