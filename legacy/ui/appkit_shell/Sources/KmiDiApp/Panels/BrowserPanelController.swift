import AppKit

final class BrowserPanelController: NSViewController {
    override func loadView() {
        let container = PanelContainerView()

        let titleLabel = PanelUIFactory.makeTitleLabel("Browser")
        let detailLabel = PanelUIFactory.makeDetailLabel(
            "Files, presets, and media library. Docked to the right; collapsible."
        )
        let placeholder = PanelUIFactory.makePlaceholder(
            color: .systemTeal,
            title: "Browser Content"
        )
        placeholder.heightAnchor.constraint(equalToConstant: 160).isActive = true

        let stack = PanelUIFactory.makeStack()
        stack.addArrangedSubview(titleLabel)
        stack.addArrangedSubview(detailLabel)
        stack.addArrangedSubview(placeholder)

        container.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 12),
            stack.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -12),
            stack.topAnchor.constraint(equalTo: container.topAnchor, constant: 12),
            stack.bottomAnchor.constraint(lessThanOrEqualTo: container.bottomAnchor, constant: -12),
        ])

        view = container
    }
}

// Registration defaults for persistence.
extension BrowserPanelController: PanelStateProviding {
    var defaultPanelState: PanelState {
        PanelState(
            panelID: "browser",
            panelTitle: "Browser",
            preferredSize: 280,
            minimumSize: 220,
            maximumSize: 440,
            isVisible: true,
            isCollapsed: false,
            canDetach: false
        )
    }
}
