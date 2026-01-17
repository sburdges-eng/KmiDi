import AppKit
import SwiftUI

/// AppKit controller that hosts Emotion Inspector SwiftUI view.
final class EmotionInspectorController: NSViewController {
    private let hostingView: NSHostingView<EmotionInspectorView>
    private let snapshotPath: String

    init(snapshotPath: String) {
        self.snapshotPath = snapshotPath
        let swiftUIView = EmotionInspectorView(snapshotPath: snapshotPath)
        self.hostingView = NSHostingView(rootView: swiftUIView)
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        view = hostingView
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        hostingView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            hostingView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            hostingView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            hostingView.topAnchor.constraint(equalTo: view.topAnchor),
            hostingView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }
}
