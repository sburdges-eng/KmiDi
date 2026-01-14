import SwiftUI
import AppKit

#if DEBUG
/// ML Debug Panel - developer-only view for inspecting ML pipeline state.
struct MLDebugPanelView: View {
    @StateObject private var viewModel: MLDebugPanelViewModel

    init(snapshotPath: String) {
        _viewModel = StateObject(wrappedValue: MLDebugPanelViewModel(snapshotPath: snapshotPath))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let snapshot = viewModel.snapshot {
                    // Raw EmotionState values
                    emotionSection(snapshot.emotion)

                    Divider()

                    // Model state
                    modelStateSection(snapshot.mlDebug)

                    Divider()

                    // Inference timing
                    timingSection(snapshot.mlDebug)

                    Divider()

                    // Output summaries
                    outputSummarySection(snapshot.mlDebug)
                } else {
                    emptyState
                }
            }
            .padding()
        }
        .background(Color(NSColor.controlBackgroundColor))
    }

    private func emotionSection(_ emotion: EmotionStateSnapshot.Emotion) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Emotion State")
                .font(.system(size: 12, weight: .semibold, design: .default))
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                debugRow(label: "Valence", value: String(format: "%.6f", emotion.valence))
                debugRow(label: "Arousal", value: String(format: "%.6f", emotion.arousal))
                debugRow(label: "Dominance", value: String(format: "%.6f", emotion.dominance))
                debugRow(label: "Complexity", value: String(format: "%.6f", emotion.complexity))
            }
            .font(.system(size: 11, weight: .regular, design: .monospaced))
        }
    }

    private func modelStateSection(_ debug: MLDebugSnapshot.MLDebugInfo) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model State")
                .font(.system(size: 12, weight: .semibold, design: .default))
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                debugRow(label: "Emotion Recognizer", value: debug.emotionRecognizerEnabled ? "✓" : "✗")
                debugRow(label: "Melody Transformer", value: debug.melodyTransformerEnabled ? "✓" : "✗")
                debugRow(label: "Harmony Predictor", value: debug.harmonyPredictorEnabled ? "✓" : "✗")
                debugRow(label: "Dynamics Engine", value: debug.dynamicsEngineEnabled ? "✓" : "✗")
                debugRow(label: "Groove Predictor", value: debug.groovePredictorEnabled ? "✓" : "✗")
            }
            .font(.system(size: 11, weight: .regular, design: .monospaced))
        }
    }

    private func timingSection(_ debug: MLDebugSnapshot.MLDebugInfo) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Inference Timing")
                .font(.system(size: 12, weight: .semibold, design: .default))
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                debugRow(label: "Time (ms)", value: String(format: "%.3f", debug.inferenceTimeMs))
                debugRow(label: "Fallback Active", value: debug.fallbackActive ? "Yes" : "No")
            }
            .font(.system(size: 11, weight: .regular, design: .monospaced))
        }
    }

    private func outputSummarySection(_ debug: MLDebugSnapshot.MLDebugInfo) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Output Summaries")
                .font(.system(size: 12, weight: .semibold, design: .default))
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                debugRow(label: "Note Prob [0:3]", value: formatArray(debug.noteProbSummary))
                debugRow(label: "Chord Prob [0:3]", value: formatArray(debug.chordProbSummary))
                debugRow(label: "Groove [0:3]", value: formatArray(debug.grooveSummary))
                debugRow(label: "Dynamics [0:3]", value: formatArray(debug.dynamicsSummary))
            }
            .font(.system(size: 11, weight: .regular, design: .monospaced))
        }
    }

    private func debugRow(label: String, value: String) -> some View {
        HStack {
            Text(label + ":")
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .foregroundColor(.primary)
        }
    }

    private func formatArray(_ arr: [Float]) -> String {
        arr.map { String(format: "%.3f", $0) }.joined(separator: ", ")
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Text("No debug data available")
                .font(.system(size: 13, weight: .regular, design: .default))
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 20)
    }
}

class MLDebugPanelViewModel: ObservableObject {
    @Published private(set) var snapshot: MLDebugSnapshot?

    private let snapshotPath: String
    private var watcher: SnapshotWatcher?
    private var cancellables = Set<AnyCancellable>()

    init(snapshotPath: String) {
        self.snapshotPath = snapshotPath
        setupWatcher()
    }

    private func setupWatcher() {
        watcher = SnapshotWatcher(filePath: snapshotPath, throttleInterval: 0.1)

        watcher?.$lastUpdate
            .compactMap { [weak self] _ in
                self?.loadSnapshot()
            }
            .assign(to: &$snapshot)
    }

    private func loadSnapshot() -> MLDebugSnapshot? {
        return MLDebugSnapshot.load(from: snapshotPath)
    }
}

#endif

#if DEBUG
final class MLDebugPanelController: NSViewController {
    private let hostingView: NSHostingView<MLDebugPanelView>
    private let snapshotPath: String

    init(snapshotPath: String) {
        self.snapshotPath = snapshotPath
        let swiftUIView = MLDebugPanelView(snapshotPath: snapshotPath)
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
#endif
