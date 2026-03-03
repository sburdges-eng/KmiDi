import SwiftUI

/// Read-only Emotion Inspector view.
/// Displays emotion state as studio meter, not mood app.
struct EmotionInspectorView: View {
    @StateObject private var viewModel: EmotionInspectorViewModel

    init(snapshotPath: String) {
        _viewModel = StateObject(wrappedValue: EmotionInspectorViewModel(snapshotPath: snapshotPath))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Emotion Summary
                if let snapshot = viewModel.snapshot {
                    emotionSummary(snapshot)

                    Divider()

                    // Dimension Bars
                    dimensionBars(snapshot.emotion)

                    // Contextual Notes (optional)
                    if viewModel.showContextualNotes, let note = viewModel.contextualNote {
                        Divider()
                        contextualNoteView(note)
                    }
                } else {
                    emptyState
                }
            }
            .padding()
        }
        .background(Color(NSColor.controlBackgroundColor))
    }

    private func emotionSummary(_ snapshot: EmotionStateSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(viewModel.primaryLabel)
                .font(.system(size: 16, weight: .medium, design: .default))
                .foregroundColor(.primary)

            if let secondary = viewModel.secondaryLabel {
                Text(secondary)
                    .font(.system(size: 13, weight: .regular, design: .default))
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func dimensionBars(_ emotion: EmotionStateSnapshot.Emotion) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            dimensionBar(
                label: "Valence",
                value: emotion.valence,
                range: -1.0...1.0,
                neutralPoint: 0.0,
                showValue: true
            )

            dimensionBar(
                label: "Arousal",
                value: emotion.arousal,
                range: 0.0...1.0,
                neutralPoint: nil,
                showValue: true
            )

            dimensionBar(
                label: "Dominance",
                value: emotion.dominance,
                range: 0.0...1.0,
                neutralPoint: nil,
                showValue: true
            )

            dimensionBar(
                label: "Complexity",
                value: emotion.complexity,
                range: 0.0...1.0,
                neutralPoint: nil,
                showValue: true
            )
        }
    }

    private func dimensionBar(
        label: String,
        value: Float,
        range: ClosedRange<Float>,
        neutralPoint: Float?,
        showValue: Bool
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.system(size: 12, weight: .regular, design: .default))
                    .foregroundColor(.secondary)

                Spacer()

                if showValue {
                    Text(String(format: "%.2f", value))
                        .font(.system(size: 11, weight: .regular, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background
                    Rectangle()
                        .fill(Color(NSColor.separatorColor))
                        .frame(height: 4)

                    // Fill
                    let normalizedValue: CGFloat
                    if let neutral = neutralPoint {
                        // Valence: map -1..1 to 0..1, then position
                        let normalized = (value - range.lowerBound) / (range.upperBound - range.lowerBound)
                        normalizedValue = CGFloat(normalized)
                    } else {
                        // 0..1 range
                        normalizedValue = CGFloat((value - range.lowerBound) / (range.upperBound - range.lowerBound))
                    }

                    Rectangle()
                        .fill(Color(NSColor.controlAccentColor).opacity(0.6))
                        .frame(width: geometry.size.width * normalizedValue, height: 4)

                    // Neutral point marker (for valence)
                    if let neutral = neutralPoint {
                        let neutralPos = CGFloat((neutral - range.lowerBound) / (range.upperBound - range.lowerBound))
                        Rectangle()
                            .fill(Color(NSColor.labelColor).opacity(0.3))
                            .frame(width: 1, height: 6)
                            .offset(x: geometry.size.width * neutralPos - 0.5)
                    }
                }
            }
            .frame(height: 4)
        }
    }

    private func contextualNoteView(_ note: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Context")
                .font(.system(size: 11, weight: .semibold, design: .default))
                .foregroundColor(.secondary)
                .textCase(.uppercase)

            Text(note)
                .font(.system(size: 12, weight: .regular, design: .default))
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 4)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Text("No emotion data available")
                .font(.system(size: 13, weight: .regular, design: .default))
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 20)
    }
}

// Accessibility support
extension EmotionInspectorView {
    var accessibilitySummary: String {
        guard let snapshot = viewModel.snapshot else {
            return "No emotion data available"
        }

        let emotion = snapshot.emotion
        return """
            Primary emotion: \(viewModel.primaryLabel). \
            Valence: \(String(format: "%.2f", emotion.valence)). \
            Arousal: \(String(format: "%.2f", emotion.arousal)). \
            Dominance: \(String(format: "%.2f", emotion.dominance)). \
            Complexity: \(String(format: "%.2f", emotion.complexity)).
            """
    }
}
