import SwiftUI

/// User Trust Controls preferences panel.
/// Allows users to enable/disable AI features globally and per-domain.
struct AITrustPreferencesView: View {
    @StateObject private var trustManager = AITrustManager.shared

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Global toggle
                globalSection

                Divider()

                // Per-domain toggles
                domainSection
            }
            .padding()
        }
        .frame(width: 500, height: 400)
        .background(Color(NSColor.controlBackgroundColor))
    }

    private var globalSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("AI Features")
                .font(.system(size: 16, weight: .semibold, design: .default))

            Toggle("Enable AI Features", isOn: $trustManager.globalAIEnabled)
                .font(.system(size: 13, weight: .regular, design: .default))

            Text("When disabled, all AI features are turned off.")
                .font(.system(size: 11, weight: .regular, design: .default))
                .foregroundColor(.secondary)
        }
    }

    private var domainSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Per-Domain Controls")
                .font(.system(size: 16, weight: .semibold, design: .default))

            Text("Control individual AI features. Global toggle must be enabled.")
                .font(.system(size: 11, weight: .regular, design: .default))
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 12) {
                domainToggle(
                    "Emotion Analysis",
                    isOn: $trustManager.emotionAnalysisEnabled,
                    description: "Analyze emotional content from audio"
                )

                domainToggle(
                    "Melody Suggestions",
                    isOn: $trustManager.melodySuggestionsEnabled,
                    description: "Generate melody probability suggestions"
                )

                domainToggle(
                    "Harmony Suggestions",
                    isOn: $trustManager.harmonySuggestionsEnabled,
                    description: "Generate chord probability suggestions"
                )

                domainToggle(
                    "Groove Suggestions",
                    isOn: $trustManager.grooveSuggestionsEnabled,
                    description: "Generate timing and groove suggestions"
                )

                domainToggle(
                    "Teaching Overlays",
                    isOn: $trustManager.teachingOverlaysEnabled,
                    description: "Show educational annotations on timeline"
                )
            }
            .disabled(!trustManager.globalAIEnabled)
            .opacity(trustManager.globalAIEnabled ? 1.0 : 0.5)
        }
    }

    private func domainToggle(
        _ title: String,
        isOn: Binding<Bool>,
        description: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Toggle(title, isOn: isOn)
                .font(.system(size: 13, weight: .medium, design: .default))

            Text(description)
                .font(.system(size: 11, weight: .regular, design: .default))
                .foregroundColor(.secondary)
                .padding(.leading, 20)
        }
    }
}

struct AITrustPreferencesController: NSViewControllerRepresentable {
    func makeNSViewController(context: Context) -> NSHostingController<AITrustPreferencesView> {
        let view = AITrustPreferencesView()
        return NSHostingController(rootView: view)
    }

    func updateNSViewController(_ nsViewController: NSHostingController<AITrustPreferencesView>, context: Context) {
        // No updates needed
    }
}
