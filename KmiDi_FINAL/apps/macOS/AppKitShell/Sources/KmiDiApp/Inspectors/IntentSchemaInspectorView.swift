import SwiftUI

/// Read-only Intent Schema Inspector view.
/// Displays Phase 0-2 intent data from Python orchestrator.
struct IntentSchemaInspectorView: View {
    @StateObject private var viewModel: IntentSchemaInspectorViewModel
    @State private var phase0Expanded = true
    @State private var phase1Expanded = true
    @State private var phase2Expanded = true

    init(snapshotPath: String) {
        _viewModel = StateObject(wrappedValue: IntentSchemaInspectorViewModel(snapshotPath: snapshotPath))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let snapshot = viewModel.snapshot {
                    // Phase 0: Core Wound/Desire
                    if let songRoot = snapshot.songRoot {
                        phase0Section(songRoot)
                    }

                    Divider()

                    // Phase 1: Emotional & Intent
                    if let songIntent = snapshot.songIntent {
                        phase1Section(songIntent)
                    }

                    Divider()

                    // Phase 2: Technical Constraints
                    if let technical = snapshot.technicalConstraints {
                        phase2Section(technical)
                    }

                    // Optional explanation
                    if let explanation = snapshot.explanation, !explanation.isEmpty {
                        Divider()
                        explanationSection(explanation)
                    }
                } else {
                    emptyState
                }
            }
            .padding()
        }
        .background(Color(NSColor.controlBackgroundColor))
    }

    private func phase0Section(_ root: IntentSchemaSnapshot.SongRoot) -> some View {
        DisclosureGroup(isExpanded: $phase0Expanded) {
            VStack(alignment: .leading, spacing: 12) {
                fieldRow(label: "Core Event", value: root.coreEvent)
                fieldRow(label: "Resistance", value: root.coreResistance)
                fieldRow(label: "Longing", value: root.coreLonging)
                fieldRow(label: "Stakes", value: root.coreStakes)
                fieldRow(label: "Transformation", value: root.coreTransformation)
            }
            .padding(.leading, 16)
            .padding(.top, 8)
        } label: {
            Text("Phase 0: Core Wound/Desire")
                .font(.system(size: 13, weight: .semibold, design: .default))
        }
    }

    private func phase1Section(_ intent: IntentSchemaSnapshot.SongIntent) -> some View {
        DisclosureGroup(isExpanded: $phase1Expanded) {
            VStack(alignment: .leading, spacing: 12) {
                fieldRow(label: "Mood", value: intent.moodPrimary)
                if let tension = intent.moodSecondaryTension {
                    fieldRow(label: "Tension", value: String(format: "%.2f", tension))
                }
                fieldRow(label: "Imagery", value: intent.imageryTexture)
                fieldRow(label: "Vulnerability", value: intent.vulnerabilityScale)
                fieldRow(label: "Narrative Arc", value: intent.narrativeArc)
            }
            .padding(.leading, 16)
            .padding(.top, 8)
        } label: {
            Text("Phase 1: Emotional & Intent")
                .font(.system(size: 13, weight: .semibold, design: .default))
        }
    }

    private func phase2Section(_ technical: IntentSchemaSnapshot.TechnicalConstraints) -> some View {
        DisclosureGroup(isExpanded: $phase2Expanded) {
            VStack(alignment: .leading, spacing: 12) {
                fieldRow(label: "Genre", value: technical.technicalGenre)
                if let tempoRange = technical.technicalTempoRange, tempoRange.count == 2 {
                    fieldRow(label: "Tempo Range", value: "\(tempoRange[0])–\(tempoRange[1]) BPM")
                }
                fieldRow(label: "Key", value: technical.technicalKey)
                fieldRow(label: "Mode", value: technical.technicalMode)
                fieldRow(label: "Groove", value: technical.technicalGrooveFeel)
                fieldRow(label: "Rule to Break", value: technical.technicalRuleToBreak)
                fieldRow(label: "Justification", value: technical.ruleBreakingJustification)
            }
            .padding(.leading, 16)
            .padding(.top, 8)
        } label: {
            Text("Phase 2: Technical Constraints")
                .font(.system(size: 13, weight: .semibold, design: .default))
        }
    }

    private func explanationSection(_ explanation: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Explanation")
                .font(.system(size: 11, weight: .semibold, design: .default))
                .foregroundColor(.secondary)
                .textCase(.uppercase)

            Text(explanation)
                .font(.system(size: 12, weight: .regular, design: .default))
                .foregroundColor(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 4)
    }

    private func fieldRow(label: String, value: String?) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text(label + ":")
                .font(.system(size: 11, weight: .medium, design: .default))
                .foregroundColor(.secondary)
                .frame(width: 100, alignment: .leading)

            if let value = value, !value.isEmpty {
                Text(value)
                    .font(.system(size: 11, weight: .regular, design: .default))
                    .foregroundColor(.primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Text("—")
                    .font(.system(size: 11, weight: .regular, design: .default))
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Text("No intent schema available")
                .font(.system(size: 13, weight: .regular, design: .default))
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 20)
    }
}
