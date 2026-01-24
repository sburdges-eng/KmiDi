import React from "react";
import { render, screen } from "@testing-library/react";
import { IntentIRInspector } from "../../src/components/IntentIRInspector";

describe("IntentIRInspector", () => {
  const testFrame = {
    meta: {
      ir_version: 1,
      intent_id: 0x123,
      session_id: 0x456,
    },
    emotion: {
      valence: 0.5,
      arousal: 0.6,
      dominance: 0.7,
      discrete_id: -1,
      intensity: 1.0,
      confidence: 0.9,
    },
    music: {
      tempo_bias: 0.2,
      rhythmic_density: 0.5,
      groove_strength: 0.6,
      harmonic_tension: 0.4,
      harmonic_motion: 0.5,
      mode_preference: 0,
      melodic_activity: 0.5,
      contour_variance: 0.5,
      dynamic_range: 0.5,
      texture_density: 0.5,
    },
    time: {
      start_bar: -1,
      end_bar: -1,
      fade_in_beats: 0.0,
      fade_out_beats: 0.0,
    },
    constraints: {
      allowed_engines_mask: 0xffffffff,
      forbidden_engines_mask: 0,
      max_cpu_cost: 1.0,
      max_event_rate: Infinity,
    },
    provenance: {
      source: 0, // UI_DIRECT
      user_override_weight: 1.0,
    },
  };

  it("displays IntentFrame correctly", () => {
    render(<IntentIRInspector frame={testFrame} />);
    expect(screen.getByText("Intent IR Inspector")).toBeInTheDocument();
    expect(screen.getByText("UI_DIRECT")).toBeInTheDocument();
  });

  it("shows provenance coloring", () => {
    render(<IntentIRInspector frame={testFrame} />);
    const badge = screen.getByText("UI_DIRECT");
    expect(badge).toHaveStyle({ backgroundColor: "#4CAF50" });
  });

  it("handles empty frame", () => {
    render(<IntentIRInspector />);
    expect(screen.getByText("No IntentFrame available")).toBeInTheDocument();
  });

  it("expands and collapses sections", () => {
    render(<IntentIRInspector frame={testFrame} />);
    const emotionHeader = screen
      .getByText("Emotion")
      .closest(".intent-ir-section-header");
    expect(emotionHeader).toBeInTheDocument();

    // Click to expand
    emotionHeader?.click();
    expect(screen.getByText(/valence:/)).toBeInTheDocument();
  });
});
