"""
UI Gate v0 + Explicit Playback v0 + MIDI Binding v0.

- A1/A2: Read-only state panel, escalation token emitters
- B1: Play Option A/B/C, Stop — playback only on user click
- B2: Playback guardrails — no playback without user action, no playback after ABSTAIN
- MIDI Binding: Play sends fixed MIDI (symbolic). Pitch/timing variation tokens gate alternate voicing/duration.
- permission=GENERATE: Programmable button gates generation. User-only, optional.
- MIDI Input: Simulate chord/empty feeds midi_notes_to_abstract_input → intent.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import ttk
from typing import Optional

# Add python/ to path for minimal_listening_core
_ui_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.dirname(_ui_dir)
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

from minimal_listening_core import (
    AbstractInput,
    IntentAspect,
    ValueType,
    intent_engine_process,
    midi_notes_to_abstract_input,
)
from ui_gate.playback_guardrails import assert_playback_allowed, can_produce_output
from ui_gate.midi_binding import write_branch_midi_to_file
from ui_gate.permission_generate import get_permission_generate, set_permission_generate


def _assert_ui_mirrors_core(ui_state: dict, core_result) -> None:
    """UI invariant: if UI state and core state diverge → assert fail in debug."""
    if __debug__ and core_result is not None:
        assert ui_state.get("metadata") == core_result.get_metadata()
        assert ui_state.get("is_abstain") == (core_result.get_metadata() == "abstain")
        assert ui_state.get("is_valid") == core_result.is_valid()


class UIGate:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("UI Gate v0 — Listen Only")
        self.root.resizable(True, True)

        self._core_result: Optional[object] = None
        self._input_type_var = tk.StringVar(value="")
        self._magnitude_var = tk.StringVar(value="0.7")
        self._is_playing = False
        self._pitch_variation = False
        self._timing_variation = False

        self._build_ui()
        # Initial: process empty input to show ABSTAIN
        self._on_process()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Input (for demo only; simulates external input) ---
        ttk.Label(main, text="Input type:").grid(row=0, column=0, sticky=tk.W, pady=2)
        input_combo = ttk.Combobox(
            main,
            textvariable=self._input_type_var,
            values=["", "chord_change", "rhythmic", "harmonic", "dynamic", "tempo"],
            state="readonly",
            width=20,
        )
        input_combo.grid(row=0, column=1, sticky=tk.W, pady=2)
        input_combo.set("")

        ttk.Label(main, text="Magnitude:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(main, textvariable=self._magnitude_var, width=8).grid(
            row=1, column=1, sticky=tk.W, pady=2
        )

        ttk.Button(main, text="Process", command=self._on_process).grid(
            row=2, column=0, columnspan=2, pady=5
        )

        # --- MIDI Input (simulate) ---
        ttk.Label(main, text="MIDI Input (simulate):", font=("", 9, "bold")).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=2
        )
        midi_frame = ttk.Frame(main)
        midi_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Button(midi_frame, text="Simulate chord", command=self._on_midi_chord).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(midi_frame, text="Simulate empty", command=self._on_midi_empty).pack(
            side=tk.LEFT
        )

        # --- Display: read-only ---
        ttk.Separator(main, orient=tk.HORIZONTAL).grid(
            row=5, column=0, columnspan=2, sticky=tk.EW, pady=10
        )
        ttk.Label(main, text="Core state (read-only):", font=("", 10, "bold")).grid(
            row=6, column=0, columnspan=2, sticky=tk.W, pady=2
        )

        self._status_label = ttk.Label(main, text="—", foreground="gray")
        self._status_label.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=2)

        self._intent_label = ttk.Label(main, text="Intent: —")
        self._intent_label.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=2)

        self._value_states_label = ttk.Label(main, text="ValueStates: —")
        self._value_states_label.grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=2)

        self._harmonic_esc_label = ttk.Label(main, text="HarmonicEscalation: —")
        self._harmonic_esc_label.grid(row=10, column=0, columnspan=2, sticky=tk.W, pady=2)

        self._rhythmic_esc_label = ttk.Label(main, text="RhythmicEscalation: —")
        self._rhythmic_esc_label.grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=2)

        # --- Escalation buttons (token emitters) ---
        ttk.Separator(main, orient=tk.HORIZONTAL).grid(
            row=12, column=0, columnspan=2, sticky=tk.EW, pady=10
        )
        ttk.Label(main, text="Emit escalation token (once):", font=("", 10, "bold")).grid(
            row=13, column=0, columnspan=2, sticky=tk.W, pady=2
        )

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=5)

        self._btn_harmonic = ttk.Button(
            btn_frame,
            text="More harmonic color",
            command=self._on_more_harmonic_color,
        )
        self._btn_harmonic.pack(side=tk.LEFT, padx=(0, 5))

        self._btn_groove = ttk.Button(
            btn_frame,
            text="More groove",
            command=self._on_more_groove,
        )
        self._btn_groove.pack(side=tk.LEFT, padx=(0, 5))

        self._btn_pitch_variation = ttk.Button(
            btn_frame,
            text="More pitch variation",
            command=self._on_pitch_variation,
        )
        self._btn_pitch_variation.pack(side=tk.LEFT, padx=(0, 5))
        self._pitch_variation_label = ttk.Label(btn_frame, text="(off)", foreground="gray")
        self._pitch_variation_label.pack(side=tk.LEFT, padx=(0, 5))

        self._btn_timing_variation = ttk.Button(
            btn_frame,
            text="More timing variation",
            command=self._on_timing_variation,
        )
        self._btn_timing_variation.pack(side=tk.LEFT, padx=(0, 5))
        self._timing_variation_label = ttk.Label(btn_frame, text="(off)", foreground="gray")
        self._timing_variation_label.pack(side=tk.LEFT, padx=(0, 5))

        # --- permission=GENERATE ---
        ttk.Separator(main, orient=tk.HORIZONTAL).grid(
            row=15, column=0, columnspan=2, sticky=tk.EW, pady=10
        )
        perm_frame = ttk.Frame(main)
        perm_frame.grid(row=16, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(perm_frame, text="permission=GENERATE:", font=("", 10, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self._btn_permission_generate = ttk.Button(
            perm_frame,
            text="Allow Generate",
            command=self._on_permission_generate,
        )
        self._btn_permission_generate.pack(side=tk.LEFT, padx=(0, 5))
        self._permission_label = ttk.Label(perm_frame, text="(revoked)", foreground="gray")
        self._permission_label.pack(side=tk.LEFT)

        # --- B1: Branch playback controls ---
        ttk.Separator(main, orient=tk.HORIZONTAL).grid(
            row=17, column=0, columnspan=2, sticky=tk.EW, pady=10
        )
        ttk.Label(main, text="Branch playback (user click only):", font=("", 10, "bold")).grid(
            row=18, column=0, columnspan=2, sticky=tk.W, pady=2
        )
        play_frame = ttk.Frame(main)
        play_frame.grid(row=19, column=0, columnspan=2, sticky=tk.W, pady=5)
        self._btn_play_a = ttk.Button(play_frame, text="Play Option A", command=lambda: self._on_play(0))
        self._btn_play_a.pack(side=tk.LEFT, padx=(0, 5))
        self._btn_play_b = ttk.Button(play_frame, text="Play Option B", command=lambda: self._on_play(1))
        self._btn_play_b.pack(side=tk.LEFT, padx=(0, 5))
        self._btn_play_c = ttk.Button(play_frame, text="Play Option C", command=lambda: self._on_play(2))
        self._btn_play_c.pack(side=tk.LEFT, padx=(0, 5))
        self._btn_stop = ttk.Button(play_frame, text="Stop", command=self._on_stop)
        self._btn_stop.pack(side=tk.LEFT)
        self._playback_status = ttk.Label(play_frame, text="", foreground="gray")
        self._playback_status.pack(side=tk.LEFT, padx=(10, 0))

        # --- Info ---
        ttk.Label(
            main,
            text="Playback only on click. permission=GENERATE user-only. No auto-play.",
            font=("", 8),
            foreground="gray",
        ).grid(row=20, column=0, columnspan=2, sticky=tk.W, pady=10)
        main.grid_columnconfigure(1, weight=1)

    def _get_input_type(self) -> str:
        return self._input_type_var.get().strip()

    def _get_magnitude(self) -> float:
        try:
            return float(self._magnitude_var.get())
        except ValueError:
            return 0.7

    def _run_core(self, escalation_token: Optional[str] = None) -> Optional[object]:
        input_type = self._get_input_type()
        aspect = IntentAspect.HARMONIC
        if input_type == "rhythmic":
            aspect = IntentAspect.RHYTHMIC
        elif input_type == "dynamic":
            aspect = IntentAspect.DYNAMIC
        elif input_type == "tempo":
            aspect = IntentAspect.TEMPO

        inp = AbstractInput(
            input_type=input_type,
            aspect=aspect,
            magnitude=self._get_magnitude(),
            escalation_token=escalation_token,
        )
        return intent_engine_process(inp, None)

    def _on_process(self) -> None:
        self._core_result = self._run_core(escalation_token=None)
        self._refresh_display()

    def _on_midi_chord(self) -> None:
        """Simulate MIDI chord → intent. Notes present → chord_change, HARMONIC."""
        notes = [(60, 80, 0.0), (64, 80, 0.0), (67, 80, 0.0)]
        inp = midi_notes_to_abstract_input(notes)
        self._core_result = intent_engine_process(inp, None)
        self._refresh_display()

    def _on_midi_empty(self) -> None:
        """Simulate empty MIDI → ABSTAIN."""
        inp = midi_notes_to_abstract_input([])
        self._core_result = intent_engine_process(inp, None)
        self._refresh_display()

    def _refresh_display(self) -> None:
        result = self._core_result
        if result is None:
            self._status_label.config(text="INVALID (result is None)", foreground="red")
            self._intent_label.config(text="Intent: —")
            self._value_states_label.config(text="ValueStates: —")
            self._harmonic_esc_label.config(text="HarmonicEscalation: —")
            self._rhythmic_esc_label.config(text="RhythmicEscalation: —")
            self._btn_harmonic.config(state=tk.DISABLED)
            self._btn_groove.config(state=tk.DISABLED)
            self._btn_pitch_variation.config(state=tk.DISABLED)
            self._btn_timing_variation.config(state=tk.DISABLED)
            self._set_play_buttons_state(0, False)
            return

        intent = result.get_intent()
        states = result.get_value_states()
        h_esc = result.get_harmonic_escalation()
        r_esc = result.get_rhythmic_escalation()
        metadata = result.get_metadata()
        is_valid = result.is_valid()

        # Status (A1: INVALID shows violation reason)
        if metadata == "abstain":
            status = "ABSTAIN"
            fg = "orange"
        elif not is_valid:
            reasons = []
            for vt in ValueType:
                s = states.get(vt)
                if s.is_invalid() and s.get_reason():
                    reasons.append(f"{vt.value}: {s.get_reason()}")
            for v in result.get_violations():
                reasons.append(v.message)
            reason_str = "; ".join(reasons) if reasons else "(violation)"
            status = f"INVALID — {reason_str}"
            fg = "red"
        else:
            status = "OK"
            fg = "green"
        self._status_label.config(text=status, foreground=fg)

        # Intent
        parts = []
        for a in IntentAspect:
            if intent.has_aspect(a):
                mag = intent.get_magnitude(a)
                parts.append(f"{a.value}={mag:.2f}")
        intent_str = ", ".join(parts) if parts else "(none)"
        self._intent_label.config(text=f"Intent: {intent_str}")

        # ValueStates
        vs_parts = []
        for vt in ValueType:
            s = states.get(vt)
            if s.is_present():
                vs_parts.append(f"{vt.value}=PRESENT({s.get_value():.2f})")
            elif s.is_missing():
                vs_parts.append(f"{vt.value}=MISSING")
            else:
                vs_parts.append(f"{vt.value}=INVALID")
        self._value_states_label.config(text="ValueStates: " + ", ".join(vs_parts))

        # Harmonic escalation
        h_list = [b.name for b in h_esc] if len(h_esc) > 0 else ["NONE"]
        self._harmonic_esc_label.config(text="HarmonicEscalation: " + ", ".join(h_list))

        # Rhythmic escalation
        r_list = [b.name for b in r_esc] if len(r_esc) > 0 else ["NONE"]
        self._rhythmic_esc_label.config(text="RhythmicEscalation: " + ", ".join(r_list))

        # Buttons: only enabled when intent matches domain
        has_harmonic = intent.has_aspect(IntentAspect.HARMONIC)
        has_rhythmic = intent.has_aspect(IntentAspect.RHYTHMIC)
        self._btn_harmonic.config(state=tk.NORMAL if has_harmonic else tk.DISABLED)
        self._btn_groove.config(state=tk.NORMAL if has_rhythmic else tk.DISABLED)
        self._btn_pitch_variation.config(state=tk.NORMAL)
        self._btn_timing_variation.config(state=tk.NORMAL)

        # B1: Play buttons enabled only when canProduceOutput and we have branches
        num_branches = max(len(h_esc), len(r_esc))
        play_enabled = can_produce_output(result) and num_branches > 0
        self._set_play_buttons_state(num_branches, play_enabled)

        # UI invariant
        ui_state = {
            "metadata": metadata,
            "is_abstain": metadata == "abstain",
            "is_valid": is_valid,
        }
        _assert_ui_mirrors_core(ui_state, result)

    def _on_more_harmonic_color(self) -> None:
        if self._core_result is None:
            return
        intent = self._core_result.get_intent()
        if not intent.has_aspect(IntentAspect.HARMONIC):
            return
        # Emit token once: re-process with token
        self._core_result = self._run_core(escalation_token="more_harmonic_color")
        self._refresh_display()

    def _on_more_groove(self) -> None:
        if self._core_result is None:
            return
        intent = self._core_result.get_intent()
        if not intent.has_aspect(IntentAspect.RHYTHMIC):
            return
        # Emit token once: re-process with token
        self._core_result = self._run_core(escalation_token="more_groove")
        self._refresh_display()

    def _on_pitch_variation(self) -> None:
        """Emit pitch variation token. Toggle alternate fixed voicing for MIDI."""
        self._pitch_variation = not self._pitch_variation
        self._pitch_variation_label.config(
            text="(on)" if self._pitch_variation else "(off)",
            foreground="green" if self._pitch_variation else "gray",
        )

    def _on_timing_variation(self) -> None:
        """Emit timing variation token. Toggle alternate fixed duration for MIDI."""
        self._timing_variation = not self._timing_variation
        self._timing_variation_label.config(
            text="(on)" if self._timing_variation else "(off)",
            foreground="green" if self._timing_variation else "gray",
        )

    def _on_permission_generate(self) -> None:
        """Programmable button: toggle permission=GENERATE. User-only."""
        granted = not get_permission_generate()
        set_permission_generate(granted)
        self._btn_permission_generate.config(text="Generating Allowed" if granted else "Allow Generate")
        self._permission_label.config(
            text="(granted)" if granted else "(revoked)",
            foreground="green" if granted else "gray",
        )

    def _set_play_buttons_state(self, num_branches: int, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self._btn_play_a.config(state=state if num_branches >= 1 else tk.DISABLED)
        self._btn_play_b.config(state=state if num_branches >= 2 else tk.DISABLED)
        self._btn_play_c.config(state=state if num_branches >= 3 else tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL if self._is_playing else tk.DISABLED)

    def _on_play(self, branch_index: int) -> None:
        """B1: Playback only on user click. B2: Assert guardrails. MIDI: symbolic only."""
        result = self._core_result
        assert_playback_allowed(
            user_initiated=True,
            result=result,
            caller="_on_play",
        )
        if not can_produce_output(result):
            return
        self._is_playing = True
        self._playback_status.config(text=f"Playing option {chr(65 + branch_index)}")
        self._btn_stop.config(state=tk.NORMAL)
        # MIDI binding: fixed structure, symbolic only. No audio from us.
        with tempfile.NamedTemporaryFile(
            suffix=".mid", delete=False, prefix="branch_option_"
        ) as f:
            path = f.name
        if write_branch_midi_to_file(branch_index, path, self._pitch_variation, self._timing_variation):
            try:
                subprocess.run(["open", path], check=False)
            except FileNotFoundError:
                pass  # open not available (e.g. Linux); file still written
        self.root.after(500, self._on_stop)

    def _on_stop(self) -> None:
        self._is_playing = False
        self._playback_status.config(text="")
        self._btn_stop.config(state=tk.DISABLED)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    app = UIGate()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
