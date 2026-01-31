"""
Utility functions exposed to the offline chatbot agent.
"""

from __future__ import annotations

def describe_auto_tune_usage() -> str:
    return (
        "To auto-tune a vocal offline, run:\n"
        "  daiw voice tune input.wav --preset transparent --key F --mode minor\n"
        "This uses the local AutoTuneProcessor without any network calls."
    )


def describe_backing_workflow() -> str:
    return (
        "Backing-track generation flow:\n"
        "1. Create/edit your intent JSON.\n"
        "2. Run `daiw backing --intent my_intent.json` (feature WIP).\n"
        "3. The offline engine will return MIDI + audio stems packaged locally."
    )

