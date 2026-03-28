"""
StructXLIP — structural audio proxies and structure-centric alignment losses for LALM training.

Experimental: use for fine-tuning when you need the model to ground temporal boundaries
(e.g. "drop at bar 16, beat 3") via Audio Edge Maps rather than dense mel-only alignment.

- audio_edges: onset envelope, spectral flux (and optional beat-synced grid).
- structure_losses: global structure–text alignment; placeholders for local and consistency.
"""

try:
    from .audio_edges import (
        extract_onset_envelope,
        extract_spectral_flux,
        extract_audio_edge_maps,
    )
    from .structure_losses import (
        global_structure_loss,
        local_structure_loss,
        consistency_edge_loss,
    )
except ImportError:
    from .audio_edges import (
        extract_onset_envelope,
        extract_spectral_flux,
        extract_audio_edge_maps,
    )
    from .structure_losses import (
        global_structure_loss,
        local_structure_loss,
        consistency_edge_loss,
    )

# Aliases for backward compatibility
compute_onset_envelope = extract_onset_envelope
compute_spectral_flux = extract_spectral_flux
extract_audio_edges = extract_audio_edge_maps

__all__ = [
    "extract_onset_envelope",
    "extract_spectral_flux",
    "extract_audio_edge_maps",
    "compute_onset_envelope",
    "compute_spectral_flux",
    "extract_audio_edges",
    "global_structure_loss",
    "local_structure_loss",
    "consistency_edge_loss",
]
