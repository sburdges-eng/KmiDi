"""
Preprocess emotion datasets for text→VAI and VAI→MIDI training.

Supported:
- EmoBank (sentence-level VAD) → normalized CSV with valence [-1,1], arousal/intensity [0,1].
- EMOPIA (MIDI + quadrant labels) → CSV with VAI labels per file.

Usage examples:
  python scripts/preprocess_emotion_datasets.py \
      --emobank-csv /path/to/affectivetext_en.csv \
      --emobank-out data/processed/emobank_vai.csv

  python scripts/preprocess_emotion_datasets.py \
      --emopia-annotations /path/to/EMOPIA_annotations.csv \
      --emopia-midi-dir /path/to/EMOPIA/midis \
      --emopia-out data/processed/emopia_vai.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def _norm_valence_1_5_to_unit(val: float) -> float:
    """Map 1–5 scale to -1..1."""
    return (float(val) - 3.0) / 2.0


def _norm_arousal_1_5_to_unit(val: float) -> float:
    """Map 1–5 scale to 0..1."""
    return (float(val) - 1.0) / 4.0


def _norm_intensity_from_arousal(arousal_01: float) -> float:
    """Use arousal as a proxy for intensity (0..1)."""
    return max(0.0, min(1.0, float(arousal_01)))


def _load_quadrant_map(
    custom_map_json: Optional[Path],
) -> Dict[str, Tuple[float, float, float]]:
    """
    Return quadrant→(valence, arousal, intensity) mapping.
    Defaults are aligned with high/low valence/arousal heuristics.
    """
    default_map = {
        "Q1": (0.8, 0.8, 0.9),    # HVHA
        "Q2": (0.8, -0.8, 0.6),   # HVLA (calm/relaxed)
        "Q3": (-0.8, -0.8, 0.7),  # LVLA (sad/depressed)
        "Q4": (-0.8, 0.8, 0.95),  # LVHA (angry/tense)
    }
    if custom_map_json and custom_map_json.exists():
        with open(custom_map_json, "r") as f:
            raw = json.load(f)
        parsed = {}
        for k, v in raw.items():
            if not isinstance(v, (list, tuple)) or len(v) != 3:
                raise ValueError(f"Quadrant map entry for {k} must be list/tuple of length 3")
            parsed[k] = (float(v[0]), float(v[1]), float(v[2]))
        return parsed
    return default_map


def process_emobank(emobank_csv: Path, output_csv: Path) -> None:
    """
    Normalize EmoBank VAD to VAI and write CSV with columns:
    text,valence,arousal,intensity
    """
    df = pd.read_csv(emobank_csv)

    # EmoBank columns are typically: id, text, V, A, D (dominance)
    text_col = "text" if "text" in df.columns else "sentence"
    val_col = "V" if "V" in df.columns else "valence"
    aro_col = "A" if "A" in df.columns else "arousal"

    if text_col not in df or val_col not in df or aro_col not in df:
        raise ValueError(f"Expected columns text/V/A (or sentence/valence/arousal) in {emobank_csv}")

    out_df = pd.DataFrame()
    out_df["text"] = df[text_col].astype(str)
    out_df["valence"] = df[val_col].apply(_norm_valence_1_5_to_unit)
    out_df["arousal"] = df[aro_col].apply(_norm_arousal_1_5_to_unit)
    out_df["intensity"] = out_df["arousal"].apply(_norm_intensity_from_arousal)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[EmoBank] Wrote {len(out_df)} rows -> {output_csv}")


def process_emopia(
    annotations_csv: Path,
    midi_dir: Path,
    output_csv: Path,
    quadrant_map_json: Optional[Path] = None,
) -> None:
    """
    Map EMOPIA quadrants to VAI and write CSV with:
    file_id,midi_path,valence,arousal,intensity,quadrant
    """
    quad_map = _load_quadrant_map(quadrant_map_json)
    df = pd.read_csv(annotations_csv)

    # Common columns: clip_id or file_id, emo_class or quadrant (1-4/Q1-Q4)
    file_col = "clip_id" if "clip_id" in df.columns else "file_id"
    quad_col = "quadrant" if "quadrant" in df.columns else "emo_class"

    if file_col not in df or quad_col not in df:
        raise ValueError(f"Expected columns clip_id/file_id and quadrant/emo_class in {annotations_csv}")

    records = []
    for _, row in df.iterrows():
        fid = str(row[file_col])
        quad_raw = str(row[quad_col])
        quad = f"Q{quad_raw}" if quad_raw.isdigit() else quad_raw.upper()
        if quad not in quad_map:
            raise ValueError(f"Unknown quadrant {quad} in row {row}")

        val, aro, inten = quad_map[quad]
        midi_path = midi_dir / f"{fid}.mid"

        records.append(
            {
                "file_id": fid,
                "midi_path": str(midi_path),
                "valence": val,
                "arousal": aro,
                "intensity": inten,
                "quadrant": quad,
            }
        )

    out_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[EMOPIA] Wrote {len(out_df)} rows -> {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess EmoBank/EMOPIA into normalized VAI CSVs.")

    parser.add_argument("--emobank-csv", type=Path, help="Path to EmoBank CSV (affectivetext_en.csv).")
    parser.add_argument("--emobank-out", type=Path, help="Output CSV for normalized EmoBank.")

    parser.add_argument("--emopia-annotations", type=Path, help="Path to EMOPIA annotations CSV.")
    parser.add_argument("--emopia-midi-dir", type=Path, help="Directory containing EMOPIA MIDI files.")
    parser.add_argument("--emopia-out", type=Path, help="Output CSV for EMOPIA VAI labels.")
    parser.add_argument("--quadrant-map", type=Path, help="Optional JSON file overriding quadrant→VAI mapping.")

    args = parser.parse_args()

    if args.emobank_csv and args.emobank_out:
        process_emobank(args.emobank_csv, args.emobank_out)

    if args.emopia_annotations and args.emopia_midi_dir and args.emopia_out:
        process_emopia(
            args.emopia_annotations,
            args.emopia_midi_dir,
            args.emopia_out,
            args.quadrant_map,
        )

    if not (
        (args.emobank_csv and args.emobank_out)
        or (args.emopia_annotations and args.emopia_midi_dir and args.emopia_out)
    ):
        parser.error("Provide EmoBank args, EMOPIA args, or both.")


if __name__ == "__main__":
    main()
