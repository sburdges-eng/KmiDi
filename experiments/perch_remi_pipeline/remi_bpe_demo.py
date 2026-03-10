#!/usr/bin/env python
"""
REMI-BPE demo for KmiDi experiments.

Demonstrates:
- Loading a REMI tokenizer (MidiTok) compatible with Maestro-REMI-bpe20k.
- Tokenizing a MIDI file.
- Running a short generation step with a Hugging Face causal model.
- Decoding tokens back to MIDI.

CLI example:

    python remi_bpe_demo.py \
        --midi-path ~/Datasets/maestro/midi/2011/track.mid \
        --model NathanFradet/Maestro-REMI-bpe20k \
        --max-new-tokens 256 \
        --output generated_track.mid
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def load_tokenizer(model_id: str):
    """
    Load MidiTok REMI tokenizer from a pretrained config, or default REMI for tokenize-only.
    """
    try:
        from miditok import REMI
    except ImportError as exc:  # noqa: PERF203
        raise SystemExit(
            "miditok is not installed. Install with `pip install miditok`."
        ) from exc

    try:
        tokenizer = REMI.from_pretrained(model_id)
    except Exception:  # noqa: BLE001
        # Offline or missing repo: use default REMI for tokenize-only round-trip tests
        tokenizer = REMI()
    return tokenizer


def load_model(model_id: str):
    """
    Load a causal LM from Hugging Face for symbolic generation.

    Keeps this script self-contained; you must install transformers
    and ensure the model is compatible with the tokenizer.
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # noqa: PERF203
        raise SystemExit(
            "transformers is not installed. Install with `pip install transformers`."
        ) from exc

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return model


def load_midi(tokens_path: Optional[Path], midi_path: Path, tokenizer) -> list[int]:
    """
    Tokenize a MIDI file into REMI-BPE ids.
    """
    if tokens_path and tokens_path.is_file():
        import json

        with tokens_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data["token_ids"]

    tok_out = tokenizer(midi_path)
    # MidiTok: may return list of TokenizedMusic or single; .ids on first sequence
    if hasattr(tok_out, "__getitem__") and len(tok_out):
        first = tok_out[0]
        return first.ids if hasattr(first, "ids") else list(first)
    if hasattr(tok_out, "ids"):
        return tok_out.ids
    raise ValueError("Unexpected tokenizer output")


def generate_tokens(
    input_ids: list[int],
    model,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> list[int]:
    """
    Run a short generation with the given model.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_tensor,
            do_sample=temperature > 0.0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=model.config.eos_token_id,
        )

    return out[0].tolist()


def decode_to_midi(token_ids: list[int], tokenizer, output_path: Path) -> None:
    """
    Decode token ids back to MIDI and save to disk.
    Supports MidiTok 3 (decode -> Score.dump_midi) and older (TokenizedMusic + tokens_to_midi).
    """
    if hasattr(tokenizer, "decode"):
        score = tokenizer.decode(token_ids)
        if hasattr(score, "dump_midi"):
            score.dump_midi(str(output_path))
        else:
            score.dump(str(output_path))
        return
    from miditok import TokenizedMusic  # type: ignore[import]

    token_obj = TokenizedMusic(tokenizer=tokenizer, ids=token_ids)
    midi = tokenizer.tokens_to_midi([token_obj])
    midi.dump(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="REMI-BPE demo with Maestro-REMI-bpe20k.")
    parser.add_argument(
        "--midi-path",
        type=str,
        required=True,
        help="Input MIDI file to tokenize and continue from.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="NathanFradet/Maestro-REMI-bpe20k",
        help="Hugging Face model / tokenizer id (e.g. NathanFradet/Maestro-REMI-bpe20k).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output MIDI file path.",
    )
    parser.add_argument(
        "--tokens-cache",
        type=str,
        default=None,
        help="Optional JSON file with precomputed token_ids to avoid re-tokenizing.",
    )
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="Only tokenize and round-trip to MIDI (no model load or generation).",
    )

    args = parser.parse_args()

    midi_path = Path(args.midi_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    tokens_cache = Path(args.tokens_cache).expanduser().resolve() if args.tokens_cache else None

    tokenizer = load_tokenizer(args.model)
    token_ids = load_midi(tokens_cache, midi_path, tokenizer)
    print(f"[remi_bpe_demo] Loaded {len(token_ids)} tokens from {midi_path}")

    if args.tokenize_only:
        full_ids = token_ids
        print("[remi_bpe_demo] Tokenize-only: skipping model, decoding back to MIDI.")
    else:
        model = load_model(args.model)
        full_ids = generate_tokens(
            input_ids=token_ids,
            model=model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"[remi_bpe_demo] Generated total sequence length: {len(full_ids)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    decode_to_midi(full_ids, tokenizer, output_path)
    print(f"[remi_bpe_demo] Wrote MIDI to {output_path}")


if __name__ == "__main__":
    main()

