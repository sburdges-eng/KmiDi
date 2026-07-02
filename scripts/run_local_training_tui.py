#!/usr/bin/env python3
"""
Local JEPA training with a Rich TUI (progress, speed, model, loss).
Runs in iTerm2, WezTerm, or any terminal with color support.

Usage:
  python scripts/run_local_training_tui.py --config config/jepa_training_local_mac.yaml
  python scripts/run_local_training_tui.py --config config/jepa_training_local_mac.yaml \\
      --model chord_jepa

Requires: pip install rich
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme
except ImportError:
    print("Install rich for the TUI: pip install rich", file=sys.stderr)
    sys.exit(1)


def _load_config(path: str):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _build_training_config(raw: dict):
    from music_brain.jepa.config import (
        AudioJEPAConfig,
        ChordJEPAConfig,
        TrainingConfig,
    )

    t = raw.get("training", {})
    training = TrainingConfig(
        epochs=int(t.get("epochs", 120)),
        batch_size=int(t.get("batch_size", 12)),
        learning_rate=float(t.get("learning_rate", 3e-4)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        warmup_epochs=int(t.get("warmup_epochs", 10)),
        save_every=int(t.get("save_every", 5)),
        early_stop_patience=int(t.get("early_stop_patience", 15)),
        mixed_precision=bool(t.get("mixed_precision", True)),
        gradient_clip=float(t.get("gradient_clip", 1.0)),
    )
    a = raw.get("audio_jepa", {})
    audio_config = AudioJEPAConfig(
        latent_dim=int(a.get("latent_dim", 256)),
        n_mels=int(a.get("n_mels", 128)),
        max_frames=int(a.get("max_frames", 512)),
        sample_rate=int(a.get("sample_rate", 22050)),
        hop_length=int(a.get("hop_length", 512)),
        mask_ratio=float(a.get("mask_ratio", 0.6)),
        mask_block_size=int(a.get("mask_block_size", 4)),
        tier=str(a.get("tier", "medium")),
    )
    c = raw.get("chord_jepa", {})
    chord_config = ChordJEPAConfig(
        d_model=int(c.get("d_model", 256)),
        num_heads=int(c.get("num_heads", 8)),
        num_layers=int(c.get("num_layers", 4)),
        seq_len=int(c.get("seq_len", 64)),
        num_chords=int(c.get("num_chords", 170)),
        dropout=float(c.get("dropout", 0.1)),
        mask_ratio=float(c.get("mask_ratio", 0.6)),
    )
    return training, audio_config, chord_config


def make_render(state: dict, theme: Theme):
    """Build the Rich layout: header, progress, speed, model uniqueness, loss table."""
    model_name = state.get("model_name", "—")
    epoch = state.get("epoch", 0)
    total_epochs = state.get("total_epochs", 1)
    batch_idx = state.get("batch_idx", 0)
    n_batches = state.get("n_batches", 1)
    loss = state.get("loss", 0.0)
    avg_loss = state.get("avg_loss") or state.get("epoch_loss", 0.0)
    n_so_far = state.get("n_batches_so_far", 0)
    if n_so_far > 0 and "epoch_loss" in state:
        avg_loss = state["epoch_loss"] / n_so_far
    best_loss = state.get("best_loss", float("inf"))
    device = state.get("device", "—")
    batch_time = state.get("batch_time", 0.0)
    epoch_done = state.get("epoch_done", False)

    batches_per_sec = 1.0 / batch_time if batch_time > 0 else 0
    pct = (100.0 * (batch_idx + 1) / n_batches) if n_batches else 0
    bar_width = 36
    filled = int(bar_width * (batch_idx + 1) / n_batches) if n_batches else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    progress_line = (
        f"Epoch [bold cyan]{epoch + 1}[/]/{total_epochs}  "
        f"[{bar}] [bold]{pct:.0f}%[/]  "
        f"[cyan]{batches_per_sec:.1f}[/] batch/s  "
        f"batch [dim]{batch_idx + 1}[/]/{n_batches}"
    )

    metrics = Table(show_header=False, box=None, padding=(0, 2))
    metrics.add_column(style="dim")
    metrics.add_column(style="bold")
    metrics.add_row("Model", model_name)
    metrics.add_row("Device", device)
    metrics.add_row("Batch loss", f"{loss:.6f}")
    metrics.add_row("Epoch avg", f"{avg_loss:.6f}")
    metrics.add_row("Best loss", f"{best_loss:.6f}")
    if epoch_done and best_loss < float("inf"):
        uniqueness = "[green]★ new best[/]" if avg_loss < best_loss else "[dim]—[/]"
    else:
        uniqueness = "[dim]—[/]"
    metrics.add_row("Uniqueness", uniqueness)

    header = Panel(
        Text.from_markup("[bold]KmiDi local training[/] — progress · speed · model"),
        border_style="blue",
    )
    progress_panel = Panel(
        Text.from_markup(progress_line),
        title="[bold]Progress[/]",
        border_style="cyan",
    )
    metrics_panel = Panel(metrics, title="[bold]Metrics[/]", border_style="green")
    from rich.columns import Columns

    grid = Columns([progress_panel, metrics_panel], equal=False, expand=True)
    return Panel(
        header,
        grid,
        title="[bold white]Local training TUI[/] (iTerm2 / WezTerm)",
        border_style="white",
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Local JEPA training with Rich TUI")
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "jepa_training_local_mac.yaml"),
        help="Config YAML",
    )
    parser.add_argument("--model", choices=["audio_jepa", "chord_jepa", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    raw = _load_config(config_path)
    training, audio_config, chord_config = _build_training_config(raw)
    if args.epochs is not None:
        training.epochs = args.epochs

    audio_dir = os.environ.get("JEPA_AUDIO_DIR", str(_REPO_ROOT / "data" / "audio"))
    midi_dir = os.environ.get("JEPA_MIDI_DIR", str(_REPO_ROOT / "data" / "midi"))
    checkpoint_root = os.environ.get("JEPA_CHECKPOINT_DIR", str(_REPO_ROOT / "checkpoints"))

    theme = Theme({"info": "dim cyan", "success": "bold green"})
    state = {"model_name": "—", "epoch": 0, "total_epochs": training.epochs}

    def progress_callback(s: dict):
        state.clear()
        state.update(s)
        # Render is called from the main thread via Live; we only update state here.
        # So we need to drive Live from the main thread. We'll run training in a thread
        # and have the main thread poll state and refresh Live.

    from torch.utils.data import DataLoader
    from music_brain.jepa.datasets import AudioMelDataset, ChordSequenceDataset
    from music_brain.jepa.trainer import train_audio_jepa, train_chord_jepa

    def _midi_files(d: str):
        exts = {".mid", ".midi"}
        out = []
        for dirpath, _, names in os.walk(d):
            for n in names:
                if Path(n).suffix.lower() in exts:
                    out.append(os.path.join(dirpath, n))
        return out

    models_to_run = ["audio_jepa", "chord_jepa"] if args.model == "both" else [args.model]
    console = Console(theme=theme)
    exception_holder = []

    def run_training():
        try:
            for model_type in models_to_run:
                if model_type == "audio_jepa":
                    if not os.path.isdir(audio_dir):
                        console.print(f"[yellow]WARNING:[/] Audio dir not found: {audio_dir}")
                        continue
                    try:
                        dataset = AudioMelDataset.from_directory(
                            audio_dir,
                            n_mels=audio_config.n_mels,
                            hop_length=audio_config.hop_length,
                            max_frames=audio_config.max_frames,
                            sr=audio_config.sample_rate,
                        )
                    except Exception as e:
                        console.print(f"[red]ERROR:[/] Audio-JEPA dataset: {e}")
                        if args.model != "both":
                            exception_holder.append(e)
                        continue
                    if len(dataset) == 0:
                        console.print(f"[red]ERROR:[/] No audio files in {audio_dir}")
                        continue
                    loader = DataLoader(
                        dataset, batch_size=training.batch_size, shuffle=True, num_workers=0
                    )
                    ckpt_dir = os.path.join(checkpoint_root, "audio_jepa")
                    export_path = (
                        os.path.join(checkpoint_root, "audio_jepa.onnx")
                        if raw.get("checkpoints", {}).get("export_onnx", True)
                        else None
                    )
                    train_audio_jepa(
                        loader,
                        audio_config,
                        training,
                        checkpoint_dir=ckpt_dir,
                        export_path=export_path,
                        progress_callback=progress_callback,
                    )
                else:
                    midi_files = _midi_files(midi_dir) if os.path.isdir(midi_dir) else []
                    dataset = ChordSequenceDataset(
                        files=midi_files if midi_files else None,
                        seq_len=chord_config.seq_len,
                        num_chords=chord_config.num_chords,
                        num_samples=256 if not midi_files else None,
                    )
                    loader = DataLoader(
                        dataset, batch_size=training.batch_size, shuffle=True, num_workers=0
                    )
                    ckpt_dir = os.path.join(checkpoint_root, "chord_jepa")
                    train_chord_jepa(
                        loader,
                        chord_config,
                        training,
                        checkpoint_dir=ckpt_dir,
                        progress_callback=progress_callback,
                    )
        except Exception as e:
            exception_holder.append(e)

    # Run training in a thread; main thread runs Live and refreshes from state
    train_thread = threading.Thread(target=run_training, daemon=True)
    refresh_interval = 0.1

    with Live(
        make_render(state, theme),
        console=console,
        refresh_per_second=10,
        transient=False,
    ) as live:
        train_thread.start()
        while train_thread.is_alive():
            import time

            live.update(make_render(state, theme))
            time.sleep(refresh_interval)
        live.update(make_render(state, theme))

    if exception_holder:
        console.print("[red]Training failed:[/]", exception_holder[0])
        sys.exit(1)
    console.print("[green]Done.[/]")


if __name__ == "__main__":
    main()
