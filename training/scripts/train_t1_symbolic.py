#!/usr/bin/env python3
"""T1 prototype — small autoregressive transformer over REMI MIDI tokens.

A pragmatic re-scoping of the original "T1 = 3-7B symbolic" target. With
a $100/mo Vertex budget the from-scratch 7B path is infeasible (~$2K),
so this script trains a much smaller (~10-30M param) GPT-style decoder
over REMI tokens — the same scale as MuseNet / Music Transformer / AMT,
which already produce real symbolic music.

Pipeline:

  midi files -> miditok REMI -> cached token shards -> small GPT

Two phases, controlled by --mode:

  tokenize   walk a MIDI corpus once, fit a REMI tokenizer, encode to a
             single concatenated uint16 stream cached under
             ~/Models/cache/t1_tokens/<corpus-name>/.

  train      load the cached stream, train a small GPT next-token model.
             Standard governance: writes run_manifest.yaml, best.pt,
             tokenizer.json, and tensorboard logs under
             ~/Models/checkpoints/<run-name>/.

Defaults are sized so a first end-to-end pass completes in a couple
hours on Mac MPS. Bump --d-model / --n-layers / --max-files when you
have a Vertex job.

Usage::

    # Step 1 — tokenize a small Lakh subset (one-off)
    python training/scripts/train_t1_symbolic.py --mode tokenize \\
        --midi-dir ~/Datasets/midi/lakh/lmd_matched --max-files 2000

    # Step 2 — train (defaults to the corpus tokenized above)
    python training/scripts/train_t1_symbolic.py --mode train --epochs 3

    # End-to-end smoke (tiny: 50 files, 1 epoch, model ~1M params)
    python training/scripts/train_t1_symbolic.py --mode all --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("train_t1_symbolic")

CACHE_ROOT = Path.home() / "Models" / "cache" / "t1_tokens"
MODELS_ROOT_DEFAULT = Path.home() / "Models"

# ── tokenization ─────────────────────────────────────────────────


def fit_and_encode(midi_dir: Path, corpus_name: str, max_files: int,
                   max_tokens_per_file: int = 8192) -> Path:
    """Walk a MIDI corpus, fit REMI tokenizer, encode to concatenated uint16
    stream. Returns the cache directory path. Idempotent: skip if cached."""
    from miditok import REMI, TokenizerConfig

    cache_dir = CACHE_ROOT / corpus_name
    train_path = cache_dir / "train.npy"
    val_path = cache_dir / "val.npy"
    tokenizer_path = cache_dir / "tokenizer.json"
    manifest_path = cache_dir / "manifest.json"

    if train_path.exists() and tokenizer_path.exists() and manifest_path.exists():
        logger.info("Tokenized cache already present at %s", cache_dir)
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning %s for MIDI files (limit %d)...", midi_dir, max_files)
    midi_files: list[Path] = []
    for ext in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
        for f in midi_dir.rglob(ext):
            midi_files.append(f)
            if len(midi_files) >= max_files:
                break
        if len(midi_files) >= max_files:
            break
    if not midi_files:
        raise SystemExit(f"No MIDI files found under {midi_dir}")
    logger.info("Found %d MIDI files", len(midi_files))

    # REMI with sane defaults: includes Pitch, Velocity, Duration, Position,
    # Bar, TimeSignature, Tempo. vocab usually 300-500.
    cfg = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=24,
        special_tokens=["PAD", "BOS", "EOS"],
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=False,
        num_tempos=24,
        tempo_range=(40, 240),
    )
    tokenizer = REMI(cfg)

    # Encode each file -> token list. Concatenate with EOS separators.
    bos_id = tokenizer.vocab["BOS_None"]
    eos_id = tokenizer.vocab["EOS_None"]

    all_tokens: list[int] = []
    skipped = 0
    pbar = tqdm(midi_files, desc="tokenize", leave=False)
    for mf in pbar:
        try:
            tok_seq = tokenizer.encode(str(mf))
            # miditok returns a TokSequence (or list of for multi-track).
            if hasattr(tok_seq, "ids"):
                ids = list(tok_seq.ids)
            elif isinstance(tok_seq, list) and tok_seq and hasattr(tok_seq[0], "ids"):
                # Multi-track: flatten with EOS between tracks.
                ids = []
                for ts in tok_seq:
                    ids.extend(ts.ids)
                    ids.append(eos_id)
            else:
                skipped += 1
                continue
            ids = ids[:max_tokens_per_file]
            if len(ids) < 32:
                skipped += 1
                continue
            all_tokens.append(bos_id)
            all_tokens.extend(ids)
            all_tokens.append(eos_id)
        except Exception as e:
            skipped += 1
            if skipped < 5:
                logger.debug("Skipped %s: %s", mf.name, e)

    vocab_size = len(tokenizer.vocab)
    logger.info("Tokenized %d files (%d skipped). Total tokens: %d. Vocab: %d.",
                len(midi_files) - skipped, skipped, len(all_tokens), vocab_size)

    # uint16 fits any vocab < 65535; REMI vocabs are ~500.
    if vocab_size >= 2**16:
        raise RuntimeError(f"Vocab {vocab_size} exceeds uint16 range")
    arr = np.asarray(all_tokens, dtype=np.uint16)

    # 95/5 split
    n = arr.shape[0]
    n_val = max(int(n * 0.05), 4096)
    train = arr[:-n_val]
    val = arr[-n_val:]

    np.save(train_path, train)
    np.save(val_path, val)
    tokenizer.save(tokenizer_path)
    manifest_path.write_text(json.dumps({
        "corpus_name": corpus_name,
        "midi_dir": str(midi_dir),
        "n_files_scanned": len(midi_files),
        "n_files_skipped": skipped,
        "n_tokens_total": int(n),
        "n_tokens_train": int(train.shape[0]),
        "n_tokens_val": int(val.shape[0]),
        "vocab_size": vocab_size,
        "max_tokens_per_file": max_tokens_per_file,
        "tokenizer": "REMI",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    logger.info("Wrote cache: %s", cache_dir)
    return cache_dir


# ── dataset ──────────────────────────────────────────────────────


class TokenStreamDataset(Dataset):
    """Memory-maps a uint16 token stream and serves random fixed-length
    windows for next-token prediction (input, target shifted by one)."""

    def __init__(self, npy_path: Path, ctx_len: int, samples_per_epoch: int):
        self.tokens = np.load(npy_path, mmap_mode="r")
        self.ctx_len = ctx_len
        self.n_samples = samples_per_epoch
        self.max_start = len(self.tokens) - ctx_len - 1
        if self.max_start <= 0:
            raise RuntimeError(
                f"Token stream too short ({len(self.tokens)}) for ctx {ctx_len}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        # Independent of idx — random window each call. RNG seeded per worker.
        start = random.randint(0, self.max_start)
        chunk = self.tokens[start:start + self.ctx_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


# ── model ────────────────────────────────────────────────────────


class SymbolicGPT(nn.Module):
    """Small decoder-only transformer. Causal mask via PyTorch's built-in
    is_causal flag in nn.MultiheadAttention via TransformerEncoderLayer with
    a causal mask — but we use a clean from-scratch block to keep MPS happy."""

    def __init__(self, vocab_size: int, d_model: int = 384, n_layers: int = 6,
                 n_heads: int = 6, d_ff: int = 1536, ctx_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.ctx_len = ctx_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        # Causal mask, additive (-inf above diagonal).
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


# ── train loop ───────────────────────────────────────────────────


def pick_device(force: str | None) -> torch.device:
    if force:
        return torch.device(force)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: SymbolicGPT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def train_loop(args: argparse.Namespace) -> int:
    cache_dir = CACHE_ROOT / args.corpus
    train_npy = cache_dir / "train.npy"
    val_npy = cache_dir / "val.npy"
    tokenizer_path = cache_dir / "tokenizer.json"
    if not train_npy.exists():
        raise SystemExit(
            f"No tokenized cache at {cache_dir}. Run --mode tokenize first.")

    # Load vocab from manifest (avoid pulling miditok at train time).
    cache_manifest = json.loads((cache_dir / "manifest.json").read_text())
    vocab_size = int(cache_manifest["vocab_size"])

    device = pick_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.smoke:
        args.epochs = 1
        args.d_model = 128
        args.n_layers = 2
        args.n_heads = 4
        args.d_ff = 512
        args.ctx_len = 128
        args.batch_size = 4
        args.steps_per_epoch = 50
        args.val_steps = 10

    train_set = TokenStreamDataset(
        train_npy, args.ctx_len, args.steps_per_epoch * args.batch_size)
    val_set = TokenStreamDataset(
        val_npy, args.ctx_len, args.val_steps * args.batch_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=False)

    model = SymbolicGPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        ctx_len=args.ctx_len,
        dropout=args.dropout,
    ).to(device)

    n_params = model.num_params()
    logger.info("Model: %.2fM params (vocab=%d, d=%d, L=%d, ctx=%d)",
                n_params / 1e6, vocab_size, args.d_model, args.n_layers, args.ctx_len)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95))

    total_steps = args.epochs * args.steps_per_epoch
    warmup = min(args.warmup_steps, max(total_steps // 20, 50))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        # Cosine decay to 10% over remaining steps.
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    run_name = args.name or f"t1-symbolic-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ckpt_dir = Path(args.models_root).expanduser() / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb"))

    # Copy tokenizer next to the checkpoint so the model is self-contained.
    if tokenizer_path.exists():
        (ckpt_dir / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())

    manifest = {
        "experiment_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "device": str(device),
        "corpus": args.corpus,
        "cache_dir": str(cache_dir),
        "vocab_size": vocab_size,
        "n_params": n_params,
        "seed": args.seed,
        "hparams": {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "ctx_len": args.ctx_len,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "steps_per_epoch": args.steps_per_epoch,
            "val_steps": args.val_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": warmup,
            "grad_clip": args.grad_clip,
        },
        "smoke": args.smoke,
        "torch_version": str(torch.__version__),
    }
    try:
        import yaml
        (ckpt_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    except ImportError:
        (ckpt_dir / "run_manifest.yaml").write_text(json.dumps(manifest, indent=2))

    logger.info("Run %s | ckpt_dir=%s | total_steps=%d | warmup=%d",
                run_name, ckpt_dir, total_steps, warmup)

    best_val = float("inf")
    global_step = 0
    start_t = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for x, y in pbar:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(global_step)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item() * x.size(0)
            n_seen += x.size(0)
            global_step += 1
            if global_step % 50 == 0:
                writer.add_scalar("loss/train_step", loss.item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

        train_loss = running / max(n_seen, 1)
        val_loss = evaluate(model, val_loader, device)
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))
        logger.info(
            "[epoch %d/%d] train_loss=%.4f train_ppl=%.2f  val_loss=%.4f val_ppl=%.2f",
            epoch + 1, args.epochs, train_loss, train_ppl, val_loss, val_ppl)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("ppl/train", train_ppl, epoch)
        writer.add_scalar("ppl/val", val_ppl, epoch)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "vocab_size": vocab_size,
                "config": manifest["hparams"],
            }, ckpt_dir / "best.pt")
            logger.info("  ↳ new best val_loss=%.4f → %s", val_loss, ckpt_dir / "best.pt")

    writer.close()
    elapsed = time.perf_counter() - start_t
    logger.info("Training complete. best_val_loss=%.4f best_val_ppl=%.2f wall=%.1fs ckpt=%s",
                best_val, math.exp(min(best_val, 20)), elapsed, ckpt_dir)
    return 0


# ── main ────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["tokenize", "train", "all"], default="train")
    ap.add_argument("--midi-dir", type=str,
                    default=str(Path.home() / "Datasets" / "midi" / "lakh" / "lmd_matched"))
    ap.add_argument("--corpus", type=str, default="lakh-2k",
                    help="Cache subdir name under ~/Models/cache/t1_tokens/")
    ap.add_argument("--max-files", type=int, default=2000)
    ap.add_argument("--max-tokens-per-file", type=int, default=8192)

    # Train hparams
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=1536)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--val-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None,
                    choices=[None, "mps", "cuda", "cpu"])
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--models-root", type=str,
                    default=os.environ.get("KELLY_MODEL_ROOT", str(MODELS_ROOT_DEFAULT)))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.max_files = 50
        args.corpus = "lakh-smoke50"

    if args.mode in ("tokenize", "all"):
        fit_and_encode(
            Path(args.midi_dir).expanduser(),
            args.corpus, args.max_files, args.max_tokens_per_file)

    if args.mode in ("train", "all"):
        return train_loop(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
