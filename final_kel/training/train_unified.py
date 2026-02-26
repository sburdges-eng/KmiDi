"""
Unified trainer: CUDA + MPS (Apple Silicon) + CPU.
Audio-JEPA encoder + LatentPredictor; checkpointing; ONNX export.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

from Brain.audio_jepa.model import AudioJEPAEncoder, LatentPredictor
from Brain.audio_jepa.masking import mask_latents
from training.dataloader_audio import AudioDataset
from training.storage_paths import (
    DATASETS_DIR,
    CHECKPOINTS_DIR,
    EXPORT_AUDIO_ONNX,
    EXPERIMENTS_DIR,
    PROCESSED_DATASET_FILE,
    ensure_storage_dirs,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_file_list(datasets_dir, exts=("wav", "mp3", "flac")):
    files = []
    if not os.path.isdir(datasets_dir):
        return files
    for root, _dirs, fnames in os.walk(datasets_dir):
        for f in fnames:
            if f.lower().split(".")[-1] in exts:
                files.append(os.path.join(root, f))
    return files


class ProcessedMelDataset(Dataset):
    """Audio samples sourced from a single preprocessed .npy dataset file."""

    def __init__(self, dataset_file, max_frames=512):
        if not os.path.isfile(dataset_file):
            raise FileNotFoundError(dataset_file)
        self.data = np.load(dataset_file, allow_pickle=True).item()
        self.keys = []
        self.max_frames = max_frames
        for key, item in self.data.items():
            if not isinstance(item, dict):
                continue
            if item.get("type") != "audio":
                continue
            mel = item.get("mel")
            if mel is None:
                continue
            mel = np.asarray(mel, dtype=np.float32)
            if mel.ndim == 2:
                self.keys.append(key)
        if not self.keys:
            raise ValueError("No audio mel entries found in dataset file: " + dataset_file)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        mel = np.asarray(self.data[key]["mel"], dtype=np.float32)
        t = mel.shape[1]
        if t >= self.max_frames:
            start = np.random.randint(0, t - self.max_frames + 1)
            mel = mel[:, start : start + self.max_frames]
        else:
            pad = np.zeros((mel.shape[0], self.max_frames - t), dtype=mel.dtype)
            mel = np.concatenate([mel, pad], axis=1)
        return torch.tensor(mel, dtype=torch.float32).unsqueeze(0)


def save_checkpoint(encoder, predictor, opt, epoch, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
    }, path)


def export_onnx(encoder, out_path, latent_dim=256, n_mels=128, max_frames=512):
    encoder.cpu().eval()
    dummy = torch.randn(1, 1, n_mels, max_frames)
    torch.onnx.export(
        encoder,
        dummy,
        out_path,
        opset_version=17,
        input_names=["mel"],
        output_names=["latent"],
    )


def train(args):
    ensure_storage_dirs()

    if getattr(args, "cpu", False):
        device = torch.device("cpu")
    else:
        device = get_device()
    print("Using device:", device)

    tracker = None
    if getattr(args, "experiment_name", None):
        try:
            from tools.experiment_tracker import ExperimentTracker
            tracker = ExperimentTracker(runs_dir=getattr(args, "experiment_dir", EXPERIMENTS_DIR))
            tracker.start_run(args.experiment_name, config={"tier": args.tier, "epochs": args.epochs, "lr": args.lr})
        except Exception:
            tracker = None

    train_data = None
    if args.dataset_file and os.path.isfile(args.dataset_file):
        try:
            train_data = ProcessedMelDataset(args.dataset_file, max_frames=512)
            print("Using single dataset file:", args.dataset_file)
        except Exception as e:
            print("Failed to load dataset file:", e)
            return

    if train_data is None:
        file_list = args.files or build_file_list(args.datasets_dir)
        if not file_list:
            print("No dataset file or audio files found.")
            print("Expected dataset file:", args.dataset_file)
            print("Or add wav/mp3/flac in:", args.datasets_dir)
            return
        train_data = AudioDataset(
            file_list,
            sr=args.sample_rate,
            n_mels=128,
            hop_length=512,
            max_frames=512,
        )

    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
    )

    encoder = AudioJEPAEncoder(tier=args.tier, latent_dim=args.latent_dim).to(device)
    predictor = LatentPredictor(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                z = encoder(batch)
                masked, _ = mask_latents(z)
                pred = predictor(masked)
                loss = F.mse_loss(pred, z)

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader) if loader else 0.0
        print(f"Epoch {epoch} | Loss {avg:.6f}")
        if tracker is not None:
            tracker.log_metric(epoch, "loss", avg)

        if (epoch + 1) % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, f"jepa_epoch_{epoch}.pt")
            save_checkpoint(encoder, predictor, opt, epoch, path)
            print("Saved", path)
            if tracker is not None:
                tracker.log_checkpoint(epoch, path)

    export_path = args.export_path or EXPORT_AUDIO_ONNX
    os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
    export_onnx(encoder, export_path, args.latent_dim)
    print("ONNX export:", export_path)


def main():
    p = argparse.ArgumentParser(description="KmiDi unified Audio-JEPA trainer (CUDA/MPS/CPU)")
    p.add_argument("--dataset-file", default=PROCESSED_DATASET_FILE, help="Single preprocessed .npy dataset file")
    p.add_argument("--datasets-dir", default=DATASETS_DIR, help="Fallback folder to scan for audio files")
    p.add_argument("--files", nargs="*", help="Explicit list of audio files")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--batch-size", type=int, default=None, help="Default 12 (cuda) / 6 (mps/cpu)")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--tier", default="medium", choices=("small", "medium", "large"))
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--checkpoint-dir", default=CHECKPOINTS_DIR)
    p.add_argument("--export-path", default=EXPORT_AUDIO_ONNX, help="ONNX output path")
    p.add_argument("--experiment-name", default=None, help="Log run to experiment tracker")
    p.add_argument("--experiment-dir", default=EXPERIMENTS_DIR, help="Experiment runs dir")
    args = p.parse_args()

    if args.batch_size is None:
        args.batch_size = 12 if torch.cuda.is_available() else 6

    train(args)


if __name__ == "__main__":
    main()
