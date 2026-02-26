"""
KmiDi Dataset Processing Pipeline v2.

Build one canonical dataset file for training.
Default output is a single .npy file on the external drive.
"""

import argparse
import os

import numpy as np
from training.storage_paths import DATASETS_DIR, PROCESSED_DATASET_FILE

try:
    import librosa
except ImportError:
    librosa = None
try:
    import pretty_midi
except ImportError:
    pretty_midi = None
try:
    import cv2
except ImportError:
    cv2 = None

DEFAULT_BASE = DATASETS_DIR
DEFAULT_OUT_FILE = PROCESSED_DATASET_FILE

def process_audio(path, sr=22050):
    if librosa is None:
        raise ImportError("librosa required. pip install librosa")
    y, sr = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    return y, sr, mel


def process_midi(path, fs=100):
    if pretty_midi is None:
        raise ImportError("pretty_midi required. pip install pretty_midi")
    pm = pretty_midi.PrettyMIDI(path)
    return pm.get_piano_roll(fs=fs)


def process_sheet(img_path):
    if cv2 is None:
        raise ImportError("opencv-python required. pip install opencv-python")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.array([])
    return cv2.Canny(img, 50, 150)


def process_tabs(txt_path):
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def analyze_environment(y, sr):
    if librosa is None:
        return {"rms": 0.0, "reverb": 0.0, "env": "unknown"}
    rms = np.mean(librosa.feature.rms(y=y))
    reverb = np.mean(librosa.feature.spectral_flatness(y=y + 1e-9))
    env = "hallway/concert" if reverb > 0.4 else "studio/close-mic"
    return {"rms": float(rms), "reverb": float(reverb), "env": env}


def master_eq_profile(y, sr):
    freqs = np.abs(np.fft.rfft(y))
    n = len(freqs)
    if n < 3:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    low = float(np.mean(freqs[: n // 3]))
    mid = float(np.mean(freqs[n // 3 : 2 * n // 3]))
    high = float(np.mean(freqs[2 * n // 3 :]))
    return {"low": low, "mid": mid, "high": high}


def process_video(path, max_frames=64, frame_stride=8):
    if cv2 is None:
        return np.array([])
    cap = cv2.VideoCapture(path)
    frames = []
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            if len(frames) >= max_frames:
                break
        frame_index += 1
    cap.release()
    return np.array(frames) if frames else np.array([])


def process_folder(folder, max_video_frames=64, video_stride=8):
    """Walk folder and process audio, MIDI, sheet, tabs, video. Returns dict path -> dict payload."""
    data = {}
    for root, _dirs, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            ext = f.lower().split(".")[-1]

            if ext in ("wav", "mp3", "flac"):
                try:
                    y, sr, mel = process_audio(p)
                    env = analyze_environment(y, sr)
                    eq = master_eq_profile(y, sr)
                    entry = {
                        "type": "audio",
                        "sr": int(sr),
                        "num_samples": int(len(y)),
                        "duration_sec": float(len(y) / sr if sr else 0.0),
                        "env": env,
                        "eq": eq,
                        "mel": np.asarray(mel, dtype=np.float32),
                    }
                    data[p] = entry
                except Exception as e:
                    data[p] = {"error": str(e)}

            elif ext in ("mid", "midi") and pretty_midi is not None:
                try:
                    pr = process_midi(p)
                    data[p] = {
                        "type": "midi",
                        "shape": list(pr.shape),
                        "pianoroll": np.asarray(pr, dtype=np.float32),
                    }
                except Exception as e:
                    data[p] = {"error": str(e)}

            elif ext in ("png", "jpg", "jpeg") and cv2 is not None:
                try:
                    edges = process_sheet(p)
                    data[p] = {
                        "type": "sheet",
                        "shape": list(edges.shape),
                        "sheet_edges": np.asarray(edges, dtype=np.uint8),
                    }
                except Exception as e:
                    data[p] = {"error": str(e)}

            elif ext == "txt":
                try:
                    tabs = process_tabs(p)
                    data[p] = {"type": "tabs", "tabs": tabs}
                except Exception as e:
                    data[p] = {"error": str(e)}

            elif ext in ("mp4", "mov", "avi") and cv2 is not None:
                try:
                    frames = process_video(p, max_frames=max_video_frames, frame_stride=video_stride)
                    data[p] = {
                        "type": "video",
                        "num_frames_sampled": int(frames.shape[0]) if frames.size else 0,
                        "shape": list(frames.shape),
                        "video_frames": np.asarray(frames, dtype=np.uint8),
                    }
                except Exception as e:
                    data[p] = {"error": str(e)}

    return data


def run_pipeline(
    base=DEFAULT_BASE,
    out_file=DEFAULT_OUT_FILE,
    save=True,
    max_video_frames=64,
    video_stride=8,
):
    """Run pipeline and save one consolidated dataset .npy file."""
    dataset = process_folder(
        base,
        max_video_frames=max_video_frames,
        video_stride=video_stride,
    )
    if save:
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        np.save(out_file, dataset, allow_pickle=True)
    return dataset


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="KmiDi multimodal dataset pipeline")
    p.add_argument("--base", default=DEFAULT_BASE, help="Input folder")
    p.add_argument("--out-file", default=DEFAULT_OUT_FILE, help="Single output .npy dataset file")
    p.add_argument("--no-save", action="store_true", help="Do not save output file")
    p.add_argument("--max-video-frames", type=int, default=64, help="Maximum frames sampled per video")
    p.add_argument("--video-stride", type=int, default=8, help="Sample every Nth video frame")
    args = p.parse_args()
    dataset = run_pipeline(
        args.base,
        args.out_file,
        save=not args.no_save,
        max_video_frames=args.max_video_frames,
        video_stride=args.video_stride,
    )
    if args.no_save:
        print("Processed", len(dataset), "items. Save disabled.")
    else:
        print("Processed", len(dataset), "items. Saved to", args.out_file)
