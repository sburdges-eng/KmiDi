"""
Download and categorize music datasets into Datasets/ for JEPA training.
Run from repo root: PYTHONPATH=. python scripts/download_datasets.py [download|list-sources] [options]
Falls back to synthetic samples if direct downloads fail. Use categorize_datasets.py to organize by category.
"""

import os
import sys
import argparse
import ssl
import subprocess
import urllib.request
import tarfile
import zipfile
import shutil

# Prefer verified SSL: use certifi's CA bundle when available (run scripts/install_certs.py).
def _ssl_verified_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

# Fallback when system/certifi verification fails (e.g. macOS Python without certs).
_SSL_UNVERIFIED = ssl.create_default_context()
_SSL_UNVERIFIED.check_hostname = False
_SSL_UNVERIFIED.verify_mode = ssl.CERT_NONE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.storage_paths import DATASETS_DIR, ensure_storage_dirs

DEFAULT_OUT = DATASETS_DIR

# Categories (match docs/MUSIC_DATASETS.md)
CATEGORIES = ["audio", "midi", "chord", "rhythm", "emotion", "multimodal", "voice"]

# Registry: name -> {category, url (None = manual), format: "tar.gz"|"zip"|"files", description}
# url = None means document-only; script will skip or print manual instructions.
SOURCES = [
    {"name": "samples_audio", "category": "audio", "url": None, "format": "files", "description": "Small WAV samples (Wikimedia fallback)"},
    {"name": "samples_midi", "category": "midi", "url": None, "format": "files", "description": "Small MIDI samples (Wikimedia fallback)"},
    {"name": "GTZAN", "category": "audio", "url": "http://opihi.cs.uvic.ca/sound/genres.tar.gz", "format": "tar.gz", "description": "1k 30s clips, 10 genres (academic)"},
    {"name": "Groove_MIDI", "category": "rhythm", "url": None, "format": "manual", "description": "Groove/E-GMD: Magenta GitHub releases"},
    {"name": "MAESTRO", "category": "midi", "url": None, "format": "manual", "description": "Audio+MIDI piano: Google form / Magenta"},
    {"name": "FMA_small", "category": "audio", "url": None, "format": "manual", "description": "FMA small: Library of Congress / GitHub mdeff/fma"},
    {"name": "DEAM", "category": "emotion", "url": None, "format": "manual", "description": "Emotion valence/arousal: research request"},
    {"name": "MagnaTagATune", "category": "audio", "url": None, "format": "manual", "description": "26k clips + tags: MIR / academic"},
    {"name": "NSynth", "category": "audio", "url": None, "format": "manual", "description": "Single notes: Magenta/Google"},
    {"name": "GTZAN_kaggle", "category": "audio", "url": None, "format": "kaggle", "kaggle_slug": "andradaolteanu/gtzan-dataset-music-genre-classification", "description": "GTZAN via Kaggle (1k 30s clips, 10 genres)"},
    {"name": "MIDI_kaggle", "category": "midi", "url": None, "format": "kaggle", "kaggle_slug": "soumikrakshit/classical-music-midi", "description": "Classical music MIDI via Kaggle"},
    {"name": "MTG_Jamendo", "category": "audio", "url": None, "format": "huggingface", "hf_dataset": "rkstgr/mtg-jamendo", "max_examples": 2000, "description": "55k full tracks, mood/instrument tags (CC), subset via HF"},
]

# Small public-domain files (downloaded as "samples_audio" / "samples_midi" when --source not set)
AUDIO_URLS = [
    ("Bicycle_bell.wav", "https://upload.wikimedia.org/wikipedia/commons/6/68/Bicycle-bell-1.wav"),
    ("Chopin_prelude_opening.wav", "https://upload.wikimedia.org/wikipedia/commons/f/fa/Chopin_Prelude_No._15%2C_opening_01.wav"),
]
MIDI_URLS = [
    ("MIDI_sample.mid", "https://upload.wikimedia.org/wikipedia/commons/5/55/MIDI_sample.mid"),
]


def download_file(url, path, timeout=60):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KmiDi/1.0"})
        try:
            ctx = _ssl_verified_context()
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                data = resp.read()
        except (ssl.SSLCertVerificationError, OSError) as e:
            if "certificate" in str(e).lower() or "ssl" in str(e).lower():
                with urllib.request.urlopen(req, timeout=timeout, context=_SSL_UNVERIFIED) as resp:
                    data = resp.read()
            else:
                raise
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print("Download failed {}: {}".format(url, e), file=sys.stderr)
        return False


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def ensure_category_dirs(base):
    for cat in CATEGORIES:
        ensure_dir(os.path.join(base, cat))


def extract_archive(archive_path, out_dir, fmt):
    try:
        if fmt == "tar.gz":
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(out_dir)
        elif fmt == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(out_dir)
        else:
            return False
        return True
    except Exception as e:
        print("Extract failed {}: {}".format(archive_path, e), file=sys.stderr)
        return False


def generate_fallback_audio(path, sr=22050, duration_sec=2.0):
    try:
        import wave
        import math
        import struct
        n = int(sr * duration_sec)
        freq = 440.0
        samples = [int(32767 * 0.3 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)]
        buf = struct.pack("<" + "h" * n, *[max(-32768, min(32767, s)) for s in samples])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(buf)
        return True
    except Exception as e:
        print("Fallback audio failed:", e, file=sys.stderr)
        return False


def generate_fallback_midi(path):
    try:
        header = bytes([0x4D, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x01, 0x00, 0x60])
        track_data = bytes([0x00, 0x90, 0x3C, 0x40, 0x81, 0x00, 0x80, 0x3C, 0x00, 0x00, 0xFF, 0x2F, 0x00])
        track_header = bytes([0x4D, 0x54, 0x72, 0x6B, 0x00, 0x00, 0x00, 0x0B]) + track_data
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(header)
            f.write(track_header)
        return True
    except Exception as e:
        print("Fallback MIDI failed:", e, file=sys.stderr)
        return False


def download_kaggle(slug, out_dir, dry_run=False, force=False):
    """Download a Kaggle dataset via Python API (avoids CLI SIGSEGV on some macOS). Requires: pip install kaggle, ~/.kaggle/kaggle.json."""
    if dry_run:
        print("[dry-run] Would download Kaggle", slug, "to", out_dir)
        return True
    if force and os.path.isdir(out_dir):
        for f in os.listdir(out_dir):
            p = os.path.join(out_dir, f)
            if f.endswith(".zip") or os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
            else:
                try:
                    import shutil
                    shutil.rmtree(p)
                except OSError:
                    pass
    ensure_dir(out_dir)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset=slug, path=out_dir, unzip=True, quiet=False)
        print("Downloaded Kaggle", slug, "to", out_dir)
        return True
    except ImportError:
        pass
    except Exception as e:
        print("Kaggle API failed:", e, file=sys.stderr)
        return False
    # Fallback: CLI (can SIGSEGV on some macOS/Python; prefer API above)
    try:
        subprocess.check_call(
            ["kaggle", "datasets", "download", "-d", slug, "-p", out_dir],
            timeout=600,
        )
        for f in os.listdir(out_dir):
            if f.endswith(".zip"):
                extract_archive(os.path.join(out_dir, f), out_dir, "zip")
                break
        print("Downloaded Kaggle", slug, "to", out_dir)
        return True
    except subprocess.CalledProcessError as e:
        print("Kaggle download failed:", e, file=sys.stderr)
        print("Install: pip install kaggle; add ~/.kaggle/kaggle.json (API key from kaggle.com/account)", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Kaggle not found. Run: pip install kaggle", file=sys.stderr)
        return False


def download_huggingface_music(hf_dataset, out_dir, max_examples=2000, dry_run=False):
    """Download a music dataset from Hugging Face (e.g. MTG Jamendo). Requires: pip install datasets soundfile."""
    if dry_run:
        print("[dry-run] Would load HF", hf_dataset, "max", max_examples, "->", out_dir)
        return True
    ensure_dir(out_dir)
    try:
        from datasets import load_dataset
        ds = load_dataset(hf_dataset, split="train", trust_remote_code=True)
        n = min(max_examples, len(ds))
        saved = 0
        for i in range(n):
            row = ds[i]
            audio = row.get("audio")
            if audio is None:
                continue
            path = os.path.join(out_dir, "track_{:06d}.wav".format(i))
            if isinstance(audio, dict):
                arr = audio.get("array")
                sr = audio.get("sampling_rate", 22050)
                if arr is not None:
                    try:
                        import soundfile as sf
                        sf.write(path, arr, sr)
                        saved += 1
                    except ImportError:
                        import scipy.io.wavfile
                        scipy.io.wavfile.write(path, sr, (arr * 32767).astype("int16"))
                        saved += 1
            if (i + 1) % 200 == 0:
                print("MTG Jamendo: saved", saved, "/", i + 1, flush=True)
        print("Downloaded HF", hf_dataset, "->", out_dir, "(", saved, "tracks)")
        return True
    except Exception as e:
        print("Hugging Face download failed:", e, file=sys.stderr)
        print("Install: pip install datasets soundfile (or scipy)", file=sys.stderr)
        return False


def download_source(name, base, dry_run=False, force=False):
    """Download a single named source into base/category/name/."""
    rec = next((s for s in SOURCES if s["name"] == name), None)
    if not rec:
        print("Unknown source:", name, file=sys.stderr)
        return False
    if rec.get("format") == "kaggle":
        cat = rec["category"]
        out_dir = os.path.join(base, cat, name)
        return download_kaggle(rec["kaggle_slug"], out_dir, dry_run=dry_run, force=force)
    if rec.get("format") == "huggingface":
        cat = rec["category"]
        out_dir = os.path.join(base, cat, name)
        return download_huggingface_music(
            rec["hf_dataset"], out_dir,
            max_examples=rec.get("max_examples", 2000),
            dry_run=dry_run,
        )
    if rec["url"] is None:
        if rec["format"] == "manual":
            print("Manual source '{}': {}".format(name, rec["description"]))
            return True
        if name == "samples_audio":
            cat_dir = os.path.join(base, "audio")
            ensure_dir(cat_dir)
            if dry_run:
                print("[dry-run] Would download samples to", cat_dir)
                return True
            for fname, url in AUDIO_URLS:
                path = os.path.join(cat_dir, fname)
                if os.path.isfile(path):
                    print("Exists:", path)
                    continue
                if download_file(url, path):
                    print("Downloaded:", path)
                else:
                    fb = os.path.join(cat_dir, "fallback_{}.wav".format(fname.replace(".wav", "")))
                    if generate_fallback_audio(fb):
                        print("Fallback:", fb)
            if not any(f.endswith((".wav", ".mp3", ".flac")) for f in os.listdir(cat_dir)):
                generate_fallback_audio(os.path.join(cat_dir, "fallback_tone.wav"))
            return True
        if name == "samples_midi":
            cat_dir = os.path.join(base, "midi")
            ensure_dir(cat_dir)
            if dry_run:
                print("[dry-run] Would download MIDI samples to", cat_dir)
                return True
            for fname, url in MIDI_URLS:
                path = os.path.join(cat_dir, fname)
                if os.path.isfile(path):
                    print("Exists:", path)
                    continue
                if download_file(url, path):
                    print("Downloaded:", path)
                else:
                    fb = os.path.join(cat_dir, "fallback_{}.mid".format(fname.replace(".mid", "")))
                    if generate_fallback_midi(fb):
                        print("Fallback:", fb)
            if not any(f.lower().endswith((".mid", ".midi")) for f in os.listdir(cat_dir)):
                generate_fallback_midi(os.path.join(cat_dir, "fallback_note.mid"))
            return True
        return True
    # Direct URL
    cat = rec["category"]
    out_dir = os.path.join(base, cat, name)
    ensure_dir(out_dir)
    fmt = rec.get("format", "tar.gz")
    archive_name = os.path.basename(rec["url"].split("?")[0])
    archive_path = os.path.join(out_dir, archive_name)
    if dry_run:
        print("[dry-run] Would download", rec["url"], "->", archive_path)
        return True
    if not download_file(rec["url"], archive_path, timeout=300):
        return False
    if fmt in ("tar.gz", "zip"):
        if extract_archive(archive_path, out_dir, fmt):
            print("Extracted to", out_dir)
        # Keep archive by default
    return True


def cmd_list_sources(args):
    ensure_category_dirs(args.out)
    print("Category dirs:", os.path.join(args.out, "<category>"))
    print("\nRegistered sources (download with --source NAME):\n")
    print("{:<20} {:<12} {:<8} {}".format("NAME", "CATEGORY", "AUTO", "DESCRIPTION"))
    print("-" * 70)
    for s in SOURCES:
        auto = "yes" if s.get("url") or s.get("format") in ("kaggle", "huggingface") else "manual"
        print("{:<20} {:<12} {:<8} {}".format(s["name"], s["category"], auto, (s["description"] or "")[:45]))
    print("\nCategories:", ", ".join(CATEGORIES))
    print("Run: python scripts/download_datasets.py download [--source NAME] [--category CAT] [--out Dir]")


def cmd_download(args):
    force = getattr(args, "force", False)
    ensure_category_dirs(args.out)
    if getattr(args, "download_all", False):
        for s in SOURCES:
            download_source(s["name"], args.out, dry_run=args.dry_run, force=force)
        return
    if args.source:
        for name in args.source:
            if name.lower() == "kaggle":
                name = "GTZAN_kaggle"
            download_source(name, args.out, dry_run=args.dry_run, force=force)
        return
    if args.category:
        to_download = [s["name"] for s in SOURCES if s["category"] == args.category and s.get("url") is not None]
        if not to_download:
            to_download = [s["name"] for s in SOURCES if s["category"] == args.category]
        for name in to_download:
            download_source(name, args.out, dry_run=args.dry_run, force=force)
        if not to_download and args.category in CATEGORIES:
            if args.category == "audio":
                download_source("samples_audio", args.out, dry_run=args.dry_run, force=force)
            elif args.category == "midi":
                download_source("samples_midi", args.out, dry_run=args.dry_run, force=force)
        return
    download_source("samples_audio", args.out, dry_run=args.dry_run, force=force)
    download_source("samples_midi", args.out, dry_run=args.dry_run, force=force)


def main():
    ensure_storage_dirs()

    p = argparse.ArgumentParser(description="Download and categorize music datasets (JEPA-ready)")
    p.add_argument("--out", default=DEFAULT_OUT, help="Base directory (default: external drive datasets)")
    sub = p.add_subparsers(dest="command", help="Command")
    # download
    dl = sub.add_parser("download", help="Download datasets")
    dl.add_argument("--source", action="append", metavar="NAME", help="Download named source (e.g. GTZAN, samples_audio)")
    dl.add_argument("--all", action="store_true", dest="download_all", help="Download every registered source (auto + manual)")
    dl.add_argument("--category", choices=CATEGORIES, help="Download all auto sources in category")
    dl.add_argument("--fallback-only", action="store_true", help="Only write fallback synthetic (no network)")
    dl.add_argument("--dry-run", action="store_true", help="Print what would be done")
    dl.add_argument("--force", action="store_true", help="Re-download even if local copy exists (Kaggle: clear target dir first)")
    dl.set_defaults(func=cmd_download)
    # list-sources
    ls = sub.add_parser("list-sources", help="List registered sources and categories")
    ls.set_defaults(func=cmd_list_sources)
    args = p.parse_args()
    if args.command == "list-sources":
        cmd_list_sources(args)
        return
    if args.command == "download" or args.command is None:
        fallback = getattr(args, "fallback_only", False)
        if fallback:
            ensure_category_dirs(args.out)
            ad = os.path.join(args.out, "audio")
            md = os.path.join(args.out, "midi")
            ensure_dir(ad)
            ensure_dir(md)
            if generate_fallback_audio(os.path.join(ad, "fallback_tone.wav")):
                print("Wrote fallback audio:", ad)
            if generate_fallback_midi(os.path.join(md, "fallback_note.mid")):
                print("Wrote fallback MIDI:", md)
        else:
            cmd_download(args)
        return
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
