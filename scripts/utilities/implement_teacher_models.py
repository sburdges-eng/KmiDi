#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path

# Define paths on the external SSD
ssd_root = Path("/Volumes/Sean's SSD/Datasets/prrot_models")
models_dir = ssd_root
cache_dir = ssd_root / ".hf_cache"

models_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

# FORCE Hugging Face to use the SSD for all logs and temporary files
# This bypasses the OS Error 17 in your home directory
env = os.environ.copy()
env["HF_HOME"] = str(cache_dir)
env["XDG_CACHE_HOME"] = str(cache_dir)

def download_model(repo, filename, target_dir):
    print(f"\n[DOWNLOAD] {repo} -> {filename}")
    cmd = [
        "huggingface-cli", "download", 
        repo, filename, 
        "--local-dir", str(target_dir),
        "--local-dir-use-symlinks", "False"
    ]
    # Run with the isolated environment
    subprocess.run(cmd, check=True, env=env)

try:
    # 1. Wav2Vec2
    wav2vec_dir = models_dir / "wav2vec2"
    download_model("facebook/wav2vec2-base", "pytorch_model.bin", wav2vec_dir)
    download_model("facebook/wav2vec2-base", "config.json", wav2vec_dir)

    # 2. Whisper
    whisper_dir = models_dir / "whisper"
    download_model("openai/whisper-base", "pytorch_model.bin", whisper_dir)
    download_model("openai/whisper-base", "config.json", whisper_dir)

    # 3. 3B Q4 Model
    download_model("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf", models_dir)
    
    # Finalize
    old_path = models_dir / "qwen2.5-3b-instruct-q4_k_m.gguf"
    new_path = models_dir / "phoneme_aligner_q4.bin"
    if old_path.exists():
        if new_path.exists(): os.remove(new_path)
        os.rename(old_path, new_path)
        print(f"\n✓ Finalized: {new_path}")

except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
