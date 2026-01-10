# Optional OpenL3 Audio Embeddings
# Generates audio embeddings using OpenL3.

import openl3
import soundfile as sf
import os


def generate_audio_embeddings(audio_path, output_path):
    """Generate OpenL3 embeddings for an audio file."""
    audio, sr = sf.read(audio_path)
    embeddings, timestamps = openl3.get_audio_embedding(
        audio, sr, content_type="music", input_repr="mel256")

    # Save embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        embeddings.tofile(f)
    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    audio_path = "example_audio.wav"
    output_path = "example_embeddings.npy"
    generate_audio_embeddings(audio_path, output_path)
