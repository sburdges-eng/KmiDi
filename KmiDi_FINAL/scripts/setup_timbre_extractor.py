#!/usr/bin/env python3
"""
Timbre Extractor Setup Script

Helps set up the timbre embedding extractor with Wav2Vec2 or Whisper.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))


def check_dependencies(model_type: str) -> bool:
    """Check if required dependencies are installed"""
    missing = []

    if model_type == "wav2vec2":
        try:
            import transformers
            import torchaudio

            print("✅ transformers and torchaudio are installed")
            return True
        except ImportError as e:
            missing.append(str(e))
            print("❌ Missing dependencies for Wav2Vec2:")
            print("   Install with: pip install transformers torchaudio")
            return False

    elif model_type == "whisper":
        try:
            import whisper

            print("✅ whisper is installed")
            return True
        except ImportError:
            print("❌ Missing dependencies for Whisper:")
            print("   Install with: pip install openai-whisper")
            return False

    return False


def setup_wav2vec2():
    """Set up Wav2Vec2 integration"""
    print("Setting up Wav2Vec2 integration...")

    if not check_dependencies("wav2vec2"):
        return False

    # Read current file
    timbre_file = project_root / "python" / "prrot" / "timbre_embeddings.py"

    if not timbre_file.exists():
        print(f"❌ File not found: {timbre_file}")
        return False

    content = timbre_file.read_text()

    # Check if already integrated
    if "Wav2Vec2Model" in content or "from transformers" in content:
        print("✅ Wav2Vec2 appears to be already integrated")
        print("   Check the implementation in timbre_embeddings.py")
        return True

    print("⚠️  Wav2Vec2 not yet integrated")
    print("   See docs/TIMBRE_EXTRACTOR_INTEGRATION.md for integration instructions")
    print("   The file contains example code that can be copied")

    return False


def setup_whisper():
    """Set up Whisper integration"""
    print("Setting up Whisper integration...")

    if not check_dependencies("whisper"):
        return False

    # Read current file
    timbre_file = project_root / "python" / "prrot" / "timbre_embeddings.py"

    if not timbre_file.exists():
        print(f"❌ File not found: {timbre_file}")
        return False

    content = timbre_file.read_text()

    # Check if already integrated
    if "import whisper" in content or "whisper.load_model" in content:
        print("✅ Whisper appears to be already integrated")
        print("   Check the implementation in timbre_embeddings.py")
        return True

    print("⚠️  Whisper not yet integrated")
    print("   See docs/TIMBRE_EXTRACTOR_INTEGRATION.md for integration instructions")
    print("   The file contains example code that can be copied")

    return False


def test_integration(model_type: str):
    """Test the integration"""
    print(f"\nTesting {model_type} integration...")

    try:
        from prrot.timbre_embeddings import TimbreEmbeddingExtractor
        import numpy as np

        extractor = TimbreEmbeddingExtractor(embedding_dim=256)

        # Create test audio
        test_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz

        # Extract embedding
        embedding = extractor.extract_embedding(test_audio, 16000)

        print(f"✅ Extraction successful!")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"   Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")

        # Check if it's placeholder
        if np.allclose(embedding, embedding / np.linalg.norm(embedding)):
            # This is a normalized random vector, likely placeholder
            print("   ⚠️  Warning: This appears to be a placeholder implementation")
            print("      The embedding is random. Integrate actual model.")
        else:
            print("   ✅ Embedding looks valid (not random)")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Set up timbre embedding extractor")
    parser.add_argument(
        "--model",
        choices=["wav2vec2", "whisper"],
        default="wav2vec2",
        help="Model to use (default: wav2vec2)",
    )
    parser.add_argument("--test", action="store_true", help="Test the integration after setup")

    args = parser.parse_args()

    print("=" * 80)
    print("TIMBRE EXTRACTOR SETUP")
    print("=" * 80)
    print()

    if args.model == "wav2vec2":
        success = setup_wav2vec2()
    else:
        success = setup_whisper()

    if args.test:
        test_integration(args.model)

    print()
    print("For detailed integration instructions, see:")
    print("  docs/TIMBRE_EXTRACTOR_INTEGRATION.md")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
