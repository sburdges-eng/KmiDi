#!/usr/bin/env bash
# After git clone + LFS pull of CREMA-D finishes in data/_cremad_clone/repo,
# run this from the experiment root (experiments/exp_002_wavjepa_emotion) to copy
# real WAVs into data/raw/emotions/cremad and remove the clone.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA="$EXP_DIR/data"
CLONE="$DATA/_cremad_clone/repo"
DEST="$DATA/raw/emotions/cremad/CREMA-D-master"

if [ ! -d "$CLONE/AudioWAV" ]; then
  echo "Clone not ready. Ensure you ran: cd $DATA/_cremad_clone && git clone https://github.com/CheyneyComputerScience/CREMA-D.git repo && cd repo && git lfs pull"
  exit 1
fi

# Check first file is real WAV
FIRST=$(find "$CLONE/AudioWAV" -name "*.wav" | head -1)
if [ -z "$FIRST" ]; then
  echo "No WAVs in $CLONE/AudioWAV. Run: cd $CLONE && git lfs pull"
  exit 1
fi
if ! head -c 4 "$FIRST" | grep -q RIFF; then
  echo "Files still LFS pointers. Run: cd $CLONE && git lfs pull"
  exit 1
fi

mkdir -p "$DEST"
echo "Copying WAVs to $DEST/AudioWAV ..."
cp -R "$CLONE/AudioWAV" "$DEST/"
echo "Removing clone directory..."
rm -rf "$DATA/_cremad_clone"
echo "Done. CREMA-D real WAVs are in $DEST/AudioWAV"
echo "Run: python run.py --encoder hubert_base --limit 20  # to test"
