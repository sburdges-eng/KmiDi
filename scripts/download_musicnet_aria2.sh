#!/usr/bin/env bash
# Fast MusicNet download via aria2 (multi-connection). ~11 GB from Zenodo.
# Runs only on /Volumes/Sean's SSD: downloads go to VOLUME/Datasets/downloads.
# Run on your machine with Ethernet; requires DNS (e.g. not in a sandbox).
# Speed: uses 16 connections (aria2 max per server) and 1M min split for better parallelism.
# Optional env overrides: ARIA2_X (default 16), ARIA2_S (default 16), ARIA2_MIN_SPLIT (default 1M).
# Usage:
#   /path/to/download_musicnet_aria2.sh           # start or restart
#   /path/to/download_musicnet_aria2.sh -c        # resume partial (server supports range)
# If a download is already running: stop it (Ctrl+C), then run with -c to resume with faster settings.
set -e
VOLUME="/Volumes/Sean's SSD"
if [[ ! -d "$VOLUME" ]]; then
  echo "Drive not mounted: $VOLUME" >&2
  exit 1
fi
if [[ -f "$VOLUME/.env" ]]; then
  set -a
  source "$VOLUME/.env"
  set +a
fi
AUDIO_ROOT="$VOLUME/Datasets"
DOWNLOAD_DIR="$AUDIO_ROOT/downloads"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"
# Use /records/ (plural) so Zenodo doesn't redirect; redirects + multi-connection can trigger 403
URL="https://zenodo.org/records/5120004/files/musicnet.tar.gz"
MUSICNET_FILE="musicnet.tar.gz"
# Zenodo MusicNet tarball exact size (bytes); skip download if already complete
MUSICNET_EXPECTED_SIZE=11097394998
if [[ -f "$MUSICNET_FILE" ]]; then
  SIZE=$(stat -f %z "$MUSICNET_FILE" 2>/dev/null || stat -c %s "$MUSICNET_FILE" 2>/dev/null || echo 0)
  if [[ -n "$SIZE" && "$SIZE" -eq "$MUSICNET_EXPECTED_SIZE" ]]; then
    echo "Data root: $AUDIO_ROOT"
    echo "$MUSICNET_FILE already complete ($SIZE bytes). Nothing to do."
    exit 0
  fi
fi
if ! command -v aria2c &>/dev/null; then
  echo "Install aria2: brew install aria2" >&2
  exit 1
fi
CONTINUE=""
[[ "${1:-}" == "-c" ]] && CONTINUE="-c"
# Stock aria2 max is 16 connections per server; -k 1M improves parallelism on slow streams
ARIA2_X="${ARIA2_X:-16}"
ARIA2_S="${ARIA2_S:-16}"
ARIA2_MIN_SPLIT="${ARIA2_MIN_SPLIT:-1M}"
echo "Data root: $AUDIO_ROOT"
echo "Downloading $MUSICNET_FILE (~11 GB) to $DOWNLOAD_DIR"
echo "Using $ARIA2_X connections, min-split $ARIA2_MIN_SPLIT. ETA will show in progress."
# Zenodo returns 403 for aria2/curl unless we send a session cookie and Referer from a prior request
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
REFERER="https://zenodo.org/records/5120004"
# Get session cookie: try full redirect chain, then first response only (cookie often on 301)
COOKIE=$(curl -sI -L -A "$UA" -H "Referer: $REFERER/" --connect-timeout 10 --max-time 15 "$URL" 2>/dev/null | grep -i set-cookie | sed -n 's/.*[sS]ession=\([^;]*\).*/\1/p' | tr -d '\r' | head -1)
[[ -z "$COOKIE" ]] && COOKIE=$(curl -sI -A "$UA" -H "Referer: $REFERER/" --connect-timeout 10 --max-time 10 "https://zenodo.org/record/5120004/files/musicnet.tar.gz" 2>/dev/null | grep -i set-cookie | sed -n 's/.*[sS]ession=\([^;]*\).*/\1/p' | tr -d '\r' | head -1)
ARIA2_EXTRA=(--header="Referer: $REFERER/")
[[ -n "$COOKIE" ]] && ARIA2_EXTRA=(--header="Referer: $REFERER/" --header="Cookie: session=$COOKIE")
if aria2c $CONTINUE -x "$ARIA2_X" -s "$ARIA2_S" -k "$ARIA2_MIN_SPLIT" --user-agent="$UA" "${ARIA2_EXTRA[@]}" -o "$MUSICNET_FILE" "$URL"; then
  true
else
  echo "aria2 failed (often 403 from Zenodo). Falling back to curl (resumable, single connection)." >&2
  CURL_EXTRA=()
  [[ -n "$COOKIE" ]] && CURL_EXTRA=(-H "Cookie: session=$COOKIE")
  curl -f -L -C - -A "$UA" -H "Referer: $REFERER/" "${CURL_EXTRA[@]}" -o "$MUSICNET_FILE" "$URL"
fi
