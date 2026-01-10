# Cloud Storage Setup for Codespace Training

## Overview

This guide sets up Backblaze B2 cloud storage to make training datasets available in GitHub Codespaces. B2 offers a generous free tier (10GB storage, 1GB/day egress) and cheap rates beyond that ($0.005/GB/mo).

```
Local Mac                    Backblaze B2                GitHub Codespace
┌────────────┐   upload     ┌────────────┐   rclone     ┌────────────┐
│ /Volumes/  │ ──────────►  │ kmidi-     │ ◄──────────► │ /data/     │
│ sbdrive/   │              │ datasets   │    mount     │ datasets/  │
│ audio/     │              │ bucket     │              │            │
│ datasets/  │              └────────────┘              └────────────┘
└────────────┘
```

## Step 1: Create Backblaze B2 Account

1. Go to https://www.backblaze.com/b2/cloud-storage.html
2. Sign up for free account
3. Create a bucket named `kmidi-datasets` (or your choice)
4. Note: First 10GB storage is free

## Step 2: Generate Application Key

1. In B2 dashboard, go to **App Keys**
2. Click **Add a New Application Key**
3. Name: `kmidi-codespace`
4. Allow access to: Your bucket (`kmidi-datasets`)
5. Type: Read and Write
6. Save the **keyID** and **applicationKey** (shown only once!)

## Step 3: Configure Local Mac (rclone)

```bash
# Install rclone
brew install rclone

# Configure B2 remote
rclone config
# n) New remote
# name: b2
# Storage: 5 (Backblaze B2)
# account: <your keyID>
# key: <your applicationKey>
# Leave other options as default

# Test connection
rclone lsd b2:
```

## Step 4: Upload Datasets from Local Mac

```bash
cd "/Volumes/sbdrive/My Mac/Desktop/KmiDi-remote"

# Upload all datasets (~27GB total)
./scripts/upload_to_b2.sh

# Or upload specific dataset
./scripts/upload_to_b2.sh m4singer

# Check status
./scripts/upload_to_b2.sh --status
```

**Upload times (approximate):**
| Dataset | Size | Time (100Mbps) |
|---------|------|----------------|
| MoodyLyrics | 640KB | <1 sec |
| WASABI | 67MB | 5 sec |
| Genius | 649MB | 1 min |
| Lakh MIDI | 7.6GB | 10 min |
| M4Singer | 11GB | 15 min |

## Step 5: Add Codespace Secrets

1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Codespaces**
2. Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `B2_ACCOUNT_ID` | Your B2 keyID |
| `B2_APPLICATION_KEY` | Your B2 applicationKey |
| `B2_BUCKET_NAME` | `kmidi-datasets` (optional) |

## Step 6: Launch Codespace

When you create/rebuild your Codespace, the post-create script will:
1. Detect the B2 credentials
2. Install rclone
3. Mount the B2 bucket at `/data/datasets`

## Using Datasets in Codespace

```python
import os

# Datasets are mounted at /data/datasets
DATA_DIR = os.environ.get('KMIDI_DATA_DIR', '/data/datasets')

# Access datasets
m4singer_path = f"{DATA_DIR}/m4singer"
lakh_midi_path = f"{DATA_DIR}/lakh_midi"
```

Or in training scripts:
```bash
python scripts/train.py --model emotion_recognizer --data /data/datasets
```

## Costs

### Free Tier
- 10GB storage
- 1GB/day download (30GB/month)
- Unlimited upload

### Beyond Free Tier
- Storage: $0.005/GB/month ($0.135/mo for 27GB)
- Download: $0.01/GB
- Transactions: $0.004 per 10,000 Class B

**Estimated monthly cost for this project: ~$0.50-$2.00**

## Troubleshooting

### "B2 credentials not found"
Check that secrets are set correctly:
```bash
echo $B2_ACCOUNT_ID  # Should show your keyID
```

### "Mount failed"
Try manual mount:
```bash
rclone mount b2:kmidi-datasets /data/datasets --daemon
```

### Slow downloads
The rclone mount uses caching (10GB max). First access may be slow, but repeated access will be fast.

### Out of cache space
Clear the cache:
```bash
rm -rf /tmp/rclone-cache/*
```

## Alternative: Streaming from HuggingFace

If you don't want to set up B2, you can stream datasets directly from HuggingFace:

```python
from datasets import load_dataset

# Stream Lakh MIDI (no download needed)
ds = load_dataset("mimbres/lakh_full", streaming=True)
for sample in ds['train']:
    # Process sample
    pass
```

This is slower but requires no setup.
