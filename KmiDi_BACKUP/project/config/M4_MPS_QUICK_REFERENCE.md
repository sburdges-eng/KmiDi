# M4/MPS Training Configuration Quick Reference
# =============================================================================
# Quick guide for selecting and using optimized training configurations
# =============================================================================

## 🚀 Quick Start

### For M4 Mac Training (Recommended)

```bash
# Production training (best performance)
python train.py --config config/m4_optimized_production.yaml

# Quick experiments
python train.py --config config/laptop_m4_small.yaml

# 16GB Mac optimized
python train.py --config config/macOS_16gb_optimized.yaml
```

### For Cloud Training

```bash
# Cloud GPU training (AWS/GCP/Azure/Lambda/RunPod)
python train.py --config config/cloud_training_optimized.yaml
```

## 📋 Configuration Comparison

| Config File                              | Use Case    | Batch Size | Memory  | Best For         |
| ---------------------------------------- | ----------- | ---------- | ------- | ---------------- |
| `m4_optimized_production.yaml`           | Production  | 8-16       | 24-48GB | Best performance |
| `laptop_m4_small.yaml`                   | Quick tests | 4-8        | 16GB+   | Fast iteration   |
| `macOS_16gb_optimized.yaml`              | Standard    | 8          | 16GB    | Balanced setup   |
| `midi_generator_training_mps_16gpu.yaml` | MIDI models | 8          | 16GB+   | Transformers     |
| `cloud_training_optimized.yaml`          | Cloud GPU   | 32+        | N/A     | Large-scale      |

## ⚙️ Key Settings (MPS)

All M4/MPS configs now include:

```yaml
device: mps
precision: fp16
num_workers: 0          # CRITICAL: Always 0 for MPS
pin_memory: false       # Not beneficial on MPS
persistent_workers: false
amp: true
amp_dtype: float16      # Must be float16 for MPS
```

## 🔧 Common Adjustments

### Increase Batch Size (if memory allows)
```yaml
dataloader:
  batch_size: 16  # Increase from 8
```

### Decrease Batch Size (if OOM)
```yaml
dataloader:
  batch_size: 4   # Decrease from 8
training:
  grad_accum_steps: 8  # Increase to maintain effective batch
```

### Adjust Memory Usage
```yaml
memory:
  max_memory_percent: 80  # Lower if system is slow
```

## 📊 Performance Expectations

### M4 Mac (24GB unified memory)
- **Batch size 8:** ~2-4 samples/sec
- **Batch size 16:** ~4-8 samples/sec
- **Effective batch 32:** Via grad_accum_steps=4

### Cloud GPU (A100)
- **Batch size 32:** ~20-40 samples/sec
- **Multi-GPU:** Linear scaling

## 🐛 Troubleshooting

### Issue: "MPS backend not available"
**Solution:** Check macOS version (12.3+) and PyTorch version (1.12+)

### Issue: "Out of memory"
**Solution:** 
- Reduce batch_size
- Increase grad_accum_steps
- Reduce segment_seconds
- Enable gradient_checkpointing

### Issue: "Slow training"
**Solution:**
- Verify num_workers=0
- Check GPU utilization in Activity Monitor
- Enable cache_mels: true
- Pre-compute spectrograms

## 📚 Full Documentation

See `M4_MPS_TRAINING_OPTIMIZATION_SUMMARY.md` for complete details.
