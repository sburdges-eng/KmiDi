# Cloud Training Quick Reference
# =============================================================================
# Quick decision guide: What to train in cloud vs locally
# =============================================================================

## 🎯 Quick Decision Table

| Model                  | Train Where | Reason                               | Est. Time | Est. Cost |
| ---------------------- | ----------- | ------------------------------------ | --------- | --------- |
| **Emotion Recognizer** | ✅ LOCAL     | Fast (12 min), good performance      | 12 min    | $0        |
| **Dynamics Engine**    | ✅ LOCAL     | Very fast (40 sec), tiny model       | 40 sec    | $0        |
| **Groove Predictor**   | ✅ LOCAL     | Fast (2 min), already perfect        | 2 min     | $0        |
| **Spectocloud ViT**    | ⚠️ LOCAL*    | Moderate size, can train locally     | 30-60 min | $0        |
| **Harmony Predictor**  | ☁️ CLOUD     | Needs improvement, long training     | 15-30 min | $2-5      |
| **Melody Transformer** | ☁️ CLOUD     | Poor performance, very long training | 30-60 min | $5-10     |
| **MIDI Generator**     | ☁️ CLOUD     | Long training, sequence generation   | 2-4 hours | $3-8      |

*Spectocloud ViT: Start locally, move to cloud if training takes >2 hours

## 📊 Priority Ranking

### 🔴 HIGH PRIORITY - Train in Cloud
1. **Melody Transformer** 
   - Current: 34.5% accuracy (POOR)
   - Needs: 200 epochs, ~4-6 hours on M4
   - Cloud: ~30-60 min, $5-10
   - **Time saved: 3.5-5.5 hours**

### 🟡 MEDIUM PRIORITY - Train in Cloud
2. **Harmony Predictor**
   - Current: 54% accuracy (needs improvement)
   - Needs: 150 epochs, ~1-2 hours on M4
   - Cloud: ~15-30 min, $2-5
   - **Time saved: 1-1.5 hours**

3. **MIDI Generator Transformer**
   - Current: New model, needs training
   - Needs: 15-30 epochs, ~8-16 hours on M4
   - Cloud: ~2-4 hours, $3-8
   - **Time saved: 6-12 hours**

### 🟢 LOW PRIORITY - Train Locally
4. **Emotion Recognizer** - Already good (89.5%), fast training
5. **Dynamics Engine** - Very fast, small model
6. **Groove Predictor** - Already perfect (100%), fast training

## 💰 Cost Summary

### Total Cloud Training Cost: **~$10-25**
- Melody Transformer: $5-10
- Harmony Predictor: $2-5
- MIDI Generator: $3-8

### Total Time Saved: **~10-18 hours**
- Melody: 3.5-5.5 hours saved
- Harmony: 1-1.5 hours saved
- MIDI Generator: 6-12 hours saved

### ROI: **Excellent**
- Your time is worth more than $1-2/hour
- Faster iteration = better models
- Can experiment more with cloud speed

## 🚀 Recommended Cloud Providers

### Best Value (Cost-Effective)
- **RunPod** - RTX 4090 @ $0.50/hour
  - Good for: All cloud models
  - Best value for money

### Balanced (Performance/Cost)
- **Lambda Labs** - A100 @ $1.10/hour
  - Good for: Large transformers
  - Professional GPUs

### Enterprise (Multi-GPU)
- **AWS** - g5.4xlarge (A10G x2) @ $1.00/hour spot
  - Good for: Distributed training
  - Multi-GPU support

## ⚙️ Quick Setup

### 1. Use Cloud Config
```bash
python train.py --config config/cloud_training_optimized.yaml
```

### 2. Set Budget Limits
```yaml
cost_optimization:
  max_budget: 50  # Hard limit: $50
  cost_alert_threshold: 25  # Alert at $25
```

### 3. Enable Spot Instances
```yaml
cloud:
  aws:
    use_spot: true  # 60-90% savings
    spot_max_price: 2.0
```

## 📝 Checklist

### Before Cloud Training:
- [ ] Choose cloud provider (RunPod/Lambda/AWS)
- [ ] Set up account and billing alerts
- [ ] Configure `cloud_training_optimized.yaml`
- [ ] Upload data to cloud storage
- [ ] Test with small run first
- [ ] Set budget limits

### During Cloud Training:
- [ ] Monitor costs regularly
- [ ] Check checkpoint saving works
- [ ] Verify spot instance handling
- [ ] Monitor training progress

### After Cloud Training:
- [ ] Download trained models
- [ ] Terminate instances (avoid idle costs)
- [ ] Export to CoreML for M4 inference
- [ ] Fine-tune locally if needed

## 🎯 Decision Rules

### Train Locally If:
- ✅ Training time < 30 minutes
- ✅ Model < 20M parameters
- ✅ Already performing well (>80%)
- ✅ Quick iteration needed

### Train in Cloud If:
- ☁️ Training time > 2 hours
- ☁️ Model > 50M parameters
- ☁️ Needs extensive experimentation
- ☁️ Long training runs (100+ epochs)
- ☁️ Poor current performance (needs work)

## 📚 Full Documentation

See `CLOUD_VS_LOCAL_TRAINING_GUIDE.md` for complete analysis and details.
