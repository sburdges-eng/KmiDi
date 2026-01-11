# iDAW Build Variants: Hardware Configuration Guide

**Purpose**: Single codebase, multiple hardware targets with optimized configurations

---

## Build Targets

```
┌─────────────────────────────────────────────────────────────┐
│                    iDAW Build Variants                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BUILD-DEV-MAC         BUILD-TRAIN-NVIDIA    BUILD-CLOUD    │
│  (Development)         (Research)            (Production)   │
│  ├─ M4 Pro            ├─ RTX 4060           ├─ AWS EC2      │
│  ├─ MPS acceleration  ├─ CUDA 12.1          ├─ GPU p3.2xl   │
│  ├─ 16GB RAM          ├─ 32GB RAM           ├─ 4× NVIDIA V100│
│  └─ Inference only    ├─ Training + Inference  └─ Scale: 1000s│
│                       └─ Fine-tuning                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Build 1: Development (M4 Pro Mac)

### Target Hardware
- **Device**: MacBook Pro 14"/16" M4 Pro/Max
- **RAM**: 16GB-32GB unified memory
- **Storage**: 512GB+ SSD
- **Network**: Internet required (data download)

### Purpose
- Local development + testing
- Inference only (no training)
- Rapid iteration + debugging
- Demo + showcase

### Build Command

```bash
# Clone repo
git clone https://github.com/yourusername/miDiKompanion.git
cd miDiKompanion

# Create environment
conda create -n idaw-dev python=3.11
conda activate idaw-dev

# Install (development mode)
pip install -e .
pip install -e ".[dev]"  # Includes pytest, black, mypy

# Install ML deps for MPS
conda install pytorch::pytorch torchvision torchaudio -c pytorch
pip install peft transformers librosa pyttsx3 peft

# Verify MPS
python -c "import torch; print(torch.backends.mps.is_available())"  # Should be True
```

### Configuration File: `config/build-dev-mac.yaml`

```yaml
build: dev-mac
device: mps
model_checkpoint_dir: /Volumes/Extreme\ SSD/kelly-project/miDiKompanion/ml_training/models/trained/checkpoints
max_batch_size: 16  # Adjusted for 16GB RAM
inference_only: true
enable_compile: true  # torch.compile for MPS
quantize: false  # Not supported on MPS
logging_level: DEBUG
demo_mode: true

cache:
  enable: true
  location: ~/.idaw_cache
  max_size_gb: 2

performance:
  target_latency_ms: 150  # Accept slower on dev
  profile_enabled: true
  memory_monitor: true
```

### Development Workflow

```bash
# 1. Run quick test
python scripts/quickstart_tier1.py

# 2. Run Streamlit demo
streamlit run app/streamlit_app.py

# 3. Run tests
pytest tests/ -v

# 4. Format code
black music_brain/ tests/

# 5. Type checking
mypy music_brain/

# 6. Profile inference
python scripts/profile_inference.py --model melody_transformer --device mps
```

### What Works
✅ All Tier 1 inference
✅ Streamlit demo
✅ API testing (non-load)
✅ Code iteration
✅ Debugging

### What Doesn't Work
❌ Tier 2 fine-tuning (training too slow on MPS)
❌ Large-scale data processing
❌ Load testing (single client only)
❌ GPU benchmarking

---

## Build 2: Training (RTX 4060 Workstation)

### Target Hardware
- **Device**: Desktop/Workstation (Ubuntu 22.04 or Windows 11 WSL2)
- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **CPU**: Ryzen 5 5600X or equivalent
- **RAM**: 32GB DDR4
- **Storage**: 1TB+ NVMe (fast I/O for data)
- **Network**: Internet required (cloud backup)

### Purpose
- Model fine-tuning (Tier 2)
- Training on custom therapy data
- Full validation loop
- Production model training

### Bill of Materials (BOM)

```
Component            | Model              | Price | Vendor
─────────────────────┼────────────────────┼───────┼──────────────
GPU                  | RTX 4060           | $250  | Amazon/Newegg
CPU                  | Ryzen 5 5600X      | $120  | Amazon/Micro Center
RAM                  | 32GB DDR4 G.Skill | $100  | Amazon
Storage              | 1TB WD Blue NVMe   | $60   | Amazon
Power Supply         | 550W Corsair       | $50   | Amazon
Case                 | NZXT H510 Flow     | $80   | Amazon
Cooling              | Arctic Freezer 34  | $30   | Amazon
───────────────────────────────────────────────────────────
TOTAL                |                    | $690  |
```

### Build Command

```bash
# 1. Install OS (Ubuntu 22.04 recommended)
# Download: https://ubuntu.com/download/desktop

# 2. Install NVIDIA drivers
sudo apt install nvidia-driver-545

# 3. Install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo bash cuda_12.1.0_530.30.02_linux.run

# 4. Install cuDNN 8.9
# Download from: https://developer.nvidia.com/cudnn
tar -xvf cudnn-linux-x86_64-8.9.0.131.tar.xz
sudo cp -r cudnn-*/include/* /usr/local/cuda/include/
sudo cp -r cudnn-*/lib/* /usr/local/cuda/lib64/

# 5. Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 6. Create environment
conda create -n idaw-train python=3.11
conda activate idaw-train

# 7. Install with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 8. Install iDAW
git clone https://github.com/yourusername/miDiKompanion.git
cd miDiKompanion
pip install -e .
pip install peft transformers librosa tensorboard wandb

# 9. Verify CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Configuration File: `config/build-train-nvidia.yaml`

```yaml
build: train-nvidia
device: cuda
cuda_device: 0
model_checkpoint_dir: /home/user/models/idaw/checkpoints
max_batch_size: 64  # Full capacity for 8GB VRAM
inference_only: false
enable_compile: true
quantize: true  # INT8 for inference optimization
logging_level: INFO
demo_mode: false

training:
  fp16: true  # Mixed precision
  gradient_checkpointing: true
  max_grad_norm: 1.0
  learning_rate: 5e-5
  num_epochs: 20
  batch_size: 16  # With gradient accumulation
  accumulation_steps: 4  # Simulate batch_size=64

data:
  dataset_path: /mnt/data/therapy_midi
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

checkpoint:
  save_every_n_epochs: 2
  keep_last_n: 3
  backup_to_s3: true

monitoring:
  tensorboard: true
  wandb: true
  profile_gpu: true
  memory_tracking: true
```

### Training Workflow

```bash
# 1. Prepare data
python scripts/prepare_therapy_dataset.py \
  --midi-dir /path/to/midi \
  --emotion-dir /path/to/emotions \
  --output /mnt/data/therapy_midi

# 2. Train Tier 2 LoRA
python scripts/train_tier2_lora.py \
  --config config/build-train-nvidia.yaml \
  --midi-dir /mnt/data/therapy_midi/train \
  --emotion-dir /mnt/data/therapy_midi/emotions \
  --epochs 20 \
  --batch-size 16

# 3. Validate
python scripts/validate_trained_model.py \
  --checkpoint ./checkpoints/tier2_lora/final \
  --test-data /mnt/data/therapy_midi/test

# 4. Merge & export
python scripts/merge_and_export.py \
  --lora-checkpoint ./checkpoints/tier2_lora/final \
  --output-path ./models/melody_transformer_therapy.pt

# 5. Upload to S3 (backup)
aws s3 cp ./models/melody_transformer_therapy.pt \
  s3://idaw-models/tier2/melody_transformer_therapy.pt
```

### Training Times (RTX 4060)

```
Model               | Dataset Size | Epochs | Batch Size | Time
────────────────────┼──────────────┼────────┼────────────┼──────
MelodyTransformer   | 100 MIDI     | 10     | 16         | 2-3 hrs
GroovePredictor     | 100 MIDI     | 10     | 16         | 30-45 min
HarmonyPredictor    | 100 MIDI     | 10     | 16         | 1-2 hrs
─────────────────────────────────────────────────────────────────
Full Pipeline       | 100 MIDI     | 10     | 16         | 3-5 hrs

With 1000 MIDI files:
MelodyTransformer   | 1000 MIDI    | 10     | 16         | 20-30 hrs
```

### What Works
✅ Tier 2 fine-tuning
✅ Full training loop
✅ Production models
✅ Data preprocessing
✅ Large datasets (1000+)
✅ GPU profiling
✅ Load testing

### What Doesn't Work
❌ Real-time audio processing (GPU not connected to audio)
❌ Inference servers (needs Tier 1 build instead)

---

## Build 3: Production (AWS EC2 + Kubernetes)

### Target Hardware
- **Instance**: AWS EC2 p3.2xlarge (8× NVIDIA V100 GPUs)
- **OS**: Amazon Linux 2 or Ubuntu 22.04
- **Storage**: 500GB+ EBS + S3 for models
- **Network**: Multi-region deployment (US, EU, Asia)
- **Orchestration**: Kubernetes or ECS

### Purpose
- Production inference serving (1000s concurrent users)
- Continuous model updating
- A/B testing (Tier 1 vs Tier 2)
- Analytics + monitoring

### Infrastructure as Code

```yaml
# terraform/main.tf
resource "aws_instance" "idaw_production" {
  ami           = "ami-0c55b159cbfafe1f0"  # Ubuntu 22.04
  instance_type = "p3.2xlarge"

  root_block_device {
    volume_size = 500
    volume_type = "gp3"
  }

  tags = {
    Name = "idaw-production"
    Env  = "prod"
  }
}

resource "aws_s3_bucket" "idaw_models" {
  bucket = "idaw-production-models"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_ecr_repository" "idaw_api" {
  name                 = "idaw-api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
```

### Docker Build

```dockerfile
# Dockerfile.prod
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git

# Install PyTorch with CUDA
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install iDAW
COPY . /app
RUN pip install -e .
RUN pip install fastapi uvicorn peft

# Expose API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Configuration File: `config/build-prod-aws.yaml`

```yaml
build: prod-aws
device: cuda
cuda_devices: [0, 1, 2, 3, 4, 5, 6, 7]  # All 8 V100s
model_checkpoint_dir: s3://idaw-models/checkpoints
max_batch_size: 512  # Huge batches for throughput
inference_only: true
enable_compile: false  # Already compiled
quantize: true  # INT8 for memory efficiency
logging_level: WARNING  # Less verbose

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  keep_alive: 300
  request_timeout: 60

inference:
  model_cache_size: 8  # Keep 8 models in memory
  batch_timeout_ms: 100  # Wait up to 100ms to batch requests
  max_queue_size: 1000
  priority_queue: true

monitoring:
  prometheus_metrics: true
  datadog_enabled: true
  cloudwatch_enabled: true
  sentry_enabled: true

deployment:
  region: us-east-1
  multi_az: true
  load_balancer: application
  auto_scaling:
    min_instances: 2
    max_instances: 20
    target_cpu: 70
```

### Production Deployment

```bash
# 1. Build Docker image
docker build -t 123456789.dkr.ecr.us-east-1.amazonaws.com/idaw-api:latest .
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/idaw-api:latest

# 2. Deploy with Terraform
cd terraform/
terraform init
terraform plan
terraform apply

# 3. Deploy to ECS/Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 4. Monitor
kubectl logs -f deployment/idaw-api
kubectl get pods
kubectl describe pod <pod-name>

# 5. Scale
kubectl scale deployment idaw-api --replicas=10
```

### Performance Targets (Production)

```
Metric                | Target      | Notes
──────────────────────┼─────────────┼──────────────────
API Latency           | <500ms p99  | MIDI + Audio gen
Throughput            | 100 req/sec | Per instance
Error Rate            | <0.1%       | 99.9% success
Uptime                | 99.95%      | 4-hour window max
Model Version Lag     | <24 hours   | Latest training
Cost per Request      | <$0.01      | Margin: 80%+
```

### Monitoring Stack

```
Prometheus (metrics collection)
  ↓
Grafana (dashboards)
  ├─ API latency
  ├─ GPU utilization
  ├─ Error rates
  └─ Cost tracking

CloudWatch (AWS monitoring)
  ├─ EC2 metrics
  ├─ S3 access logs
  └─ Lambda logs

Sentry (error tracking)
  └─ Exceptions + stack traces

DataDog (APM)
  ├─ Request tracing
  ├─ Dependency mapping
  └─ Performance insights
```

---

## Build Configuration Switching

### Automatic Detection

```python
# idaw/config.py
import platform
import os

def detect_build():
    """Auto-detect best build based on environment"""

    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            return "build-train-nvidia"
        elif torch.backends.mps.is_available():
            return "build-dev-mac"
    except ImportError:
        pass

    # Check for AWS
    if os.path.exists("/var/lib/cloud"):
        return "build-prod-aws"

    # Check for Mac
    if platform.system() == "Darwin":
        return "build-dev-mac"

    return "build-dev-linux"  # Default fallback

def load_config(build=None):
    if build is None:
        build = detect_build()

    with open(f"config/{build}.yaml") as f:
        return yaml.safe_load(f)
```

### Runtime Override

```bash
# Override detected build
export IDAW_BUILD=build-train-nvidia
python scripts/train_tier2_lora.py

# Or in code
from idaw.config import load_config
config = load_config(build="build-train-nvidia")
```

---

## Deployment Checklist

### Development Build (M4 Pro)
- [ ] Conda environment created
- [ ] PyTorch MPS verified
- [ ] `quickstart_tier1.py` runs successfully
- [ ] Streamlit demo accessible
- [ ] Config file: `config/build-dev-mac.yaml`

### Training Build (RTX 4060)
- [ ] Ubuntu 22.04 installed
- [ ] NVIDIA drivers + CUDA 12.1 installed
- [ ] cuDNN 8.9 installed
- [ ] PyTorch CUDA verified (`torch.cuda.is_available() == True`)
- [ ] Model training script runs
- [ ] Config file: `config/build-train-nvidia.yaml`
- [ ] Training times match expectations

### Production Build (AWS)
- [ ] Terraform variables configured
- [ ] Docker image builds successfully
- [ ] ECR repository created + pushed
- [ ] ECS/Kubernetes cluster ready
- [ ] Load balancer configured
- [ ] SSL/TLS certificates installed
- [ ] Monitoring stack deployed
- [ ] Config file: `config/build-prod-aws.yaml`
- [ ] Health checks passing

---

## Cost Comparison

| Build | Hardware Cost | Monthly Cost | Use Case |
|-------|---|---|---|
| **Dev-Mac** | $1,800 (laptop) | $0 | Development |
| **Train-NVIDIA** | $690 | $200-400 (power) | Fine-tuning |
| **Prod-AWS** | $0 (cloud) | $1,500-5,000 | Production (1000s users) |

---

**See**: `IMPLEMENTATION_PLAN.md` for phases when each build is active
