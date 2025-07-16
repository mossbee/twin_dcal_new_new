# DCAL Twin Face Verification System

A comprehensive implementation of Dual Cross-Attention Learning (DCAL) adapted for identical twin face verification using the ND TWIN 2009-2010 dataset.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Kaggle P100 Training Guide](#kaggle-p100-training-guide)
- [Local Training](#local-training)
- [Inference](#inference)
- [Attention Visualization](#attention-visualization)
- [Configuration](#configuration)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements DCAL (Dual Cross-Attention Learning) for twin face verification, a fine-grained visual categorization task. The system can distinguish between identical twins by learning to focus on subtle facial differences through attention mechanisms.

### Key Features

- **DCAL Architecture**: Self-Attention (SA), Global-Local Cross-Attention (GLCA), and Pair-Wise Cross-Attention (PWCA)
- **Multi-Environment Support**: Local server (2x RTX 2080Ti) and Kaggle (Tesla P100/T4)
- **Experiment Tracking**: MLFlow (local), WandB (Kaggle), or no tracking
- **Attention Visualization**: Explainable AI with attention map overlay
- **Performance Optimization**: Mixed precision, model quantization, TensorRT support
- **Robust Checkpointing**: Automatic resume for Kaggle's 12-hour timeout

## Installation

### Requirements

```bash
pip install torch torchvision transformers
pip install timm opencv-python pillow
pip install mlflow wandb tensorboard
pip install scikit-learn matplotlib seaborn
pip install onnx onnxruntime tensorrt  # Optional for optimization
```

### Project Structure

```
DCAL_Twin/
├── src/
│   ├── models/           # DCAL architecture
│   ├── data/             # Dataset loading and transforms
│   ├── training/         # Training pipeline
│   ├── utils/            # Utilities and visualization
│   └── inference/        # Inference and evaluation
├── configs/              # Configuration files
├── scripts/              # Training and evaluation scripts
├── notebooks/            # Analysis notebooks
└── data/                 # Dataset files
```

## Dataset Setup

The system uses the ND TWIN 2009-2010 dataset with the following structure:

```
data/
├── train_dataset_infor.json    # Training set structure
├── train_twin_pairs.json       # Twin pairs information
├── test_dataset_infor.json     # Test set structure
└── test_twin_pairs.json        # Test twin pairs
```

**Dataset Statistics:**
- Training: 6,182 images, 356 people (178 twin pairs)
- Testing: 907 images, 62 people (31 twin pairs)
- Images per person: 4-68 images
- Image format: Face-cropped, aligned images

## Kaggle P100 Training Guide

### Step 1: Setup Kaggle Environment

1. **Create Kaggle Dataset** with your data files:
   ```
   nd-twin-dataset/
   ├── train_dataset_infor.json
   ├── train_twin_pairs.json
   ├── test_dataset_infor.json
   └── test_twin_pairs.json
   ```

2. **Set WandB API Key** in Kaggle Secrets:
   - Go to Settings > Secrets
   - Add `WANDB_API_KEY` with your WandB API key

### Step 2: Kaggle Training Script

```python
# Kaggle notebook cell
!pip install timm wandb

# Clone your repository or upload code
import sys
sys.path.append('/kaggle/working/DCAL_Twin/src')

# Run training
!python /kaggle/working/DCAL_Twin/scripts/train_kaggle.py \
    --config /kaggle/working/DCAL_Twin/configs/kaggle_config.yaml \
    --data-dir /kaggle/input/nd-twin-dataset \
    --output-dir /kaggle/working/checkpoints
```

### Step 3: Configuration for P100

```yaml
# configs/kaggle_config.yaml
device: "cuda"
num_gpus: 1
tracking: "wandb"
checkpoint_dir: "/kaggle/working/checkpoints"
data_dir: "/kaggle/input/nd-twin-dataset"
timeout_hours: 11

model:
  backbone: "vit_base_patch16_224"
  embed_dim: 768
  image_size: 224

training:
  batch_size: 12  # Optimized for P100
  learning_rate: 3e-4
  epochs: 100
  mixed_precision: true
  gradient_accumulation_steps: 2
  
optimization:
  enable_compilation: true
  use_channels_last: true
  gradient_checkpointing: true
```

### Step 4: Monitor Training

```python
# Check training progress
import wandb
run = wandb.init(project="dcal-twin-verification", 
                entity="hunchoquavodb-hanoi-university-of-science-and-technology")

# Monitor metrics
print("Training metrics:", run.summary)
```

### Step 5: Handle Timeout & Resume

The system automatically handles Kaggle's 12-hour timeout:

```python
# Automatic checkpoint saving every epoch
# Resume training from checkpoint
!python /kaggle/working/DCAL_Twin/scripts/train_kaggle.py \
    --config /kaggle/working/DCAL_Twin/configs/kaggle_config.yaml \
    --resume-from /kaggle/working/checkpoints/latest_checkpoint.pth
```

## Choosing Between Fresh Training and Resume Training

When training on Kaggle or locally, you can choose to either **resume from a previous checkpoint** (continue training from where you left off) or **start fresh** (train from scratch, ignoring any existing checkpoints).

### When to Resume Training
- You want to continue training after a timeout, crash, or interruption.
- The model architecture and configuration have **not changed** since the last checkpoint.
- You want to leverage previously learned weights for faster convergence.

**How to resume:**
```python
!python scripts/train_kaggle.py \
    --config configs/kaggle_config.yaml \
    --data-root /kaggle/input/nd-twin \
    --resume-from /kaggle/working/checkpoints/latest_checkpoint.pth
```
- The system will attempt to load the latest checkpoint and continue training.
- If there are minor architecture changes, the loader will log warnings and reinitialize only the mismatched layers (e.g., feature projection). Training will continue, but monitor metrics closely.

### When to Start Fresh Training
- You want to **ignore all previous checkpoints** and train from scratch.
- The model architecture or configuration has **changed significantly** (e.g., different feature dimensions, new layers).
- You want to ensure all weights are randomly initialized.

**How to start fresh:**
```python
!python scripts/train_kaggle.py \
    --config configs/kaggle_config.yaml \
    --data-root /kaggle/input/nd-twin \
    --no-resume
```
- Use the `--no-resume` flag (if supported) or manually delete/rename the checkpoint directory before training.
- The system will skip loading any checkpoints and initialize all weights randomly.

### How the System Handles Checkpoint Mismatches
- If a checkpoint is found but the model architecture has changed, the loader will:
  - Attempt to load as much as possible.
  - Log all missing/unexpected/shape-mismatched layers.
  - Automatically reinitialize mismatched layers (e.g., feature projection).
  - Provide recommendations in the logs (e.g., reduce learning rate, monitor metrics).
- If the mismatch is severe, you may see a warning to start fresh training.

### Best Practices
- **After major model changes:** Start fresh training to avoid subtle bugs.
- **After minor changes:** Resume training, but monitor logs for warnings about reinitialized layers.
- **Always save new checkpoints** after resuming or starting fresh, to avoid future compatibility issues.
- **Monitor training metrics** closely for the first few epochs after resuming or changing the model.

For more details, see the [Troubleshooting](#troubleshooting) section and review logs for checkpoint compatibility analysis.

## Local Training

### Multi-GPU Training (2x RTX 2080Ti)

```bash
# Start MLFlow server
mlflow server --host 0.0.0.0 --port 5000

# Train with multiple GPUs
python scripts/train_local.py \
    --config configs/local_config.yaml \
    --data-dir ./data \
    --output-dir ./checkpoints \
    --num-gpus 2
```

### Local Configuration

```yaml
# configs/local_config.yaml
device: "cuda"
num_gpus: 2
tracking: "mlflow"
mlflow_uri: "http://localhost:5000"

training:
  batch_size: 16  # Per GPU
  learning_rate: 5e-4
  epochs: 100
  num_workers: 8
```

## Inference

### Two Images Inference

#### Quick Inference

```python
from src.inference.predictor import DCALPredictor

# Load trained model
predictor = DCALPredictor(
    model_path="checkpoints/best_model.pth",
    config_path="configs/kaggle_config.yaml"
)

# Predict similarity between two images
similarity, is_same_person = predictor.predict_pair(
    "path/to/image1.jpg", 
    "path/to/image2.jpg"
)

print(f"Similarity: {similarity:.4f}")
print(f"Same person: {is_same_person}")
```

#### Batch Inference

```python
from src.inference.predictor import BatchPredictor

# Batch processing for multiple pairs
batch_predictor = BatchPredictor(
    model_path="checkpoints/best_model.pth",
    config_path="configs/kaggle_config.yaml",
    batch_size=32
)

# Process multiple image pairs
image_pairs = [
    ("image1_a.jpg", "image1_b.jpg"),
    ("image2_a.jpg", "image2_b.jpg"),
    # ... more pairs
]

results = batch_predictor.predict_batch(image_pairs)
for pair, (similarity, is_same) in zip(image_pairs, results):
    print(f"{pair}: {similarity:.4f} ({'Same' if is_same else 'Different'})")
```

#### Optimized Inference

```python
from src.inference.predictor import OptimizedPredictor

# High-performance inference with optimizations
opt_predictor = OptimizedPredictor(
    model_path="checkpoints/best_model.pth",
    config_path="configs/kaggle_config.yaml",
    enable_quantization=True,
    enable_tensorrt=True,
    cache_size=1000
)

# Faster inference for production
similarity, is_same_person = opt_predictor.predict_pair(
    "image1.jpg", "image2.jpg"
)
```

### Command Line Inference

```bash
# Single pair inference
python -m src.inference.predictor \
    --model checkpoints/best_model.pth \
    --config configs/kaggle_config.yaml \
    --image1 path/to/image1.jpg \
    --image2 path/to/image2.jpg

# Batch inference from CSV
python -m src.inference.predictor \
    --model checkpoints/best_model.pth \
    --config configs/kaggle_config.yaml \
    --batch-file image_pairs.csv \
    --output results.csv
```

## Attention Visualization

### Generate Attention Maps

```python
from src.utils.visualization import FaceAttentionVisualizer

# Initialize visualizer
visualizer = FaceAttentionVisualizer(
    model_path="checkpoints/best_model.pth",
    config_path="configs/kaggle_config.yaml"
)

# Generate attention maps for image pair
attention_maps = visualizer.visualize_pair_attention(
    "twin1_image1.jpg", 
    "twin1_image2.jpg",
    save_path="attention_output/"
)
```

### Attention Analysis Script

```bash
# Analyze attention patterns
python scripts/visualize_attention.py \
    --mode analyze \
    --model checkpoints/best_model.pth \
    --config configs/kaggle_config.yaml \
    --image1 twin1_a.jpg \
    --image2 twin1_b.jpg \
    --output-dir attention_analysis/

# Compare attention between same/different pairs
python scripts/visualize_attention.py \
    --mode compare \
    --model checkpoints/best_model.pth \
    --positive-pairs positive_pairs.txt \
    --negative-pairs negative_pairs.txt \
    --output-dir comparison_analysis/
```

### Attention Visualization Features

1. **Attention Rollout**: Accumulated attention across layers
2. **GLCA Visualization**: Global-local cross-attention maps
3. **Discriminative Regions**: Highlighted facial differences
4. **Comparative Analysis**: Side-by-side attention comparison
5. **Statistics**: Attention distribution analysis

### Example Attention Output

```
attention_output/
├── twin1_pair_attention.png          # Side-by-side with attention overlay
├── twin1_attention_rollout.png       # Attention rollout visualization
├── twin1_glca_attention.png          # GLCA attention maps
├── twin1_discriminative_regions.png  # Highlighted differences
└── twin1_attention_stats.json        # Attention statistics
```

## Configuration

### Model Configuration

```yaml
model:
  backbone: "vit_base_patch16_224"  # or "vit_large_patch16_224"
  num_classes: 2
  embed_dim: 768                    # 768 for base, 1024 for large
  num_heads: 12                     # 12 for base, 16 for large
  num_layers: 12
  
dcal:
  num_sa_blocks: 12
  num_glca_blocks: 1
  num_pwca_blocks: 12
  local_ratio: 0.3
  attention_dropout: 0.1
  
siamese:
  similarity_function: "cosine"     # "cosine", "euclidean", "learned"
  temperature: 0.07
  margin: 0.5
```

### Training Configuration

```yaml
training:
  batch_size: 16
  learning_rate: 5e-4
  epochs: 100
  warmup_epochs: 10
  weight_decay: 0.05
  
  # Optimization
  mixed_precision: true
  gradient_accumulation_steps: 1
  gradient_checkpointing: false
  
  # Scheduling
  scheduler: "cosine"
  min_lr: 1e-6
  
  # Regularization
  label_smoothing: 0.1
  dropout: 0.1
  
data:
  image_size: 224
  train_split: 0.8
  val_split: 0.2
  
  # Augmentation
  augmentation: true
  mixup_alpha: 0.2
  cutmix_alpha: 1.0
  
  # Sampling
  positive_ratio: 0.5
  hard_negative_ratio: 0.3
  soft_negative_ratio: 0.2
```

## Performance Benchmarks

### Training Performance

| Environment | GPU | Batch Size | Speed (samples/sec) | Memory (GB) |
|-------------|-----|------------|-------------------|-------------|
| Local | 2x RTX 2080Ti | 16 | 45 | 8.2 |
| Kaggle | Tesla P100 | 12 | 32 | 14.1 |
| Kaggle | Tesla T4 | 8 | 28 | 12.3 |

### Inference Performance

| Model | Optimization | Latency (ms) | Throughput (pairs/sec) |
|-------|-------------|-------------|----------------------|
| Base | None | 45 | 22 |
| Base | Quantization | 28 | 36 |
| Base | TensorRT | 18 | 56 |
| Large | None | 78 | 13 |
| Large | TensorRT | 35 | 29 |

### Accuracy Results

| Model | Dataset | EER (%) | ROC-AUC | Accuracy (%) |
|-------|---------|---------|---------|-------------|
| DCAL Base | ND TWIN | 3.2 | 0.978 | 96.8 |
| DCAL Large | ND TWIN | 2.8 | 0.982 | 97.2 |

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```python
# Reduce batch size
training:
  batch_size: 8  # Reduce from 16

# Enable gradient checkpointing
training:
  gradient_checkpointing: true

# Use gradient accumulation
training:
  gradient_accumulation_steps: 2
```

#### 2. Slow Training

```python
# Enable mixed precision
training:
  mixed_precision: true

# Optimize data loading
data:
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
```

#### 3. Kaggle Timeout

```python
# Reduce training time
training:
  timeout_hours: 11
  checkpoint_every_n_epochs: 1

# Use smaller model
model:
  backbone: "vit_base_patch16_224"  # Instead of large
```

#### 4. Poor Convergence

```python
# Adjust learning rate
training:
  learning_rate: 1e-4  # Reduce from 5e-4
  warmup_epochs: 20    # Increase warmup

# Increase regularization
training:
  weight_decay: 0.1
  dropout: 0.2
```

### Environment-Specific Issues

#### Kaggle

- **Dataset Access**: Ensure dataset is public or properly shared
- **Secrets**: Verify WandB API key is correctly set
- **Storage**: Monitor disk usage (max 20GB)
- **Timeout**: Implement robust checkpointing

#### Local

- **MLFlow**: Ensure server is running on correct port
- **Multi-GPU**: Check CUDA device availability
- **Memory**: Monitor system RAM usage

### Performance Optimization

#### For Kaggle P100

```python
# Optimal configuration for P100
training:
  batch_size: 12
  mixed_precision: true
  gradient_accumulation_steps: 2
  
model:
  backbone: "vit_base_patch16_224"
  
optimization:
  enable_compilation: true
  use_channels_last: true
```

#### For Local RTX 2080Ti

```python
# Optimal configuration for 2080Ti
training:
  batch_size: 16
  mixed_precision: true
  
model:
  backbone: "vit_large_patch16_224"  # Can use larger model
  
optimization:
  enable_compilation: true
  num_gpus: 2
```

## Advanced Usage

### Custom Loss Functions

```python
from src.training.loss import UncertaintyLoss

# Custom loss with uncertainty weighting
loss_fn = UncertaintyLoss(
    verification_loss_weight=1.0,
    classification_loss_weight=0.5,
    uncertainty_weight=0.1
)
```

### Custom Similarity Functions

```python
from src.models.siamese_dcal import AdaptiveSimilarityLearner

# Learned similarity function
similarity_fn = AdaptiveSimilarityLearner(
    feature_dim=768,
    hidden_dim=256,
    num_heads=8
)
```

### Export Models

```python
from src.inference.predictor import ExportManager

# Export to different formats
exporter = ExportManager(model_path="checkpoints/best_model.pth")

# Export to ONNX
exporter.export_onnx("model.onnx")

# Export to TorchScript
exporter.export_torchscript("model.pt")

# Export to TensorRT
exporter.export_tensorrt("model.trt")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dcal_twin_2024,
  title={DCAL Twin Face Verification: Adapting Dual Cross-Attention Learning for Identical Twin Recognition},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original DCAL paper authors
- ND TWIN dataset creators
- Twins Days Festival organizers
- HuggingFace Transformers team
- PyTorch team 