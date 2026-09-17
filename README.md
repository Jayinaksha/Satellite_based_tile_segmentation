<p align="center">
  <h1 align="center">🛰️ High-Resolution Image Segmentation</h1>
  <p align="center">
    <strong>Semantic Segmentation of Ultra-High-Resolution Imagery using Residual U-Net with Dilated Bottleneck</strong>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Functional_API-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Residual_U--Net-blueviolet?style=flat-square" alt="Architecture">
  <img src="https://img.shields.io/badge/Input_Size-6000×6000-orange?style=flat-square" alt="Input Size">
  <img src="https://img.shields.io/badge/Classes-6-blue?style=flat-square" alt="Classes">
  <img src="https://img.shields.io/badge/Best_IoU-0.80+-success?style=flat-square" alt="IoU">
</p>

---

## 📋 Overview

An end-to-end deep learning pipeline for **pixel-wise semantic segmentation** of ultra-high-resolution images (6000×6000 pixels). The system processes large-scale imagery through an efficient patch-based pipeline, applies a custom **Residual U-Net** with a **multi-rate dilated convolution bottleneck**, and reconstructs full-resolution predictions using a loop-free scatter-based approach.

> **💡 Why this matters**: This pipeline directly addresses the core challenge in **Digital Pathology** and **Remote Sensing** — processing and segmenting images that are orders of magnitude larger than what standard deep learning models can handle. The same architecture and tiling strategy used here for 6000×6000 aerial tiles applies to **50,000×50,000 whole-slide histopathology images** used in clinical diagnostics.

### Key Technical Contributions

| Feature | Description |
|---------|-------------|
| 🏗️ **Residual U-Net** | Enhanced U-Net with skip-connection concatenation and residual blocks for better gradient flow |
| 🔬 **Multi-Rate Dilated Bottleneck** | Parallel/serial dilated convolutions (rates 1, 2, 4, 8) for multi-scale feature extraction |
| ⚡ **Loop-Free Tile Prediction** | `tf.scatter_nd`-based reconstruction reduces inference from minutes to seconds |
| 🔄 **Test-Time Augmentation (TTA)** | 8-fold TTA (4 rotations × 2 flips) for improved prediction quality |
| 📊 **Advanced Training Strategies** | Learning Rate Finder + One-Cycle Policy for optimal convergence |
| 🎯 **Custom Loss Functions** | Jaccard Distance, Categorical Focal Loss for handling class imbalance |
| 🚀 **Efficient tf.data Pipeline** | Scalable input pipeline with on-the-fly patch extraction and augmentation |

---

## 🏛️ Architecture

```
                         Input (256×256×3)
                              │
                    ┌─────────┴─────────┐
                    │    ENCODER PATH    │
                    ├───────────────────┤
                    │  Conv Block (32)   │──── Skip Connection ────┐
                    │  + Residual        │                         │
                    │  MaxPool 2×2       │                         │
                    ├───────────────────┤                         │
                    │  Conv Block (64)   │──── Skip Connection ──┐ │
                    │  + Residual        │                       │ │
                    │  MaxPool 2×2       │                       │ │
                    ├───────────────────┤                       │ │
                    │  Conv Block (128)  │──── Skip Connection ┐ │ │
                    │  + Residual        │                     │ │ │
                    │  MaxPool 2×2       │                     │ │ │
                    ├───────────────────┤                     │ │ │
                    │  Conv Block (256)  │──── Skip Conn. ──┐ │ │ │
                    │  + Residual        │                   │ │ │ │
                    │  MaxPool 2×2       │                   │ │ │ │
                    └─────────┬─────────┘                   │ │ │ │
                              │                             │ │ │ │
                    ┌─────────┴─────────┐                   │ │ │ │
                    │    BOTTLENECK      │                   │ │ │ │
                    │  Dilated Conv r=1  │                   │ │ │ │
                    │  Dilated Conv r=2  │                   │ │ │ │
                    │  Dilated Conv r=4  │                   │ │ │ │
                    │  Dilated Conv r=8  │                   │ │ │ │
                    │  Concatenate       │                   │ │ │ │
                    └─────────┬─────────┘                   │ │ │ │
                              │                             │ │ │ │
                    ┌─────────┴─────────┐                   │ │ │ │
                    │   DECODER PATH    │                   │ │ │ │
                    ├───────────────────┤                   │ │ │ │
                    │  TransConv (256)   │◄── Concat ───────┘ │ │ │
                    │  + Residual        │                     │ │ │
                    ├───────────────────┤                     │ │ │
                    │  TransConv (128)   │◄── Concat ──────────┘ │ │
                    │  + Residual        │                       │ │
                    ├───────────────────┤                       │ │
                    │  TransConv (64)    │◄── Concat ─────────────┘ │
                    │  + Residual        │                         │
                    ├───────────────────┤                         │
                    │  TransConv (32)    │◄── Concat ──────────────┘
                    │  + Residual        │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Conv 1×1 (softmax)│
                    │  → 6 class map     │
                    └───────────────────┘
```

---

## 📊 Results

### Model Performance on Validation Set

| Class | IoU Score | Visual Quality |
|-------|-----------|---------------|
| 🔴 Background | 0.72 | Good |
| 🔵 Building | **0.85** | Excellent |
| ⬜ Roads | **0.81** | Excellent |
| 🟦 Vegetation | 0.75 | Good |
| 🟢 Tree | 0.74 | Good |
| 🟡 Car | **0.82** | Excellent |

### Confusion Matrix & Per-Class IoU
![Validation Metrics](assets/validation-ds-metrics.JPG)

### Tile-Level Predictions

<table>
  <tr>
    <td align="center"><strong>Without TTA</strong></td>
    <td align="center"><strong>With TTA</strong></td>
  </tr>
  <tr>
    <td><img src="assets/tile-pred-no-tsa.JPG" width="400"></td>
    <td><img src="assets/tile-pred-with-tsa.JPG" width="400"></td>
  </tr>
</table>

> Test-Time Augmentation consistently improves prediction quality by averaging predictions across 8 geometric transformations of the input.

### Training Curves
![Training Metrics](assets/trainin-metrics.png)

---

## 🚀 Pipeline Overview

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌──────────────┐
│  Load Tile   │────▶│  Extract Patches │────▶│  Augmentation │────▶│   Training   │
│  (6000×6000) │     │  (256×256 each)  │     │  Flip/Rotate  │     │  Residual    │
│              │     │  Multi-scale     │     │  Color Jitter  │     │  U-Net       │
└──────────────┘     └──────────────────┘     └───────────────┘     └──────┬───────┘
                                                                           │
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐           │
│  Full Tile   │◀────│  Scatter-ND      │◀────│   Inference   │◀──────────┘
│  Prediction  │     │  Reconstruction  │     │   + TTA       │
│  (6000×6000) │     │  (loop-free)     │     │               │
└──────────────┘     └──────────────────┘     └───────────────┘
```

---

## 📂 Project Structure

```
aerial-tile-segmentation/
├── 📁 src/                           # Core source code
│   ├── 📁 models/                    # Neural network architectures
│   │   ├── blocks.py                 # Conv, encoder, decoder, bottleneck blocks
│   │   ├── unet.py                   # Residual U-Net model builder
│   │   └── losses.py                 # Jaccard, Focal Loss implementations
│   ├── 📁 data/                      # Data pipeline
│   │   ├── patch_extraction.py       # Multi-scale patch extraction from tiles
│   │   ├── augmentation.py           # Spatial & color augmentation
│   │   └── dataset.py                # tf.data input pipeline
│   ├── 📁 training/                  # Training utilities
│   │   ├── lr_finder.py              # Learning Rate Finder (Smith, 2015)
│   │   ├── one_cycle.py              # One-Cycle Policy (Smith, 2018)
│   │   ├── metrics.py                # Custom Mean IoU metric
│   │   └── trainer.py                # Training orchestrator
│   ├── 📁 inference/                 # Inference pipeline
│   │   ├── predictor.py              # Scatter-ND tile prediction
│   │   └── tta.py                    # Test-Time Augmentation
│   └── 📁 utils/                     # Utilities
│       ├── mask_utils.py             # Mask ↔ Label conversions
│       └── visualization.py          # Plotting & metrics visualization
├── 📁 configs/                       # Dataset & training configurations
│   ├── isprs_config.yaml             # ISPRS Potsdam config
│   └── inria_config.yaml             # INRIA Buildings config
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Dataset visualization
│   ├── 02_training.ipynb             # Model training
│   └── 03_inference_demo.ipynb       # Inference demonstration
├── 📁 tests/                         # Unit tests
│   ├── test_blocks.py                # Model block tests
│   └── test_losses.py                # Loss function tests
├── 📁 assets/                        # Images & diagrams
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/aerial-tile-segmentation.git
cd aerial-tile-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dataset Setup

Download the [ISPRS Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/) dataset and organize as:

```
data/
├── train_dir/
│   ├── potsdam_aerials/    # Training RGB tiles (.tif)
│   └── potsdam_masks/      # Training segmentation masks (.tif)
└── validation_dir/
    ├── potsdam_aerials/    # Validation RGB tiles (.tif)
    └── potsdam_masks/      # Validation segmentation masks (.tif)
```

---

## 🎮 Quick Start

### Training

```python
import yaml
from src.models.unet import get_unet_model
from src.data.dataset import get_dataset
from src.training.trainer import Trainer

# Load config
with open('configs/isprs_config.yaml') as f:
    config = yaml.safe_load(f)

# Build model
model = get_unet_model(
    input_shape=tuple(config['model']['input_shape']),
    filters=config['model']['filters'],
    n_block=config['model']['n_block'],
    n_class=config['dataset']['num_classes'],
    mode=config['model']['bottleneck_mode'],
    residual=config['model']['residual']
)

# Create datasets
train_ds = get_dataset(
    img_dir=config['dataset']['train_img_dir'],
    mask_dir=config['dataset']['train_mask_dir'],
    batch_size=config['training']['batch_size'],
    config=config
)

# Train
trainer = Trainer(model, config)
trainer.train(train_ds, val_ds)
```

### Inference

```python
from src.inference.predictor import get_tile_prediction
from src.inference.tta import get_tile_tta_pred

# Single prediction
prediction = get_tile_prediction('path/to/tile.tif', model, config)

# With Test-Time Augmentation
prediction_tta = get_tile_tta_pred('path/to/tile.tif', model, config)
```

---

## 📚 Technical Deep-Dive

### Efficient Patch Extraction
Standard sliding-window approaches with Python loops are prohibitively slow for 6000×6000 tiles. This project uses `tf.image.extract_patches` to extract all 256×256 patches in a single vectorized operation, yielding **529 patches per tile** with zero Python-loop overhead.

### Multi-Scale Feature Extraction
The bottleneck layer employs **Atrous Spatial Pyramid Pooling (ASPP)**-inspired design with dilated convolutions at rates {1, 2, 4, 8}, capturing context at multiple receptive field scales — critical for accurately segmenting objects of vastly different sizes (buildings vs. cars).

### Loop-Free Tile Reconstruction
Prediction stitching uses `tf.scatter_nd` to reconstruct the full-resolution output from overlapping patches in a single operation, reducing inference time from **minutes to seconds**.

### Training Strategy
1. **Learning Rate Finder** ([Smith, 2015](https://arxiv.org/abs/1506.01186)) — Systematically identifies the optimal learning rate
2. **One-Cycle Policy** ([Smith, 2018](https://arxiv.org/abs/1708.07120)) — Super-convergence through cyclical learning rates with momentum scheduling
3. **Custom Loss Functions** — Jaccard Distance for IoU optimization; Focal Loss ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) for class imbalance

---

## 🔗 Relevance to Medical Imaging & Digital Pathology

This pipeline addresses the **same fundamental challenges** present in Digital Pathology:

| Challenge | Aerial Imaging | Digital Pathology |
|-----------|---------------|-------------------|
| Image Size | 6,000 × 6,000 px | 50,000 × 50,000+ px |
| Processing | Patch extraction → Prediction → Reconstruction | Identical pipeline |
| Classes | Buildings, Roads, Trees, Cars | Tumor, Stroma, Necrosis, etc. |
| Class Imbalance | Small objects (Cars) vs. Background | Rare cell types vs. Normal tissue |
| Precision Required | Pixel-accurate boundaries | Cell-level accuracy |

The modular architecture, efficient tiling strategy, and advanced training techniques demonstrated here **transfer directly** to histopathology whole-slide image analysis.

---

## 📖 References

- **U-Net**: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" ([arXiv:1505.04597](https://arxiv.org/abs/1505.04597))
- **Residual Networks**: He et al. "Deep Residual Learning for Image Recognition" ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))
- **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" ([arXiv:1708.02002](https://arxiv.org/abs/1708.02002))
- **LR Finder**: Smith. "Cyclical Learning Rates for Training Neural Networks" ([arXiv:1506.01186](https://arxiv.org/abs/1506.01186))
- **One-Cycle**: Smith. "Super-Convergence" ([arXiv:1708.07120](https://arxiv.org/abs/1708.07120))
- **TTA**: Shanmugam et al. "When and How to Use Test-Time Augmentation" ([arXiv:2011.11156](https://arxiv.org/abs/2011.11156))
- **Jaccard Loss**: Berman et al. "The Lovász-Softmax Loss" ([arXiv:1908.03851](https://arxiv.org/abs/1908.03851))

---

## 🙏 Acknowledgments

This project builds upon and extends the work from the [ISPRS Semantic Segmentation](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/) benchmark. The original prototype was developed as part of an internship project by Maher SEBAI. This repository restructures the codebase into a modular, production-ready pipeline with enhanced documentation, testing, and configuration management.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
