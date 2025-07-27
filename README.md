# AdaptQNet

**AdaptQNet** is a hardware-aware neural network optimization system for microcontrollers, focusing on mixed-precision quantization, memory/computation constraints, and efficient deployment on resource-limited devices.

## Features

- Mixed-precision quantization (FP32/FP16/INT8/INT4/INT2)
- Hardware-aware search and optimization (RAM, Flash, cores, etc.)
- Support for MobileNetV2, MobileNetV3, EfficientNet
- Automatic parallel execution planning
- PACT activation quantization, DoReFa-Net weight quantization
- Visualization tools for quantization, memory, and performance
- Flexible dataset processing and splitting (e.g., Mini-ImageNet 7:2:1 split)

## Project Structure

```
adaptqnet/
├── benchmark/                  # Benchmarking scripts (e.g., loss, precision analysis)
│   └── Loss_benchmark.py
├── data/                       # Data and dataset processing
│   ├── MiniImagenet/           # Original Mini-ImageNet (train/val/test, non-overlapping classes)
│   ├── MiniImagenet_merged/    # Merged Mini-ImageNet (all classes, 7:2:1 split)
│   ├── cifar-10-batches-py/    # CIFAR-10 raw data
│   ├── process_miniimagenet.py # Mini-ImageNet preprocessing
│   └── process_miniimagenet_split.py # Script for 7:2:1 split
├── models/                     # Model definitions
│   ├── base.py                 # Base classes for quantization/mixed-precision
│   ├── mobilenetv2.py
│   ├── mobilenetv3.py
│   └── efficientnet.py
├── plot/                       # Visualization scripts and figures
│   ├── plot_latency.py
│   ├── plot_profile.py
│   ├── plot_2_stage.py
│   ├── plot_combined.py
│   ├── plot_distribution.py
│   ├── single-dual-search-compare.png
│   └── Draw/
│       ├── draw.py
│       └── operation_latency_comparison_optimized_v3.png
├── Search_32x32_10_CIFAR10/    # NAS/search experiments for CIFAR-10
│   ├── train_dnas_cifar10.py
│   ├── evaluate_cifar10.py
│   ├── pretrain_cifar10_efficientnet.py
│   ├── search_results_*.pt
│   └── ...
├── Search_84x84_100_MiniImagenet/ # NAS/search experiments for Mini-ImageNet
│   ├── train_dnas_miniimagenet.py
│   ├── pretrain_miniimagenet_*.py
│   └── ...
├── utils/                      # Utility modules
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── visualization.py
│   └── config_utils.py
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/adaptqnet.git
cd adaptqnet
pip install -r requirements.txt
```

## Dataset Preparation

### Mini-ImageNet

- The original Mini-ImageNet is split into train/val/test with non-overlapping classes.
- To use a 7:2:1 split (all classes in each split), run:

```bash
python data/process_miniimagenet_split.py \
    --data_root data/MiniImagenet \
    --out_root data/MiniImagenet_merged
```

- The new `MiniImagenet_merged` directory will have 100 classes in each of train/val/test, with a 7:2:1 ratio per class.

### CIFAR-10

- Place the raw CIFAR-10 data in `data/cifar-10-batches-py/` (downloaded automatically if missing).

## Usage

### 1. Pre-training

Train a model in FP32 (example for Mini-ImageNet):

```bash
python Search_84x84_100_MiniImagenet/pretrain_miniimagenet_mobilenetv2.py --config path/to/config.yaml
```

### 2. Mixed-Precision NAS/Search

Run NAS/mixed-precision search (example for Mini-ImageNet):

```bash
python Search_84x84_100_MiniImagenet/train_dnas_miniimagenet.py \
    --model mobilenetv2 \
    --pretrained-path path/to/pretrained.pth \
    --config path/to/config.yaml
```

### 3. Evaluation

Evaluate searched models (see `evaluate_cifar10.py` or similar scripts).

### 4. Visualization

Generate plots for latency, memory, precision distribution, etc.:

```bash
python plot/plot_latency.py
python plot/plot_distribution.py
# ... and other scripts in plot/
```

## Configuration

### Hardware Configuration

Edit `configs/hardware_configs.yaml` to set RAM, Flash, cores, etc.

### Model Configuration

Edit `configs/model_configs.yaml` for model type, width multiplier, number of classes, training hyperparameters, etc.

## Benchmarking

- Use `benchmark/Loss_benchmark.py` for precision/loss analysis and ablation studies.

## Supported Models

- MobileNetV2 (adaptqnet/models/mobilenetv2.py)
- MobileNetV3 (adaptqnet/models/mobilenetv3.py)
- EfficientNet (adaptqnet/models/efficientnet.py)

## Quantization Methods

- Weights: DoReFa-Net quantization
- Activations: PACT quantization
- FC layers: fixed INT8

## Hardware Constraints

- RAM, Flash, cores, DMA, cache, etc. are all considered in the search and deployment.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{adaptqnet2024,
  title={AdaptQNet: Hardware-Aware Mixed-Precision Neural Network Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/adaptqnet}
}
```

