# Aq-Artifact

**This system** is a comprehensive hardware-aware neural network optimization system designed for microcontrollers and resource-constrained devices. It focuses on mixed-precision quantization, memory/computation constraints, and efficient deployment through differentiable neural architecture search (DNAS).

## Project Overview

This project is a complete implementation, featuring:

- **Mixed-Precision Quantization**: Support for FP32/FP16/INT8/INT4/INT2 precision levels
- **Hardware-Aware Search**: Consideration of RAM and computational constraints
- **Supported Models**: MobileNetV2 (with extensible architecture for MobileNetV3, EfficientNet)
- **DNAS Algorithm**: Differentiable neural architecture search with temperature annealing
- **Quantization Methods**: PACT activation quantization, DoReFa-Net weight quantization
- **Tensor Analysis**: Memory usage analysis and latency prediction
- **Flexible Dataset Processing**: Support for CIFAR-10 and other datasets

## Project Structure

```
adaptqnet-artifact/
├── models/                     # Model definitions and quantization layers
│   ├── base.py                 # MixedPrecisionLayer and base classes
│   ├── mobilenetv2.py          # MobileNetV2 implementation
│   └── __init__.py             # Model exports
├── Search_CIFAR10/             # CIFAR-10 experiments
│   ├── dnas/                   # DNAS search implementation
│   │   ├── dnas_search_cifar10_mbv2.py  # Main search script
│   │   ├── utils/              # Utility functions
│   │   │   ├── data_utils.py   # Data loaders
│   │   │   ├── model_utils.py  # Model utilities
│   │   │   ├── training_utils.py # Training helpers
│   │   │   ├── save_utils.py   # Model saving utilities
│   │   │   ├── visualization_utils.py # Plotting functions
│   │   │   └── __init__.py     # Utility exports
│   │   ├── data/               # Local data directory
│   │   ├── run_all_dnas.sh     # Batch execution script
│   ├── dnas_multicore/         # Multi-core experiments
│   ├── tensor_analysis_result/ # Analysis results
├── tensor_analysis/            # Tensor analysis and latency prediction
│   ├── tensor_latency_predictor.py  # Latency prediction tool
│   ├── tensor_memory_filter.py      # Memory analysis tool
│   ├── tensor_memory_filter_dualcore.py # Dual-core analysis
│   └── README.md               # Analysis tool documentation
├── data/                       # Dataset storage and processing
├── requirements.txt            # Python dependencies
├── __init__.py                 # Package initialization
└── README.md                   # This file
```

## Core Features

### 1. Mixed-Precision Quantization
- **Precision Options**: FP32, FP16, INT8, INT4, INT2
- **Layer-wise Precision Assignment**: Independent precision selection per layer
- **Hardware-Aware Selection**: Automatic precision optimization based on constraints

### 2. DNAS Search Algorithm
- **Two-Stage Optimization**: Weight training + architecture parameter training
- **Temperature Annealing**: Exponential decay temperature control
- **Hardware Penalty**: Computation cost and memory constraints consideration

### 3. Hardware Constraint Modeling
- **Memory Constraints**: RAM usage analysis
- **Computational Constraints**: MMACs and latency requirements
- **Precision Constraints**: Available precision options per layer

### 4. Quantization Methods
- **Weight Quantization**: DoReFa-Net method
- **Activation Quantization**: PACT method
- **Shared Weights**: All precision options share the same underlying weights

## Installation

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage Examples

### 1. Pre-training
```bash
cd Search_CIFAR10/pretrain
python pretrain_cifar10_mobilenetv2.py --batch_size 256
```

### 2. DNAS Search
```bash
cd Search_CIFAR10/dnas
nohup env PYTHONPATH=/root python dnas_search_cifar10_mbv2.py \
    --tensor_analysis_json ../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json \
    --input_size 32 > mbv2_mixed_32x32_dnas_search.log 2>&1 &
```

### 3. INT-only Search
```bash
nohup env PYTHONPATH=/root python dnas_search_cifar10_mbv2.py \
    --tensor_analysis_json ../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json \
    --input_size 32 \
    --int_only > mbv2_int_only_32x32_dnas_search.log 2>&1 &
```

### 4. Tensor Memory Analysis
```bash
cd ../../tensor_analysis
python tensor_memory_filter.py --model_type mobilenetv2 --memory_threshold 512 --input_size 112
```

### 5. Latency Analysis
```bash
python tensor_latency_predictor.py \
    --model_type mobilenetv2 \
    --input_size 112 \
    --precision_mapping_file example_precision_mapping.json
```

### 6. Batch Execution
```bash
cd Search_CIFAR10/dnas
bash run_all_dnas.sh
```

## Configuration

### Model Configuration
- **Input Resolution**: Support for 32x32, 112x112, 144x144, etc.
- **Width Multiplier**: 0.5x, 1.0x, etc.
- **Precision Options**: Customizable available precision per layer

### Search Parameters
- **Temperature Annealing**: Exponential decay, cosine decay, etc.
- **Learning Rates**: Weight learning rate and alpha learning rate
- **Hardware Penalty Weight**: Balance between accuracy and efficiency

### Hardware Constraints
- **Memory Limits**: RAM usage constraints
- **Computational Limits**: latency requirements
- **Precision Constraints**: Available precision options per layer

## Experimental Results

### Convergence Mechanisms
This system employs multiple mechanisms to ensure search convergence:

1. **Temperature Annealing**: Exponential decay temperature control
2. **Alternating Optimization**: Separation of weight and architecture training
3. **Smart Initialization**: Alpha parameters biased towards INT8
4. **Hardware-Aware Loss**: Computation cost consideration
6. **Gradient Clipping**: Prevention of gradient explosion

### Performance Optimization
- **Progressive Precision Reduction**: From FP16 to INT4/INT2 gradually
- **Critical Layer Protection**: Important depthwise layers maintain higher precision
- **Computational Efficiency**: Extensive use of INT4 in later layers

## Key Files

### Core Implementation
- `models/base.py`: MixedPrecisionLayer implementation, core quantization component
- `models/mobilenetv2.py`: MobileNetV2 architecture definition
- `Search_CIFAR10/dnas/dnas_search_cifar10_mbv2.py`: Main search script
- `tensor_analysis/tensor_latency_predictor.py`: Latency analysis tool
- `tensor_analysis/tensor_memory_filter.py`: Memory analysis tool

### Configuration Files
- `tensor_analysis/example_precision_mapping.json`: Precision mapping configuration
- `requirements.txt`: Python dependencies

## Technical Details

### DNAS Search Process
1. **Stage 1 Step 1**: Train model weights on training set
2. **Stage 1 Step 2**: Train alpha parameters on validation set
3. **Evaluation**: Assess performance on test set (avoiding information leakage)
4. **Stage 2**: Dual-Core enhancement search.

### Quantization Implementation
- **PACT Quantization**: Activation quantization with learnable parameters
- **DoReFa Quantization**: Weight quantization maintaining gradient flow
- **Mixed Precision**: Independent optimal precision per layer

### Hardware Modeling
- **Latency Prediction**: Based on precision and hardware characteristics
- **Memory Analysis**: Consideration of weight and activation storage
- **Computational Complexity**: MMACs calculation

## Dataset Usage

### CIFAR-10
- **Training Set**: 50,000 images for weight training
- **Validation Set**: 6,000 images for alpha parameter training
- **Test Set**: 4,000 images for performance evaluation

### Data Processing
- **Input Resolution**: Configurable (32x32, 112x112, 144x144)
- **Data Augmentation**: Random horizontal flip, rotation, color jitter
- **Normalization**: CIFAR-10 standard normalization

## Analysis Tools

### Tensor Memory Filter
- Analyzes tensor memory usage per layer
- Filters available precision options based on memory constraints
- Generates detailed analysis reports

### Latency Predictor
- Predicts model latency based on precision mapping
- Calculates MMACs and computational complexity
- Supports different hardware configurations

### Compression Calculator
- Calculates model compression ratios
- Analyzes precision distribution
- Generates compression reports

## Contributing

We welcome contributions through issues and pull requests to improve this project.

## License

This project is licensed under the MIT License.

