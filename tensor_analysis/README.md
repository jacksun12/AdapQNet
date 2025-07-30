# Tensor Analysis Tools

This repository contains tools for analyzing neural network models, including tensor memory usage analysis.

## Tools Overview

### 1. Tensor Memory Filter (`tensor_memory_filter.py`)
Analyzes tensor memory usage of each layer in neural network models and filters available quantization precision options based on given memory thresholds.

### 2. Dual-Core Tensor Memory Filter (`tensor_memory_filter_dualcore.py`)
Analyzes tensor memory usage for dual-core architectures, supporting dual-precision combinations and filtering available quantization options based on memory constraints.

## Features

### Tensor Memory Filter
1. Analyze input and output tensor sizes for each layer
2. Support multiple precision options (FP32/FP16/INT8/INT4/INT2)
3. Automatically filter precision options that meet memory constraints
4. Generate detailed analysis reports
5. Support JSON format result export

### Dual-Core Tensor Memory Filter
1. All features from single-core version
2. Support dual-precision combinations (e.g., fp32+fp16, int8+int4)
3. Analyze memory requirements for dual-core architectures
4. Generate comprehensive dual-core analysis reports
5. Support both single-core and dual-core precision options

## Usage

### Tensor Memory Filter

#### Basic Usage

```bash
python tensor_memory_filter.py --model_type mobilenetv2 --memory_threshold 512
```

#### Complete Parameter Description

- `--model_type`: Model type (required)
  - Options: mobilenetv2, mobilenetv3, efficientnet
- `--batch_size`: Batch size (default: 1)
- `--input_size`: Input image size (default: 84)
- `--memory_threshold`: Memory threshold (KB) (default: 512.0)
- `--output_file`: Result output file (default: mbv2_cifar10_tensor_analysis_results.json)
- `--mode`: MobileNetV3 mode (default: small)
  - Options: large, small
- `--model_version`: EfficientNet version (default: b0)

#### Examples

1. Analyze MobileNetV2 model (512KB memory limit):
```bash
python tensor_memory_filter.py --model_type mobilenetv2 --memory_threshold 512
```

2. Analyze MobileNetV3 model (1MB memory limit):
```bash
python tensor_memory_filter.py --model_type mobilenetv3 --memory_threshold 1024 --mode small
```

3. Analyze EfficientNet model (2MB memory limit):
```bash
python tensor_memory_filter.py --model_type efficientnet --memory_threshold 2048 --model_version b0
```

### Dual-Core Tensor Memory Filter

#### Basic Usage

```bash
python tensor_memory_filter_dualcore.py --model_type mobilenetv2 --memory_threshold 512
```

#### Complete Parameter Description

Parameters are the same as the single-core version, but analysis results include dual-core options.

#### Examples

1. Analyze MobileNetV2 dual-core memory requirements:
```bash
python tensor_memory_filter_dualcore.py --model_type mobilenetv2 --memory_threshold 512
```

## Output Description

### Tensor Memory Filter

#### Console Output

The tool will print detailed analysis reports to the console, including:
- Name of each layer
- Input/output tensor shapes
- Memory usage under different precisions
- Available precision options
- Memory limit warnings

#### JSON Output

The result file contains the following information:
- Model type
- Input tensor shape
- Memory threshold
- Available precision options for each layer

### Dual-Core Tensor Memory Filter

#### Console Output

The tool will print detailed dual-core analysis reports to the console, including:
- Single-core tensor analysis
- Dual-core tensor analysis with precision combinations
- Available single-core precision options
- Available dual-core precision combinations
- Memory limit warnings for both modes

#### JSON Output

The result file contains the following information:
- Model type
- Input tensor shape
- Memory threshold
- Available single-core precision options for each layer
- Available dual-core precision combinations for each layer

## Precision Options

Both tools support the following precision options:
- `fp32`: 32-bit floating point
- `fp16`: 16-bit floating point
- `int8`: 8-bit integer
- `int4`: 4-bit integer
- `int2`: 2-bit integer

## Notes

### Tensor Memory Filter
1. Memory threshold should consider actual hardware limitations
2. Batch size significantly affects tensor size
3. Some layers may exceed memory limits under all precision options
4. It's recommended to test with larger thresholds first, then gradually adjust

### Dual-Core Tensor Memory Filter
1. All notes from single-core version apply
2. Dual-core combinations include all possible pairs of precision options
3. Memory requirements for dual-core are the sum of two precision requirements
4. Provides comprehensive analysis for both single-core and dual-core architectures

## Use Cases

### Tensor Memory Filter
1. Feasibility analysis before model quantization
2. Hardware compatibility assessment
3. Memory optimization scheme design
4. Quantization strategy formulation

### Dual-Core Tensor Memory Filter
1. All use cases from single-core version
2. Dual-core hardware optimization
3. Mixed-precision strategy design
4. Advanced quantization scheme development 
