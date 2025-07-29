# Tensor Analysis Tools

This repository contains tools for analyzing neural network models, including tensor memory usage and latency prediction.

## Tools Overview

### 1. Tensor Memory Filter (`tensor_memory_filter.py`)
Analyzes tensor memory usage of each layer in neural network models and filters available quantization precision options based on given memory thresholds.

### 2. Tensor Latency Predictor (`tensor_latency_predictor.py`)
Predicts model latency based on layer-wise precision mapping and hardware-specific latency coefficients.

## Features

### Tensor Memory Filter
1. Analyze input and output tensor sizes for each layer
2. Support multiple precision options (FP32/FP16/INT8/INT4/INT2)
3. Automatically filter precision options that meet memory constraints
4. Generate detailed analysis reports
5. Support JSON format result export

### Tensor Latency Predictor
1. Calculate MMACs (Multiply-Accumulate operations) for each layer
2. Support precision-specific latency coefficients for different operation types
3. Predict latency based on actual hardware performance data
4. Generate layer-wise and total latency reports
5. Support custom precision mapping via JSON files

## Usage

### Tensor Memory Filter

#### Basic Usage

```bash
python tensor_memory_filter.py --model_type mobilenetv2 --memory_threshold 524288
```

#### Complete Parameter Description

- `--model_type`: Model type (required)
  - Options: mobilenetv2
- `--batch_size`: Batch size (default: 1)
- `--input_size`: Input image size (default: 84)
- `--memory_threshold`: Memory threshold (bytes) (default: 512KB)
- `--output_file`: Result output file (default: tensor_analysis_results.json)

#### Examples

1. Analyze MobileNetV2 model (512KB memory limit):
```bash
python tensor_memory_filter.py --model_type mobilenetv2 --memory_threshold 524288
```

### Tensor Latency Predictor

#### Basic Usage

```bash
python tensor_latency_predictor.py --model_type mobilenetv2 --input_size 224
```

#### Complete Parameter Description

- `--model_type`: Model type (required)
  - Options: mobilenetv2, mobilenetv3, efficientnet
- `--batch_size`: Batch size (default: 1)
- `--input_size`: Input image size (default: 112)
- `--precision_mapping_file`: Precision mapping file path (JSON format)
- `--output_file`: Result output file (default: latency_analysis_results.json)
- `--mode`: MobileNetV3 mode (default: small)
  - Options: large, small
- `--model_version`: EfficientNet version (default: b0)

#### Examples

1. Use model default precision:
```bash
python tensor_latency_predictor.py --model_type mobilenetv2 --input_size 224
```

2. Use custom precision mapping file:
```bash
python tensor_latency_predictor.py --model_type mobilenetv2 --input_size 224 \
  --precision_mapping_file example_precision_mapping.json
```


#### Precision Mapping File Format (JSON)

```json
{
  "layer_name": "precision",
  "features.0.0": "fp32",
  "features.1.conv.0.0": "fp16",
  "features.1.conv.1": "int8"
}
```

**Supported precisions**: fp32, fp16, int8, int4, int2

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

### Tensor Latency Predictor

#### Console Output

The tool will print detailed latency analysis reports to the console, including:
- Layer name and type
- Input/output tensor shapes
- MMACs (Multiply-Accumulate operations)
- Precision and latency for each layer
- Total latency and MMACs
- Precision distribution statistics

#### JSON Output

The result file contains the following information:
- Input shape
- Precision mapping
- Total latency (ns and ms)
- Precision statistics (count and latency per precision)
- Layer details (type, shapes, MMACs, precision, latency)

#### Latency Calculation Formula

```
Latency(ms) = MMACs / (2M) × Precision Coefficient
```

The precision coefficients represent the latency time (milliseconds) per 2M MACs for different operation types:
- **Conv**: Standard convolution operations
- **DWConv**: Depth-wise convolution operations  
- **Linear**: Fully connected layer operations

## Notes

### Tensor Memory Filter
1. Memory threshold should consider actual hardware limitations
2. Batch size significantly affects tensor size
3. Some layers may exceed memory limits under all precision options
4. It's recommended to test with larger thresholds first, then gradually adjust

### Tensor Latency Predictor
1. Latency coefficients are based on actual hardware performance data
2. MMACs calculation considers layer-specific parameters (kernel size, stride, padding, groups)
3. Precision mapping should match the actual model architecture
4. Results provide both layer-wise and total latency analysis

## Use Cases

### Tensor Memory Filter
1. Feasibility analysis before model quantization
2. Hardware compatibility assessment
3. Memory optimization scheme design
4. Quantization strategy formulation

### Tensor Latency Predictor
1. Performance evaluation of different precision configurations
2. Hardware-aware model optimization
3. Latency-accuracy trade-off analysis
4. Real-time inference system design 
