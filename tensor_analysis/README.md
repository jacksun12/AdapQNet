# 张量内存分析工具

这个工具用于分析神经网络模型中各层的张量内存使用情况，并根据给定的内存阈值筛选可用的量化精度选项。

## 功能特点

1. 分析每一层的输入和输出张量大小
2. 支持多种精度选项（FP32/FP16/INT8/INT4/INT2）
3. 自动筛选满足内存限制的精度选项
4. 生成详细的分析报告
5. 支持JSON格式结果导出

## 使用方法

### 基本用法

```bash
python tensor_memory_filter.py --model_type mobilenetv2 --memory_threshold 524288
```

### 完整参数说明

- `--model_type`: 模型类型 (必需)
  - 可选值: mobilenetv2, mobilenetv3, efficientnet
- `--batch_size`: 批次大小 (默认: 1)
- `--input_size`: 输入图像大小 (默认: 84)
- `--memory_threshold`: 内存阈值（字节）(默认: 512KB)
- `--output_file`: 结果输出文件 (默认: tensor_analysis_results.json)
- `--mode`: MobileNetV3的模式 (默认: large)
  - 可选值: large, small
- `--model_version`: EfficientNet的版本 (默认: b0)

### 示例

1. 分析MobileNetV2模型（512KB内存限制）：
```bash
python tensor_memory_filter.py --model_type mobilenetv2 --memory_threshold 524288
```

2. 分析MobileNetV3-Small模型（1MB内存限制）：
```bash
python tensor_memory_filter.py --model_type mobilenetv3 --mode small --memory_threshold 1048576
```

3. 分析EfficientNet-B0模型（自定义输入大小）：
```bash
python tensor_memory_filter.py --model_type efficientnet --input_size 96 --memory_threshold 524288
```

## 输出说明

### 控制台输出

工具会在控制台打印详细的分析报告，包括：
- 每一层的名称
- 输入/输出张量的形状
- 不同精度下的内存使用
- 可用的精度选项
- 内存超限警告

### JSON输出

结果文件包含以下信息：
- 模型类型
- 输入张量形状
- 内存阈值
- 每层可用的精度选项

## 注意事项

1. 内存阈值应考虑实际硬件限制
2. 批次大小会显著影响张量大小
3. 某些层可能在所有精度下都超出内存限制
4. 建议先用较大的阈值测试，再逐步调整

## 使用场景

1. 模型量化前的可行性分析
2. 硬件兼容性评估
3. 内存优化方案设计
4. 量化策略制定 