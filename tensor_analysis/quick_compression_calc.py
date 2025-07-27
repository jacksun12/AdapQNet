#!/usr/bin/env python3
"""
快速计算压缩比的脚本
基于您提供的精度配置
"""

import json

def get_precision_bits(precision):
    """获取精度对应的位数"""
    if precision.startswith('fp'):
        return int(precision.replace('fp', ''))
    elif precision.startswith('int'):
        return int(precision.replace('int', ''))
    else:
        raise ValueError(f"Unknown precision: {precision}")

def calculate_compression_ratio():
    """计算您提供的精度配置的压缩比"""
    
    # 基于最新DNAS搜索结果的精度配置（从日志中提取）
    current_config = {
        "features.0.conv_bn_relu": "int2",
        "features.1.depthwise.conv_bn_relu": "int4",
        "features.1.project.conv_bn": "fp16",
        "features.2.expand.conv_bn_relu": "int8",
        "features.2.depthwise.conv_bn_relu": "int8",
        "features.2.project.conv_bn": "int8",
        "features.3.expand.conv_bn_relu": "int2",
        "features.3.depthwise.conv_bn_relu": "fp16",
        "features.3.project.conv_bn": "int8",
        "features.4.expand.conv_bn_relu": "int8",
        "features.4.depthwise.conv_bn_relu": "fp16",
        "features.4.project.conv_bn": "fp16",
        "features.5.expand.conv_bn_relu": "int4",
        "features.5.depthwise.conv_bn_relu": "int8",
        "features.5.project.conv_bn": "int4",
        "features.6.expand.conv_bn_relu": "int8",
        "features.6.depthwise.conv_bn_relu": "int4",
        "features.6.project.conv_bn": "fp16",
        "features.7.expand.conv_bn_relu": "int4",
        "features.7.depthwise.conv_bn_relu": "fp16",
        "features.7.project.conv_bn": "fp16",
        "features.8.expand.conv_bn_relu": "int4",
        "features.8.depthwise.conv_bn_relu": "int4",
        "features.8.project.conv_bn": "int4",
        "features.9.expand.conv_bn_relu": "int4",
        "features.9.depthwise.conv_bn_relu": "int8",
        "features.9.project.conv_bn": "int4",
        "features.10.expand.conv_bn_relu": "int4",
        "features.10.depthwise.conv_bn_relu": "int4",
        "features.10.project.conv_bn": "int4",
        "features.11.expand.conv_bn_relu": "int4",
        "features.11.depthwise.conv_bn_relu": "int8",
        "features.11.project.conv_bn": "int4",
        "features.12.expand.conv_bn_relu": "int4",
        "features.12.depthwise.conv_bn_relu": "int4",
        "features.12.project.conv_bn": "int4",
        "features.13.expand.conv_bn_relu": "int4",
        "features.13.depthwise.conv_bn_relu": "int4",
        "features.13.project.conv_bn": "int4",
        "features.14.expand.conv_bn_relu": "int4",
        "features.14.depthwise.conv_bn_relu": "fp16",
        "features.14.project.conv_bn": "int2",
        "features.15.expand.conv_bn_relu": "int4",
        "features.15.depthwise.conv_bn_relu": "int4",
        "features.15.project.conv_bn": "int4",
        "features.16.expand.conv_bn_relu": "int4",
        "features.16.depthwise.conv_bn_relu": "int4",
        "features.16.project.conv_bn": "int4",
        "features.17.expand.conv_bn_relu": "int4",
        "features.17.depthwise.conv_bn_relu": "int4",
        "features.17.project.conv_bn": "int4",
        "features.18.conv_bn_relu": "int4",
        "classifier.1": "fp16"
    }
    
    # 更激进的精度配置（目标：超过INT8的4x压缩比）
    aggressive_config = {
        "features.0.conv_bn_relu": "int8",      # 输入层可以用INT8
        "features.1.depthwise.conv_bn_relu": "int4",
        "features.1.project.conv_bn": "int4",
        "features.2.expand.conv_bn_relu": "int8",
        "features.2.depthwise.conv_bn_relu": "int4",
        "features.2.project.conv_bn": "int8",
        "features.3.expand.conv_bn_relu": "int8",
        "features.3.depthwise.conv_bn_relu": "int4",
        "features.3.project.conv_bn": "int4",
        "features.4.expand.conv_bn_relu": "int8",
        "features.4.depthwise.conv_bn_relu": "int4",
        "features.4.project.conv_bn": "int8",
        "features.5.expand.conv_bn_relu": "int8",
        "features.5.depthwise.conv_bn_relu": "int2",
        "features.5.project.conv_bn": "int4",
        "features.6.expand.conv_bn_relu": "int2",
        "features.6.depthwise.conv_bn_relu": "int4",
        "features.6.project.conv_bn": "int4",
        "features.7.expand.conv_bn_relu": "int8",
        "features.7.depthwise.conv_bn_relu": "int4",
        "features.7.project.conv_bn": "int8",
        "features.8.expand.conv_bn_relu": "int8",
        "features.8.depthwise.conv_bn_relu": "int4",
        "features.8.project.conv_bn": "int8",
        "features.9.expand.conv_bn_relu": "int2",
        "features.9.depthwise.conv_bn_relu": "int4",
        "features.9.project.conv_bn": "int4",
        "features.10.expand.conv_bn_relu": "int2",
        "features.10.depthwise.conv_bn_relu": "int4",
        "features.10.project.conv_bn": "int4",
        "features.11.expand.conv_bn_relu": "int8",
        "features.11.depthwise.conv_bn_relu": "int4",
        "features.11.project.conv_bn": "int2",
        "features.12.expand.conv_bn_relu": "int8",
        "features.12.depthwise.conv_bn_relu": "int4",
        "features.12.project.conv_bn": "int2",
        "features.13.expand.conv_bn_relu": "int2",
        "features.13.depthwise.conv_bn_relu": "int2",
        "features.13.project.conv_bn": "int4",
        "features.14.expand.conv_bn_relu": "int4",
        "features.14.depthwise.conv_bn_relu": "int4",
        "features.14.project.conv_bn": "int2",
        "features.15.expand.conv_bn_relu": "int2",
        "features.15.depthwise.conv_bn_relu": "int4",
        "features.15.project.conv_bn": "int4",
        "features.16.expand.conv_bn_relu": "int2",
        "features.16.depthwise.conv_bn_relu": "int4",
        "features.16.project.conv_bn": "int2",
        "features.17.expand.conv_bn_relu": "int4",
        "features.17.depthwise.conv_bn_relu": "int2",
        "features.17.project.conv_bn": "int2",
        "features.18.conv_bn_relu": "int4",
        "classifier.1": "int2"
    }
    
    # 选择要分析的配置
    configs = {
        "当前DNAS结果": current_config,
        "激进配置": aggressive_config
    }
    
    # MobileNetV2的层参数量（基于实际架构计算）
    layer_params = {
        # features.0: 3 -> 32, kernel=3x3, stride=2
        "features.0.conv_bn_relu": 896,  # 3*32*3*3 + 32 = 864 + 32 = 896
        
        # features.1: t=1, c=16, n=1, s=1
        # input: 32, output: 16, t=1 (no expansion)
        "features.1.depthwise.conv_bn_relu": 320,  # 32*3*3 + 32 = 288 + 32 = 320 (depthwise卷积，groups=32)
        "features.1.project.conv_bn": 528,  # 32*16*1*1 + 16 = 512 + 16 = 528
        
        # features.2: t=6, c=24, n=2, s=2 (第一个block)
        # input: 16, output: 24, t=6 (expansion: 16*6=96)
        "features.2.expand.conv_bn_relu": 1536,  # 16*96*1*1 + 96 = 1536 + 96 = 1632
        "features.2.depthwise.conv_bn_relu": 864,  # 96*3*3 + 96 = 864 + 96 = 960
        "features.2.project.conv_bn": 2304,  # 96*24*1*1 + 24 = 2304 + 24 = 2328
        
        # features.3: t=6, c=24, n=2, s=2 (第二个block)
        # input: 24, output: 24, t=6 (expansion: 24*6=144)
        "features.3.expand.conv_bn_relu": 3456,  # 24*144*1*1 + 144 = 3456 + 144 = 3600
        "features.3.depthwise.conv_bn_relu": 1296,  # 144*3*3 + 144 = 1296 + 144 = 1440
        "features.3.project.conv_bn": 3456,  # 144*24*1*1 + 24 = 3456 + 24 = 3480
        
        # features.4: t=6, c=32, n=3, s=2 (第一个block)
        # input: 24, output: 32, t=6 (expansion: 24*6=144)
        "features.4.expand.conv_bn_relu": 3456,  # 24*144*1*1 + 144 = 3456 + 144 = 3600
        "features.4.depthwise.conv_bn_relu": 1296,  # 144*3*3 + 144 = 1296 + 144 = 1440
        "features.4.project.conv_bn": 4608,  # 144*32*1*1 + 32 = 4608 + 32 = 4640
        
        # features.5: t=6, c=32, n=3, s=2 (第二个block)
        # input: 32, output: 32, t=6 (expansion: 32*6=192)
        "features.5.expand.conv_bn_relu": 6144,  # 32*192*1*1 + 192 = 6144 + 192 = 6336
        "features.5.depthwise.conv_bn_relu": 1728,  # 192*3*3 + 192 = 1728 + 192 = 1920
        "features.5.project.conv_bn": 6144,  # 192*32*1*1 + 32 = 6144 + 32 = 6176
        
        # features.6: t=6, c=32, n=3, s=2 (第三个block)
        # input: 32, output: 32, t=6 (expansion: 32*6=192)
        "features.6.expand.conv_bn_relu": 6144,  # 32*192*1*1 + 192 = 6144 + 192 = 6336
        "features.6.depthwise.conv_bn_relu": 1728,  # 192*3*3 + 192 = 1728 + 192 = 1920
        "features.6.project.conv_bn": 6144,  # 192*32*1*1 + 32 = 6144 + 32 = 6176
        
        # features.7: t=6, c=64, n=4, s=2 (第一个block)
        # input: 32, output: 64, t=6 (expansion: 32*6=192)
        "features.7.expand.conv_bn_relu": 6144,  # 32*192*1*1 + 192 = 6144 + 192 = 6336
        "features.7.depthwise.conv_bn_relu": 1728,  # 192*3*3 + 192 = 1728 + 192 = 1920
        "features.7.project.conv_bn": 12288,  # 192*64*1*1 + 64 = 12288 + 64 = 12352
        
        # features.8: t=6, c=64, n=4, s=2 (第二个block)
        # input: 64, output: 64, t=6 (expansion: 64*6=384)
        "features.8.expand.conv_bn_relu": 24576,  # 64*384*1*1 + 384 = 24576 + 384 = 24960
        "features.8.depthwise.conv_bn_relu": 3456,  # 384*3*3 + 384 = 3456 + 384 = 3840
        "features.8.project.conv_bn": 24576,  # 384*64*1*1 + 64 = 24576 + 64 = 24640
        
        # features.9: t=6, c=64, n=4, s=2 (第三个block)
        # input: 64, output: 64, t=6 (expansion: 64*6=384)
        "features.9.expand.conv_bn_relu": 24576,  # 64*384*1*1 + 384 = 24576 + 384 = 24960
        "features.9.depthwise.conv_bn_relu": 3456,  # 384*3*3 + 384 = 3456 + 384 = 3840
        "features.9.project.conv_bn": 24576,  # 384*64*1*1 + 64 = 24576 + 64 = 24640
        
        # features.10: t=6, c=64, n=4, s=2 (第四个block)
        # input: 64, output: 64, t=6 (expansion: 64*6=384)
        "features.10.expand.conv_bn_relu": 24576,  # 64*384*1*1 + 384 = 24576 + 384 = 24960
        "features.10.depthwise.conv_bn_relu": 3456,  # 384*3*3 + 384 = 3456 + 384 = 3840
        "features.10.project.conv_bn": 36864,  # 384*96*1*1 + 96 = 36864 + 96 = 36960
        
        # features.11: t=6, c=96, n=3, s=1 (第一个block)
        # input: 96, output: 96, t=6 (expansion: 96*6=576)
        "features.11.expand.conv_bn_relu": 55296,  # 96*576*1*1 + 576 = 55296 + 576 = 55872
        "features.11.depthwise.conv_bn_relu": 5184,  # 576*3*3 + 576 = 5184 + 576 = 5760
        "features.11.project.conv_bn": 55296,  # 576*96*1*1 + 96 = 55296 + 96 = 55392
        
        # features.12: t=6, c=96, n=3, s=1 (第二个block)
        # input: 96, output: 96, t=6 (expansion: 96*6=576)
        "features.12.expand.conv_bn_relu": 55296,  # 96*576*1*1 + 576 = 55296 + 576 = 55872
        "features.12.depthwise.conv_bn_relu": 5184,  # 576*3*3 + 576 = 5184 + 576 = 5760
        "features.12.project.conv_bn": 55296,  # 576*96*1*1 + 96 = 55296 + 96 = 55392
        
        # features.13: t=6, c=96, n=3, s=1 (第三个block)
        # input: 96, output: 96, t=6 (expansion: 96*6=576)
        "features.13.expand.conv_bn_relu": 55296,  # 96*576*1*1 + 576 = 55296 + 576 = 55872
        "features.13.depthwise.conv_bn_relu": 5184,  # 576*3*3 + 576 = 5184 + 576 = 5760
        "features.13.project.conv_bn": 92160,  # 576*160*1*1 + 160 = 92160 + 160 = 92320
        
        # features.14: t=6, c=160, n=3, s=2 (第一个block)
        # input: 160, output: 160, t=6 (expansion: 160*6=960)
        "features.14.expand.conv_bn_relu": 153600,  # 160*960*1*1 + 960 = 153600 + 960 = 154560
        "features.14.depthwise.conv_bn_relu": 8640,  # 960*3*3 + 960 = 8640 + 960 = 9600
        "features.14.project.conv_bn": 153600,  # 960*160*1*1 + 160 = 153600 + 160 = 153760
        
        # features.15: t=6, c=160, n=3, s=2 (第二个block)
        # input: 160, output: 160, t=6 (expansion: 160*6=960)
        "features.15.expand.conv_bn_relu": 153600,  # 160*960*1*1 + 960 = 153600 + 960 = 154560
        "features.15.depthwise.conv_bn_relu": 8640,  # 960*3*3 + 960 = 8640 + 960 = 9600
        "features.15.project.conv_bn": 153600,  # 960*160*1*1 + 160 = 153600 + 160 = 153760
        
        # features.16: t=6, c=160, n=3, s=2 (第三个block)
        # input: 160, output: 160, t=6 (expansion: 160*6=960)
        "features.16.expand.conv_bn_relu": 153600,  # 160*960*1*1 + 960 = 153600 + 960 = 154560
        "features.16.depthwise.conv_bn_relu": 8640,  # 960*3*3 + 960 = 8640 + 960 = 9600
        "features.16.project.conv_bn": 307200,  # 960*320*1*1 + 320 = 307200 + 320 = 307520
        
        # features.17: t=6, c=320, n=1, s=1 (只有一个block)
        # input: 320, output: 320, t=6 (expansion: 320*6=1920)
        "features.17.expand.conv_bn_relu": 614400,  # 320*1920*1*1 + 1920 = 614400 + 1920 = 616320
        "features.17.depthwise.conv_bn_relu": 17280,  # 1920*3*3 + 1920 = 17280 + 1920 = 19200
        "features.17.project.conv_bn": 614400,  # 1920*320*1*1 + 320 = 614400 + 320 = 614720
        
        # features.18: 320 -> 1280, kernel=1x1
        "features.18.conv_bn_relu": 409600,  # 320*1280*1*1 + 1280 = 409600 + 1280 = 410880
        
        # classifier: 1280 -> 10 (CIFAR-10)
        "classifier.1": 12810,  # 1280*10 + 10 = 12800 + 10 = 12810
    }
    
    for config_name, precision_config in configs.items():
        print(f"\n{'='*80}")
        print(f"{config_name} - 模型压缩比分析结果")
        print(f"{'='*80}")
        
        total_fp32_params = 0
        total_quantized_params = 0
        precision_counts = {}
        
        print(f"{'层名称':<35} {'精度':<8} {'参数量':<10} {'FP32大小(bits)':<15} {'量化大小(bits)':<15} {'压缩比':<10}")
        print("-" * 80)
        
        for layer_name, precision in precision_config.items():
            if layer_name in layer_params:
                param_count = layer_params[layer_name]
                fp32_size = param_count * 32
                quantized_bits = get_precision_bits(precision)
                quantized_size = param_count * quantized_bits
                
                total_fp32_params += fp32_size
                total_quantized_params += quantized_size
                
                compression_ratio = fp32_size / quantized_size if quantized_size > 0 else 1.0
                
                print(f"{layer_name:<35} {precision:<8} {param_count:<10} {fp32_size:<15} {quantized_size:<15} {compression_ratio:<10.2f}x")
                
                # 统计精度分布
                if precision not in precision_counts:
                    precision_counts[precision] = 0
                precision_counts[precision] += 1
        
        overall_compression_ratio = total_fp32_params / total_quantized_params if total_quantized_params > 0 else 1.0
        
        print("-" * 80)
        print(f"总体压缩比: {overall_compression_ratio:.2f}x")
        print(f"FP32模型大小: {total_fp32_params / 8 / 1024 / 1024:.2f} MB")
        print(f"量化后模型大小: {total_quantized_params / 8 / 1024 / 1024:.2f} MB")
        print(f"大小减少: {(total_fp32_params - total_quantized_params) / 8 / 1024 / 1024:.2f} MB")
        print(f"大小减少百分比: {(1 - total_quantized_params / total_fp32_params) * 100:.1f}%")
        
        print("\n精度分布:")
        for precision, count in sorted(precision_counts.items()):
            print(f"  {precision}: {count} 层")
        
        # 保存结果
        result = {
            'config_name': config_name,
            'overall_compression_ratio': overall_compression_ratio,
            'total_fp32_size_mb': total_fp32_params / 8 / 1024 / 1024,
            'total_quantized_size_mb': total_quantized_params / 8 / 1024 / 1024,
            'size_reduction_mb': (total_fp32_params - total_quantized_params) / 8 / 1024 / 1024,
            'size_reduction_percentage': (1 - total_quantized_params / total_fp32_params) * 100,
            'precision_distribution': precision_counts
        }
        
        with open(f'compression_result_{config_name.replace(" ", "_")}.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n结果已保存到: compression_result_{config_name.replace(' ', '_')}.json")
    
    total_fp32_params = 0
    total_quantized_params = 0
    precision_counts = {}
    
    print("=" * 80)
    print("模型压缩比分析结果")
    print("=" * 80)
    print(f"{'层名称':<35} {'精度':<8} {'参数量':<10} {'FP32大小(bits)':<15} {'量化大小(bits)':<15} {'压缩比':<10}")
    print("-" * 80)
    
    for layer_name, precision in precision_config.items():
        if layer_name in layer_params:
            param_count = layer_params[layer_name]
            fp32_size = param_count * 32
            quantized_bits = get_precision_bits(precision)
            quantized_size = param_count * quantized_bits
            
            total_fp32_params += fp32_size
            total_quantized_params += quantized_size
            
            compression_ratio = fp32_size / quantized_size if quantized_size > 0 else 1.0
            
            print(f"{layer_name:<35} {precision:<8} {param_count:<10} {fp32_size:<15} {quantized_size:<15} {compression_ratio:<10.2f}x")
            
            # 统计精度分布
            if precision not in precision_counts:
                precision_counts[precision] = 0
            precision_counts[precision] += 1
    
    overall_compression_ratio = total_fp32_params / total_quantized_params if total_quantized_params > 0 else 1.0
    
    print("-" * 80)
    print(f"总体压缩比: {overall_compression_ratio:.2f}x")
    print(f"FP32模型大小: {total_fp32_params / 8 / 1024 / 1024:.2f} MB")
    print(f"量化后模型大小: {total_quantized_params / 8 / 1024 / 1024:.2f} MB")
    print(f"大小减少: {(total_fp32_params - total_quantized_params) / 8 / 1024 / 1024:.2f} MB")
    print(f"大小减少百分比: {(1 - total_quantized_params / total_fp32_params) * 100:.1f}%")
    
    print("\n精度分布:")
    for precision, count in sorted(precision_counts.items()):
        print(f"  {precision}: {count} 层")
    
    # 保存结果
    result = {
        'overall_compression_ratio': overall_compression_ratio,
        'total_fp32_size_mb': total_fp32_params / 8 / 1024 / 1024,
        'total_quantized_size_mb': total_quantized_params / 8 / 1024 / 1024,
        'size_reduction_mb': (total_fp32_params - total_quantized_params) / 8 / 1024 / 1024,
        'size_reduction_percentage': (1 - total_quantized_params / total_fp32_params) * 100,
        'precision_distribution': precision_counts
    }
    
    with open('compression_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存到: compression_result.json")

if __name__ == '__main__':
    calculate_compression_ratio() 