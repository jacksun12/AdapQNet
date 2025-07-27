import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.mobilenetv2 import AdaptQMobileNetV2

# 精度对应的延迟系数（单位：ms/2M MACs，基于实际硬件性能）
# 这些系数表示每2M MACs的延迟时间（毫秒）
OP_PRECISION_LATENCY_COEF = {
    "Conv": {
        "fp32": 100,     # 2M MACs = 1ms (基准)
        "fp16": 50,     # fp16约为fp32的一半
        "int8": 28,    # int8约为fp32的1/50
        "int4": 17.4,   # int4约为fp32的1/67
        "int2": 11     # int2约为fp32的1/100
    },
    "DWConv": {
        "fp32": 200,     # 深度卷积通常比标准卷积快
        "fp16": 100,     # fp16约为fp32的一半
        "int8": 46,   # int8约为fp32的1/50
        "int4": 27,   # int4约为fp32的1/67
        "int2": 16.8    # int2约为fp32的1/100
    },
    "Linear": {
        "fp32": 50,     # 线性层通常比卷积层稍慢
        "fp16": 25,    # fp16约为fp32的一半
        "int8": 14,   # int8约为fp32的1/50
        "int4": 8.7,   # int4约为fp32的1/67
        "int2": 5.5    # int2约为fp32的1/100
    }
}

@dataclass
class LayerLatencyInfo:
    """层延迟信息"""
    layer_name: str
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    mmacs: float
    precision: str
    latency_ns: float
    latency_ms: float

def calc_conv2d_mmacs(module: nn.Conv2d, input_shape: Tuple[int, ...]) -> int:
    """计算Conv2d的MMACs（乘累加运算次数）"""
    batch, in_c, in_h, in_w = input_shape
    out_c = module.out_channels
    k_h, k_w = module.kernel_size
    stride_h, stride_w = module.stride
    pad_h, pad_w = module.padding
    groups = module.groups
    
    # 计算输出尺寸
    out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1
    
    # 计算MMACs（乘累加运算次数）
    if groups == 1:
        # 标准卷积
        mmacs = k_h * k_w * in_c * out_c * out_h * out_w
    else:
        # 分组卷积（包括深度卷积）
        mmacs = k_h * k_w * in_c * out_c * out_h * out_w // groups
    
    return mmacs

def calc_linear_mmacs(module: nn.Linear, input_shape: Tuple[int, ...]) -> int:
    """计算Linear层的MMACs（乘累加运算次数）"""
    batch = input_shape[0]
    in_features = module.in_features
    out_features = module.out_features
    return batch * in_features * out_features

def get_op_type(module: nn.Module) -> str:
    """获取操作类型"""
    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels and module.in_channels == module.out_channels:
            return "DWConv"
        else:
            return "Conv"
    elif isinstance(module, nn.Linear):
        return "Linear"
    else:
        return None

def get_layer_precision(module: nn.Module) -> str:
    """获取层的实际精度"""
    # 检查模块是否有精度属性
    if hasattr(module, 'precision'):
        return module.precision
    elif hasattr(module, 'current_precision'):
        return module.current_precision
    elif hasattr(module, 'get_precision'):
        return module.get_precision()
    else:
        # 默认精度
        return "fp32"

class LatencyPredictor:
    """延迟预测器"""
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...], 
                 precision_mapping: Optional[Dict[str, str]] = None):
        self.model = model
        self.input_shape = input_shape
        self.precision_mapping = precision_mapping or {}  # 层名到精度的映射
        self.layer_latency: Dict[str, LayerLatencyInfo] = {}
        self.total_latency_ns = 0.0
        self.handles = []

    def set_precision_mapping(self, precision_mapping: Dict[str, str]):
        """设置精度映射"""
        self.precision_mapping = precision_mapping

    def _register_hooks(self):
        """注册钩子以收集层信息"""
        def hook_fn(name: str):
            def hook(module, input_tensor, output_tensor):
                op_type = get_op_type(module)
                if op_type is None:
                    return  # 只统计支持的层类型
                
                # 获取输入输出形状
                if isinstance(input_tensor, tuple):
                    in_shape = input_tensor[0].shape
                else:
                    in_shape = input_tensor.shape
                out_shape = output_tensor.shape
                
                # 计算MMACs
                if isinstance(module, nn.Conv2d):
                    mmacs = calc_conv2d_mmacs(module, in_shape)
                elif isinstance(module, nn.Linear):
                    mmacs = calc_linear_mmacs(module, in_shape)
                else:
                    mmacs = 0
                
                # 获取层的实际精度
                precision = self.precision_mapping.get(name, get_layer_precision(module))
                
                # 计算延迟（基于2M MACs的系数）
                coef = OP_PRECISION_LATENCY_COEF[op_type].get(precision, None)
                if coef is None:
                    print(f"警告: 层 {name} 的精度 {precision} 不支持，使用fp32")
                    precision = "fp32"
                    coef = OP_PRECISION_LATENCY_COEF[op_type]["fp32"]
                
                # 延迟计算：MMACs / (2M) * 系数 = 延迟(ms)
                latency_ms = (mmacs / (2e6)) * coef
                latency_ns = latency_ms * 1e6  # 转换为纳秒
                
                info = LayerLatencyInfo(
                    layer_name=name,
                    layer_type=op_type,
                    input_shape=in_shape,
                    output_shape=out_shape,
                    mmacs=mmacs,
                    precision=precision,
                    latency_ns=latency_ns,
                    latency_ms=latency_ms
                )
                
                self.layer_latency[name] = info
                self.total_latency_ns += latency_ns
            return hook
        
        # 为所有支持的层注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(hook_fn(name))
                self.handles.append(handle)

    def analyze_latency(self):
        """分析模型延迟"""
        self.model.eval()
        self._register_hooks()
        
        try:
            with torch.no_grad():
                dummy_input = torch.randn(self.input_shape)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                    self.model = self.model.cuda()
                self.model(dummy_input)
        except Exception as e:
            print(f"前向传播过程中出现错误: {str(e)}")
            raise
        finally:
            # 移除钩子
            for handle in self.handles:
                handle.remove()

    def print_report(self):
        """打印延迟分析报告"""
        print("\n模型延迟分析报告")
        print("=" * 100)
        print(f"输入形状: {self.input_shape}")
        print("延迟计算: 延迟(ms) = MMACs / (2M) × 精度系数")
        print("精度系数: 每2M MACs的延迟时间（毫秒）")
        print("=" * 100)
        
        total_mmacs = 0
        precision_stats = {}
        
        # 按层打印延迟信息
        for layer_name, info in self.layer_latency.items():
            total_mmacs += info.mmacs
            
            # 统计精度分布
            if info.precision not in precision_stats:
                precision_stats[info.precision] = {"count": 0, "latency_ns": 0.0}
            precision_stats[info.precision]["count"] += 1
            precision_stats[info.precision]["latency_ns"] += info.latency_ns
            
            print(f"层: {info.layer_name:25s} | "
                  f"类型: {info.layer_type:8s} | "
                  f"精度: {info.precision:6s} | "
                  f"MMACs: {info.mmacs:10.2e} | "
                  f"延迟: {info.latency_ms:8.3f} ms")
        
        total_latency_ms = self.total_latency_ns / 1e6
        print(f"\n总MMACs: {total_mmacs:.2e}")
        print(f"总延迟: {total_latency_ms:.3f} ms ({self.total_latency_ns:.2e} ns)")
        
        # 打印精度统计
        print(f"\n精度分布:")
        for precision, stats in precision_stats.items():
            latency_ms = stats["latency_ns"] / 1e6
            print(f"  {precision}: {stats['count']} 层, {latency_ms:.3f} ms")
        
        print("=" * 100)

    def save_report(self, output_file: str):
        """保存延迟分析报告"""
        precision_stats = {}
        for info in self.layer_latency.values():
            if info.precision not in precision_stats:
                precision_stats[info.precision] = {"count": 0, "latency_ns": 0.0}
            precision_stats[info.precision]["count"] += 1
            precision_stats[info.precision]["latency_ns"] += info.latency_ns
        
        result = {
            "input_shape": list(self.input_shape),
            "precision_mapping": self.precision_mapping,
            "total_latency": {
                "latency_ns": self.total_latency_ns,
                "latency_ms": self.total_latency_ns / 1e6
            },
            "precision_stats": {
                precision: {
                    "count": stats["count"],
                    "latency_ns": stats["latency_ns"],
                    "latency_ms": stats["latency_ns"] / 1e6
                } for precision, stats in precision_stats.items()
            },
            "layer_details": {
                layer_name: {
                    "layer_type": info.layer_type,
                    "input_shape": list(info.input_shape),
                    "output_shape": list(info.output_shape),
                    "mmacs": info.mmacs,
                    "precision": info.precision,
                    "latency_ns": info.latency_ns,
                    "latency_ms": info.latency_ms
                } for layer_name, info in self.layer_latency.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n延迟分析结果已保存到: {output_file}")

    def get_latency_summary(self) -> Dict[str, float]:
        """获取延迟总结"""
        return {
            "latency_ns": self.total_latency_ns,
            "latency_ms": self.total_latency_ns / 1e6
        }

    def get_precision_distribution(self) -> Dict[str, Dict[str, float]]:
        """获取精度分布"""
        precision_stats = {}
        for info in self.layer_latency.values():
            if info.precision not in precision_stats:
                precision_stats[info.precision] = {"count": 0, "latency_ns": 0.0}
            precision_stats[info.precision]["count"] += 1
            precision_stats[info.precision]["latency_ns"] += info.latency_ns
        
        return {
            precision: {
                "count": stats["count"],
                "latency_ns": stats["latency_ns"],
                "latency_ms": stats["latency_ns"] / 1e6
            } for precision, stats in precision_stats.items()
        }


def main():
    parser = argparse.ArgumentParser(
        description="模型延迟分析工具（基于实际精度）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用模型默认精度
  python tensor_latency_predictor.py --model_type mobilenetv2 --input_size 224

  # 使用自定义精度映射文件
  python tensor_latency_predictor.py --model_type mobilenetv2 --input_size 224 \\
    --precision_mapping_file example_precision_mapping.json

  # 分析MobileNetV3
  python tensor_latency_predictor.py --model_type mobilenetv3 --mode large --input_size 224

  # 分析EfficientNet
  python tensor_latency_predictor.py --model_type efficientnet --model_version b0 --input_size 224

精度映射文件格式 (JSON):
  {
    "layer_name": "precision",
    "features.0.0": "fp32",
    "features.1.conv.0.0": "fp16",
    "features.1.conv.1": "int8",
    ...
  }

支持的精度: fp32, fp16, int8, int4, int2
        """
    )
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["mobilenetv2", "mobilenetv3", "efficientnet"],
                       help="模型类型")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--input_size", type=int, default=112,
                       help="输入图像大小（默认112x112）")
    parser.add_argument("--precision_mapping_file", type=str, default=None,
                       help="精度映射文件路径（JSON格式），如果不指定则使用模型默认精度")
    parser.add_argument("--output_file", type=str, default="latency_analysis_results.json",
                       help="结果输出文件")
    parser.add_argument("--mode", type=str, default="small",
                       choices=["large", "small"],
                       help="MobileNetV3的模式")
    parser.add_argument("--model_version", type=str, default="b0",
                       help="EfficientNet的版本")
    parser.add_argument("--show_help", action="store_true",
                       help="显示详细使用说明")
    args = parser.parse_args()
    
    if args.show_help:
        print("延迟预测器使用说明:")
        print("=" * 60)
        print("1. 基本用法:")
        print("   python tensor_latency_predictor.py --model_type mobilenetv2 --input_size 224")
        print()
        print("2. 使用精度映射文件:")
        print("   python tensor_latency_predictor.py --model_type mobilenetv2 --input_size 224 \\")
        print("     --precision_mapping_file my_precision.json")
        print()
        print("3. 精度映射文件格式 (JSON):")
        print("   {")
        print('     "layer_name": "precision",')
        print('     "features.0.0": "fp32",')
        print('     "features.1.conv.0.0": "fp16",')
        print('     "features.1.conv.1": "int8"')
        print("   }")
        print()
        print("4. 支持的精度: fp32, fp16, int8, int4, int2")
        print("5. 延迟计算公式: 延迟(ms) = MMACs / (2M) × 精度系数")
        print("6. 精度系数: 每2M MACs的延迟时间（毫秒）")
        print("7. 输出包含: 每层延迟、总延迟、精度分布统计")
        print("=" * 60)
        return
    
    try:
        # 加载精度映射
        precision_mapping = {}
        if args.precision_mapping_file:
            with open(args.precision_mapping_file, 'r') as f:
                precision_mapping = json.load(f)
            print(f"加载精度映射: {len(precision_mapping)} 层")
        
        # 创建模型
        if args.model_type == "mobilenetv2":
            model = AdaptQMobileNetV2(
                num_classes=100,
                width_mult=0.5,
                precision_options=["fp32", "fp16", "int8", "int4", "int2"],
                hardware_constraints=None,
                pretrain_mode=False
            )
        elif args.model_type == "mobilenetv3":
            model = AdaptQMobileNetV3(
                mode=args.mode,
                num_classes=100,
                width_mult=1.0,
                precision_options=["fp32", "fp16", "int8", "int4", "int2"],
                hardware_constraints=None,
                dropout=0.2,
                pretrain_mode=False
            )
        elif args.model_type == "efficientnet":
            model = AdaptQEfficientNet(
                model_name=args.model_version,
                num_classes=100,
                precision_options=["fp32", "fp16", "int8", "int4", "int2"],
                hardware_constraints=None,
                pretrain_mode=False
            )
        else:
            raise ValueError("不支持的模型类型")

        input_shape = (args.batch_size, 3, args.input_size, args.input_size)
        
        print(f"模型类型: {args.model_type}")
        print(f"输入形状: {input_shape}")
        print(f"精度映射: {precision_mapping if precision_mapping else '使用模型默认精度'}")
        
        # 创建延迟预测器
        predictor = LatencyPredictor(model, input_shape, precision_mapping)
        
        # 分析延迟
        print("\n开始分析模型延迟...")
        predictor.analyze_latency()
        
        # 打印报告
        predictor.print_report()
        
        # 保存报告
        predictor.save_report(args.output_file)
        
        # 打印总结
        summary = predictor.get_latency_summary()
        print(f"\n总延迟: {summary['latency_ms']:.3f} ms")
        
        # 打印精度分布
        precision_dist = predictor.get_precision_distribution()
        print(f"\n精度分布:")
        for precision, stats in precision_dist.items():
            print(f"  {precision}: {stats['count']} 层, {stats['latency_ms']:.3f} ms")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请检查输入参数和模型配置是否正确")
        sys.exit(1)

if __name__ == "__main__":
    main() 