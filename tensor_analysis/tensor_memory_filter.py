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
from models.mobilenetv3 import AdaptQMobileNetV3
from models.efficientnet import AdaptQEfficientNet

@dataclass
class TensorInfo:
    """张量信息类"""
    shape: Tuple[int, ...]
    precision: str
    size_kb: float  # 改为KB
    layer_name: str
    is_input: bool = True

def get_precision_bits(precision: str) -> int:
    """获取精度对应的位数"""
    if precision == "fp32":
        return 32
    elif precision == "fp16":
        return 16
    elif precision.startswith("int"):
        return int(precision.replace("int", ""))
    else:
        raise ValueError(f"Unsupported precision: {precision}")

def calculate_tensor_size_kb(shape: Tuple[int, ...], precision: str) -> float:
    """计算张量大小（KB）"""
    num_elements = np.prod(shape)
    bits_per_element = get_precision_bits(precision)
    bytes_size = int(math.ceil((num_elements * bits_per_element) / 8))
    return bytes_size / 1024  # 转换为KB

class TensorAnalyzer:
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...],
                 precision_options: List[str], memory_threshold_kb: float):
        self.model = model
        self.input_shape = input_shape
        self.precision_options = precision_options
        self.memory_threshold_kb = memory_threshold_kb
        self.tensor_info: Dict[str, List[TensorInfo]] = {}
        self.filtered_precisions: Dict[str, List[str]] = {}
        
    def _register_hooks(self):
        """注册钩子以收集张量信息"""
        def hook_fn(name: str):
            def hook(module: nn.Module, input_tensor: Tuple[torch.Tensor, ...],
                    output_tensor: torch.Tensor):
                if name not in self.tensor_info:
                    self.tensor_info[name] = []
                
                # 记录输入张量信息
                if isinstance(input_tensor, tuple):
                    input_shape = input_tensor[0].shape
                else:
                    input_shape = input_tensor.shape
                
                self.tensor_info[name].append(
                    TensorInfo(shape=input_shape,
                             precision="fp32",  # 临时精度，后续会针对不同精度计算
                             size_kb=0.0,  # 临时大小，后续会更新
                             layer_name=name,
                             is_input=True)
                )
                
                # 记录输出张量信息
                self.tensor_info[name].append(
                    TensorInfo(shape=output_tensor.shape,
                             precision="fp32",
                             size_kb=0.0,
                             layer_name=name,
                             is_input=False)
                )
            return hook
        
        # 为所有卷积和线性层注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.register_forward_hook(hook_fn(name))
    
    def analyze_tensors(self):
        """分析模型中的张量大小"""
        # 设置为评估模式
        self.model.eval()
        
        self._register_hooks()
        
        try:
            # 执行一次前向传播以触发钩子
            with torch.no_grad():
                dummy_input = torch.randn(self.input_shape)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                    self.model = self.model.cuda()
                self.model(dummy_input)
        except Exception as e:
            print(f"前向传播过程中出现错误: {str(e)}")
            print("尝试调整输入大小或检查模型配置")
            raise
        
        # 计算每层在不同精度下的张量大小
        for layer_name, tensors in self.tensor_info.items():
            self.filtered_precisions[layer_name] = []
            
            for precision in self.precision_options:
                max_tensor_size_kb = 0.0
                
                # 更新张量大小信息
                for tensor in tensors:
                    tensor.precision = precision
                    tensor.size_kb = calculate_tensor_size_kb(tensor.shape, precision)
                    max_tensor_size_kb = max(max_tensor_size_kb, tensor.size_kb)
                
                # 如果最大张量大小在阈值内，则保留该精度选项
                if max_tensor_size_kb <= self.memory_threshold_kb:
                    self.filtered_precisions[layer_name].append(precision)
    
    def get_filtered_precisions(self) -> Dict[str, List[str]]:
        """获取筛选后的精度选项"""
        return self.filtered_precisions
    
    def get_tensor_info(self) -> Dict[str, List[TensorInfo]]:
        """获取张量信息"""
        return self.tensor_info
    
    def print_analysis(self):
        """打印分析结果"""
        print("\n张量内存分析报告")
        print("=" * 80)
        print(f"内存阈值: {self.memory_threshold_kb:.2f} KB")
        print("=" * 80)
        
        for layer_name, tensors in self.tensor_info.items():
            print(f"\n层名称: {layer_name}")
            print("-" * 40)
            
            # 打印张量信息
            for tensor in tensors:
                tensor_type = "输入" if tensor.is_input else "输出"
                print(f"{tensor_type}张量:")
                print(f"  形状: {tensor.shape}")
                
                # 打印不同精度下的大小
                for precision in self.precision_options:
                    size_kb = calculate_tensor_size_kb(tensor.shape, precision)
                    status = "通过" if size_kb <= self.memory_threshold_kb else "超出限制"
                    print(f"  {precision}: {size_kb:.2f} KB ({status})")
            
            # 打印可用精度选项
            allowed_precisions = self.filtered_precisions[layer_name]
            if allowed_precisions:
                print(f"\n可用精度选项: {', '.join(allowed_precisions)}")
            else:
                print("\n警告: 该层在任何精度下都超出内存限制!")
        
        print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="模型张量内存分析工具")
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["mobilenetv2", "mobilenetv3", "efficientnet"],
                       help="模型类型")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--input_size", type=int, default=84,
                       help="输入图像大小")
    parser.add_argument("--memory_threshold", type=float, default=512.0,
                       help="内存阈值（KB）")
    parser.add_argument("--output_file", type=str, default="tensor_analysis_results.json",
                       help="结果输出文件")
    parser.add_argument("--mode", type=str, default="small",
                       choices=["large", "small"],
                       help="MobileNetV3的模式")
    parser.add_argument("--model_version", type=str, default="b0",
                       help="EfficientNet的版本")
    args = parser.parse_args()
    
    # 创建模型
    if args.model_type == "mobilenetv2":
        model = AdaptQMobileNetV2(
            num_classes=100,
            width_mult=1.0,
            precision_options=["fp32"],
            hardware_constraints=None,
            pretrain_mode=False  # 设置为False以避免训练模式
        )
    elif args.model_type == "mobilenetv3":
        model = AdaptQMobileNetV3(
            mode=args.mode,
            num_classes=100,
            width_mult=1.0,
            precision_options=["fp32"],
            hardware_constraints=None,
            pretrain_mode=False  # 设置为False以避免训练模式
        )
    else:  # efficientnet
        model = AdaptQEfficientNet(
            model_name=args.model_version,
            num_classes=100,
            precision_options=["fp32"],
            hardware_constraints=None,
            pretrain_mode=False  # 设置为False以避免训练模式
        )
    
    # 设置输入形状
    input_shape = (args.batch_size, 3, args.input_size, args.input_size)
    
    # 设置精度选项
    precision_options = ["fp32", "fp16", "int8", "int4", "int2"]
    
    try:
        # 创建分析器
        analyzer = TensorAnalyzer(
            model=model,
            input_shape=input_shape,
            precision_options=precision_options,
            memory_threshold_kb=args.memory_threshold
        )
        
        # 执行分析
        analyzer.analyze_tensors()
        
        # 打印分析结果
        analyzer.print_analysis()
        
        # 保存结果
        results = {
            "model_type": args.model_type,
            "input_shape": list(input_shape),  # 转换为list以便JSON序列化
            "memory_threshold_kb": args.memory_threshold,
            "filtered_precisions": analyzer.get_filtered_precisions()
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n分析结果已保存到: {args.output_file}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请检查输入参数和模型配置是否正确")
        sys.exit(1)

if __name__ == "__main__":
    main() 