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
import itertools

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

@dataclass
class DualCoreTensorInfo:
    """双核张量信息类"""
    shape: Tuple[int, ...]
    precision1: str
    precision2: str
    size_kb: float  # 双核组合的总大小
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

def calculate_dual_core_tensor_size_kb(shape: Tuple[int, ...], precision1: str, precision2: str) -> float:
    """计算双核张量大小（KB）- 两个精度的总和"""
    size1 = calculate_tensor_size_kb(shape, precision1)
    size2 = calculate_tensor_size_kb(shape, precision2)
    return size1 + size2

class DualCoreTensorAnalyzer:
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...],
                 precision_options: List[str], memory_threshold_kb: float):
        self.model = model
        self.input_shape = input_shape
        self.precision_options = precision_options
        self.memory_threshold_kb = memory_threshold_kb
        self.tensor_info: Dict[str, List[TensorInfo]] = {}
        self.dual_core_tensor_info: Dict[str, List[DualCoreTensorInfo]] = {}
        self.filtered_precisions: Dict[str, List[str]] = {}
        self.filtered_dual_core_options: Dict[str, List[str]] = {}
        
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
    
    def _create_dual_core_options(self) -> List[str]:
        """创建双核选项：纯精度组合"""
        # 双精度组合：C(5,2) = 10种组合
        dual_combinations = []
        for i in range(len(self.precision_options)):
            for j in range(i+1, len(self.precision_options)):
                dual_combinations.append(f"{self.precision_options[i]}+{self.precision_options[j]}")
        
        # 总共10个双核选项
        return dual_combinations
    
    def analyze_tensors(self):
        """分析模型中的张量大小（单核和双核）"""
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
        
        # 创建双核选项
        dual_core_options = self._create_dual_core_options()
        
        # 分析每层的单核和双核内存需求
        for layer_name, tensors in self.tensor_info.items():
            self.filtered_precisions[layer_name] = []
            self.filtered_dual_core_options[layer_name] = []
            self.dual_core_tensor_info[layer_name] = []
            
            # 分析单核精度选项
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
            
            # 分析双核选项（纯精度组合）
            for option in dual_core_options:
                # 双精度选项：检查两个精度的组合
                prec1, prec2 = option.split('+')
                max_dual_tensor_size_kb = 0.0
                
                # 计算双核组合的最大张量大小
                for tensor in tensors:
                    dual_size = calculate_dual_core_tensor_size_kb(tensor.shape, prec1, prec2)
                    max_dual_tensor_size_kb = max(max_dual_tensor_size_kb, dual_size)
                    
                    # 记录双核张量信息
                    self.dual_core_tensor_info[layer_name].append(
                        DualCoreTensorInfo(
                            shape=tensor.shape,
                            precision1=prec1,
                            precision2=prec2,
                            size_kb=dual_size,
                            layer_name=layer_name,
                            is_input=tensor.is_input
                        )
                    )
                
                # 如果双核组合在阈值内，则保留该选项
                if max_dual_tensor_size_kb <= self.memory_threshold_kb:
                    self.filtered_dual_core_options[layer_name].append(option)
    
    def get_filtered_precisions(self) -> Dict[str, List[str]]:
        """获取筛选后的单核精度选项"""
        return self.filtered_precisions
    
    def get_filtered_dual_core_options(self) -> Dict[str, List[str]]:
        """获取筛选后的双核选项"""
        return self.filtered_dual_core_options
    
    def get_tensor_info(self) -> Dict[str, List[TensorInfo]]:
        """获取单核张量信息"""
        return self.tensor_info
    
    def get_dual_core_tensor_info(self) -> Dict[str, List[DualCoreTensorInfo]]:
        """获取双核张量信息"""
        return self.dual_core_tensor_info
    
    def print_analysis(self):
        """打印分析结果"""
        print("\n双核张量内存分析报告")
        print("=" * 100)
        print(f"内存阈值: {self.memory_threshold_kb:.2f} KB")
        print("=" * 100)
        
        for layer_name, tensors in self.tensor_info.items():
            print(f"\n层名称: {layer_name}")
            print("-" * 60)
            
            # 打印单核张量信息
            print("单核张量分析:")
            for tensor in tensors:
                tensor_type = "输入" if tensor.is_input else "输出"
                print(f"  {tensor_type}张量形状: {tensor.shape}")
                
                # 打印不同精度下的大小
                for precision in self.precision_options:
                    size_kb = calculate_tensor_size_kb(tensor.shape, precision)
                    status = "通过" if size_kb <= self.memory_threshold_kb else "超出限制"
                    print(f"    {precision}: {size_kb:.2f} KB ({status})")
            
            # 打印单核可用精度选项
            allowed_precisions = self.filtered_precisions[layer_name]
            if allowed_precisions:
                print(f"\n单核可用精度选项: {', '.join(allowed_precisions)}")
            else:
                print("\n警告: 该层在任何单核精度下都超出内存限制!")
            
            # 打印双核张量信息
            print("\n双核张量分析:")
            dual_tensors = self.dual_core_tensor_info.get(layer_name, [])
            for dual_tensor in dual_tensors:
                tensor_type = "输入" if dual_tensor.is_input else "输出"
                print(f"  {tensor_type}张量形状: {dual_tensor.shape}")
                print(f"    组合: {dual_tensor.precision1} + {dual_tensor.precision2}")
                print(f"    总大小: {dual_tensor.size_kb:.2f} KB")
            
            # 打印双核可用选项
            allowed_dual_options = self.filtered_dual_core_options[layer_name]
            if allowed_dual_options:
                print(f"\n双核可用选项: {', '.join(allowed_dual_options)}")
            else:
                print("\n警告: 该层在任何双核组合下都超出内存限制!")
        
        print("\n" + "=" * 100)
    
    def print_summary(self):
        """打印总结报告"""
        print("\n双核内存分析总结")
        print("=" * 80)
        
        total_layers = len(self.tensor_info)
        layers_with_single_options = sum(1 for options in self.filtered_precisions.values() if options)
        layers_with_dual_options = sum(1 for options in self.filtered_dual_core_options.values() if options)
        
        print(f"总层数: {total_layers}")
        print(f"支持单核的层数: {layers_with_single_options}")
        print(f"支持双核的层数: {layers_with_dual_options}")
        
        # 统计双核选项分布（纯精度组合）
        dual_option_counts = {}
        for layer_name, options in self.filtered_dual_core_options.items():
            for option in options:
                if option not in dual_option_counts:
                    dual_option_counts[option] = 0
                dual_option_counts[option] += 1
        
        print(f"\n双核精度组合支持统计:")
        for option, count in sorted(dual_option_counts.items()):
            print(f"  {option}: {count} 层支持")
        
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="双核模型张量内存分析工具")
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["mobilenetv2", "mobilenetv3", "efficientnet"],
                       help="模型类型")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批次大小")
    parser.add_argument("--input_size", type=int, default=84,
                       help="输入图像大小")
    parser.add_argument("--memory_threshold", type=float, default=512.0,
                       help="内存阈值（KB）")
    parser.add_argument("--output_file", type=str, default="dual_core_tensor_analysis_results.json",
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
        analyzer = DualCoreTensorAnalyzer(
            model=model,
            input_shape=input_shape,
            precision_options=precision_options,
            memory_threshold_kb=args.memory_threshold
        )
        
        # 执行分析
        analyzer.analyze_tensors()
        
        # 打印分析结果
        analyzer.print_analysis()
        analyzer.print_summary()
        
        # 保存结果
        results = {
            "model_type": args.model_type,
            "input_shape": list(input_shape),  # 转换为list以便JSON序列化
            "memory_threshold_kb": args.memory_threshold,
            "filtered_precisions": analyzer.get_filtered_precisions(),
            "filtered_dual_core_options": analyzer.get_filtered_dual_core_options()
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