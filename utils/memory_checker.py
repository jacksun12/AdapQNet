import torch
from typing import Dict, List, Tuple
from models.quantization import QuantizedConvBNReLU
import numpy as np

class MemoryChecker:
    def __init__(self, model: torch.nn.Module, input_shape: tuple):
        """
        Args:
            model: 要分析的模型
            input_shape: (N, C, H, W) - N应该是1，用于推理
        """
        # 强制将batch_size设为1用于内存分析
        self.model = model
        self.input_shape = (1, input_shape[1], input_shape[2], input_shape[3])
        self.feature_shapes = {}
        self._trace_feature_shapes()

    def _trace_feature_shapes(self):
        """追踪每一层的特征图尺寸"""
        N, C, H, W = self.input_shape
        current_shape = (C, H, W)  # 不包含batch维度
        
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizedConvBNReLU):
                self.feature_shapes[name] = current_shape
                # 更新下一层的输入shape
                C_in, H_in, W_in = current_shape
                H_out = ((H_in + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0]) + 1
                W_out = ((W_in + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[1]) + 1
                current_shape = (module.conv.out_channels, H_out, W_out)

    def calculate_layer_ram(self, name, module, bits):
        """计算单个层的RAM使用量（bytes）"""
        if name not in self.feature_shapes:
            return 0
            
        N = self.input_shape[0]  # batch_size
        C_in, H_in, W_in = self.feature_shapes[name]
        
        # 计算输出特征图大小
        H_out = ((H_in + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0]) + 1
        W_out = ((W_in + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[1]) + 1
        
        # 1. 输入特征图
        input_size = N * C_in * H_in * W_in * bits / 8
        
        # 2. 输出特征图
        output_size = N * module.conv.out_channels * H_out * W_out * bits / 8
        
        # 3. 中间特征图 (如果是扩展层)
        if hasattr(module, 'expand_ratio'):
            expanded_size = N * (C_in * module.expand_ratio) * H_in * W_in * bits / 8
            return max(input_size, expanded_size, output_size)
        
        return max(input_size, output_size)

    def check_ram_constraints(self, threshold_bytes):
        """检查RAM约束"""
        max_ram = 0
        ram_per_layer = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizedConvBNReLU):
                # 获取当前精度
                precision_idx = torch.argmax(module.alpha)
                precision = module.precision_options[precision_idx]
                bits = {
                    'fp32': 32,
                    'fp16': 16,
                    'int8': 8,
                    'int4': 4,
                    'int2': 2,
                    'int1': 1
                }[precision]
                
                ram_usage = self.calculate_layer_ram(name, module, bits)
                ram_per_layer[name] = ram_usage
                max_ram = max(max_ram, ram_usage)
        
        return max_ram <= threshold_bytes, {
            'max_ram': max_ram,
            'per_layer_ram': ram_per_layer
        }

    def check_flash_constraints(self, threshold_mb):
        """检查Flash约束"""
        flash_usage = 0
        flash_info = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizedConvBNReLU):
                # 获取当前精度
                precision_idx = torch.argmax(module.alpha)
                precision = module.precision_options[precision_idx]
                bits = {
                    'fp32': 32,
                    'fp16': 16,
                    'int8': 8,
                    'int4': 4,
                    'int2': 2,
                    'int1': 1
                }[precision]
                
                params_size = sum(p.numel() * bits / 8 for p in module.parameters())
                flash_usage_mb = params_size / (1024 * 1024)
                flash_usage += flash_usage_mb
                flash_info[name] = bits
        
        return flash_usage <= threshold_mb, {
            'total_flash': flash_usage,
            'per_layer_bits': flash_info
        }

    def get_feasible_precisions(self, ram_threshold_bytes):
        """获取每层可行的精度数量"""
        feasible_precisions = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizedConvBNReLU):
                feasible_count = 0
                for precision in module.precision_options:
                    bits = {
                        'fp32': 32,
                        'fp16': 16,
                        'int8': 8,
                        'int4': 4,
                        'int2': 2,
                        'int1': 1
                    }[precision]
                    
                    ram_usage = self.calculate_layer_ram(name, module, bits)
                    if ram_usage <= ram_threshold_bytes:
                        feasible_count += 1
                
                feasible_precisions[name] = feasible_count
        
        return feasible_precisions

    def print_memory_analysis(self, flash_threshold_mb, ram_threshold_mb):
        """打印内存分析报告"""
        print("\n=== Memory Analysis Report ===\n")
        
        # RAM分析
        ram_ok, ram_info = self.check_ram_constraints(ram_threshold_mb * 1024 * 1024)  # 转换为bytes
        print("RAM Analysis:")
        print(f"Maximum RAM Usage: {ram_info['max_ram']/1024/1024:.2f} MB")  # 转换为MB
        print(f"RAM Threshold: {ram_threshold_mb:.2f} MB")
        print(f"RAM Constraints: {'Satisfied' if ram_ok else 'Violated'}\n")
        
        print("Per Layer RAM Usage:")
        for name, ram in ram_info['per_layer_ram'].items():
            print(f"{name}: {ram/1024/1024:.2f} MB")  # 转换为MB
        
        # 可行精度分析
        feasible_precisions = self.get_feasible_precisions(ram_threshold_mb * 1024 * 1024)
        print("\nFeasible Precisions per Layer:")
        for name, count in feasible_precisions.items():
            print(f"{name}: {count} feasible precisions")