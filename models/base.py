import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from abc import ABC, abstractmethod
import torch.nn.functional as F
import copy
import os
from pathlib import Path

class MixedPrecisionLayer(nn.Module):
    """Base class for mixed-precision layers with shared weights"""
    def __init__(self, 
                 module: nn.Module,
                 precision_options: List[str],
                 temperature: float = 1.0):
        super().__init__()
        self.precision_options = precision_options
        self.temperature = temperature
        
        # Initialize alpha parameters for precision selection with bias towards INT8
        num_options = len(precision_options)
        alpha_init = torch.ones(num_options) * 0.1  # 基础值
        
        # 为INT8精度设置更高的初始值
        for i, precision in enumerate(precision_options):
            if precision == 'int8':
                alpha_init[i] = 1.0  # INT8获得更高的初始值
            elif precision == 'int4':
                alpha_init[i] = 0.5  # INT4获得中等初始值
            elif precision == 'int2':
                alpha_init[i] = 0.2  # INT2获得较低初始值
            elif precision == 'fp16':
                alpha_init[i] = 0.3  # FP16获得较低初始值
            elif precision == 'fp32':
                alpha_init[i] = 0.1  # FP32获得最低初始值
        
        # 确保没有NaN
        alpha_init = torch.clamp(alpha_init, 1e-6, 10.0)
        self.alpha = nn.Parameter(alpha_init)
        
        # 共享权重：只保留一个基础模块
        self.base_module = module
        
        # 为每个精度路径创建独立的PACT截断值
        # 获取设备信息：如果module是Sequential，找到第一个有weight的子模块
        def get_device_from_module(module):
            if hasattr(module, 'weight'):
                return module.weight.device
            elif isinstance(module, nn.Sequential):
                for submodule in module:
                    if hasattr(submodule, 'weight'):
                        return submodule.weight.device
            # 如果都找不到，使用CPU作为默认值
            return torch.device('cpu')
        
        device = get_device_from_module(module)
        self.alpha_pact_dict = nn.ParameterDict({
            precision: nn.Parameter(torch.tensor(1.0, device=device)) for precision in precision_options
        })

    def _get_quantized_module(self, precision: str) -> nn.Module:
        """根据精度动态创建量化模块（共享权重）"""
        if precision == 'fp32':
            # FP32: 直接返回原始模块
            return self.base_module
        elif precision == 'fp16':
            # FP16: 创建临时模块，权重转换为FP16
            if isinstance(self.base_module, nn.Sequential):
                return self._create_fp16_sequential()
            else:
                return self._create_fp16_module()
        elif precision.startswith('int'):
            # INT量化: 创建临时模块，权重量化
            bits = int(precision.replace('int', ''))
            if isinstance(self.base_module, nn.Sequential):
                return self._create_int_sequential(bits)
            else:
                return self._create_int_module(bits)
        else:
            raise ValueError(f"Unsupported precision: {precision}")

    def _create_fp16_sequential(self) -> nn.Sequential:
        """创建FP16版本的Sequential模块（共享权重）"""
        layers = []
        for layer in self.base_module:
            if isinstance(layer, nn.Conv2d):
                # 创建新的Conv2d层，但共享权重
                new_layer = nn.Conv2d(
                    layer.in_channels, layer.out_channels, layer.kernel_size,
                    layer.stride, layer.padding, layer.dilation, layer.groups,
                    layer.bias is not None, layer.padding_mode
                )
                # 共享权重，但转换为FP16
                new_layer.weight.data = layer.weight.data.half().float()
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data.half().float()
                layers.append(new_layer)
            elif isinstance(layer, nn.BatchNorm2d):
                # 创建新的BatchNorm2d层，但共享权重
                new_layer = nn.BatchNorm2d(layer.num_features, layer.eps, layer.momentum,
                                         layer.affine, layer.track_running_stats)
                if layer.affine:
                    new_layer.weight.data = layer.weight.data.half().float()
                    new_layer.bias.data = layer.bias.data.half().float()
                if layer.track_running_stats:
                    new_layer.running_mean.data = layer.running_mean.data.float()
                    new_layer.running_var.data = layer.running_var.data.float()
                layers.append(new_layer)
            elif isinstance(layer, nn.Linear):
                # 创建新的Linear层，但共享权重
                new_layer = nn.Linear(layer.in_features, layer.out_features, layer.bias is not None)
                new_layer.weight.data = layer.weight.data.half().float()
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data.half().float()
                layers.append(new_layer)
            else:
                # 其他层直接复制
                layers.append(copy.deepcopy(layer))
        return nn.Sequential(*layers)

    def _create_fp16_module(self) -> nn.Module:
        """创建FP16版本的单个模块（共享权重）"""
        if isinstance(self.base_module, nn.Conv2d):
            new_module = nn.Conv2d(
                self.base_module.in_channels, self.base_module.out_channels, 
                self.base_module.kernel_size, self.base_module.stride, 
                self.base_module.padding, self.base_module.dilation, 
                self.base_module.groups, self.base_module.bias is not None, 
                self.base_module.padding_mode
            )
            new_module.weight.data = self.base_module.weight.data.half().float()
            if self.base_module.bias is not None:
                new_module.bias.data = self.base_module.bias.data.half().float()
            return new_module
        elif isinstance(self.base_module, nn.Linear):
            new_module = nn.Linear(self.base_module.in_features, self.base_module.out_features, 
                                 self.base_module.bias is not None)
            new_module.weight.data = self.base_module.weight.data.half().float()
            if self.base_module.bias is not None:
                new_module.bias.data = self.base_module.bias.data.half().float()
            return new_module
        elif isinstance(self.base_module, nn.BatchNorm2d):
            new_module = nn.BatchNorm2d(self.base_module.num_features, self.base_module.eps, 
                                      self.base_module.momentum, self.base_module.affine, 
                                      self.base_module.track_running_stats)
            if self.base_module.affine:
                new_module.weight.data = self.base_module.weight.data.half().float()
                new_module.bias.data = self.base_module.bias.data.half().float()
            if self.base_module.track_running_stats:
                new_module.running_mean.data = self.base_module.running_mean.data.float()
                new_module.running_var.data = self.base_module.running_var.data.float()
            return new_module
        else:
            return copy.deepcopy(self.base_module)

    def _create_int_sequential(self, bits: int) -> nn.Sequential:
        """创建INT量化版本的Sequential模块（共享权重）"""
        layers = []
        for layer in self.base_module:
            if isinstance(layer, nn.Conv2d):
                # 创建新的Conv2d层，但共享权重
                new_layer = nn.Conv2d(
                    layer.in_channels, layer.out_channels, layer.kernel_size,
                    layer.stride, layer.padding, layer.dilation, layer.groups,
                    layer.bias is not None, layer.padding_mode
                )
                # 共享权重，但量化
                new_layer.weight.data = self._dorefa_quantize_weights(layer.weight.data, bits)
                if layer.bias is not None:
                    new_layer.bias.data = self._dorefa_quantize_weights(layer.bias.data, bits)
                layers.append(new_layer)
            elif isinstance(layer, nn.BatchNorm2d):
                # 创建新的BatchNorm2d层，但共享权重
                new_layer = nn.BatchNorm2d(layer.num_features, layer.eps, layer.momentum,
                                         layer.affine, layer.track_running_stats)
                if layer.affine:
                    new_layer.weight.data = self._dorefa_quantize_weights(layer.weight.data, bits)
                    new_layer.bias.data = self._dorefa_quantize_weights(layer.bias.data, bits)
                if layer.track_running_stats:
                    new_layer.running_mean.data = layer.running_mean.data
                    new_layer.running_var.data = layer.running_var.data
                layers.append(new_layer)
            elif isinstance(layer, nn.Linear):
                # 创建新的Linear层，但共享权重
                new_layer = nn.Linear(layer.in_features, layer.out_features, layer.bias is not None)
                new_layer.weight.data = self._dorefa_quantize_weights(layer.weight.data, bits)
                if layer.bias is not None:
                    new_layer.bias.data = self._dorefa_quantize_weights(layer.bias.data, bits)
                layers.append(new_layer)
            else:
                # 其他层直接复制
                layers.append(copy.deepcopy(layer))
        return nn.Sequential(*layers)

    def _create_int_module(self, bits: int) -> nn.Module:
        """创建INT量化版本的单个模块（共享权重）"""
        if isinstance(self.base_module, nn.Conv2d):
            new_module = nn.Conv2d(
                self.base_module.in_channels, self.base_module.out_channels, 
                self.base_module.kernel_size, self.base_module.stride, 
                self.base_module.padding, self.base_module.dilation, 
                self.base_module.groups, self.base_module.bias is not None, 
                self.base_module.padding_mode
            )
            new_module.weight.data = self._dorefa_quantize_weights(self.base_module.weight.data, bits)
            if self.base_module.bias is not None:
                new_module.bias.data = self._dorefa_quantize_weights(self.base_module.bias.data, bits)
            return new_module
        elif isinstance(self.base_module, nn.Linear):
            new_module = nn.Linear(self.base_module.in_features, self.base_module.out_features, 
                                 self.base_module.bias is not None)
            new_module.weight.data = self._dorefa_quantize_weights(self.base_module.weight.data, bits)
            if self.base_module.bias is not None:
                new_module.bias.data = self._dorefa_quantize_weights(self.base_module.bias.data, bits)
            return new_module
        elif isinstance(self.base_module, nn.BatchNorm2d):
            new_module = nn.BatchNorm2d(self.base_module.num_features, self.base_module.eps, 
                                      self.base_module.momentum, self.base_module.affine, 
                                      self.base_module.track_running_stats)
            if self.base_module.affine:
                new_module.weight.data = self._dorefa_quantize_weights(self.base_module.weight.data, bits)
                new_module.bias.data = self._dorefa_quantize_weights(self.base_module.bias.data, bits)
            if self.base_module.track_running_stats:
                new_module.running_mean.data = self.base_module.running_mean.data
                new_module.running_var.data = self.base_module.running_var.data
            return new_module
        else:
            return copy.deepcopy(self.base_module)

    def _dorefa_quantize_weights(self, w: torch.Tensor, k: int) -> torch.Tensor:
        """Improved DoReFa-Net weight quantization with STE for QAT (Per-Channel)"""
        if k == 32:
            return w
        
        # 添加数值稳定性检查
        if torch.isnan(w).any() or torch.isinf(w).any():
            print("Warning: NaN or Inf detected in weights before quantization")
            return w
            
        # 更严格的梯度裁剪防止梯度爆炸
        if w.requires_grad:
            w = torch.clamp(w, -3.0, 3.0)  # 更严格的裁剪范围
        
        # Per-channel 量化
        if len(w.shape) == 4:  # Conv2d weights: [out_channels, in_channels, kernel_h, kernel_w]
            # 计算每个输出通道的最大值
            orig_max = torch.amax(torch.abs(w), dim=(1, 2, 3), keepdim=True)
        elif len(w.shape) == 2:  # Linear weights: [out_features, in_features]
            # 计算每个输出特征的最大值
            orig_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
        else:
            # 对于其他形状，使用全局最大值
            orig_max = torch.max(torch.abs(w))
        
        # 确保orig_max不为零且合理
        orig_max = torch.clamp(orig_max, 1e-6, 5.0)  # 更保守的范围
        
        # 处理零值情况
        if torch.all(orig_max == 0):
            return w
        
        # 安全的归一化
        w_normalized = w / orig_max
        w_scaled = (w_normalized + 1) / 2  # [0,1]
        
        # 确保w_scaled在有效范围内
        w_scaled = torch.clamp(w_scaled, 0.0, 1.0)
        
        # 使用STE：前向传播时量化，反向传播时直通
        w_quantized = torch.round(w_scaled * (2**k - 1)) / (2**k - 1)
        w_quantized = w_scaled + (w_quantized - w_scaled).detach()  # STE
        
        w_recovered = 2 * w_quantized - 1  # [-1,1]
        result = w_recovered * orig_max
        
        # 最终检查并裁剪结果
        result = torch.clamp(result, -5.0, 5.0)  # 更保守的范围
        
        # 检查结果
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN or Inf detected in quantized weights")
            return w
            
        return result

    def _pact_quantize_activations(self, x: torch.Tensor, k: int, precision: str = None) -> torch.Tensor:
        """PACT activation quantization with STE for QAT (Global)"""
        if k >= 32:
            return x
            
        # 添加数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in activations before quantization")
            return x
            
        # 更严格的梯度裁剪防止梯度爆炸
        if x.requires_grad:
            x = torch.clamp(x, -3.0, 3.0)  # 更严格的裁剪范围
            
        # 使用精度特定的PACT截断值
        if precision and precision in self.alpha_pact_dict:
            alpha_pact = F.softplus(self.alpha_pact_dict[precision])
        else:
            # 如果没有指定精度，使用默认值（向后兼容）
            alpha_pact = F.softplus(torch.tensor(1.0, device=x.device))
            
        # 确保alpha_pact不为零且合理
        alpha_pact = torch.clamp(alpha_pact, 1e-6, 2.0)  # 更保守的范围
        
        # 全局激活量化（不需要per-channel）
        # 计算全局最大值
        global_max = torch.amax(torch.abs(x))
        # 使用全局最大值和PACT截断值的较小值
        alpha_pact = torch.min(alpha_pact, global_max)
        
        # 确保alpha_pact不为零
        alpha_pact = torch.clamp(alpha_pact, 1e-6, 2.0)  # 更保守的范围
        
        # 安全的PACT截断
        x_clip = torch.where(torch.abs(x) <= alpha_pact, x, alpha_pact * torch.sign(x))
        x_norm = x_clip / alpha_pact
        
        # 确保x_norm在有效范围内
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        # 使用STE：前向传播时量化，反向传播时直通
        x_q = torch.round(x_norm * (2**k - 1)) / (2**k - 1)
        x_q = x_norm + (x_q - x_norm).detach()  # STE
        
        result = x_q * alpha_pact
        
        # 最终检查并裁剪结果
        result = torch.clamp(result, -5.0, 5.0)  # 更保守的范围
        
        # 检查结果
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN or Inf detected in quantized activations")
            return x
            
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Gumbel-Softmax for precision selection with shared weights"""
        # 确保alpha参数大小与精度选项数量匹配
        if len(self.alpha) != len(self.precision_options):
            print(f"Warning: Alpha size ({len(self.alpha)}) doesn't match precision options count ({len(self.precision_options)})")
            print(f"Precision options: {self.precision_options}")
            # 重新初始化alpha参数
            device = self.alpha.device
            new_alpha = torch.zeros(len(self.precision_options), device=device)
            for i, precision in enumerate(self.precision_options):
                if precision == 'int8':
                    new_alpha[i] = 1.0
                elif precision == 'int4':
                    new_alpha[i] = 0.5
                elif precision == 'int2':
                    new_alpha[i] = 0.2
                elif precision == 'fp16':
                    new_alpha[i] = 0.3
                elif precision == 'fp32':
                    new_alpha[i] = 0.1
                else:
                    new_alpha[i] = 0.1
            new_alpha = torch.clamp(new_alpha, 1e-6, 10.0)
            self.alpha = nn.Parameter(new_alpha)
        
        if self.training:
            weights = F.gumbel_softmax(self.alpha, tau=max(self.temperature, 0.1), hard=False)
        else:
            weights = F.softmax(self.alpha / max(self.temperature, 0.1), dim=0)
        
        outputs = []
        for precision_idx, precision in enumerate(self.precision_options):
            try:
                # 获取当前精度路径的模块（动态创建，共享权重）
                module = self._get_quantized_module(precision)
                
                # 前向传播
                current_x = x  # 保持FP32输入
                if isinstance(module, nn.Sequential):
                    for layer in module:
                        current_x = layer(current_x)
                else:
                    current_x = module(current_x)
                
                # 激活量化（如果需要）
                if precision.startswith('int'):
                    bits = int(precision.replace('int', ''))
                    current_x = self._pact_quantize_activations(current_x, bits, precision)
                elif precision == 'fp16':
                    current_x = current_x.half().float() 

                outputs.append(current_x * weights[precision_idx])
                
            except Exception as e:
                print(f"Error in forward pass for precision {precision}: {str(e)}")
                print(f"Input shape: {x.shape}")
                print(f"Module type: {type(module)}")
                if hasattr(module, 'weight'):
                    print(f"Module weight shape: {module.weight.shape}")
                # 不要立即抛出异常，而是跳过这个精度路径
                print(f"Skipping precision {precision} due to error")
                continue
        
        # 确保至少有一个输出
        if not outputs:
            print("Warning: No valid outputs from any precision path")
            print(f"Precision options: {self.precision_options}")
            print(f"Weights: {weights}")
            print(f"Alpha: {self.alpha}")
            
            # 直接抛出异常，让错误暴露出来
            raise RuntimeError(f"No valid outputs from any precision path for layer with input shape {x.shape}")
        
        # 加权平均所有精度路径的结果
        result = sum(outputs)
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN or Inf detected in output")
            print(f"Weights: {weights}")
            print(f"Alpha: {self.alpha}")
            return x
        return result

    def set_precision(self, precision: str):
        """Set the layer to use the specified precision"""
        if precision not in self.precision_options:
            raise ValueError(f"Unsupported precision: {precision}")
        idx = self.precision_options.index(precision)
        self.alpha.data = torch.zeros_like(self.alpha)
        self.alpha.data[idx] = 100.0

    def get_current_precision(self) -> str:
        """Get the currently used precision"""
        weights = F.softmax(self.alpha / self.temperature, dim=0)
        idx = weights.argmax().item()
        return self.precision_options[idx]

    def get_compute_cost(self, precision: str) -> float:
        """返回该层参数量 × 精度位数²（不依赖输入尺寸）
        Args:
            precision: 精度类型
        """
        # 使用基础模块计算参数量
        total_params = 0
        if isinstance(self.base_module, nn.Sequential):
            for layer in self.base_module:
                # 只考虑卷积层和线性层
                if isinstance(layer, nn.Conv2d):
                    # 卷积层参数量
                    if hasattr(layer, 'groups') and layer.groups > 1:
                        params = (layer.in_channels // layer.groups) * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                    else:
                        params = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                    
                    if layer.bias is not None:
                        params += layer.out_channels
                    total_params += params
                elif isinstance(layer, nn.Linear):
                    # 线性层参数量
                    params = layer.in_features * layer.out_features
                    if layer.bias is not None:
                        params += layer.out_features
                    total_params += params
        else:
            # 单个模块的情况
            if isinstance(self.base_module, nn.Conv2d):
                # 卷积层参数量
                if hasattr(self.base_module, 'groups') and self.base_module.groups > 1:
                    params = (self.base_module.in_channels // self.base_module.groups) * self.base_module.out_channels * self.base_module.kernel_size[0] * self.base_module.kernel_size[1]
                else:
                    params = self.base_module.in_channels * self.base_module.out_channels * self.base_module.kernel_size[0] * self.base_module.kernel_size[1]
                
                if self.base_module.bias is not None:
                    params += self.base_module.out_channels
                total_params = params
            elif isinstance(self.base_module, nn.Linear):
                # 线性层参数量
                params = self.base_module.in_features * self.base_module.out_features
                if self.base_module.bias is not None:
                    params += self.base_module.out_features
                total_params = params
        
        # 计算位数
        if precision.startswith('fp'):
            bits = int(precision.replace('fp', ''))
        elif precision.startswith('int'):
            bits = int(precision.replace('int', ''))
        else:
            raise ValueError(f"Unknown precision: {precision}")
        
        return total_params * (bits ** 2)

    def prune_precision_by_memory(self, input_shape, output_shape, ram_limit):
        """
        Prune precision options that exceed memory limit based on input/output tensor shape and precision.
        input_shape/output_shape: (batch, channel, height, width)
        ram_limit: in bytes
        """
        valid_precisions = []
        for p in self.precision_options:
            bits = int(p.replace('fp', '').replace('int', ''))
            num_in = np.prod(input_shape)
            num_out = np.prod(output_shape)
            mem_in = num_in * bits // 8
            mem_out = num_out * bits // 8
            if mem_in + mem_out < ram_limit:
                valid_precisions.append(p)
        if set(valid_precisions) != set(self.precision_options):
            print(f"[Prune] Layer {self.__class__.__name__}: {self.precision_options} -> {valid_precisions}")
        
        # 更新精度选项
        self.precision_options = valid_precisions
        
        # 更新PACT截断值
        self.alpha_pact_dict = nn.ParameterDict({
            p: self.alpha_pact_dict[p] for p in valid_precisions if p in self.alpha_pact_dict
        })
        
        # 重新初始化alpha参数以匹配新的精度选项数量
        if len(valid_precisions) != len(self.alpha):
            device = self.alpha.device
            # 创建新的alpha参数，保持与精度选项的对应关系
            new_alpha = torch.zeros(len(valid_precisions), device=device)
            
            # 为保留的精度选项设置合理的初始值
            for i, precision in enumerate(valid_precisions):
                if precision == 'int8':
                    new_alpha[i] = 1.0
                elif precision == 'int4':
                    new_alpha[i] = 0.5
                elif precision == 'int2':
                    new_alpha[i] = 0.2
                elif precision == 'fp16':
                    new_alpha[i] = 0.3
                elif precision == 'fp32':
                    new_alpha[i] = 0.1
                else:
                    new_alpha[i] = 0.1
            
            # 确保没有NaN
            new_alpha = torch.clamp(new_alpha, 1e-6, 10.0)
            self.alpha = nn.Parameter(new_alpha)


class BaseModel(nn.Module):
    """Base model class"""
    def __init__(self, precision_options=None, hardware_constraints=None):
        super().__init__()
        self.precision_options = precision_options or ["fp32"]
        self.hardware_constraints = hardware_constraints or {}
        self.mixed_precision_layers: List[MixedPrecisionLayer] = []


    def get_compute_cost(self) -> float:
        """Get model compute cost"""
        total_cost = 0
        for layer in self.mixed_precision_layers:
            precision = layer.get_current_precision()
            total_cost += layer.get_compute_cost(precision)
        return total_cost

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def set_layer_precision(self, layer_name: str, precision: str):
        """Set the precision of the specified layer
        Args:
            layer_name: name of the layer
            precision: target precision
        """
        for name, module in self.named_modules():
            if name == layer_name and isinstance(module, MixedPrecisionLayer):
                module.set_precision(precision)
                break

    def copy_weights_from_pretrained(self, pretrained_model):
        """Copy weights from pretrained model to quantized model (shared weights mode)"""
        def _get_base_name(name):
            parts = name.split('.')
            if parts[-1].isdigit():
                return '.'.join(parts[:-1])
            return name
            
        def _match_modules(pretrained_modules, target_name):
            """Match pretrained modules with target module name"""
            base_name = target_name.replace('.base_module', '')
            # 尝试多种匹配策略
            candidates = []
            
            # 策略1：直接匹配
            if base_name in pretrained_modules:
                candidates.extend(pretrained_modules[base_name])
            
            # 策略2：去掉数字后缀匹配
            base_name_no_num = '.'.join([p for p in base_name.split('.') if not p.isdigit()])
            for name, modules in pretrained_modules.items():
                name_no_num = '.'.join([p for p in name.split('.') if not p.isdigit()])
                if name_no_num == base_name_no_num:
                    candidates.extend(modules)
            
            # 策略3：部分匹配
            for name, modules in pretrained_modules.items():
                if base_name in name or name in base_name:
                    candidates.extend(modules)
            
            return candidates
        
        print("\nStarting weight copying (shared weights mode)...")
        pretrained_modules = {}
        for name, module in pretrained_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                base_name = _get_base_name(name)
                if base_name not in pretrained_modules:
                    pretrained_modules[base_name] = []
                pretrained_modules[base_name].append((int(name.split('.')[-1]) if name.split('.')[-1].isdigit() else 0, module))
                print(f"Found pretrained module: {name} -> {base_name}")
        
        for name, module in self.named_modules():
            if isinstance(module, MixedPrecisionLayer):
                source_modules = _match_modules(pretrained_modules, name)
                if source_modules:
                    print(f"\nCopying weights for {name} (shared weights mode)")
                    source_modules.sort(key=lambda x: x[0])
                    source_modules = [m for _, m in source_modules]
                    
                    # 直接复制到base_module
                    base_module = module.base_module
                    if isinstance(base_module, nn.Sequential):
                        for i, (layer, source_module) in enumerate(zip(base_module, source_modules)):
                            if isinstance(layer, type(source_module)):
                                print(f"  Copying to base_module[{i}]")
                                if hasattr(layer, 'weight'):
                                    try:
                                        if layer.weight.data.shape != source_module.weight.data.shape:
                                            print(f"    Skipping base_module[{i}] due to shape mismatch: {layer.weight.data.shape} vs {source_module.weight.data.shape}")
                                            continue
                                        layer.weight.data.copy_(source_module.weight.data)
                                    except Exception as e:
                                        print(f"    Error copying weight to base_module[{i}]: {str(e)}")
                                if hasattr(layer, 'bias') and layer.bias is not None:
                                    try:
                                        if hasattr(source_module, 'bias') and source_module.bias is not None:
                                            layer.bias.data.copy_(source_module.bias.data)
                                    except Exception as e:
                                        print(f"    Error copying bias to base_module[{i}]: {str(e)}")
                                if isinstance(layer, nn.BatchNorm2d):
                                    try:
                                        layer.running_mean.data.copy_(source_module.running_mean.data)
                                        layer.running_var.data.copy_(source_module.running_var.data)
                                    except Exception as e:
                                        print(f"    Error copying BN stats to base_module[{i}]: {str(e)}")
                    else:
                        source_module = source_modules[0]
                        if isinstance(base_module, type(source_module)):
                            print(f"  Copying to base_module")
                            if hasattr(base_module, 'weight'):
                                try:
                                    if base_module.weight.data.shape != source_module.weight.data.shape:
                                        print(f"    Skipping base_module due to shape mismatch: {base_module.weight.data.shape} vs {source_module.weight.data.shape}")
                                        continue
                                    base_module.weight.data.copy_(source_module.weight.data)
                                except Exception as e:
                                    print(f"    Error copying weight to base_module: {str(e)}")
                            if hasattr(base_module, 'bias') and base_module.bias is not None:
                                try:
                                    if hasattr(source_module, 'bias') and source_module.bias is not None:
                                        base_module.bias.data.copy_(source_module.bias.data)
                                except Exception as e:
                                    print(f"    Error copying bias to base_module: {str(e)}")
                            if isinstance(base_module, nn.BatchNorm2d):
                                try:
                                    base_module.running_mean.data.copy_(source_module.running_mean.data)
                                    base_module.running_var.data.copy_(source_module.running_var.data)
                                except Exception as e:
                                    print(f"    Error copying BN stats to base_module: {str(e)}")
                else:
                    print(f"Warning: No matching pretrained module found for {name}")
                    base_name = name.replace('.base_module', '')
                    print(f"  Looking for: {base_name}")
                    print(f"  Available keys: {list(pretrained_modules.keys())[:10]}...")

    def prune_all_precisions_by_memory(self, input_size, batch_size, ram_limit):
        """
        Prune all mixed-precision layers by memory.
        input_size: (C, H, W)
        batch_size: int
        ram_limit: in bytes
        """
        x = torch.randn(batch_size, *input_size)
        hooks = []
        layer_io_shapes = {}
        def hook_fn(module, inp, outp):
            layer_io_shapes[module] = (inp[0].shape, outp.shape)
        for m in self.modules():
            if isinstance(m, MixedPrecisionLayer):
                hooks.append(m.register_forward_hook(hook_fn))
        self.eval()
        with torch.no_grad():
            self(x)
        for m in self.modules():
            if isinstance(m, MixedPrecisionLayer) and m in layer_io_shapes:
                in_shape = layer_io_shapes[m][0]
                out_shape = layer_io_shapes[m][1]
                m.prune_precision_by_memory(in_shape, out_shape, ram_limit)
        for h in hooks:
            h.remove()