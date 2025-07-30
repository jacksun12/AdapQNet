import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from abc import ABC, abstractmethod
import torch.nn.functional as F
import copy
import os
import logging
from pathlib import Path

class MixedPrecisionLayer(nn.Module):
    """Base class for mixed-precision layers with independent weights for each precision"""
    def __init__(self, 
                 module: nn.Module,
                 precision_options: List[str],
                 temperature: float = 1.0):
        super().__init__()
        self.precision_options = precision_options
        self.temperature = temperature
        
        num_options = len(precision_options)
        alpha_init = torch.ones(num_options) * 0.1
        
        for i, precision in enumerate(precision_options):
            if precision == 'int8':
                alpha_init[i] = 1.0
            elif precision == 'int4':
                alpha_init[i] = 0.5
            elif precision == 'int2':
                alpha_init[i] = 0.2
            elif precision == 'fp16':
                alpha_init[i] = 0.3
            elif precision == 'fp32':
                alpha_init[i] = 0.1
        
        alpha_init = torch.clamp(alpha_init, 1e-6, 10.0)
        self.alpha = nn.Parameter(alpha_init)
        
        self.precision_modules = nn.ModuleDict()
        
        def get_device_from_module(module):
            if hasattr(module, 'weight'):
                return module.weight.device
            elif isinstance(module, nn.Sequential):
                for submodule in module:
                    if hasattr(submodule, 'weight'):
                        return submodule.weight.device
            return torch.device('cpu')
        
        device = get_device_from_module(module)
        
        for precision in precision_options:
            if precision == 'fp32':
                self.precision_modules[precision] = self._copy_module(module)
            elif precision == 'fp16':
                self.precision_modules[precision] = self._create_fp16_module(module)
            elif precision.startswith('int'):
                bits = int(precision.replace('int', ''))
                self.precision_modules[precision] = self._create_int_module(module, bits)
            else:
                raise ValueError(f"Unsupported precision: {precision}")
        
        self.alpha_pact_dict = nn.ParameterDict({
            precision: nn.Parameter(torch.tensor(1.0, device=device)) for precision in precision_options
        })

    def _copy_module(self, module: nn.Module) -> nn.Module:
        """Copy module while keeping weights independent"""
        if isinstance(module, nn.Sequential):
            return self._copy_sequential(module)
        else:
            return self._copy_single_module(module)

    def _copy_sequential(self, module: nn.Sequential) -> nn.Sequential:
        """Copy Sequential module"""
        layers = []
        for layer in module:
            if isinstance(layer, nn.Conv2d):
                new_layer = nn.Conv2d(
                    layer.in_channels, layer.out_channels, layer.kernel_size,
                    layer.stride, layer.padding, layer.dilation, layer.groups,
                    layer.bias is not None, layer.padding_mode
                )
                new_layer.weight.data = layer.weight.data.clone()
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data.clone()
                layers.append(new_layer)
            elif isinstance(layer, nn.BatchNorm2d):
                new_layer = nn.BatchNorm2d(layer.num_features, layer.eps, layer.momentum,
                                         layer.affine, layer.track_running_stats)
                if layer.affine:
                    new_layer.weight.data = layer.weight.data.clone()
                    new_layer.bias.data = layer.bias.data.clone()
                if layer.track_running_stats:
                    new_layer.running_mean.data = layer.running_mean.data.clone()
                    new_layer.running_var.data = layer.running_var.data.clone()
                layers.append(new_layer)
            elif isinstance(layer, nn.Linear):
                new_layer = nn.Linear(layer.in_features, layer.out_features, layer.bias is not None)
                new_layer.weight.data = layer.weight.data.clone()
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data.clone()
                layers.append(new_layer)
            else:
                layers.append(copy.deepcopy(layer))
        return nn.Sequential(*layers)

    def _copy_single_module(self, module: nn.Module) -> nn.Module:
        """Copy single module"""
        if isinstance(module, nn.Conv2d):
            new_module = nn.Conv2d(
                module.in_channels, module.out_channels, 
                module.kernel_size, module.stride, 
                module.padding, module.dilation, 
                module.groups, module.bias is not None, 
                module.padding_mode
            )
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            return new_module
        elif isinstance(module, nn.Linear):
            new_module = nn.Linear(module.in_features, module.out_features, 
                                 module.bias is not None)
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            return new_module
        elif isinstance(module, nn.BatchNorm2d):
            new_module = nn.BatchNorm2d(module.num_features, module.eps, 
                                      module.momentum, module.affine, 
                                      module.track_running_stats)
            if module.affine:
                new_module.weight.data = module.weight.data.clone()
                new_module.bias.data = module.bias.data.clone()
            if module.track_running_stats:
                new_module.running_mean.data = module.running_mean.data.clone()
                new_module.running_var.data = module.running_var.data.clone()
            return new_module
        else:
            return copy.deepcopy(module)

    def _get_quantized_module(self, precision: str) -> nn.Module:
        """Get corresponding independent weight module based on precision"""
        if precision not in self.precision_modules:
            raise ValueError(f"Precision {precision} not found in precision_modules")
        return self.precision_modules[precision]

    def _create_fp16_module(self, module: nn.Module) -> nn.Module:
        """Create FP16 version of module (independent weights)"""
        if isinstance(module, nn.Sequential):
            return self._create_fp16_sequential(module)
        else:
            return self._create_fp16_single_module(module)

    def _create_fp16_sequential(self, module: nn.Sequential) -> nn.Sequential:
        """Create FP16 version of Sequential module (independent weights)"""
        layers = []
        for layer in module:
            if isinstance(layer, nn.Conv2d):
                new_layer = nn.Conv2d(
                    layer.in_channels, layer.out_channels, layer.kernel_size,
                    layer.stride, layer.padding, layer.dilation, layer.groups,
                    layer.bias is not None, layer.padding_mode
                )
                new_layer.weight.data = layer.weight.data.half().float()
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data.half().float()
                layers.append(new_layer)
            elif isinstance(layer, nn.BatchNorm2d):
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
                new_layer = nn.Linear(layer.in_features, layer.out_features, layer.bias is not None)
                new_layer.weight.data = layer.weight.data.half().float()
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data.half().float()
                layers.append(new_layer)
            else:
                layers.append(copy.deepcopy(layer))
        return nn.Sequential(*layers)

    def _create_fp16_single_module(self, module: nn.Module) -> nn.Module:
        """Create FP16 version of single module (independent weights)"""
        if isinstance(module, nn.Conv2d):
            new_module = nn.Conv2d(
                module.in_channels, module.out_channels, 
                module.kernel_size, module.stride, 
                module.padding, module.dilation, 
                module.groups, module.bias is not None, 
                module.padding_mode
            )
            new_module.weight.data = module.weight.data.half().float()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.half().float()
            return new_module
        elif isinstance(module, nn.Linear):
            new_module = nn.Linear(module.in_features, module.out_features, 
                                 module.bias is not None)
            new_module.weight.data = module.weight.data.half().float()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.half().float()
            return new_module
        elif isinstance(module, nn.BatchNorm2d):
            new_module = nn.BatchNorm2d(module.num_features, module.eps, 
                                      module.momentum, module.affine, 
                                      module.track_running_stats)
            if module.affine:
                new_module.weight.data = module.weight.data.half().float()
                new_module.bias.data = module.bias.data.half().float()
            if module.track_running_stats:
                new_module.running_mean.data = module.running_mean.data.float()
                new_module.running_var.data = module.running_var.data.float()
            return new_module
        else:
            return copy.deepcopy(module)

    def _create_int_module(self, module: nn.Module, bits: int) -> nn.Module:
        """Create INT quantized version of module (independent weights)"""
        if isinstance(module, nn.Sequential):
            return self._create_int_sequential(module, bits)
        else:
            return self._create_int_single_module(module, bits)

    def _create_int_sequential(self, module: nn.Sequential, bits: int) -> nn.Sequential:
        """Create INT quantized version of Sequential module (independent weights)"""
        layers = []
        for layer in module:
            if isinstance(layer, nn.Conv2d):
                new_layer = nn.Conv2d(
                    layer.in_channels, layer.out_channels, layer.kernel_size,
                    layer.stride, layer.padding, layer.dilation, layer.groups,
                    layer.bias is not None, layer.padding_mode
                )
                new_layer.weight.data = self._dorefa_quantize_weights(layer.weight.data, bits)
                if layer.bias is not None:
                    new_layer.bias.data = self._dorefa_quantize_weights(layer.bias.data, bits)
                layers.append(new_layer)
            elif isinstance(layer, nn.BatchNorm2d):
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
                new_layer = nn.Linear(layer.in_features, layer.out_features, layer.bias is not None)
                new_layer.weight.data = self._dorefa_quantize_weights(layer.weight.data, bits)
                if layer.bias is not None:
                    new_layer.bias.data = self._dorefa_quantize_weights(layer.bias.data, bits)
                layers.append(new_layer)
            else:
                layers.append(copy.deepcopy(layer))
        return nn.Sequential(*layers)

    def _create_int_single_module(self, module: nn.Module, bits: int) -> nn.Module:
        """Create INT quantized version of single module (independent weights)"""
        if isinstance(module, nn.Conv2d):
            new_module = nn.Conv2d(
                module.in_channels, module.out_channels, 
                module.kernel_size, module.stride, 
                module.padding, module.dilation, 
                module.groups, module.bias is not None, 
                module.padding_mode
            )
            new_module.weight.data = self._dorefa_quantize_weights(module.weight.data, bits)
            if module.bias is not None:
                new_module.bias.data = self._dorefa_quantize_weights(module.bias.data, bits)
            return new_module
        elif isinstance(module, nn.Linear):
            new_module = nn.Linear(module.in_features, module.out_features, 
                                 module.bias is not None)
            new_module.weight.data = self._dorefa_quantize_weights(module.weight.data, bits)
            if module.bias is not None:
                new_module.bias.data = self._dorefa_quantize_weights(module.bias.data, bits)
            return new_module
        elif isinstance(module, nn.BatchNorm2d):
            new_module = nn.BatchNorm2d(module.num_features, module.eps, 
                                      module.momentum, module.affine, 
                                      module.track_running_stats)
            if module.affine:
                new_module.weight.data = self._dorefa_quantize_weights(module.weight.data, bits)
                new_module.bias.data = self._dorefa_quantize_weights(module.bias.data, bits)
            if module.track_running_stats:
                new_module.running_mean.data = module.running_mean.data
                new_module.running_var.data = module.running_var.data
            return new_module
        else:
            return copy.deepcopy(module)

    def _dorefa_quantize_weights(self, w: torch.Tensor, k: int) -> torch.Tensor:
        """Improved DoReFa-Net weight quantization with STE for QAT (Per-Channel)"""
        if k == 32:
            return w
        
        if torch.isnan(w).any() or torch.isinf(w).any():
            print("Warning: NaN or Inf detected in weights before quantization")
            return w
            
        if w.requires_grad:
            w = torch.clamp(w, -3.0, 3.0)
        
        if len(w.shape) == 4:
            orig_max = torch.amax(torch.abs(w), dim=(1, 2, 3), keepdim=True)
        elif len(w.shape) == 2:
            orig_max = torch.amax(torch.abs(w), dim=1, keepdim=True)
        else:
            orig_max = torch.max(torch.abs(w))
        
        orig_max = torch.clamp(orig_max, 1e-6, 5.0)
        
        if torch.all(orig_max == 0):
            return w
        
        w_normalized = w / orig_max
        w_scaled = (w_normalized + 1) / 2
        
        w_scaled = torch.clamp(w_scaled, 0.0, 1.0)
        
        w_quantized = torch.floor(w_scaled * (2**k - 1)) / (2**k - 1)
        w_quantized = w_scaled + (w_quantized - w_scaled).detach()
        
        w_recovered = 2 * w_quantized - 1
        result = w_recovered * orig_max
        
        result = torch.clamp(result, -5.0, 5.0)
        
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN or Inf detected in quantized weights")
            return w
            
        return result

    def _pact_quantize_activations(self, x: torch.Tensor, k: int, precision: str = None) -> torch.Tensor:
        """PACT activation quantization with STE for QAT (Global)"""
        if k >= 32:
            return x
            
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in activations before quantization")
            return x
            
        if x.requires_grad:
            x = torch.clamp(x, -3.0, 3.0)
            
        if precision and precision in self.alpha_pact_dict:
            alpha_pact = F.softplus(self.alpha_pact_dict[precision])
        else:
            alpha_pact = F.softplus(torch.tensor(1.0, device=x.device))
            
        alpha_pact = torch.clamp(alpha_pact, 1e-6, 2.0)
        
        global_max = torch.amax(torch.abs(x))
        alpha_pact = torch.min(alpha_pact, global_max)
        
        alpha_pact = torch.clamp(alpha_pact, 1e-6, 2.0)
        
        x_clip = torch.where(torch.abs(x) <= alpha_pact, x, alpha_pact * torch.sign(x))
        x_norm = x_clip / alpha_pact
        
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        x_q = torch.floor(x_norm * (2**k - 1)) / (2**k - 1)
        x_q = x_norm + (x_q - x_norm).detach()
        
        result = x_q * alpha_pact
        
        result = torch.clamp(result, -5.0, 5.0)
        
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("Warning: NaN or Inf detected in quantized activations")
            return x
            
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Gumbel-Softmax for precision selection with shared weights - Weighted Version"""
        if len(self.alpha) != len(self.precision_options):
            print(f"Warning: Alpha size ({len(self.alpha)}) doesn't match precision options count ({len(self.precision_options)})")
            print(f"Precision options: {self.precision_options}")
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
                module = self._get_quantized_module(precision)
                
                current_x = x
                if isinstance(module, nn.Sequential):
                    for layer in module:
                        if isinstance(layer, nn.BatchNorm2d) and not self.training:
                            layer.eval()
                        current_x = layer(current_x)
                else:
                    if isinstance(module, nn.BatchNorm2d) and not self.training:
                        module.eval()
                    current_x = module(current_x)
                
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
                outputs.append(torch.zeros_like(x) * weights[precision_idx])
                continue
        
        if not outputs:
            print("Warning: No valid outputs from any precision path")
            print(f"Precision options: {self.precision_options}")
            print(f"Weights: {weights}")
            print(f"Alpha: {self.alpha}")
            return x
        
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
        """Args:
            precision: precision type
        """
        module = self.precision_modules[precision]
        total_params = 0
        
        if isinstance(module, nn.Sequential):
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    if hasattr(layer, 'groups') and layer.groups > 1:
                        params = (layer.in_channels // layer.groups) * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                    else:
                        params = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                    
                    if layer.bias is not None:
                        params += layer.out_channels
                    total_params += params
                elif isinstance(layer, nn.Linear):
                    params = layer.in_features * layer.out_features
                    if layer.bias is not None:
                        params += layer.out_features
                    total_params += params
        else:
            if isinstance(module, nn.Conv2d):
                if hasattr(module, 'groups') and module.groups > 1:
                    params = (module.in_channels // module.groups) * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
                else:
                    params = module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
                
                if module.bias is not None:
                    params += module.out_channels
                total_params = params
            elif isinstance(module, nn.Linear):
                params = module.in_features * module.out_features
                if module.bias is not None:
                    params += module.out_features
                total_params = params
        
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
        
        self.precision_options = valid_precisions
        
        self.alpha_pact_dict = nn.ParameterDict({
            p: self.alpha_pact_dict[p] for p in valid_precisions if p in self.alpha_pact_dict
        })
        
        self.precision_modules = nn.ModuleDict({
            p: self.precision_modules[p] for p in valid_precisions if p in self.precision_modules
        })
        
        if len(valid_precisions) != len(self.alpha):
            device = self.alpha.device
            new_alpha = torch.zeros(len(valid_precisions), device=device)
            
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
            
            new_alpha = torch.clamp(new_alpha, 1e-6, 10.0)
            self.alpha = nn.Parameter(new_alpha)


class BaseModel(nn.Module):
    """Base model class with training strategies"""
    def __init__(self, precision_options=None, hardware_constraints=None):
        super().__init__()
        self.precision_options = precision_options or ["fp32"]
        self.hardware_constraints = hardware_constraints or {}
        self.mixed_precision_layers: List[MixedPrecisionLayer] = []
        
        self.training_strategy = {
            'patience': 2,
            'min_delta': 1e-4,
            'best_loss': float('inf'),
            'patience_counter': 0,
            'epoch': 0
        }


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

    def copy_weights_from_pretrained(self, pretrained_model, skip_classifier=False):
        """Copy weights from pretrained model to quantized model (shared weights mode)
        Args:
            pretrained_model: pretrained model
            skip_classifier: whether to skip classifier layer
        """
        def _get_base_name(name):
            parts = name.split('.')
            if parts[-1].isdigit():
                return '.'.join(parts[:-1])
            return name
            
        def _match_modules(pretrained_modules, target_name):
            """Match pretrained modules with target module name"""
            base_name = target_name.replace('.base_module', '')
            candidates = []
            
            if base_name in pretrained_modules:
                candidates.extend(pretrained_modules[base_name])
            
            base_name_no_num = '.'.join([p for p in base_name.split('.') if not p.isdigit()])
            for name, modules in pretrained_modules.items():
                name_no_num = '.'.join([p for p in name.split('.') if not p.isdigit()])
                if name_no_num == base_name_no_num:
                    candidates.extend(modules)
            
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
            if skip_classifier and ('classifier' in name or 'fc' in name or 'linear' in name):
                print(f"Skipping classifier layer: {name}")
                continue
                
            if isinstance(module, MixedPrecisionLayer):
                source_modules = _match_modules(pretrained_modules, name)
                if source_modules:
                    print(f"\nCopying weights for {name} (independent weights mode)")
                    source_modules.sort(key=lambda x: x[0])
                    source_modules = [m for _, m in source_modules]
                    
                    for precision, precision_module in module.precision_modules.items():
                        print(f"  Copying to {precision} module")
                        
                        if isinstance(precision_module, nn.Sequential):
                            if len(source_modules) >= len(precision_module):
                                for i, (layer, source_module) in enumerate(zip(precision_module, source_modules)):
                                    if isinstance(layer, type(source_module)):
                                        print(f"    Copying to {precision}_module[{i}]")
                                        if hasattr(layer, 'weight'):
                                            try:
                                                if layer.weight.data.shape != source_module.weight.data.shape:
                                                    print(f"      Skipping {precision}_module[{i}] due to shape mismatch: {layer.weight.data.shape} vs {source_module.weight.data.shape}")
                                                    continue
                                                layer.weight.data.copy_(source_module.weight.data)
                                            except Exception as e:
                                                print(f"      Error copying weight to {precision}_module[{i}]: {str(e)}")
                                        if hasattr(layer, 'bias') and layer.bias is not None:
                                            try:
                                                if hasattr(source_module, 'bias') and source_module.bias is not None:
                                                    layer.bias.data.copy_(source_module.bias.data)
                                            except Exception as e:
                                                print(f"      Error copying bias to {precision}_module[{i}]: {str(e)}")
                                        if isinstance(layer, nn.BatchNorm2d):
                                            try:
                                                layer.running_mean.data.copy_(source_module.running_mean.data)
                                                layer.running_var.data.copy_(source_module.running_var.data)
                                            except Exception as e:
                                                print(f"      Error copying BN stats to {precision}_module[{i}]: {str(e)}")
                            else:
                                source_module = source_modules[0]
                                for i, layer in enumerate(precision_module):
                                    if isinstance(layer, type(source_module)):
                                        print(f"    Copying to {precision}_module[{i}] (using first source)")
                                        if hasattr(layer, 'weight'):
                                            try:
                                                if layer.weight.data.shape != source_module.weight.data.shape:
                                                    print(f"      Skipping {precision}_module[{i}] due to shape mismatch: {layer.weight.data.shape} vs {source_module.weight.data.shape}")
                                                    continue
                                                layer.weight.data.copy_(source_module.weight.data)
                                            except Exception as e:
                                                print(f"      Error copying weight to {precision}_module[{i}]: {str(e)}")
                                        if hasattr(layer, 'bias') and layer.bias is not None:
                                            try:
                                                if hasattr(source_module, 'bias') and source_module.bias is not None:
                                                    layer.bias.data.copy_(source_module.bias.data)
                                            except Exception as e:
                                                print(f"      Error copying bias to {precision}_module[{i}]: {str(e)}")
                                        if isinstance(layer, nn.BatchNorm2d):
                                            try:
                                                layer.running_mean.data.copy_(source_module.running_mean.data)
                                                layer.running_var.data.copy_(source_module.running_var.data)
                                            except Exception as e:
                                                print(f"      Error copying BN stats to {precision}_module[{i}]: {str(e)}")
                        else:
                            source_module = source_modules[0]
                            if isinstance(precision_module, type(source_module)):
                                print(f"    Copying to {precision}_module")
                                if hasattr(precision_module, 'weight'):
                                    try:
                                        if precision_module.weight.data.shape != source_module.weight.data.shape:
                                            print(f"      Skipping {precision}_module due to shape mismatch: {precision_module.weight.data.shape} vs {source_module.weight.data.shape}")
                                            continue
                                        precision_module.weight.data.copy_(source_module.weight.data)
                                    except Exception as e:
                                        print(f"      Error copying weight to {precision}_module: {str(e)}")
                                if hasattr(precision_module, 'bias') and precision_module.bias is not None:
                                    try:
                                        if hasattr(source_module, 'bias') and source_module.bias is not None:
                                            precision_module.bias.data.copy_(source_module.bias.data)
                                    except Exception as e:
                                        print(f"      Error copying bias to {precision}_module: {str(e)}")
                                if isinstance(precision_module, nn.BatchNorm2d):
                                    try:
                                        precision_module.running_mean.data.copy_(source_module.running_mean.data)
                                        precision_module.running_var.data.copy_(source_module.running_var.data)
                                    except Exception as e:
                                        print(f"      Error copying BN stats to {precision}_module: {str(e)}")
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
    
    def calibrate_bn(self, calibration_loader, num_batches=100):
        """
        Calibrate BatchNorm statistics for all mixed precision layers
        Args:
            calibration_loader: DataLoader for calibration
            num_batches: Number of batches to use for calibration
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting BN calibration with {num_batches} batches...")
        
        original_training = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                original_training[name] = module.training
                module.train()
        
        self.eval()
        
        batch_count = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(calibration_loader):
                if batch_count >= num_batches:
                    break
                
                _ = self(images)
                batch_count += 1
                
                if batch_count % 10 == 0:
                    logger.info(f"BN calibration progress: {batch_count}/{num_batches}")
        
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in original_training:
                module.train(original_training[name])
        
        logger.info(f"BN calibration completed, processed {batch_count} batches")
    
    def calibrate_bn_for_subnet(self, subnet_model, calibration_loader, num_batches=100):
        """
        Calibrate BatchNorm statistics for a specific subnet model
        Args:
            subnet_model: The subnet model to calibrate
            calibration_loader: DataLoader for calibration
            num_batches: Number of batches to use for calibration
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting subnet BN calibration with {num_batches} batches...")
        
        original_training = {}
        for name, module in subnet_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                original_training[name] = module.training
                module.train()
        
        subnet_model.eval()
        
        batch_count = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(calibration_loader):
                if batch_count >= num_batches:
                    break
                
                _ = subnet_model(images)
                batch_count += 1
                
                if batch_count % 10 == 0:
                    logger.info(f"Subnet BN calibration progress: {batch_count}/{num_batches}")
        
        for name, module in subnet_model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in original_training:
                module.train(original_training[name])
        
        logger.info(f"Subnet BN calibration completed, processed {batch_count} batches")
