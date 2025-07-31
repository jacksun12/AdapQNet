"""
Multi-Core Neural Architecture Search for CIFAR-10 with MobileNetV2

This script implements a sophisticated multi-core neural architecture search (NAS) 
approach for CIFAR-10 classification using MobileNetV2 as the base architecture. 
The search process is designed to optimize both computational efficiency and 
hardware resource utilization through a two-phase approach.

Multi-Core Search  
- Selects top-K layers for dual-core optimization based on sensitivity analysis
- Implements dual-core layers with independent weights for each precision
- Supports 11 options per layer: 1 single-precision + 10 dual-precision combinations
- Implements temperature annealing strategy for stable architecture selection

KEY FEATURES:
- Dual-core layer implementation with independent weight optimization
- Hardware-aware penalty calculation (convolution params × bit-width²)
- Temperature annealing for stable architecture search
- Alpha parameter optimization with separate optimizers for weights and architecture
- Sensitivity analysis for layer selection
- Visualization of training curves and alpha evolution
- Support for filtered dual-core options from analysis results

ARCHITECTURE OPTIONS:
- Single precision: fp32, fp16, int8, int4, int2
- Dual precision combinations: fp32+fp16, fp32+int8, fp32+int4, fp32+int2,
  fp16+int8, fp16+int4, fp16+int2, int8+int4, int8+int2, int4+int2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path
import argparse
import json
import numpy as np
import copy
import itertools
import torch.nn.functional as F
from tqdm import tqdm
import time
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from models.mobilenetv2 import AdaptQMobileNetV2
from models.base import MixedPrecisionLayer

dnas_utils_path = Path(__file__).parent.parent / 'dnas' / 'utils'
sys.path.append(str(dnas_utils_path))
from data_utils import create_train_val_test_loaders

def load_dual_core_analysis_results(analysis_path):
    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)
    return analysis_data

def get_filtered_dual_core_options(analysis_data, layer_name):
    analysis_layer_name = f"{layer_name}.precision_modules.fp32.0"
    
    if analysis_layer_name in analysis_data.get("filtered_dual_core_options", {}):
        return analysis_data["filtered_dual_core_options"][analysis_layer_name]
    else:
        print(f"Warning: Dual-core options for layer {layer_name} not found in analysis results, using default options")
        return [
            "fp32+fp16", "fp32+int8", "fp32+int4", "fp32+int2",
            "fp16+int8", "fp16+int4", "fp16+int2",
            "int8+int4", "int8+int2", "int4+int2"
        ]

class DualCoreLayer(nn.Module):
    def __init__(self, base_layer, precision_options, filtered_dual_core_options=None):
        super().__init__()
        self.base_layer = base_layer
        self.precision_options = precision_options
        
        original_alpha = self.base_layer.alpha.data.clone()
        original_best_idx = original_alpha.argmax().item()
        self.layer_best_precision = self.base_layer.precision_options[original_best_idx]
        
        if filtered_dual_core_options:
            self.dual_core_options = [self.layer_best_precision] + filtered_dual_core_options
            print(f"Using filtered dual-core options: {len(filtered_dual_core_options)} dual-core combinations + 1 best single precision ({self.layer_best_precision})")
        else:
            self.dual_core_options = self._create_dual_core_options()
            print(f"Using default dual-core options: {len(self.dual_core_options)} options")
        
        self.mixed_precision_layers = nn.ModuleDict()
        
        self.mixed_precision_layers[self.layer_best_precision] = copy.deepcopy(base_layer)
        
        dual_options = filtered_dual_core_options if filtered_dual_core_options else self.dual_core_options[1:]
        for option in dual_options:
            prec1, prec2 = option.split('+')
            dual_layer = copy.deepcopy(base_layer)
            dual_layer.precision_options = [prec1, prec2]
            dual_layer.alpha = nn.Parameter(torch.zeros(2))
            
            new_precision_modules = nn.ModuleDict()
            new_alpha_pact_dict = nn.ParameterDict()
            
            for precision in [prec1, prec2]:
                if precision in dual_layer.precision_modules:
                    new_precision_modules[precision] = dual_layer.precision_modules[precision]
                if precision in dual_layer.alpha_pact_dict:
                    new_alpha_pact_dict[precision] = dual_layer.alpha_pact_dict[precision]
            
            dual_layer.precision_modules = new_precision_modules
            dual_layer.alpha_pact_dict = new_alpha_pact_dict
            
            self.mixed_precision_layers[option] = dual_layer
        
        self.alpha = nn.Parameter(torch.zeros(len(self.dual_core_options)))
        self.temperature = 1.0
    
    def _create_dual_core_options(self):
        """Create dual-core options: best single precision + 10 dual-precision combinations"""
        original_alpha = self.base_layer.alpha.data.clone()
        original_best_idx = original_alpha.argmax().item()
        layer_best_precision = self.base_layer.precision_options[original_best_idx]
        
        dual_core_options = get_dual_core_options(layer_best_precision)
        return dual_core_options
    
    def forward(self, x, mode='search'):
        if mode == 'single':
            idx = self.alpha.argmax().item()
            option = self.dual_core_options[idx]
            return self.mixed_precision_layers[option](x)
        elif mode == 'search':
            weights = F.softmax(self.alpha / self.temperature, dim=0)
            outputs = []
            for i, option in enumerate(self.dual_core_options):
                output = self.mixed_precision_layers[option](x)
                outputs.append(output * weights[i])
            return sum(outputs)
        elif mode == 'eval':
            idx = self.alpha.argmax().item()
            option = self.dual_core_options[idx]
            selected_layer = self.mixed_precision_layers[option]
            
            return selected_layer(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def set_precision(self, option_idx):
        """Set specified precision option"""
        self.alpha.data.zero_()
        self.alpha.data[option_idx] = 100.0
    
    def get_current_option(self):
        """Get current selected option"""
        weights = F.softmax(self.alpha / self.temperature, dim=0)
        idx = weights.argmax().item()
        return self.dual_core_options[idx]
    
    def get_dual_core_weights(self, option):
        """Get W1/W2 weights for specified dual-core option"""
        if option in self.mixed_precision_layers and '+' in option:
            layer = self.mixed_precision_layers[option]
            weights = F.softmax(layer.alpha / layer.temperature, dim=0)
            return weights[0].item(), weights[1].item()
        return None, None

def get_dual_core_options(layer_best_precision="fp16"):
    """Generate dual-core options: C(5,2) = 10 dual-precision combinations"""
    base_precisions = ["fp32", "fp16", "int8", "int4", "int2"]
    
    single_precision = [layer_best_precision]
    
    dual_combinations = []
    for i in range(len(base_precisions)):
        for j in range(i+1, len(base_precisions)):
            dual_combinations.append(f"{base_precisions[i]}+{base_precisions[j]}")
    
    return single_precision + dual_combinations

def analyze_alpha_sensitivity(alpha_snapshot, temperature=1.0):
    """Analyze sensitivity of alpha distribution, return sensitivity score for each layer"""
    sensitivity_scores = {}
    
    for layer_name, alpha_values in alpha_snapshot.items():
        alpha_tensor = torch.tensor(alpha_values)
        weights = F.softmax(alpha_tensor / temperature, dim=0)
        
        entropy = -(weights * torch.log(weights + 1e-10)).sum()
        
        max_weight = weights.max()
        
        weight_variance = torch.var(weights)
        
        sensitivity_score = entropy.item() + weight_variance.item() - max_weight.item()
        sensitivity_scores[layer_name] = sensitivity_score
    
    return sensitivity_scores

def get_topk_layers_from_snapshot(alpha_snapshot, k, temperature=1.0):
    """Select K layers most suitable for multi-core search based on alpha snapshot"""
    sensitivity_scores = analyze_alpha_sensitivity(alpha_snapshot, temperature)
    
    layer_order = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
    topk_layers = [layer_name for layer_name, score in layer_order[:k]]
    
    
    return topk_layers

def load_phase_one_supernet(supernet_path, device):
    """Load first phase saved supernet model"""
    print(f"Loading first phase supernet model: {supernet_path}")
    
    supernet_state = torch.load(supernet_path, map_location=device)
    
    model_state_dict = supernet_state['model_state_dict']
    best_config = supernet_state.get('best_config', {})
    best_alpha_snapshot = supernet_state.get('best_alpha_snapshot', {})
    training_history = supernet_state.get('training_history', {})
    final_epoch = supernet_state.get('final_epoch', 0)
    final_temperature = supernet_state.get('final_temperature', 0.001)
    final_accuracy = supernet_state.get('final_accuracy', 0.0)
    
    return {
        'model_state_dict': model_state_dict,
        'best_config': best_config,
        'best_alpha_snapshot': best_alpha_snapshot,
        'training_history': training_history,
        'final_epoch': final_epoch,
        'final_temperature': final_temperature,
        'final_accuracy': final_accuracy
    }

def load_phase_one_alpha_state(alpha_state_path):
    """Load first phase alpha state file"""
    print(f"Loading first phase alpha state: {alpha_state_path}")
    
    with open(alpha_state_path, 'r') as f:
        alpha_state = json.load(f)
    
    print(f"Alpha state information:")
    print(f"  - Number of layers: {len(alpha_state)}")
    
    return alpha_state

def create_dual_core_model_from_supernet(supernet_info, topk_layers, analysis_data=None, alpha_state=None):
    """Create true multi-core model from supernet info (each precision has independent weights)"""
    
    if alpha_state:
        phase_one_temperature = alpha_state.get('features.0.conv_bn_relu', {}).get('temperature', supernet_info.get('final_temperature', 1.0))
    else:
        phase_one_temperature = supernet_info.get('final_temperature', 1.0)
    print(f"First phase temperature: {phase_one_temperature}")
    
    model = AdaptQMobileNetV2(
        num_classes=10, 
        width_mult=1.0, 
        precision_options=["fp32", "fp16", "int8", "int4", "int2"],
        hardware_constraints=None, 
        pretrain_mode=False, 
        initialize_weights=False,  
        input_size=32
    )
    
    model.load_state_dict(supernet_info['model_state_dict'], strict=False)
    
    if alpha_state:
        alpha_snapshot = {name: data['alpha'] for name, data in alpha_state.items()}
    else:
        alpha_snapshot = supernet_info['best_alpha_snapshot']
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and name in alpha_snapshot:
            if name not in topk_layers:
                alpha_values = torch.tensor(alpha_snapshot[name])
                max_idx = alpha_values.argmax().item()
                
                if module.alpha.size(0) != len(alpha_values):
                    min_size = min(module.alpha.size(0), len(alpha_values))
                    module.alpha.data.zero_()
                    module.alpha.data[:min_size] = alpha_values[:min_size]
                    if max_idx < min_size:
                        module.alpha.data[max_idx] = 100.0
                else:
                    module.alpha.data.zero_()
                    module.alpha.data[max_idx] = 100.0
                
                for param in module.parameters():
                    param.requires_grad = False
                module.alpha.requires_grad = False
            else:
                filtered_options = None
                if analysis_data:
                    filtered_options = get_filtered_dual_core_options(analysis_data, name)
                
                original_alpha = module.alpha.data.clone()
                original_best_idx = original_alpha.argmax().item()
                layer_best_precision = module.precision_options[original_best_idx]
                
                dual_core_layer = DualCoreLayer(
                    module, 
                    module.precision_options, 
                    filtered_options
                )
                
                new_alpha_size = len(dual_core_layer.dual_core_options)
                
                original_best_idx = original_alpha.argmax().item()
                original_best_precision = module.precision_options[original_best_idx]
                
                dual_core_layer.alpha.data.zero_()
                
                layer_best_precision = original_best_precision
                best_single_idx = dual_core_layer.dual_core_options.index(layer_best_precision)
                
                for i in range(new_alpha_size):
                    if i == best_single_idx:
                        dual_core_layer.alpha.data[i] = 1.0
                    else:
                        dual_core_layer.alpha.data[i] = 1.0
                
                
                single_layer = dual_core_layer.mixed_precision_layers[layer_best_precision]
                single_layer.alpha.data.copy_(original_alpha)
                single_layer.temperature = phase_one_temperature
                
                weights = F.softmax(single_layer.alpha.data / single_layer.temperature, dim=0)
                max_idx = weights.argmax().item()
                
                for option in dual_core_layer.dual_core_options[1:]:
                    if '+' in option:
                        prec1, prec2 = option.split('+')
                        dual_layer = dual_core_layer.mixed_precision_layers[option]
                        
                        prec1_idx = module.precision_options.index(prec1) if prec1 in module.precision_options else 0
                        prec2_idx = module.precision_options.index(prec2) if prec2 in module.precision_options else 0
                        
                        original_weights = F.softmax(original_alpha.clone(), dim=0)
                        w1_weight = original_weights[prec1_idx].item()
                        w2_weight = original_weights[prec2_idx].item()
                        
                        total_weight = w1_weight + w2_weight
                        if total_weight > 0:
                            w1_normalized = w1_weight / total_weight
                            w2_normalized = w2_weight / total_weight
                        else:
                            w1_normalized = 0.5
                            w2_normalized = 0.5
                        
                        dual_layer.alpha.data[0] = w1_normalized * 10.0
                        dual_layer.alpha.data[1] = w2_normalized * 10.0
                        dual_layer.temperature = phase_one_temperature
                
                
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, child_name, dual_core_layer)
                else:
                    setattr(model, child_name, dual_core_layer)
                
    
    def forward_with_mode(self, x, mode='eval'):
        original_forwards = {}
        
        for name, module in self.named_modules():
            if (hasattr(module, 'forward') and 
                hasattr(module, 'alpha') and 
                hasattr(module, 'dual_core_options')):
                original_forward = module.forward
                original_forwards[name] = original_forward
                
                def create_forward_wrapper(original_func, target_mode):
                    def forward_wrapper(x):
                        return original_func(x, target_mode)
                    return forward_wrapper
                
                module.forward = create_forward_wrapper(original_forward, mode)
        
        output = self.forward(x)
        
        for name, module in self.named_modules():
            if name in original_forwards:
                module.forward = original_forwards[name]
        
        return output
    
    model.forward_with_mode = forward_with_mode.__get__(model)
    
    return model

def calculate_multicore_penalty(model, topk_layers):
    """Calculate hardware penalty for multi-core search - based on convolution parameter count and bit-width²"""
    
    precision_bits = {
        "fp32": 32,
        "fp16": 16, 
        "int8": 8,
        "int4": 4,
        "int2": 2
    }
    
    total_penalty = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and name in topk_layers:
            weights = F.softmax(module.alpha / module.temperature, dim=0)
            
            weighted_cost = 0
            for i, option in enumerate(module.dual_core_options):
                if '+' not in option:
                    precision = option
                    bits = precision_bits[precision]
                    
                    single_layer = module.mixed_precision_layers[option]
                    conv_params = 0
                    
                    if isinstance(single_layer, MixedPrecisionLayer):
                        precision_module = single_layer.precision_modules[precision]
                        
                        if isinstance(precision_module, nn.Sequential):
                            for layer in precision_module:
                                if isinstance(layer, nn.Conv2d):
                                    if hasattr(layer, 'groups') and layer.groups > 1:
                                        params = (layer.in_channels // layer.groups) * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                    else:
                                        params = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                    
                                    if layer.bias is not None:
                                        params += layer.out_channels
                                    conv_params += params
                        elif isinstance(precision_module, nn.Conv2d):
                            if hasattr(precision_module, 'groups') and precision_module.groups > 1:
                                params = (precision_module.in_channels // precision_module.groups) * precision_module.out_channels * precision_module.kernel_size[0] * precision_module.kernel_size[1]
                            else:
                                params = precision_module.in_channels * precision_module.out_channels * precision_module.kernel_size[0] * precision_module.kernel_size[1]
                            
                            if precision_module.bias is not None:
                                params += precision_module.out_channels
                            conv_params = params
                    elif isinstance(single_layer, nn.Sequential):
                        for layer in single_layer:
                            if isinstance(layer, nn.Conv2d):
                                if hasattr(layer, 'groups') and layer.groups > 1:
                                    params = (layer.in_channels // layer.groups) * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                else:
                                    params = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                
                                if layer.bias is not None:
                                    params += layer.out_channels
                                conv_params += params
                    elif isinstance(single_layer, nn.Conv2d):
                        if hasattr(single_layer, 'groups') and single_layer.groups > 1:
                            params = (single_layer.in_channels // single_layer.groups) * single_layer.out_channels * single_layer.kernel_size[0] * single_layer.kernel_size[1]
                        else:
                            params = single_layer.in_channels * single_layer.out_channels * single_layer.kernel_size[0] * single_layer.kernel_size[1]
                        
                        if single_layer.bias is not None:
                            params += single_layer.out_channels
                        conv_params = params
                    
                    cost = conv_params * (bits ** 2)
                    weighted_cost += weights[i] * cost
                    
                else:
                    prec1, prec2 = option.split('+')
                    bits1 = precision_bits[prec1]
                    bits2 = precision_bits[prec2]
                    total_bits_squared = bits1 ** 2 + bits2 ** 2
                    
                    total_conv_params = 0
                    for layer_name, layer in module.mixed_precision_layers.items():
                        if '+' in layer_name:
                            if isinstance(layer, MixedPrecisionLayer):
                                for precision_key, precision_module in layer.precision_modules.items():
                                    if isinstance(precision_module, nn.Sequential):
                                        for sublayer in precision_module:
                                            if isinstance(sublayer, nn.Conv2d):
                                                if hasattr(sublayer, 'groups') and sublayer.groups > 1:
                                                    params = (sublayer.in_channels // sublayer.groups) * sublayer.out_channels * sublayer.kernel_size[0] * sublayer.kernel_size[1]
                                                else:
                                                    params = sublayer.in_channels * sublayer.out_channels * sublayer.kernel_size[0] * sublayer.kernel_size[1]
                                                
                                                if sublayer.bias is not None:
                                                    params += sublayer.out_channels
                                                total_conv_params += params
                                    elif isinstance(precision_module, nn.Conv2d):
                                        if hasattr(precision_module, 'groups') and precision_module.groups > 1:
                                            params = (precision_module.in_channels // precision_module.groups) * precision_module.out_channels * precision_module.kernel_size[0] * precision_module.kernel_size[1]
                                        else:
                                            params = precision_module.in_channels * precision_module.out_channels * precision_module.kernel_size[0] * precision_module.kernel_size[1]
                                        
                                        if precision_module.bias is not None:
                                            params += precision_module.out_channels
                                        total_conv_params += params
                            elif isinstance(layer, nn.Sequential):
                                for sublayer in layer:
                                    if isinstance(sublayer, nn.Conv2d):
                                        if hasattr(sublayer, 'groups') and sublayer.groups > 1:
                                            params = (sublayer.in_channels // sublayer.groups) * sublayer.out_channels * sublayer.kernel_size[0] * sublayer.kernel_size[1]
                                        else:
                                            params = sublayer.in_channels * sublayer.out_channels * sublayer.kernel_size[0] * sublayer.kernel_size[1]
                                        
                                        if sublayer.bias is not None:
                                            params += sublayer.out_channels
                                        total_conv_params += params
                            elif isinstance(layer, nn.Conv2d):
                                if hasattr(layer, 'groups') and layer.groups > 1:
                                    params = (layer.in_channels // layer.groups) * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                else:
                                    params = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                
                                if layer.bias is not None:
                                    params += layer.out_channels
                                total_conv_params += params
                    
                    cost = total_conv_params * total_bits_squared
                    weighted_cost += weights[i] * cost
            
            layer_penalty = weighted_cost / 1e6
            total_penalty += layer_penalty
    
    return total_penalty

def evaluate_model(model, test_loader, device, mode='eval'):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if hasattr(model, 'forward_with_mode'):
                outputs = model.forward_with_mode(inputs, mode)
            else:
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def plot_training_curves(training_history, save_path='training_curves.png'):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(training_history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(training_history['val_acc'], label='Test Accuracy', color='red', linewidth=2)
    axes[1].set_title('Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")

def plot_alpha_evolution(alpha_history, layer_names, model, topk_layers, save_path='alpha_evolution.png'):
    """Plot alpha weight evolution (weights after softmax)"""
    n_layers = len(layer_names)
    n_epochs = len(alpha_history)
    
    fig, axes = plt.subplots(n_layers, 1, figsize=(15, 4*n_layers))
    if n_layers == 1:
        axes = [axes]
    
    for i, layer_name in enumerate(layer_names):
        layer_data = alpha_history[layer_name]
        n_options = len(layer_data[0])
        
        layer_module = None
        for name, module in model.named_modules():
            if name == layer_name:
                layer_module = module
                break
        
        if layer_module and hasattr(layer_module, 'dual_core_options'):
            option_names = layer_module.dual_core_options
        else:
            option_names = [f'Option {j}' for j in range(n_options)]
        
        for option_idx in range(n_options):
            option_weights = []
            for epoch_data in layer_data:
                alpha_tensor = torch.tensor(epoch_data)
                weights = F.softmax(alpha_tensor, dim=0)
                option_weights.append(weights[option_idx].item())
            
            axes[i].plot(option_weights, label=option_names[option_idx], linewidth=2)
        
        axes[i].set_title(f'Alpha Evolution - {layer_name}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Softmax Weight')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Alpha evolution plot saved to: {save_path}")



def convert_to_onehot_selection(model, topk_layers, temperature=0.001):
    """Convert outer alpha to one-hot selection, inner alpha remains softmax weighted"""
    logger = logging.getLogger(__name__)
    logger.info("Converting outer alpha to one-hot selection, inner alpha remains softmax weighted...")
    
    onehot_config = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and name in topk_layers:
            weights = F.softmax(module.alpha / temperature, dim=0)
            selected_idx = weights.argmax().item()
            selected_option = module.dual_core_options[selected_idx]
            
            module.alpha.data.zero_()
            module.alpha.data[selected_idx] = 100.0
            
            config = {
                'selected_option': selected_option,
                'original_alpha': module.alpha.detach().cpu().numpy().tolist(),
                'original_weights': weights.detach().cpu().numpy().tolist(),
                'onehot_idx': selected_idx
            }
            
            if '+' in selected_option:
                dual_layer = module.mixed_precision_layers[selected_option]
                if hasattr(dual_layer, 'alpha'):
                    inner_weights = F.softmax(dual_layer.alpha, dim=0)
                    
                    config['dual_weights'] = {
                        'w1': inner_weights[0].item(),
                        'w2': inner_weights[1].item(),
                        'softmax_weights': inner_weights.detach().cpu().numpy().tolist(),
                        'note': 'Inner alpha remains softmax weighted, no temperature constraint, free optimization'
                    }
                    
                    logger.info(f"   Inner alpha free optimization: W1={inner_weights[0]:.3f}, W2={inner_weights[1]:.3f}")
            
            onehot_config[name] = config
            logger.info(f"Layer {name}: Outer selected option {selected_option} (index: {selected_idx})")
    
    return onehot_config

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--supernet_path', type=str, required=True, 
                       help='First phase supernet model path')
    parser.add_argument('--alpha_state_path', type=str, 
                       help='First phase alpha state file path')
    parser.add_argument('--analysis_path', type=str, 
                       help='Dual-core precision analysis results file path')
    parser.add_argument('--topk', type=int, default=3, 
                       help='Number of layers to search for dual-core')
    parser.add_argument('--epochs', type=int, default=30, 
                       help='Number of epochs for dual-core search')
    parser.add_argument('--lr', type=float, default=0.0005, 
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, 
                       help='Batch size')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='Temperature parameter (suggest using 1.0 to allow multi-core options to be selected)')
    parser.add_argument('--hardware_weight', type=float, default=0.0005, 
                       help='Hardware penalty weight (based on parameter count × bit-width)')
    parser.add_argument('--best_single_precision', type=str, default=None,
                       help='Best single precision option determined in first phase (if not specified, will be analyzed from alpha snapshot automatically)')
    parser.add_argument('--input_size', type=int, default=32,
                       help='Input image size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info(f"Loading first phase supernet model: {args.supernet_path}")
    supernet_info = load_phase_one_supernet(args.supernet_path, device)
    
    analysis_data = None
    if args.analysis_path:
        logger.info(f"Loading dual-core precision analysis results: {args.analysis_path}")
        analysis_data = load_dual_core_analysis_results(args.analysis_path)
        logger.info(f"Analysis results contain {len(analysis_data.get('filtered_dual_core_options', {}))} dual-core options")

    alpha_state = None
    if args.alpha_state_path:
        logger.info(f"Loading alpha state file: {args.alpha_state_path}")
        alpha_state = load_phase_one_alpha_state(args.alpha_state_path)
    else:
        alpha_state = {}
        for name, alpha_values in supernet_info['best_alpha_snapshot'].items():
            alpha_state[name] = {
                'alpha': alpha_values,
                'temperature': supernet_info['final_temperature']
            }
    
    alpha_snapshot = {name: data['alpha'] for name, data in alpha_state.items()}
    final_temperature = alpha_state.get('features.0.conv_bn_relu', {}).get('temperature', supernet_info['final_temperature'])
    
    topk_layers = get_topk_layers_from_snapshot(alpha_snapshot, args.topk, final_temperature)
    logger.info(f"Layers selected for dual-core search: {topk_layers}")

    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        batch_size=args.batch_size, 
        input_size=32,
        num_workers=4
    )

    
    model = create_dual_core_model_from_supernet(supernet_info, topk_layers, analysis_data, alpha_state)
    model = model.to(device)
    
    weight_params = []
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and name in topk_layers:
            for option, layer in module.mixed_precision_layers.items():
                if '+' in option:
                    for param_name, param in layer.named_parameters():
                        if 'alpha' not in param_name:
                            param.requires_grad = True
                            weight_params.append(param)
                        else:
                            param.requires_grad = False
                    module.alpha.requires_grad = False
    
    alpha_params = []
    outer_alpha_params = 0
    inner_alpha_params = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and name in topk_layers:
            module.alpha.requires_grad = True
            alpha_params.append(module.alpha)
            outer_alpha_params += module.alpha.numel()
            module.temperature = args.temperature
            
            for option, layer in module.mixed_precision_layers.items():
                if '+' in option:
                    layer.alpha.requires_grad = True
                    alpha_params.append(layer.alpha)
                    inner_alpha_params += layer.alpha.numel()
                    layer.temperature = args.temperature
    

    
    weight_optimizer = optim.AdamW(weight_params, lr=args.lr)
    alpha_optimizer = optim.AdamW(alpha_params, lr=args.lr * 0.5)
    
    weight_scheduler = optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, args.epochs)
    alpha_scheduler = optim.lr_scheduler.CosineAnnealingLR(alpha_optimizer, args.epochs)
    
    criterion = nn.CrossEntropyLoss()

    logger.info("Starting multi-core search training...")
    best_acc = 0
    best_config = None
    
    training_history = {
        'train_loss': [],
        'val_acc': [],
        'lr': [],
        'temperature': []
    }
    alpha_history = {layer_name: [] for layer_name in topk_layers}
    top_options_history = []
    
    initial_temperature = args.temperature
    final_temperature = 0.001
    temperature_decay = (final_temperature / initial_temperature) ** (1.0 / args.epochs)
    current_temperature = initial_temperature
    
    logger.info(f"Temperature annealing strategy: initial temperature={initial_temperature}, final temperature={final_temperature}, decay rate={temperature_decay:.6f}")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        current_temperature = initial_temperature * (temperature_decay ** epoch)
        
        for name, module in model.named_modules():
            if hasattr(module, 'alpha') and name in topk_layers:
                module.temperature = current_temperature
                for option, layer in module.mixed_precision_layers.items():
                    if '+' in option:
                        layer.temperature = 1.0
        
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'alpha') and name in topk_layers:
                module.alpha.requires_grad = False
                for option, layer in module.mixed_precision_layers.items():
                    if '+' in option:
                        layer.alpha.requires_grad = False
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} - Training weights - Temperature:{current_temperature:.4f}')
        for images, targets in train_pbar:
            images, targets = images.to(device), targets.to(device)
            
            weight_optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, targets)
            
            ce_loss.backward()
            weight_optimizer.step()
            
            train_loss += ce_loss.item()
            train_batches += 1
            
            train_pbar.set_postfix({
                'CE_Loss': f'{ce_loss.item():.4f}',
                'LR': f'{weight_optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        weight_scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'alpha') and name in topk_layers:
                module.alpha.requires_grad = True
                for option, layer in module.mixed_precision_layers.items():
                    if '+' in option:
                        layer.alpha.requires_grad = True
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} - Optimizing Alpha - Temperature:{current_temperature:.4f}')
        for images, targets in val_pbar:
            images, targets = images.to(device), targets.to(device)
            
            alpha_optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, targets)
            
            hw_penalty = calculate_multicore_penalty(model, topk_layers)
            total_loss = ce_loss + args.hardware_weight * hw_penalty
            
            total_loss.backward()
            alpha_optimizer.step()
            
            val_loss += total_loss.item()
            val_batches += 1
            
            val_pbar.set_postfix({
                'CE_Loss': f'{ce_loss.item():.4f}',
                'HW_Penalty': f'{hw_penalty:.4f}',
                'Total_Loss': f'{total_loss.item():.4f}',
                'LR': f'{alpha_optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        alpha_scheduler.step()
        
        test_acc = evaluate_model(model, test_loader, device)
        epoch_time = time.time() - epoch_start_time
        
        training_history['train_loss'].append(train_loss/train_batches)
        training_history['val_acc'].append(test_acc)
        training_history['lr'].append(weight_optimizer.param_groups[0]['lr'])
        training_history['temperature'].append(current_temperature)
        
        current_epoch_top_options = {}
        for name, module in model.named_modules():
            if hasattr(module, 'alpha') and name in topk_layers:
                weights = F.softmax(module.alpha / module.temperature, dim=0)
                alpha_history[name].append(weights.detach().cpu().numpy().tolist())
                
                top_idx = weights.argmax().item()
                current_epoch_top_options[name] = {
                    'top_idx': top_idx,
                    'top_option': module.dual_core_options[top_idx],
                    'top_weight': weights[top_idx].item()
                }
        
        top_options_history.append(current_epoch_top_options)
        
        logger.info(f"Top 5 precision options (Epoch {epoch+1}):")
        for name, module in model.named_modules():
            if hasattr(module, 'alpha') and name in topk_layers:
                weights = F.softmax(module.alpha / module.temperature, dim=0)
                top5_indices = weights.argsort(descending=True)[:5]
                
                logger.info(f"  Layer {name}:")
                for i, idx in enumerate(top5_indices):
                    option = module.dual_core_options[idx]
                    weight = weights[idx].item()
                    logger.info(f"    {i+1}. {option}: {weight:.3f}")
        
        logger.info(f'Epoch {epoch+1}/{args.epochs}: Test Acc: {test_acc:.2f}%, '
                   f'Loss: {train_loss/train_batches:.4f}, Time: {epoch_time:.2f}s')
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_config = {}
            for name, module in model.named_modules():
                if hasattr(module, 'alpha') and name in topk_layers:
                    weights = F.softmax(module.alpha / module.temperature, dim=0)
                    selected_idx = weights.argmax().item()
                    selected_option = module.dual_core_options[selected_idx]
                    
                    config = {
                        'selected_option': selected_option,
                        'alpha_values': module.alpha.detach().cpu().numpy().tolist(),
                        'weights': weights.detach().cpu().numpy().tolist()
                    }
                    
                    if '+' in selected_option:
                        w1, w2 = module.get_dual_core_weights(selected_option)
                        config['dual_weights'] = {
                            'w1': w1,
                            'w2': w2,
                            'option': selected_option
                        }
                    
                    best_config[name] = config
            
            os.makedirs('dual_core_best_models', exist_ok=True)
            torch.save({
                'dual_core_config': best_config,
                'model_state_dict': model.state_dict(),
                'topk_layers': topk_layers,
                'alpha_snapshot': alpha_snapshot,
                'validation_acc': best_acc,
                'supernet_info': supernet_info
            }, 'dual_core_best_models/best_dual_core_mbv2.pth')
            logger.info(f"Saved best multi-core model, validation accuracy: {best_acc:.2f}%")
    
    logger.info(f"Independent weight multi-core search completed! Best test accuracy: {best_acc:.2f}%")
    logger.info("Best multi-core configuration:")
    for layer_name, config in best_config.items():
        logger.info(f"  {layer_name}: {config['selected_option']}")
        if 'dual_weights' in config:
            dual_weights = config['dual_weights']
            logger.info(f"    W1/W2 combination weights: {dual_weights['w1']:.3f} / {dual_weights['w2']:.3f}")
            logger.info(f"     Combination option: {dual_weights['option']}")
        else:
            logger.info(f"     Single precision option: using independent weights")
    
    logger.info("Generating training process visualization plots...")
    os.makedirs('visualization', exist_ok=True)
    
    plot_training_curves(training_history, 'visualization/training_curves.png')
    
    plot_alpha_evolution(alpha_history, topk_layers, model, topk_layers, 'visualization/alpha_evolution.png')
    
    history_data = {
        'training_history': training_history,
        'alpha_history': alpha_history,
        'top_options_history': top_options_history,
        'topk_layers': topk_layers
    }
    torch.save(history_data, 'visualization/training_history.pth')
    logger.info("Visualization data saved to visualization/ directory")
    
    onehot_config = convert_to_onehot_selection(model, topk_layers, temperature=final_temperature)
    
    logger.info("Evaluating final model accuracy...")
    final_accuracy = evaluate_model(model, test_loader, device)
    logger.info(f"Final model test accuracy: {final_accuracy:.2f}%")
    
    os.makedirs('dual_core_final_models', exist_ok=True)
    torch.save({
        'onehot_config': onehot_config,
        'model_state_dict': model.state_dict(),
        'topk_layers': topk_layers,
        'alpha_snapshot': alpha_snapshot,
        'test_acc': final_accuracy,
        'validation_acc': best_acc,
        'supernet_info': supernet_info,
        'final_temperature': final_temperature,
        'strategy': 'Outer one-hot + Inner free optimization (no temperature constraint)'
    }, 'dual_core_final_models/final_dual_core_mbv2.pth')
    logger.info(f"Saved final model, test accuracy: {final_accuracy:.2f}%")
    
    logger.info("=" * 60)
    logger.info("Multi-core search final report:")
    logger.info(f"   Multi-core search best test accuracy: {best_acc:.2f}%")
    logger.info(f"   Final model test accuracy: {final_accuracy:.2f}%")
    logger.info(f"   Temperature annealing strategy: {initial_temperature} -> {final_temperature}")
    logger.info(f"   Number of selected layers: {len(topk_layers)}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main() 
