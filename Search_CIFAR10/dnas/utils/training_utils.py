import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import time
import logging
import sys

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from models.mobilenetv2 import MixedPrecisionLayer
import torch.nn as nn


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def calculate_hardware_penalty(model, use_fixed_alpha=False):
    """Calculate hardware penalty"""
    total_penalty = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=False)
    
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            if use_fixed_alpha:
                # Use fixed alpha distribution (when training weights)
                weights = F.softmax(module.alpha.detach() / module.temperature, dim=0)
            else:
                # Use current alpha distribution (when training alpha)
                weights = F.softmax(module.alpha / module.temperature, dim=0)
            
            # Force weights to alpha's device
            weights = weights.to(module.alpha.device)
            
            # Directly use each layer's get_compute_cost method
            for i, precision in enumerate(module.precision_options):
                try:
                    # Use the layer's get_compute_cost method to get computation cost
                    cost = module.get_compute_cost(precision)
                    cost = torch.as_tensor(cost, dtype=weights.dtype, device=weights.device)
                    total_penalty = total_penalty + weights[i] * cost
                except Exception as e:
                    # If get_compute_cost fails, use default parameter count calculation
                    print(f"Warning: get_compute_cost failed for {name} with precision {precision}: {e}")
                    
                    # Calculate parameter count for this layer as fallback
                    total_params = 0
                    # In independent weight mode, use the first precision module to calculate parameter count
                    if module.precision_modules:
                        first_precision = list(module.precision_modules.keys())[0]
                        precision_module = module.precision_modules[first_precision]
                        
                        if isinstance(precision_module, nn.Sequential):
                            # For Sequential modules, calculate parameters for all layers
                            for layer in precision_module:
                                if hasattr(layer, 'weight'):
                                    if hasattr(layer, 'in_channels'):
                                        # Convolution layer parameter count
                                        if hasattr(layer, 'groups') and layer.groups > 1:
                                            params = (layer.in_channels // layer.groups) * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                        else:
                                            params = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                        
                                        if layer.bias is not None:
                                            params += layer.out_channels
                                        total_params += params
                                    else:
                                        # Linear layer parameter count
                                        params = layer.in_features * layer.out_features
                                        if layer.bias is not None:
                                            params += layer.out_features
                                        total_params += params
                        else:
                            # Single module case
                            if hasattr(precision_module, 'weight'):
                                if hasattr(precision_module, 'in_channels'):
                                    # Convolution layer parameter count
                                    if hasattr(precision_module, 'groups') and precision_module.groups > 1:
                                        params = (precision_module.in_channels // precision_module.groups) * precision_module.out_channels * precision_module.kernel_size[0] * precision_module.kernel_size[1]
                                    else:
                                        params = precision_module.in_channels * precision_module.out_channels * precision_module.kernel_size[0] * precision_module.kernel_size[1]
                                    
                                    if precision_module.bias is not None:
                                        params += precision_module.out_channels
                                    total_params = params
                                else:
                                    # Linear layer parameter count
                                    params = precision_module.in_features * precision_module.out_features
                                    if precision_module.bias is not None:
                                        params += precision_module.out_features
                                    total_params = params
                    
                    # If no parameters found, use default value
                    if total_params == 0:
                        total_params = 1000  # Default parameter count
                    
                    # Calculate precision bits
                    if precision.startswith('fp'):
                        bits = int(precision.replace('fp', ''))
                    elif precision.startswith('int'):
                        bits = int(precision.replace('int', ''))
                    else:
                        bits = 8  # Default 8 bits
                    
                    # Calculate hardware penalty for this precision = parameter count * precision bits squared
                    cost = total_params * (bits ** 2)
                    cost = torch.as_tensor(cost, dtype=weights.dtype, device=weights.device)
                    total_penalty = total_penalty + weights[i] * cost
    
    return total_penalty


def update_temperature(model, epoch, total_epochs, initial_temp=5.0, min_temp=0.1, decay_type='exponential'):
    progress = epoch / total_epochs
    
    if decay_type == 'exponential':
        # Exponential decay: smoother temperature decrease
        temp = min_temp + (initial_temp - min_temp) * np.exp(-3 * progress)
    elif decay_type == 'cosine':
        # Cosine decay: gentler decrease
        temp = min_temp + (initial_temp - min_temp) * 0.5 * (1 + np.cos(np.pi * progress))
    elif decay_type == 'linear':
        # Linear decay: simplest strategy
        temp = min_temp + (initial_temp - min_temp) * (1 - progress)
    elif decay_type == 'quadratic':
        # Quadratic decay: more aggressive decrease
        temp = min_temp + (initial_temp - min_temp) * (1 - progress)**2
    elif decay_type == 'cubic':
        # Cubic decay: very aggressive decrease, ensures full convergence to one-hot in later stages
        temp = min_temp + (initial_temp - min_temp) * (1 - progress)**3
    elif decay_type == 'step':
        # Step decay: rapid decrease at specific stages
        if progress < 0.3:
            temp = initial_temp
        elif progress < 0.6:
            temp = initial_temp * 0.5
        elif progress < 0.8:
            temp = initial_temp * 0.1
        else:
            temp = min_temp
    else:  # Default to cubic decay
        # Cubic decay: very aggressive decrease
        temp = min_temp + (initial_temp - min_temp) * (1 - progress)**3
    
    for module in model.modules():
        if isinstance(module, MixedPrecisionLayer):
            module.temperature = temp
    return temp
