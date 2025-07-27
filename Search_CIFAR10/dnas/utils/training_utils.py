import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import time
import logging
from adaptqnet.models.mobilenetv2 import MixedPrecisionLayer
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
    """计算硬件惩罚"""
    total_penalty = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=False)
    
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            if use_fixed_alpha:
                # 使用固定的alpha分布（训练权重时）
                weights = F.softmax(module.alpha.detach() / module.temperature, dim=0)
            else:
                # 使用当前的alpha分布（训练alpha时）
                weights = F.softmax(module.alpha / module.temperature, dim=0)
            
            # 强制weights到alpha的device
            weights = weights.to(module.alpha.device)
            
            # 直接使用每层的get_compute_cost方法
            for i, precision in enumerate(module.precision_options):
                try:
                    # 使用该层的get_compute_cost方法获取计算成本
                    cost = module.get_compute_cost(precision)
                    cost = torch.as_tensor(cost, dtype=weights.dtype, device=weights.device)
                    total_penalty = total_penalty + weights[i] * cost
                except Exception as e:
                    # 如果get_compute_cost失败，使用默认的参数量计算
                    print(f"Warning: get_compute_cost failed for {name} with precision {precision}: {e}")
                    
                    # 计算该层的参数量作为fallback
                    total_params = 0
                    # 在共享权重模式下，使用基础模块计算参数量
                    base_module = module.base_module
                    if hasattr(base_module, 'weight'):
                        if hasattr(base_module, 'in_channels'):
                            # 卷积层参数量
                            if hasattr(base_module, 'groups') and base_module.groups > 1:
                                params = (base_module.in_channels // base_module.groups) * base_module.out_channels * base_module.kernel_size[0] * base_module.kernel_size[1]
                            else:
                                params = base_module.in_channels * base_module.out_channels * base_module.kernel_size[0] * base_module.kernel_size[1]
                            
                            if base_module.bias is not None:
                                params += base_module.out_channels
                            total_params = params
                        else:
                            # 线性层参数量
                            params = base_module.in_features * base_module.out_features
                            if base_module.bias is not None:
                                params += base_module.out_features
                            total_params = params
                    elif isinstance(base_module, nn.Sequential):
                        # 对于Sequential模块，计算所有层的参数量
                        for layer in base_module:
                            if hasattr(layer, 'weight'):
                                if hasattr(layer, 'in_channels'):
                                    # 卷积层参数量
                                    if hasattr(layer, 'groups') and layer.groups > 1:
                                        params = (layer.in_channels // layer.groups) * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                    else:
                                        params = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
                                    
                                    if layer.bias is not None:
                                        params += layer.out_channels
                                    total_params += params
                                else:
                                    # 线性层参数量
                                    params = layer.in_features * layer.out_features
                                    if layer.bias is not None:
                                        params += layer.out_features
                                    total_params += params
                    
                    # 如果没有找到参数，使用默认值
                    if total_params == 0:
                        total_params = 1000  # 默认参数量
                    
                    # 计算精度位数
                    if precision.startswith('fp'):
                        bits = int(precision.replace('fp', ''))
                    elif precision.startswith('int'):
                        bits = int(precision.replace('int', ''))
                    else:
                        bits = 8  # 默认8位
                    
                    # 计算该精度的硬件惩罚 = 参数量 * 精度位数的平方
                    cost = total_params * (bits ** 2)
                    cost = torch.as_tensor(cost, dtype=weights.dtype, device=weights.device)
                    total_penalty = total_penalty + weights[i] * cost
    
    return total_penalty


def update_temperature(model, epoch, total_epochs, initial_temp=5.0, min_temp=0.1, decay_type='exponential'):
    progress = epoch / total_epochs
    
    if decay_type == 'exponential':
        # 指数衰减：更平滑的温度下降
        temp = min_temp + (initial_temp - min_temp) * np.exp(-3 * progress)
    elif decay_type == 'cosine':
        # 余弦衰减：更温和的下降
        temp = min_temp + (initial_temp - min_temp) * 0.5 * (1 + np.cos(np.pi * progress))
    elif decay_type == 'linear':
        # 线性衰减：最简单的策略
        temp = min_temp + (initial_temp - min_temp) * (1 - progress)
    else:  # quadratic (原来的策略)
        # 二次衰减：更激进的下降
        temp = min_temp + (initial_temp - min_temp) * (1 - progress)**2
    
    for module in model.modules():
        if isinstance(module, MixedPrecisionLayer):
            module.temperature = temp
    return temp


def should_sample_subnet(epoch, total_epochs, current_sampling_count, max_sampling_times, sampling_schedule='dynamic'):
    """决定是否应该采样子网"""
    if current_sampling_count >= max_sampling_times:
        return False
    
    progress = epoch / total_epochs
    
    if sampling_schedule == 'dynamic':
        # 动态采样：开始密集，后期稀疏
        if progress < 0.3:  # 前30%每5代采样一次
            return epoch % 5 == 0
        elif progress < 0.6:  # 30%-60%每10代采样一次
            return epoch % 10 == 0
        else:  # 60%以后每15代采样一次
            return epoch % 15 == 0
    elif sampling_schedule == 'uniform':
        # 均匀采样
        sampling_interval = total_epochs // max_sampling_times
        return epoch % sampling_interval == 0
    else:
        # 固定间隔采样
        return epoch % 10 == 0


def sample_and_save_subnet(model, epoch, current_temp, device, sampled_subnets, save_dir='mbv2_sampled_subnets'):
    """采样并保存子网"""
    # 创建logger
    logger = logging.getLogger(__name__)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取当前精度配置和alpha分布
    current_config = {}
    alpha_snapshot = {}
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            weights = F.softmax(module.alpha / current_temp, dim=0)
            # 确保selected_idx不超出precision_options的范围
            selected_idx = min(weights.argmax().item(), len(module.precision_options) - 1)
            current_config[name] = module.precision_options[selected_idx]
            # 保存原始alpha分布
            alpha_snapshot[name] = module.alpha.detach().cpu().numpy().tolist()
    
    # 创建模型副本并应用精度配置
    model_copy = copy.deepcopy(model)
    from .model_utils import apply_precision_config
    apply_precision_config(model_copy, current_config)
    
    # 保存子网信息
    subnet_info = {
        'epoch': epoch,
        'temperature': current_temp,
        'precision_config': current_config,
        'alpha_snapshot': alpha_snapshot,  # 新增：保存alpha分布
        'model_state_dict': model_copy.state_dict(),
        'sampling_time': time.time()
    }
    
    # 保存到文件
    subnet_path = os.path.join(save_dir, f'subnet_epoch_{epoch:03d}.pth')
    torch.save(subnet_info, subnet_path)
    
    # 添加到内存列表
    sampled_subnets.append(subnet_info)
    
    logger.info(f"✓ 采样并保存子网 (Epoch {epoch}): {subnet_path}")
    return subnet_info 