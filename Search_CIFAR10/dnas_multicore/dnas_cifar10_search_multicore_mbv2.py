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

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from models.mobilenetv2 import AdaptQMobileNetV2

# 添加DNAS工具函数
dnas_utils_path = Path(__file__).parent.parent / 'dnas' / 'utils'
sys.path.append(str(dnas_utils_path))
from data_utils import create_train_val_test_loaders

def load_dual_core_analysis_results(analysis_path):
    """加载双核精度分析结果"""
    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)
    return analysis_data

def get_filtered_dual_core_options(analysis_data, layer_name):
    """获取指定层的过滤后双核选项"""
    # 将层名转换为分析结果中的格式
    analysis_layer_name = f"{layer_name}.precision_modules.fp32.0"
    
    if analysis_layer_name in analysis_data.get("filtered_dual_core_options", {}):
        return analysis_data["filtered_dual_core_options"][analysis_layer_name]
    else:
        # 如果没有找到，返回默认的双核选项
        print(f"警告: 在分析结果中未找到层 {layer_name} 的双核选项，使用默认选项")
        return [
            "fp32+fp16", "fp32+int8", "fp32+int4", "fp32+int2",
            "fp16+int8", "fp16+int4", "fp16+int2",
            "int8+int4", "int8+int2", "int4+int2"
        ]

class DualCoreLayer(nn.Module):
    """双核层：11个MixedPrecisionLayer（1个单精度 + 10个双精度组合）"""
    def __init__(self, base_layer, precision_options, best_single_precision="fp16", filtered_dual_core_options=None):
        super().__init__()
        self.precision_options = precision_options
        self.best_single_precision = best_single_precision
        
        # 创建双核组合选项：使用过滤后的选项或默认选项
        if filtered_dual_core_options:
            # 添加该层的最佳单精度选项到过滤后的双核选项中
            self.dual_core_options = [best_single_precision] + filtered_dual_core_options
            print(f"使用过滤后的双核选项: {len(filtered_dual_core_options)} 个双核组合 + 1个最佳单精度")
        else:
            # 使用默认的双核选项
            self.dual_core_options = self._create_dual_core_options()
            print(f"使用默认双核选项: {len(self.dual_core_options)} 个选项")
        
        # 创建11个MixedPrecisionLayer
        self.mixed_precision_layers = nn.ModuleDict()
        
        # 1. 单精度层：直接复制第一阶段的one-hot结果
        self.mixed_precision_layers[best_single_precision] = copy.deepcopy(base_layer)
        print(f"DEBUG: 单精度层 {best_single_precision}:")
        print(f"  - 精度选项: {self.mixed_precision_layers[best_single_precision].precision_options}")
        print(f"  - Alpha形状: {self.mixed_precision_layers[best_single_precision].alpha.shape}")
        print(f"  - Alpha值: {self.mixed_precision_layers[best_single_precision].alpha.data}")
        
        # 2. 双精度层：每个都有两个精度选项
        dual_options = filtered_dual_core_options if filtered_dual_core_options else self.dual_core_options[1:]
        print(f"DEBUG: 创建 {len(dual_options)} 个双精度层")
        for option in dual_options:
            prec1, prec2 = option.split('+')
            # 创建双精度MixedPrecisionLayer，只有两个精度选项
            dual_layer = copy.deepcopy(base_layer)
            # 修改精度选项为只有两个
            dual_layer.precision_options = [prec1, prec2]
            # 重新初始化alpha参数（只有2个）
            dual_layer.alpha = nn.Parameter(torch.zeros(2))
            self.mixed_precision_layers[option] = dual_layer
            print(f"DEBUG: 双精度层 {option}:")
            print(f"  - 精度选项: {dual_layer.precision_options}")
            print(f"  - Alpha形状: {dual_layer.alpha.shape}")
            print(f"  - Alpha值: {dual_layer.alpha.data}")
        
        # 外层Alpha参数：控制11个MixedPrecisionLayer的选择
        self.alpha = nn.Parameter(torch.zeros(len(self.dual_core_options)))
        self.temperature = 1.0
    
    def _create_dual_core_options(self):
        """创建双核选项：1个最佳单精度 + 10个双精度组合"""
        base_precisions = ["fp32", "fp16", "int8", "int4", "int2"]
        
        # 最佳单精度选项
        single_precision = [self.best_single_precision]
        
        # 双精度组合：C(5,2) = 10种组合
        dual_combinations = []
        for i in range(len(base_precisions)):
            for j in range(i+1, len(base_precisions)):
                dual_combinations.append(f"{base_precisions[i]}+{base_precisions[j]}")
        
        # 总共11个选项：1个最佳单精度 + 10个双精度组合
        return single_precision + dual_combinations
    
    def forward(self, x, mode='search'):
        if mode == 'single':
            # 单精度模式：选择alpha最大的选项
            idx = self.alpha.argmax().item()
            option = self.dual_core_options[idx]
            return self.mixed_precision_layers[option](x)
        elif mode == 'search':
            # 搜索模式：加权所有选项
            weights = F.softmax(self.alpha / self.temperature, dim=0)
            outputs = []
            for i, option in enumerate(self.dual_core_options):
                output = self.mixed_precision_layers[option](x)
                outputs.append(output * weights[i])
            return sum(outputs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def set_precision(self, option_idx):
        """设置指定的精度选项"""
        self.alpha.data.zero_()
        self.alpha.data[option_idx] = 100.0
    
    def get_current_option(self):
        """获取当前选择的选项"""
        weights = F.softmax(self.alpha / self.temperature, dim=0)
        idx = weights.argmax().item()
        return self.dual_core_options[idx]
    
    def get_dual_core_weights(self, option):
        """获取指定双核选项的W1/W2权重"""
        if option in self.mixed_precision_layers and '+' in option:
            layer = self.mixed_precision_layers[option]
            weights = F.softmax(layer.alpha / layer.temperature, dim=0)
            return weights[0].item(), weights[1].item()
        return None, None

def get_dual_core_options(best_single_precision="fp16"):
    """生成双核选项：C(5,2) = 10种双精度组合"""
    base_precisions = ["fp32", "fp16", "int8", "int4", "int2"]
    
    # 最佳单精度选项
    single_precision = [best_single_precision]
    
    # 双精度组合：C(5,2) = 10种组合
    dual_combinations = []
    for i in range(len(base_precisions)):
        for j in range(i+1, len(base_precisions)):
            dual_combinations.append(f"{base_precisions[i]}+{base_precisions[j]}")
    
    # 总共11个选项：1个最佳单精度 + 10个双精度组合
    return single_precision + dual_combinations

def analyze_alpha_sensitivity(alpha_snapshot, temperature=1.0):
    """分析alpha分布的敏感性，返回每层的敏感性分数"""
    sensitivity_scores = {}
    
    for layer_name, alpha_values in alpha_snapshot.items():
        alpha_tensor = torch.tensor(alpha_values)
        # 先应用温度进行softmax
        weights = F.softmax(alpha_tensor / temperature, dim=0)
        
        # 计算熵 - 熵越高表示越不确定，越适合多核搜索
        entropy = -(weights * torch.log(weights + 1e-10)).sum()
        
        # 计算最大权重 - 权重越分散表示越适合多核搜索
        max_weight = weights.max()
        
        # 计算权重方差 - 方差越大表示越适合多核搜索
        weight_variance = torch.var(weights)
        
        # 综合敏感性分数：熵 + 方差 - 最大权重（归一化）
        sensitivity_score = entropy.item() + weight_variance.item() - max_weight.item()
        sensitivity_scores[layer_name] = sensitivity_score
        
        # 打印调试信息
        print(f"层 {layer_name}: 熵={entropy.item():.4f}, 最大权重={max_weight.item():.4f}, 方差={weight_variance.item():.4f}, 敏感性={sensitivity_score:.4f}")
    
    return sensitivity_scores

def get_topk_layers_from_snapshot(alpha_snapshot, k, temperature=1.0):
    """基于alpha快照选择最适合多核搜索的K个层"""
    print(f"使用温度 {temperature} 进行Alpha敏感性分析...")
    sensitivity_scores = analyze_alpha_sensitivity(alpha_snapshot, temperature)
    
    # 按敏感性分数降序排序，选择最高的K个
    layer_order = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
    topk_layers = [layer_name for layer_name, score in layer_order[:k]]
    
    print(f"Alpha敏感性分析结果 (温度={temperature}):")
    for layer_name, score in layer_order[:k]:
        print(f"  {layer_name}: {score:.4f}")
    
    return topk_layers

def get_best_single_precision_from_alpha_snapshot(alpha_snapshot):
    """从alpha快照中提取最佳单精度选项"""
    precision_options = ["fp32", "fp16", "int8", "int4", "int2"]
    
    # 统计所有层选择最多的精度
    precision_counts = {precision: 0 for precision in precision_options}
    
    for layer_name, alpha_values in alpha_snapshot.items():
        alpha_tensor = torch.tensor(alpha_values)
        max_idx = alpha_tensor.argmax().item()
        if max_idx < len(precision_options):
            selected_precision = precision_options[max_idx]
            precision_counts[selected_precision] += 1
    
    # 选择被选择最多的精度作为最佳单精度
    best_precision = max(precision_counts.items(), key=lambda x: x[1])[0]
    
    print(f"从alpha快照分析最佳单精度:")
    for precision, count in precision_counts.items():
        print(f"  {precision}: {count} 层选择")
    print(f"最佳单精度: {best_precision}")
    
    return best_precision

def load_phase_one_supernet(supernet_path, device):
    """加载第一阶段保存的超网模型"""
    print(f"加载第一阶段超网模型: {supernet_path}")
    
    # 加载超网状态
    supernet_state = torch.load(supernet_path, map_location=device)
    
    # 提取关键信息
    model_state_dict = supernet_state['model_state_dict']
    best_config = supernet_state.get('best_config', {})
    best_alpha_snapshot = supernet_state.get('best_alpha_snapshot', {})
    training_history = supernet_state.get('training_history', {})
    final_epoch = supernet_state.get('final_epoch', 0)
    final_temperature = supernet_state.get('final_temperature', 0.001)
    final_accuracy = supernet_state.get('final_accuracy', 0.0)
    
    print(f"超网信息:")
    print(f"  - 最终轮数: {final_epoch}")
    print(f"  - 最终温度: {final_temperature}")
    print(f"  - 最终准确率: {final_accuracy:.2f}%")
    print(f"  - Alpha快照层数: {len(best_alpha_snapshot)}")
    print(f"  - 最佳配置层数: {len(best_config)}")
    
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
    """加载第一阶段的alpha状态文件"""
    print(f"加载第一阶段alpha状态: {alpha_state_path}")
    
    with open(alpha_state_path, 'r') as f:
        alpha_state = json.load(f)
    
    print(f"Alpha状态信息:")
    print(f"  - 层数: {len(alpha_state)}")
    
    return alpha_state

def create_dual_core_model_from_supernet(supernet_info, topk_layers, best_single_precision="fp16", analysis_data=None, alpha_state=None):
    """从超网信息创建真正的多核模型（每个精度有独立权重）"""
    print("创建多核模型（独立权重）...")
    
    # 获取第一阶段的温度
    if alpha_state:
        phase_one_temperature = alpha_state.get('features.0.conv_bn_relu', {}).get('temperature', supernet_info.get('final_temperature', 1.0))
    else:
        phase_one_temperature = supernet_info.get('final_temperature', 1.0)
    print(f"第一阶段温度: {phase_one_temperature}")
    
    # 创建基础模型（使用与超网相同的配置）
    model = AdaptQMobileNetV2(
        num_classes=10, 
        width_mult=1.0, 
        precision_options=["fp32", "fp16", "int8", "int4", "int2"],
        hardware_constraints=None, 
        pretrain_mode=False, 
        initialize_weights=False,  # 不初始化权重，直接加载超网权重
        input_size=32
    )
    
    # 加载超网权重
    model.load_state_dict(supernet_info['model_state_dict'], strict=False)
    print("成功加载超网权重")
    
    # 应用alpha快照到非topk层，确保精度设置正确
    if alpha_state:
        alpha_snapshot = {name: data['alpha'] for name, data in alpha_state.items()}
    else:
        alpha_snapshot = supernet_info['best_alpha_snapshot']
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and name in alpha_snapshot:
            if name not in topk_layers:
                # 非topk层：应用one-hot编码的精度选择并冻结
                alpha_values = torch.tensor(alpha_snapshot[name])
                max_idx = alpha_values.argmax().item()
                
                # 检查alpha参数大小是否匹配
                if module.alpha.size(0) != len(alpha_values):
                    print(f"警告: 层 {name} 的alpha大小不匹配 - 模型: {module.alpha.size(0)}, 快照: {len(alpha_values)}")
                    # 如果大小不匹配，只复制最小长度的部分
                    min_size = min(module.alpha.size(0), len(alpha_values))
                    module.alpha.data.zero_()  # 清零
                    module.alpha.data[:min_size] = alpha_values[:min_size]
                    # 确保最大值位置为1
                    if max_idx < min_size:
                        module.alpha.data[max_idx] = 100.0
                else:
                    # 大小匹配，直接设置one-hot
                    module.alpha.data.zero_()  # 清零
                    module.alpha.data[max_idx] = 100.0  # 设置one-hot
                
                # 冻结参数
                for param in module.parameters():
                    param.requires_grad = False
                module.alpha.requires_grad = False
                print(f"冻结层 {name}: 选择精度 {module.precision_options[max_idx] if max_idx < len(module.precision_options) else 'unknown'}")
            else:
                # topk层：替换为多核层（每个精度有独立权重）
                # 获取该层的过滤后双核选项
                filtered_options = None
                if analysis_data:
                    filtered_options = get_filtered_dual_core_options(analysis_data, name)
                
                # 获取该层自己的最佳精度（直接从模型state_dict中获取alpha）
                original_alpha = module.alpha.data.clone()  # 直接从当前模块获取alpha
                original_best_idx = original_alpha.argmax().item()
                layer_best_precision = module.precision_options[original_best_idx]
                
                print(f"DEBUG: 创建DualCoreLayer前的module alpha: {module.alpha.data}")
                dual_core_layer = DualCoreLayer(
                    module, 
                    module.precision_options, 
                    layer_best_precision,  # 使用该层自己的最佳精度
                    filtered_options
                )
                
                # 初始化alpha参数（基于原始alpha分布，更偏向原始单精度）
                new_alpha_size = len(dual_core_layer.dual_core_options)
                
                # 获取原始最佳单精度索引
                original_best_idx = original_alpha.argmax().item()
                original_best_precision = module.precision_options[original_best_idx]
                
                # 初始化外层alpha：给该层自己的最佳单精度选项很高的权重，其他选项较低
                dual_core_layer.alpha.data.zero_()  # 初始化为0
                
                # 找到该层自己的最佳单精度在新选项中的位置
                layer_best_precision = original_best_precision  # 该层自己的最佳精度
                best_single_idx = dual_core_layer.dual_core_options.index(layer_best_precision)
                dual_core_layer.alpha.data[best_single_idx] = 10.0  # 给该层最佳单精度很高的权重
                
                # 其他选项给予较低的初始权重
                for i in range(new_alpha_size):
                    if i != best_single_idx:
                        dual_core_layer.alpha.data[i] = -2.0
                
                print(f"多核层 {name}: 原始最佳精度 {original_best_precision} -> 新alpha大小 {new_alpha_size}")
                print(f"  外层最佳单精度权重: {dual_core_layer.alpha.data[best_single_idx]:.2f}")
                print(f"  外层其他选项权重: {dual_core_layer.alpha.data[1:].mean():.2f}")
                
                # 初始化单精度层的alpha（直接copy第一阶段的alpha和温度）
                single_layer = dual_core_layer.mixed_precision_layers[layer_best_precision]
                print(f"DEBUG: 初始化单精度层alpha:")
                print(f"  - 原始alpha: {original_alpha}")
                print(f"  - 原始alpha形状: {original_alpha.shape}")
                # 直接copy原始alpha（保持该层的one-hot状态）
                single_layer.alpha.data.copy_(original_alpha)
                print(f"  - 复制后alpha: {single_layer.alpha.data}")
                # 设置第一阶段的温度
                single_layer.temperature = phase_one_temperature
                print(f"  - 设置温度: {phase_one_temperature} (第一阶段温度)")
                
                # 验证one-hot状态
                weights = F.softmax(single_layer.alpha.data / single_layer.temperature, dim=0)
                max_idx = weights.argmax().item()
                print(f"  - Softmax权重: {weights}")
                print(f"  - 最大权重索引: {max_idx}")
                print(f"  - 选择的精度: {single_layer.precision_options[max_idx]}")
                print(f"  单精度层: 直接copy原始alpha和温度（保持该层的one-hot状态）")
                
                # 初始化双精度层的alpha（基于原始alpha分布）
                for option in dual_core_layer.dual_core_options[1:]:  # 跳过第一个（最佳单精度）
                    if '+' in option:
                        prec1, prec2 = option.split('+')
                        dual_layer = dual_core_layer.mixed_precision_layers[option]
                        
                        # 获取原始alpha中这两个精度的权重
                        prec1_idx = module.precision_options.index(prec1) if prec1 in module.precision_options else 0
                        prec2_idx = module.precision_options.index(prec2) if prec2 in module.precision_options else 0
                        
                        # 基于原始alpha权重初始化双精度层的alpha
                        original_weights = F.softmax(original_alpha.clone(), dim=0)
                        w1_weight = original_weights[prec1_idx].item()
                        w2_weight = original_weights[prec2_idx].item()
                        
                        # 归一化W1/W2权重
                        total_weight = w1_weight + w2_weight
                        if total_weight > 0:
                            w1_normalized = w1_weight / total_weight
                            w2_normalized = w2_weight / total_weight
                        else:
                            w1_normalized = 0.5
                            w2_normalized = 0.5
                        
                        # 设置双精度层的alpha（对应W1/W2权重）
                        dual_layer.alpha.data[0] = w1_normalized * 10.0  # 转换为alpha值
                        dual_layer.alpha.data[1] = w2_normalized * 10.0
                        # 设置第一阶段的温度
                        dual_layer.temperature = phase_one_temperature
                        
                        print(f"  双核选项 {option}: W1={w1_normalized:.3f}, W2={w2_normalized:.3f} (基于原始权重)")
                        print(f"    - 设置alpha: {dual_layer.alpha.data}")
                        print(f"    - 温度: {phase_one_temperature} (第一阶段温度)")
                
                # 替换模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, child_name, dual_core_layer)
                else:
                    setattr(model, child_name, dual_core_layer)
                
                print(f"多核层 {name}: 设置搜索模式，温度: {dual_core_layer.temperature}")
                print(f"多核选项: {dual_core_layer.dual_core_options}")
                print(f"MixedPrecisionLayer数量: {len(dual_core_layer.mixed_precision_layers)}")
                print(f"双精度层数量: {len([opt for opt in dual_core_layer.dual_core_options if '+' in opt])}")
    
    return model

def calculate_multicore_penalty(model, topk_layers):
    """计算多核搜索的硬件惩罚 - 基于参数量和位宽"""
    
    # 定义各精度的位宽
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
            # 获取alpha权重分布
            weights = F.softmax(module.alpha / module.temperature, dim=0)
            
            # 计算该层的参数量
            layer_params = 0
            if hasattr(module, 'mixed_precision_layers'):
                # 多核层：计算所有MixedPrecisionLayer的参数量
                for layer in module.mixed_precision_layers.values():
                    layer_params += sum(p.numel() for p in layer.parameters())
            else:
                # 普通层：计算当前层的参数量
                layer_params = sum(p.numel() for p in module.parameters())
            
            # 计算加权位宽
            weighted_bits = 0
            for i, option in enumerate(module.dual_core_options):
                if '+' not in option:
                    # 单核选项：参数量 × 位宽
                    precision = option
                    bits = precision_bits.get(precision, 32)
                    weighted_bits += weights[i] * layer_params * bits
                else:
                    # 双核选项：参数量 × (位宽1 + 位宽2)
                    prec1, prec2 = option.split('+')
                    bits1 = precision_bits.get(prec1, 32)
                    bits2 = precision_bits.get(prec2, 32)
                    total_bits = bits1 + bits2
                    weighted_bits += weights[i] * layer_params * total_bits
            
            # 硬件惩罚：加权位宽（归一化到合理范围）
            # 将位宽除以1e6来缩小数值范围
            total_penalty += weighted_bits / 1e6
    
    return total_penalty

def calculate_layer_complexity(layer_name, module, precision_bits):
    """计算单个层的复杂度（用于调试）"""
    if not hasattr(module, 'alpha'):
        return None
    
    # 计算参数量
    layer_params = 0
    if hasattr(module, 'mixed_precision_layers'):
        for layer in module.mixed_precision_layers.values():
            layer_params += sum(p.numel() for p in layer.parameters())
    else:
        layer_params = sum(p.numel() for p in module.parameters())
    
    # 获取alpha权重
    weights = F.softmax(module.alpha / module.temperature, dim=0)
    
    # 计算各选项的复杂度
    complexities = {}
    for i, option in enumerate(module.dual_core_options):
        if '+' not in option:
            # 单核选项
            precision = option
            bits = precision_bits.get(precision, 32)
            complexity = layer_params * bits
            complexities[option] = {
                'type': 'single',
                'precision': precision,
                'bits': bits,
                'params': layer_params,
                'complexity': complexity,
                'weight': weights[i].item()
            }
        else:
            # 双核选项
            prec1, prec2 = option.split('+')
            bits1 = precision_bits.get(prec1, 32)
            bits2 = precision_bits.get(prec2, 32)
            total_bits = bits1 + bits2
            complexity = layer_params * total_bits
            complexities[option] = {
                'type': 'dual',
                'precisions': [prec1, prec2],
                'bits': [bits1, bits2],
                'total_bits': total_bits,
                'params': layer_params,
                'complexity': complexity,
                'weight': weights[i].item()
            }
    
    return {
        'layer_name': layer_name,
        'layer_params': layer_params,
        'complexities': complexities,
        'total_weighted_complexity': sum(comp['complexity'] * comp['weight'] for comp in complexities.values())
    }

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
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

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--supernet_path', type=str, required=True, 
                       help='第一阶段超网模型路径')
    parser.add_argument('--alpha_state_path', type=str, 
                       help='第一阶段alpha状态文件路径')
    parser.add_argument('--analysis_path', type=str, 
                       help='双核精度分析结果文件路径')
    parser.add_argument('--topk', type=int, default=3, 
                       help='选择进行双核搜索的层数')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='双核搜索的训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=384, 
                       help='批次大小')
    parser.add_argument('--temperature', type=float, default=0.0001, 
                       help='温度参数（建议使用0.0001确保one-hot收敛）')
    parser.add_argument('--hardware_weight', type=float, default=0.001, 
                       help='硬件惩罚权重（基于参数量×位宽计算）')
    parser.add_argument('--best_single_precision', type=str, default=None,
                       help='第一阶段确定的最佳单精度选项（如果不指定，将从alpha快照自动分析）')
    parser.add_argument('--input_size', type=int, default=32,
                       help='输入图像尺寸')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载第一阶段超网模型
    logger.info(f"加载第一阶段超网模型: {args.supernet_path}")
    supernet_info = load_phase_one_supernet(args.supernet_path, device)
    
    # 加载双核精度分析结果（如果提供）
    analysis_data = None
    if args.analysis_path:
        logger.info(f"加载双核精度分析结果: {args.analysis_path}")
        analysis_data = load_dual_core_analysis_results(args.analysis_path)
        logger.info(f"分析结果包含 {len(analysis_data.get('filtered_dual_core_options', {}))} 层的双核选项")

    # 加载alpha状态（如果提供）
    alpha_state = None
    if args.alpha_state_path:
        logger.info(f"加载alpha状态文件: {args.alpha_state_path}")
        alpha_state = load_phase_one_alpha_state(args.alpha_state_path)
    else:
        # 如果没有提供alpha_state_path，从supernet_info中提取alpha快照
        logger.info("未提供alpha_state_path，从supernet_info中提取alpha快照")
        alpha_state = {}
        # 从超网信息中构建alpha_state格式
        for name, alpha_values in supernet_info['best_alpha_snapshot'].items():
            alpha_state[name] = {
                'alpha': alpha_values,
                'temperature': supernet_info['final_temperature']
            }
    
    # 基于alpha快照选择topk层
    alpha_snapshot = {name: data['alpha'] for name, data in alpha_state.items()}
    final_temperature = alpha_state.get('features.0.conv_bn_relu', {}).get('temperature', supernet_info['final_temperature'])
    
    logger.info(f"使用第一阶段最终温度: {final_temperature}")
    topk_layers = get_topk_layers_from_snapshot(alpha_snapshot, args.topk, final_temperature)
    logger.info(f"选择进行双核搜索的层: {topk_layers}")

    # 自动分析最佳单精度选项
    if args.best_single_precision is None:
        best_single_precision = get_best_single_precision_from_alpha_snapshot(alpha_snapshot)
    else:
        best_single_precision = args.best_single_precision
    logger.info(f"使用最佳单精度选项: {best_single_precision}")

    # 数据集加载
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        batch_size=args.batch_size, 
        input_size=32,
        num_workers=4
    )

    # 首先测试第一阶段超网的准确率
    logger.info("测试第一阶段超网的准确率...")
    logger.info(f"第一阶段搜索报告的最佳准确率: {supernet_info['final_accuracy']:.2f}%")
    
    # 直接使用第一阶段搜索的模型状态进行测试
    phase_one_model = AdaptQMobileNetV2(
        num_classes=10, 
        width_mult=1.0, 
        precision_options=["fp32", "fp16", "int8", "int4", "int2"],
        hardware_constraints=None, 
        pretrain_mode=False, 
        initialize_weights=False,
        input_size=32
    )
    # 加载完整的模型状态（包括alpha参数）
    phase_one_model.load_state_dict(supernet_info['model_state_dict'], strict=False)
    phase_one_model = phase_one_model.to(device)
    
    # 确保模型处于评估模式
    phase_one_model.eval()
    
    # 显示当前模型的alpha分布（不修改，直接使用加载的状态）
    logger.info("使用加载的模型状态进行测试...")
    logger.info(f"使用最终温度: {supernet_info['final_temperature']}")
    
    # 显示当前模型的alpha分布
    for name, module in phase_one_model.named_modules():
        if hasattr(module, 'alpha'):
            weights = F.softmax(module.alpha / supernet_info['final_temperature'], dim=0)
            max_idx = weights.argmax().item()
            selected_precision = module.precision_options[max_idx]
            logger.info(f"层 {name}: 选择精度 {selected_precision} (权重: {weights[max_idx]:.4f})")

    
    # 设置为评估模式并测试
    phase_one_model.eval()
    phase_one_accuracy = evaluate_model(phase_one_model, test_loader, device)
    logger.info(f"第一阶段超网准确率（测试集）: {phase_one_accuracy:.2f}%")
    
    # 创建多核模型（每个精度有独立权重）
    dual_core_options = get_dual_core_options(best_single_precision)
    logger.info(f"多核选项（共{len(dual_core_options)}个）: {dual_core_options}")
    logger.info(f"架构说明: 独立权重 + W1/W2组合 + Alpha路径选择")
    
    model = create_dual_core_model_from_supernet(supernet_info, topk_layers, best_single_precision, analysis_data, alpha_state)
    model = model.to(device)
    
    # 测试初始模型的准确率
    logger.info("测试初始多核模型的准确率...")
    initial_accuracy = evaluate_model(model, val_loader, device)
    logger.info(f"初始多核模型准确率: {initial_accuracy:.2f}%")
    
    # 如果准确率太低，可能需要检查初始化
    if initial_accuracy < 70.0:
        logger.warning(f"初始准确率较低 ({initial_accuracy:.2f}%)，可能需要检查模型初始化")
    
    # 冻结所有权重参数，只训练alpha
    for name, param in model.named_parameters():
        if 'alpha' not in name:  # 冻结所有非alpha参数
            param.requires_grad = False
    
    # 只收集alpha参数进行训练
    trainable_params = []
    outer_alpha_params = 0
    inner_alpha_params = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and name in topk_layers:
            # 外层alpha（控制11个MixedPrecisionLayer的选择）
            trainable_params.append(module.alpha)
            outer_alpha_params += module.alpha.numel()  # 统计alpha参数数量
            module.temperature = args.temperature
            
            # 内层alpha（控制双精度层的W1/W2）
            for option, layer in module.mixed_precision_layers.items():
                if '+' in option:  # 双精度层
                    trainable_params.append(layer.alpha)
                    inner_alpha_params += layer.alpha.numel()  # 统计alpha参数数量
                    layer.temperature = args.temperature
    
    logger.info(f"可训练参数统计:")
    logger.info(f"  - 外层Alpha参数（控制11个MixedPrecisionLayer）: {outer_alpha_params} 层")
    logger.info(f"  - 内层Alpha参数（控制双精度层W1/W2）: {inner_alpha_params} 个")
    logger.info(f"  - 总可训练参数数量: {len(trainable_params)}")
    logger.info(f"  - 所有权重参数已冻结，只训练alpha")
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # 训练循环
    logger.info("开始多核搜索训练（独立权重）...")
    best_acc = 0
    best_config = None
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} - 独立权重多核搜索')
        for images, targets in train_pbar:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, targets)
            
            # 硬件惩罚
            hw_penalty = calculate_multicore_penalty(model, topk_layers)
            total_loss = ce_loss + args.hardware_weight * hw_penalty
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_batches += 1
            
            train_pbar.set_postfix({
                'CE_Loss': f'{ce_loss.item():.4f}',
                'HW_Penalty': f'{hw_penalty:.4f}',
                'Total_Loss': f'{total_loss.item():.4f}'
            })
        
        scheduler.step()
        
        # 验证阶段
        val_acc = evaluate_model(model, val_loader, device)
        epoch_time = time.time() - epoch_start_time
        
        # 计算并显示硬件惩罚详细信息（每10个epoch显示一次）
        if epoch % 10 == 0:
            logger.info(f"硬件惩罚详细分析 (Epoch {epoch+1}):")
            precision_bits = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4, "int2": 2}
            
            for name, module in model.named_modules():
                if hasattr(module, 'alpha') and name in topk_layers:
                    complexity_info = calculate_layer_complexity(name, module, precision_bits)
                    if complexity_info:
                        logger.info(f"  层 {name}:")
                        logger.info(f"    参数量: {complexity_info['layer_params']:,}")
                        logger.info(f"    加权复杂度: {complexity_info['total_weighted_complexity']:,.0f}")
                        
                        # 显示前3个最高权重的选项
                        sorted_options = sorted(
                            complexity_info['complexities'].items(),
                            key=lambda x: x[1]['weight'],
                            reverse=True
                        )[:3]
                        
                        for option, info in sorted_options:
                            if info['type'] == 'single':
                                logger.info(f"    {option}: {info['precision']} ({info['bits']}位) - 权重: {info['weight']:.3f}")
                            else:
                                logger.info(f"    {option}: {info['precisions'][0]}({info['bits'][0]}位) + {info['precisions'][1]}({info['bits'][1]}位) - 权重: {info['weight']:.3f}")
        
        logger.info(f'Epoch {epoch+1}/{args.epochs}: Val Acc: {val_acc:.2f}%, '
                   f'Loss: {train_loss/train_batches:.4f}, Time: {epoch_time:.2f}s')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳配置
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
                    
                    # 如果是双核选项，保存W1和W2权重
                    if '+' in selected_option:
                        w1, w2 = module.get_dual_core_weights(selected_option)
                        config['dual_weights'] = {
                            'w1': w1,
                            'w2': w2,
                            'option': selected_option
                        }
                    
                    best_config[name] = config
            
            # 保存最佳模型
            os.makedirs('dual_core_best_models', exist_ok=True)
            torch.save({
                'dual_core_config': best_config,
                'model_state_dict': model.state_dict(),
                'topk_layers': topk_layers,
                'alpha_snapshot': alpha_snapshot,
                'validation_acc': best_acc,
                'supernet_info': supernet_info
            }, 'dual_core_best_models/best_dual_core_mbv2.pth')
            logger.info(f"保存最佳多核模型，验证准确率: {best_acc:.2f}%")
    
    logger.info(f"独立权重多核搜索完成！最佳验证准确率: {best_acc:.2f}%")
    logger.info("最佳多核配置（独立权重 + W1/W2组合）:")
    for layer_name, config in best_config.items():
        logger.info(f"  {layer_name}: {config['selected_option']}")
        if 'dual_weights' in config:
            dual_weights = config['dual_weights']
            logger.info(f"    W1/W2组合权重: {dual_weights['w1']:.3f} / {dual_weights['w2']:.3f}")
            logger.info(f"    组合选项: {dual_weights['option']}")
        else:
            logger.info(f"    单精度选项: 使用独立权重")

if __name__ == '__main__':
    main() 
