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
from adaptqnet.models.mobilenetv2 import AdaptQMobileNetV2

class SharedWeightDualCoreLayer(nn.Module):
    """共享权重的双核层：W1/W2组合 + Alpha路径选择"""
    def __init__(self, base_layer, precision_options, best_single_precision="fp16"):
        super().__init__()
        self.precision_options = precision_options  # 原始5个精度选项
        self.best_single_precision = best_single_precision
        
        # 共享的基础层（所有精度共享同一套权重）
        self.shared_layer = base_layer
        
        # 创建双核组合选项：1个最佳单精度 + 10个双精度组合
        self.dual_core_options = self._create_dual_core_options()
        
        # 每个双核选项的W1/W2权重参数
        self.dual_core_weights = nn.ParameterDict()
        for option in self.dual_core_options[1:]:  # 跳过第一个（最佳单精度）
            self.dual_core_weights[option] = nn.Parameter(torch.tensor([0.5, 0.5]))  # 初始权重各0.5
        
        # Alpha参数：控制11个选项的路径选择
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
            return self._compute_option(x, option)
        elif mode == 'search':
            # 搜索模式：加权所有选项
            weights = F.softmax(self.alpha / self.temperature, dim=0)
            outputs = []
            for i, option in enumerate(self.dual_core_options):
                output = self._compute_option(x, option)
                outputs.append(output * weights[i])
            return sum(outputs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_option(self, x, option):
        """计算指定选项的输出"""
        if '+' not in option:
            # 单精度选项：直接使用共享层
            return self._compute_with_precision(x, option)
        else:
            # 双精度选项：使用W1/W2组合merge
            prec1, prec2 = option.split('+')
            
            # 获取W1/W2权重
            dual_weights = self.dual_core_weights[option]
            w1, w2 = F.softmax(dual_weights, dim=0)  # 归一化权重
            
            # 两个精度的计算（使用共享权重）
            output1 = self._compute_with_precision(x, prec1)
            output2 = self._compute_with_precision(x, prec2)
            
            # W1/W2组合merge
            merged_output = w1 * output1 + w2 * output2
            return merged_output
    
    def _compute_with_precision(self, x, precision):
        """使用指定精度计算输出（共享权重）"""
        # 根据精度类型对输入和权重进行相应处理
        if precision == "fp32":
            return self.shared_layer(x)
        elif precision == "fp16":
            # 将输入转换为fp16进行计算
            with torch.cuda.amp.autocast():
                return self.shared_layer(x.half()).float()
        elif precision == "int8":
            # 模拟int8量化（实际实现中需要真正的量化）
            # 这里简化处理，实际应该对权重进行int8量化
            return self.shared_layer(x)
        elif precision == "int4":
            # 模拟int4量化
            return self.shared_layer(x)
        elif precision == "int2":
            # 模拟int2量化
            return self.shared_layer(x)
        else:
            return self.shared_layer(x)
    
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
        if option in self.dual_core_weights:
            weights = self.dual_core_weights[option]
            w1, w2 = F.softmax(weights, dim=0)
            return w1.item(), w2.item()
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
    
    return sensitivity_scores

def get_topk_layers_from_snapshot(alpha_snapshot, k, temperature=1.0):
    """基于alpha快照选择最适合多核搜索的K个层"""
    sensitivity_scores = analyze_alpha_sensitivity(alpha_snapshot, temperature)
    
    # 按敏感性分数降序排序，选择最高的K个
    layer_order = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
    topk_layers = [layer_name for layer_name, score in layer_order[:k]]
    
    print(f"Alpha敏感性分析结果:")
    for layer_name, score in layer_order[:k]:
        print(f"  {layer_name}: {score:.4f}")
    
    return topk_layers

def load_phase_one_state_with_snapshot(model, phase_one_ckpt, topk_layers, alpha_snapshot):
    """加载第一阶段状态，并应用alpha快照"""
    ckpt = torch.load(phase_one_ckpt, map_location='cpu', weights_only=False)
    
    # 只加载模型权重，不加载alpha参数（因为尺寸不匹配）
    model_state_dict = ckpt['model_state_dict']
    # 过滤掉alpha参数和alpha_pact_dict
    filtered_state_dict = {}
    for k, v in model_state_dict.items():
        if not k.endswith('.alpha') and not 'alpha_pact_dict' in k:
            filtered_state_dict[k] = v
    
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # 应用alpha快照到非topk层，确保精度设置正确
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
                
                # 确保精度设置正确
                if hasattr(module, 'set_precision'):
                    # 根据实际的精度选项来确定精度字符串
                    if hasattr(module, 'precision_options'):
                        precision_options = module.precision_options
                        if max_idx < len(precision_options):
                            precision_str = precision_options[max_idx]
                            module.set_precision(precision_str)
                            print(f"冻结层 {name}: 选择精度 {precision_str} (索引 {max_idx})")
                        else:
                            print(f"警告: 层 {name} 的max_idx {max_idx} 超出精度选项范围 {len(precision_options)}")
                    else:
                        # 使用默认的精度映射
                        original_precisions = ["fp32", "fp16", "int8", "int4", "int2"]
                        if max_idx < len(original_precisions):
                            precision_str = original_precisions[max_idx]
                            module.set_precision(precision_str)
                            print(f"冻结层 {name}: 选择精度 {precision_str} (索引 {max_idx})")
                
                # 冻结参数
                for param in module.parameters():
                    param.requires_grad = False
                module.alpha.requires_grad = False
            else:
                # topk层：保持可训练状态（将在replace_and_freeze_layers中处理）
                pass

def replace_and_freeze_layers(model, topk_layers, alpha_snapshot, best_single_precision="fp16"):
    """替换topk层为共享权重的双核层，并冻结其他层"""
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and name in alpha_snapshot:
            if name in topk_layers:
                # 创建共享权重的双核层
                dual_core_layer = SharedWeightDualCoreLayer(
                    module, 
                    module.precision_options, 
                    best_single_precision
                )
                
                # 初始化alpha参数（基于原始alpha分布）
                original_alpha = torch.tensor(alpha_snapshot[name])
                # 将原始alpha映射到双核选项
                new_alpha_size = len(dual_core_layer.dual_core_options)
                dual_core_layer.alpha.data.zero_()  # 初始化为0
                
                # 检查大小匹配
                if len(original_alpha) != new_alpha_size:
                    print(f"双核层 {name}: 原始alpha大小 {len(original_alpha)} -> 新alpha大小 {new_alpha_size}")
                    # 如果大小不匹配，采用不同的初始化策略
                    min_size = min(len(original_alpha), new_alpha_size)
                    # 将原始alpha的前min_size个值复制到新alpha的前min_size个位置
                    dual_core_layer.alpha.data[:min_size] = original_alpha[:min_size]
                    # 对于新增的选项，给予较小的初始值
                    if new_alpha_size > len(original_alpha):
                        dual_core_layer.alpha.data[len(original_alpha):] = -2.0  # 较小的初始值
                else:
                    # 大小匹配，直接复制
                    dual_core_layer.alpha.data[:] = original_alpha[:]
                
                # 确保双核层使用正确的计算模式
                dual_core_layer.temperature = 1.0
                print(f"共享权重双核层 {name}: 设置搜索模式，温度: {dual_core_layer.temperature}")
                print(f"双核选项: {dual_core_layer.dual_core_options}")
                print(f"W1/W2权重参数: {len(dual_core_layer.dual_core_weights)} 个双核组合")
                
                # 替换模块
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, child_name, dual_core_layer)
                else:
                    setattr(model, child_name, dual_core_layer)
            # 非topk层已经在load_phase_one_state_with_snapshot中处理过了，这里不需要重复处理

def calculate_dual_core_penalty(model, topk_layers):
    """计算双核搜索的硬件惩罚"""
    total_penalty = 0
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and name in topk_layers:
            weights = F.softmax(module.alpha / module.temperature, dim=0)
            # 简化的硬件惩罚：基于alpha权重的方差
            weight_variance = torch.var(weights)
            total_penalty += weight_variance.item()
    return total_penalty

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
    parser.add_argument('--phase_one_ckpt', type=str, required=True, 
                       help='第一阶段的最佳模型检查点路径')
    parser.add_argument('--topk', type=int, default=3, 
                       help='选择进行双核搜索的层数')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='双核搜索的训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='批次大小')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='温度参数')
    parser.add_argument('--hardware_weight', type=float, default=0.01, 
                       help='硬件惩罚权重')
    parser.add_argument('--best_single_precision', type=str, default='fp16',
                       help='第一阶段确定的最佳单精度选项')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载第一阶段检查点
    logger.info(f"加载第一阶段检查点: {args.phase_one_ckpt}")
    ckpt = torch.load(args.phase_one_ckpt, map_location=device, weights_only=False)
    
    # 获取alpha快照
    if 'alpha_snapshot' in ckpt:
        alpha_snapshot = ckpt['alpha_snapshot']
        logger.info(f"成功加载alpha快照，包含 {len(alpha_snapshot)} 层")
    else:
        logger.error("检查点中未找到alpha_snapshot，请确保使用修改后的DNAS脚本")
        return

    # 基于alpha快照选择topk层
    topk_layers = get_topk_layers_from_snapshot(alpha_snapshot, args.topk, args.temperature)
    logger.info(f"选择进行双核搜索的层: {topk_layers}")

    # 数据集加载
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set = CIFAR10(root='../../data', train=True, download=True, transform=transform)
    val_set = CIFAR10(root='../../data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 构建共享权重的双核supernet
    dual_core_options = get_dual_core_options(args.best_single_precision)
    logger.info(f"双核选项（共{len(dual_core_options)}个）: {dual_core_options}")
    logger.info(f"最佳单精度选项: {args.best_single_precision}")
    logger.info(f"架构说明: 共享权重 + W1/W2组合 + Alpha路径选择")
    
    model = AdaptQMobileNetV2(
        num_classes=10, 
        width_mult=1.0, 
        precision_options=["fp32", "fp16", "int8", "int4", "int2"],  # 原始5个精度
        hardware_constraints=None, 
        pretrain_mode=False, 
        initialize_weights=True
    )
    
    # 加载第一阶段状态并应用alpha快照
    load_phase_one_state_with_snapshot(model, args.phase_one_ckpt, topk_layers, alpha_snapshot)
    replace_and_freeze_layers(model, topk_layers, alpha_snapshot, args.best_single_precision)
    model = model.to(device)
    
    # 只收集topk层的参数进行训练
    trainable_params = []
    alpha_params = 0
    w1w2_params = 0
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and name in topk_layers:
            trainable_params.append(module.alpha)
            alpha_params += 1
            # 添加W1/W2权重参数
            if hasattr(module, 'dual_core_weights'):
                for weight_param in module.dual_core_weights.values():
                    trainable_params.append(weight_param)
                    w1w2_params += 1
            module.temperature = args.temperature
    
    logger.info(f"可训练参数统计:")
    logger.info(f"  - Alpha参数（路径选择）: {alpha_params} 层")
    logger.info(f"  - W1/W2参数（组合权重）: {w1w2_params} 个")
    logger.info(f"  - 总可训练参数数量: {len(trainable_params)}")
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # 训练循环
    logger.info("开始双核搜索训练...")
    best_acc = 0
    best_config = None
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} - 共享权重双核搜索')
        for images, targets in train_pbar:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, targets)
            
            # 硬件惩罚
            hw_penalty = calculate_dual_core_penalty(model, topk_layers)
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
                'validation_acc': best_acc
            }, 'dual_core_best_models/best_dual_core_mbv2.pth')
            logger.info(f"保存最佳双核模型，验证准确率: {best_acc:.2f}%")
    
    logger.info(f"共享权重双核搜索完成！最佳验证准确率: {best_acc:.2f}%")
    logger.info("最佳双核配置（共享权重 + W1/W2组合）:")
    for layer_name, config in best_config.items():
        logger.info(f"  {layer_name}: {config['selected_option']}")
        if 'dual_weights' in config:
            dual_weights = config['dual_weights']
            logger.info(f"    W1/W2组合权重: {dual_weights['w1']:.3f} / {dual_weights['w2']:.3f}")
            logger.info(f"    组合选项: {dual_weights['option']}")
        else:
            logger.info(f"    单精度选项: 使用共享权重")

if __name__ == '__main__':
    main() 