#!/usr/bin/env python3
"""
测试共享权重双核层的功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from dnas_cifar10_search_multicore_mbv2 import SharedWeightDualCoreLayer

class SimpleTestLayer(nn.Module):
    """简单的测试层"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.linear(x))

def test_shared_weight_dual_core():
    """测试共享权重双核层"""
    print("=== 测试共享权重双核层 ===")
    
    # 创建基础层
    base_layer = SimpleTestLayer(10, 5)
    precision_options = ["fp32", "fp16", "int8", "int4", "int2"]
    
    # 创建双核层
    dual_core_layer = SharedWeightDualCoreLayer(
        base_layer, 
        precision_options, 
        best_single_precision="fp16"
    )
    
    print(f"双核选项数量: {len(dual_core_layer.dual_core_options)}")
    print(f"双核选项: {dual_core_layer.dual_core_options}")
    print(f"W1/W2权重参数数量: {len(dual_core_layer.dual_core_weights)}")
    
    # 测试输入
    x = torch.randn(2, 10)
    
    # 测试单精度模式
    print("\n=== 测试单精度模式 ===")
    dual_core_layer.set_precision(0)  # 选择第一个选项（fp16）
    output_single = dual_core_layer(x, mode='single')
    print(f"单精度输出形状: {output_single.shape}")
    print(f"当前选择选项: {dual_core_layer.get_current_option()}")
    
    # 测试双精度模式
    print("\n=== 测试双精度模式 ===")
    dual_core_layer.set_precision(1)  # 选择第二个选项（fp32+fp16）
    output_dual = dual_core_layer(x, mode='single')
    print(f"双精度输出形状: {output_dual.shape}")
    print(f"当前选择选项: {dual_core_layer.get_current_option()}")
    
    # 获取W1/W2权重
    w1, w2 = dual_core_layer.get_dual_core_weights("fp32+fp16")
    print(f"W1/W2权重: {w1:.3f} / {w2:.3f}")
    
    # 测试搜索模式
    print("\n=== 测试搜索模式 ===")
    output_search = dual_core_layer(x, mode='search')
    print(f"搜索模式输出形状: {output_search.shape}")
    
    # 测试参数数量
    total_params = sum(p.numel() for p in dual_core_layer.parameters())
    trainable_params = sum(p.numel() for p in dual_core_layer.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"总参数数量: {total_params}")
    print(f"可训练参数数量: {trainable_params}")
    
    print("\n=== 测试完成 ===")

if __name__ == '__main__':
    test_shared_weight_dual_core() 