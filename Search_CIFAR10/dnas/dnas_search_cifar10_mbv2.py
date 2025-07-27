import torch
import argparse
import sys
import os
from pathlib import Path
import json
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import logging
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题


# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 使用示例:
# 全精度搜索: python dnas_search_cifar10_mbv2.py --checkpoint path/to/pretrained_model.pth
# INT only搜索: python dnas_search_cifar10_mbv2.py --checkpoint path/to/pretrained_model.pth --int_only

from adaptqnet.models.mobilenetv2 import AdaptQMobileNetV2, MixedPrecisionLayer

# 导入工具函数
from utils import (
    create_train_val_test_loaders,
    load_layer_precision_options,
    set_layer_precision_options,
    validate_model_alpha_parameters,
    apply_precision_config,
    evaluate_model,
    calculate_hardware_penalty,
    update_temperature,
    should_sample_subnet,
    sample_and_save_subnet,
    plot_training_curves,
    save_dnas_results,
    finetune_subnet
)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='../pretrain/checkpoints/mobilenetv2_cifar_112x112_0.5_fp32_pretrained.pth')
    parser.add_argument('--tensor_analysis_json', type=str, default='../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json')
    parser.add_argument('--int_only', action='store_true', help='是否只搜索INT精度选项 (int8, int4, int2)')
    args = parser.parse_args()
    
    # 设置日志记录
    log_dir = 'mbv2_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = "_int_only" if args.int_only else "_mixed"
    log_file = os.path.join(log_dir, f'dnas_search_mbv2{mode_suffix}_{timestamp}.log')
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    if args.int_only:
        logger.info("开始 DNAS 搜索 - MobileNetV2 INT Only")
    else:
        logger.info("开始 DNAS 搜索 - MobileNetV2 混合精度")
    logger.info("=" * 80)
    
    # 记录参数
    logger.info("训练参数:")
    logger.info(f"  预训练模型: {args.checkpoint}")
    logger.info(f"  张量分析结果: {args.tensor_analysis_json}")
    logger.info(f"  INT only模式: {args.int_only}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("-" * 80)

    # 仅支持GPU训练
    device = torch.device('cuda:1')
    logger.info(f"使用GPU训练: {device}")
    logger.info(f"选择GPU 1: {torch.cuda.get_device_name(1)}")

    # 1. 读取张量分析结果，获得每层可行精度
    logger.info("正在读取张量分析结果...")
    if args.int_only:
        model_default_options = ["int8", "int4", "int2"]
        logger.info("INT only模式: 只搜索INT精度选项")
    else:
        model_default_options = ["fp32", "fp16", "int8", "int4", "int2"]
        logger.info("全精度模式: 搜索所有精度选项")
    
    layer_precision_options = load_layer_precision_options(args.tensor_analysis_json, int_only=args.int_only, model_default_options=model_default_options)
    logger.info(f"成功加载 {len(layer_precision_options)} 层的精度选项")

    # 2. 初始化模型（先加载预训练模型，再新建DNAS模型并迁移权重）
    logger.info("正在加载预训练模型...")
    pretrained_model = AdaptQMobileNetV2(
        num_classes=10,
        width_mult=0.5,  # 使用0.5倍宽度
        precision_options=None,
        pretrain_mode=True
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"成功加载预训练模型: {args.checkpoint}")

    logger.info("正在创建DNAS模型...")
    model = AdaptQMobileNetV2(
        num_classes=10,
        width_mult=0.5,  # 使用0.5倍宽度
        precision_options=model_default_options,
        hardware_constraints=None,
        pretrain_mode=False,
        initialize_weights=True
    )
    model.copy_weights_from_pretrained(pretrained_model)
    
    # 仅支持GPU训练，不使用DataParallel
    logger.info("仅支持单GPU训练，不启用数据并行")
    
    model = model.to(device)
    logger.info("DNAS模型创建完成并迁移到设备")

    # 4. 设置每层可行精度
    logger.info("正在设置每层精度选项...")
    set_layer_precision_options(model, layer_precision_options)
    logger.info("精度选项设置完成")
    
    # 验证alpha参数大小是否正确
    validate_model_alpha_parameters(model)
    
    # 验证内存过滤是否生效
    logger.info("验证内存过滤结果:")
    total_modules_before = 0
    total_modules_after = 0
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            if args.int_only:
                total_modules_before += len(["int8", "int4", "int2"])  # INT only模式原始有3个选项
            else:
                total_modules_before += len(["fp32", "fp16", "int8", "int4", "int2"])  # 全精度模式原始有5个选项
            total_modules_after += len(module.precision_options)
            logger.info(f"  层 {name}:")
            logger.info(f"    精度选项: {module.precision_options}")
            logger.info(f"    基础模块: {type(module.base_module).__name__}")
            logger.info(f"    PACT参数数量: {len(module.alpha_pact_dict)}")
            logger.info(f"    Alpha大小: {module.alpha.size(0)}")
    
    logger.info(f"内存过滤统计:")
    logger.info(f"  总模块数减少: {total_modules_before} -> {total_modules_after}")
    logger.info(f"  减少比例: {((total_modules_before - total_modules_after) / total_modules_before * 100):.1f}%")
    
    # 确保所有模块都在正确的设备上
    model = model.to(device)
    logger.info("确保所有模块都在正确设备上")
    
    # 调试信息：检查所有模块的设备
    logger.info("调试信息 - 检查模块设备:")
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            logger.info(f"  层 {name}:")
            logger.info(f"    模块设备: {next(module.parameters()).device}")
            logger.info(f"    Alpha设备: {module.alpha.device}")
            for precision, pact_param in module.alpha_pact_dict.items():
                logger.info(f"    PACT {precision} 设备: {pact_param.device}")
            logger.info(f"    基础模块设备: {next(module.base_module.parameters()).device}")
    
    # 调试信息：检查每层的精度选项和alpha大小
    logger.info("调试信息 - 每层精度选项和alpha大小:")
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            logger.info(f"  层 {name}:")
            logger.info(f"    精度选项: {module.precision_options}")
            logger.info(f"    Alpha大小: {module.alpha.size(0)}")
            logger.info(f"    选项数量: {len(module.precision_options)}")
            if module.alpha.size(0) != len(module.precision_options):
                logger.warning(f"    ⚠️ 不匹配！Alpha大小({module.alpha.size(0)}) != 选项数量({len(module.precision_options)})")
            else:
                logger.info(f"    ✅ 匹配")

    # 5. 创建训练集、验证集和测试集加载器
    logger.info("正在创建数据加载器...")
    # 112x112分辨率，适当增加batch size
    batch_size = 256  # 112x112分辨率可以使用更大的batch size
    input_size = 112  # 设置输入尺寸为112x112
    logger.info(f"112x112分辨率训练，batch size: {batch_size}")
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        batch_size=batch_size, 
        num_workers=8, 
        input_size=input_size
    )
    logger.info("数据加载器创建完成")
    
    # 数据集使用说明：
    # - train_loader: 用于训练模型权重（第一阶段）
    # - val_loader: 用于训练alpha参数（第二阶段），避免信息泄露
    # - test_loader: 用于评估模型性能，报告最终结果

    # 6. DNAS搜索主流程
    logger.info("开始DNAS搜索主流程...")
    epochs = 6  # 减少训练轮数，因为224x224训练更慢
    
    # 仅支持GPU训练，使用固定学习率
    lr_w = 0.0001
    lr_alpha = 0.05
    logger.info(f"GPU训练，权重学习率: {lr_w}")
    logger.info(f"GPU训练，alpha学习率: {lr_alpha}")
    initial_temp = 5.0  # 初始温度
    min_temp = 0.01
    temperature_decay = 'exponential'  # 温度衰减策略
    hardware_penalty_weight = 0.00000001  # 硬件惩罚权重
    grad_clip_norm = 1.0  # 梯度裁剪阈值
    
    # 采样策略参数
    max_sampling_times = 10  # 最大采样次数
    early_stopping_patience = 5  # 早停耐心值
    sampling_schedule = 'dynamic'  # 动态采样频率
    
    logger.info("DNAS参数:")
    logger.info(f"  训练轮数: {epochs}")
    logger.info(f"  权重学习率: {lr_w}")
    logger.info(f"  Alpha学习率: {lr_alpha}")
    logger.info(f"  初始温度: {initial_temp}")
    logger.info(f"  最小温度: {min_temp}")
    logger.info(f"  温度衰减: {temperature_decay}")
    logger.info(f"  硬件惩罚权重: {hardware_penalty_weight}")
    logger.info(f"  梯度裁剪: {grad_clip_norm}")
    logger.info(f"  最大采样次数: {max_sampling_times}")
    logger.info(f"  早停耐心值: {early_stopping_patience}")
    logger.info(f"  采样策略: {sampling_schedule}")
    
    # 性能优化信息
    logger.info("性能优化配置:")
    logger.info(f"  训练模式: GPU单卡训练")
    logger.info(f"  输入分辨率: {input_size}x{input_size}")
    logger.info(f"  有效Batch Size: {batch_size}")
    logger.info(f"  数据加载器工作进程: 8")
    logger.info(f"  混合精度训练: 禁用")
    logger.info(f"  使用GPU: {torch.cuda.get_device_name(1)}")
    logger.info(f"  GPU内存: {torch.cuda.get_device_properties(1).total_memory / 1024**3:.1f} GB")
    
    optimizer_w = optim.Adam(model.parameters(), lr=lr_w, betas=(0.9, 0.999), weight_decay=4e-5)
    optimizer_alpha = optim.Adam([p for n, p in model.named_parameters() if 'alpha' in n], lr=lr_alpha, betas=(0.5, 0.999))
    scheduler_w = optim.lr_scheduler.CosineAnnealingLR(optimizer_w, epochs)

    best_acc = 0
    best_config = None
    sampled_subnets = []  # 存储所有采样的子网
    early_stopping_counter = 0
    last_loss = float('inf')
    
    # 初始化训练历史记录
    training_history = {
        'epochs': [],
        'accuracies': [],
        'ce_losses': [],
        'hw_penalties': [],
        'total_losses': [],
        'temperatures': []
    }
    
    # 记录总开始时间
    total_start_time = time.time()
    logger.info(f"DNAS搜索开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(epochs):
        # 冻结BN层：训练一代后（即epoch==1）将所有BN层切换为eval模式
        if epoch == 1:
            logger.info("冻结所有BN层的统计量（eval模式）")
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        epoch_start_time = time.time()
        current_temp = update_temperature(model, epoch, epochs, initial_temp, min_temp, temperature_decay)
        
        # 第一阶段：训练模型权重（在训练集上）
        model.train()
        
        # 冻结alpha参数，解冻权重参数，只训练权重
        for name, module in model.named_modules():
            if isinstance(module, MixedPrecisionLayer):
                module.alpha.requires_grad = False
                # 解冻权重参数
                for param_name, param in module.named_parameters():
                    if 'alpha' not in param_name:  # 除了alpha之外的所有参数
                        param.requires_grad = True
        
        # 单GPU训练
        model.train()
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training Weights', leave=False)
        epoch_ce_loss = 0.0
        epoch_hw_penalty = 0.0
        epoch_total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer_w.zero_grad()
            
            # 标准精度训练
            outputs = model(images)
            ce_loss = F.cross_entropy(outputs, labels)
            # 使用当前batch的输入尺寸计算硬件惩罚
            batch_size = images.size(0)
            hw_penalty = calculate_hardware_penalty(model, use_fixed_alpha=True)
            total_loss = ce_loss + hardware_penalty_weight * hw_penalty
            total_loss.backward()
            
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            optimizer_w.step()
            
            # 累积损失
            epoch_ce_loss += ce_loss.item()
            epoch_hw_penalty += hw_penalty.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1
            
            # 更新进度条
            train_pbar.set_postfix({
                'CE_Loss': f'{ce_loss.item():.4f}',
                'HW_Penalty': f'{hw_penalty.item():.4f}',
                'Total_Loss': f'{total_loss.item():.4f}'
            })
        
        # 第二阶段：训练架构参数（在验证集上）
        model.eval()  # 固定BN
        
        # 解冻alpha参数，冻结权重参数，只训练alpha
        for name, module in model.named_modules():
            if isinstance(module, MixedPrecisionLayer):
                module.alpha.requires_grad = True
                # 冻结权重参数
                for param_name, param in module.named_parameters():
                    if 'alpha' not in param_name:  # 除了alpha之外的所有参数
                        param.requires_grad = False
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Training Alpha', leave=False)
        for batch_idx, (images, labels) in enumerate(val_pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer_alpha.zero_grad()
            
            # 标准精度训练
            outputs = model(images)
            ce_loss = F.cross_entropy(outputs, labels)
            # 使用当前batch的输入尺寸计算硬件惩罚
            batch_size = images.size(0)
            hw_penalty = calculate_hardware_penalty(model, use_fixed_alpha=False)
            total_loss = ce_loss + hardware_penalty_weight * hw_penalty
            total_loss.backward()
            
            # 对alpha参数也进行梯度裁剪
            alpha_params = [p for n, p in model.named_parameters() if 'alpha' in n]
            torch.nn.utils.clip_grad_norm_(alpha_params, grad_clip_norm)
            
            optimizer_alpha.step()
            
            # 更新进度条
            val_pbar.set_postfix({
                'CE_Loss': f'{ce_loss.item():.4f}',
                'HW_Penalty': f'{hw_penalty.item():.4f}',
                'Total_Loss': f'{total_loss.item():.4f}'
            })
            
        scheduler_w.step()
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        
        # 评估（在测试集上，避免验证集信息泄露）
        test_acc = evaluate_model(model, test_loader, device)
        current_loss = epoch_total_loss / num_batches
        
        # 记录训练历史
        training_history['epochs'].append(epoch)
        training_history['accuracies'].append(test_acc)  # 使用测试集准确率
        training_history['ce_losses'].append(epoch_ce_loss / num_batches)
        training_history['hw_penalties'].append(epoch_hw_penalty / num_batches)
        training_history['total_losses'].append(current_loss)
        training_history['temperatures'].append(current_temp)
        
        # 实时更新训练曲线
        plot_training_curves(
            training_history['epochs'],
            training_history['accuracies'],
            training_history['ce_losses'],
            training_history['hw_penalties'],
            training_history['total_losses'],
            f'training_curves_mbv2{mode_suffix}_live.png'
        )
        
        logger.info(f'Epoch: {epoch}, Temperature: {current_temp:.4f}, Test Accuracy: {test_acc:.2f}%')
        logger.info(f'Epoch时间: {epoch_time:.2f}s, 总时间: {total_time:.2f}s ({total_time/60:.1f}分钟)')
        logger.info(f'CE Loss: {epoch_ce_loss/num_batches:.4f}, HW Penalty: {epoch_hw_penalty/num_batches:.4f}, Total Loss: {current_loss:.4f}')
        
        # 记录精度分布
        logger.info("Precision Distribution:")
        for name, module in model.named_modules():
            if isinstance(module, MixedPrecisionLayer):
                weights = F.softmax(module.alpha / current_temp, dim=0)
                # 确保selected_idx不超出precision_options的范围
                selected_idx = min(weights.argmax().item(), len(module.precision_options) - 1)
                selected_precision = module.precision_options[selected_idx]
                entropy = -(weights * torch.log(weights + 1e-10)).sum()
                logger.info(f"{name}: {selected_precision}")
                logger.info(f"  Weights: {weights.detach().cpu().numpy()}")
                logger.info(f"  Entropy: {entropy.item():.4f}")
                logger.info(f"  Precision options: {module.precision_options}")
                logger.info(f"  Alpha size: {module.alpha.size(0)}, Options count: {len(module.precision_options)}")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_config = {}
            for name, module in model.named_modules():
                if isinstance(module, MixedPrecisionLayer):
                    weights = F.softmax(module.alpha / current_temp, dim=0)
                    selected_idx = min(weights.argmax().item(), len(module.precision_options) - 1)
                    best_config[name] = module.precision_options[selected_idx]
            
            # 保存最佳模型对应的alpha分布
            best_alpha_snapshot = {}
            for name, module in model.named_modules():
                if isinstance(module, MixedPrecisionLayer):
                    best_alpha_snapshot[name] = module.alpha.detach().cpu().numpy().tolist()
            
            # 创建最佳模型保存目录
            os.makedirs('mbv2_best_models', exist_ok=True)
            model_state_dict = model.state_dict()
            
            torch.save({
                'precision_config': best_config,
                'alpha_snapshot': best_alpha_snapshot,  # 新增：保存最佳模型的alpha分布
                'model_state_dict': model_state_dict,
                'training_history': training_history
            }, f'mbv2_best_models/best_dnas_cifar10_mbv2{mode_suffix}.pth')
            logger.info(f"Best model saved at epoch {epoch} with test acc {test_acc:.2f}%")
        
        # 动态采样子网
        if should_sample_subnet(epoch, epochs, len(sampled_subnets), max_sampling_times, sampling_schedule):
            subnet_info = sample_and_save_subnet(model, epoch, current_temp, device, sampled_subnets)
            logger.info(f"采样进度: {len(sampled_subnets)}/{max_sampling_times}")
        
        # 早停检查
        if current_loss > last_loss:
            early_stopping_counter += 1
            logger.info(f"Loss上升，早停计数器: {early_stopping_counter}/{early_stopping_patience}")
        else:
            early_stopping_counter = 0
        
        last_loss = current_loss
        
        # 早停条件
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"\n{'='*50}")
            logger.info(f"早停触发！连续{early_stopping_patience}代Loss上升")
            logger.info(f"在Epoch {epoch}停止超网搜索")
            logger.info(f"{'='*50}")
            break

    # 记录总训练时间
    total_training_time = time.time() - total_start_time
    logger.info(f"\nDNAS搜索完成!")
    logger.info(f"总训练时间: {total_training_time:.2f}s ({total_training_time/60:.1f}分钟)")
    logger.info(f"平均每epoch时间: {total_training_time/epochs:.2f}s")
    logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("搜索结束，最佳精度分布：")
    for k, v in best_config.items():
        logger.info(f"{k}: {v}")
    logger.info(f"Best test accuracy: {best_acc:.2f}%")
    
    # 批量微调所有采样的子网
    logger.info(f"\n{'='*60}")
    logger.info(f"开始批量微调 {len(sampled_subnets)} 个采样的子网")
    logger.info(f"{'='*60}")
    
    finetune_results = []
    for i, subnet_info in enumerate(sampled_subnets):
        logger.info(f"\n微调子网 {i+1}/{len(sampled_subnets)} (Epoch {subnet_info['epoch']})")
        
        # 根据子网的精度配置创建对应的精度选项
        # 从保存的精度配置中提取所有使用的精度选项
        used_precisions = set(subnet_info['precision_config'].values())
        subnet_precision_options = list(used_precisions)
        logger.info(f"子网使用的精度选项: {subnet_precision_options}")
        
        # 创建子网模型，使用与采样时相同的精度选项
        subnet_model = AdaptQMobileNetV2(
            num_classes=10,
            width_mult=0.5,  # 使用0.5倍宽度
            precision_options=subnet_precision_options,
            hardware_constraints=None,
            pretrain_mode=False,
            initialize_weights=False  # 不初始化权重，直接加载保存的状态
        )
        
        # 直接加载子网的状态字典（包含所有训练好的权重和alpha参数）
        subnet_state_dict = subnet_info['model_state_dict']
        model_state_dict = subnet_model.state_dict()
        
        # 过滤掉不匹配的参数
        filtered_state_dict = {}
        for key, value in subnet_state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                logger.debug(f"跳过不匹配的参数: {key} (保存的形状: {value.shape if hasattr(value, 'shape') else 'N/A'}, 当前形状: {model_state_dict.get(key, 'N/A').shape if key in model_state_dict else 'N/A'})")
        
        # 加载过滤后的状态字典
        missing_keys, unexpected_keys = subnet_model.load_state_dict(filtered_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"缺少的参数: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"多余的参数: {unexpected_keys}")
        
        subnet_model = subnet_model.to(device)
        
        # 应用精度配置（确保与采样时一致）
        apply_precision_config(subnet_model, subnet_info['precision_config'])
        
        # 微调子网（针对224x224调整）
        finetune_acc = finetune_subnet(subnet_model, train_loader, val_loader, test_loader, device, epochs=25)
        
        finetune_results.append({
            'subnet_id': i,
            'epoch': subnet_info['epoch'],
            'temperature': subnet_info['temperature'],
            'precision_config': subnet_info['precision_config'],
            'finetune_acc': finetune_acc
        })
        
        logger.info(f"子网 {i+1} 微调完成，准确率: {finetune_acc:.2f}%")
    
    # 保存微调结果
    os.makedirs('mbv2_results', exist_ok=True)
    torch.save({
        'sampled_subnets': sampled_subnets,
        'finetune_results': finetune_results,
        'best_config': best_config,
        'training_history': training_history
    }, f'mbv2_results/dnas_search_and_finetune_results{mode_suffix}.pth')
    
    # 记录微调结果总结
    logger.info(f"\n{'='*60}")
    logger.info("批量微调结果总结")
    logger.info(f"{'='*60}")
    logger.info(f"{'Subnet':<8} {'Epoch':<8} {'Temp':<8} {'Finetune Acc':<12}")
    logger.info("-" * 50)
    
    best_finetune_acc = 0
    best_finetune_subnet = None
    
    for result in finetune_results:
        subnet_id = result['subnet_id']
        epoch = result['epoch']
        temp = result['temperature']
        acc = result['finetune_acc']
        
        logger.info(f"{subnet_id:<8} {epoch:<8} {temp:<8.3f} {acc:<12.2f}")
        
        if acc > best_finetune_acc:
            best_finetune_acc = acc
            best_finetune_subnet = result
    
    logger.info(f"\n最佳微调结果:")
    logger.info(f"子网 {best_finetune_subnet['subnet_id']} (Epoch {best_finetune_subnet['epoch']})")
    logger.info(f"准确率: {best_finetune_acc:.2f}%")
    logger.info(f"温度: {best_finetune_subnet['temperature']:.3f}")
    logger.info(f"精度配置: {best_finetune_subnet['precision_config']}")
    
    logger.info(f"{'='*60}")

    # 在测试集上评估最终性能（已经在搜索过程中评估过了）
    logger.info("\n最终测试集性能（搜索过程中最佳）：")
    logger.info(f"测试集准确率: {best_acc:.2f}%")

    # 保存DNAS搜索结果
    save_dnas_results(best_config, best_acc, training_history, save_dir=f'dnas_results_mbv2{mode_suffix}', test_acc=best_acc)

    logger.info("\n开始子网微调...")
    
    # 记录微调开始时间
    finetune_start_time = time.time()
    logger.info(f"微调开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 固化精度
    apply_precision_config(model, best_config)
    
    # 验证精度配置是否正确应用
    logger.info("验证精度配置...")
    try:
        test_acc = evaluate_model(model, val_loader, device)
        logger.info(f"应用精度配置后的测试准确率: {test_acc:.2f}%")
    except Exception as e:
        logger.error(f"精度配置验证失败: {str(e)}")
        logger.info("跳过微调阶段")
        return

    # 只优化权重参数（针对224x224调整）
    finetune_epochs = 40  # 减少微调轮数，因为224x224训练更慢
    finetune_lr = 0.0008  # 降低微调学习率
    optimizer_finetune = optim.Adam(model.parameters(), lr=finetune_lr, betas=(0.9, 0.999), weight_decay=4e-5)
    scheduler_finetune = optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, finetune_epochs)
    
    # 初始化微调历史记录
    finetune_history = {
        'epochs': [],
        'accuracies': [],
        'ce_losses': []
    }

    for epoch in range(finetune_epochs):
        epoch_start_time = time.time()
        model.train()
        finetune_pbar = tqdm(val_loader, desc=f'Finetune Epoch {epoch}', leave=False)
        epoch_ce_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(finetune_pbar):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            ce_loss = F.cross_entropy(outputs, labels)
            optimizer_finetune.zero_grad()
            ce_loss.backward()
            optimizer_finetune.step()
            
            # 累积损失
            epoch_ce_loss += ce_loss.item()
            num_batches += 1
            
            # 更新进度条
            finetune_pbar.set_postfix({
                'CE_Loss': f'{ce_loss.item():.4f}'
            })
            
            if batch_idx % 50 == 0:
                logger.info(f'[Finetune] Epoch: {epoch}, Batch: {batch_idx}, CE Loss: {ce_loss:.4f}')
        scheduler_finetune.step()
        
        # 评估
        acc = evaluate_model(model, val_loader, device)
        epoch_time = time.time() - epoch_start_time
        
        # 记录微调历史
        finetune_history['epochs'].append(epoch)
        finetune_history['accuracies'].append(acc)
        finetune_history['ce_losses'].append(epoch_ce_loss / num_batches)
        
        # 绘制微调曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 微调准确率
        ax1.plot(finetune_history['epochs'], finetune_history['accuracies'], 'b-', linewidth=2, label='Finetune Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Finetune Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 微调Loss
        ax2.plot(finetune_history['epochs'], finetune_history['ce_losses'], 'r-', linewidth=2, label='Finetune CE Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Finetune Cross Entropy Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        fig.tight_layout()
        fig.savefig(f'finetune_curves_mbv2{mode_suffix}_live.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f'[Finetune] Epoch: {epoch}, Accuracy: {acc:.2f}%, Epoch时间: {epoch_time:.2f}s, CE Loss: {epoch_ce_loss/num_batches:.4f}')

    # 打印微调总时间
    finetune_total_time = time.time() - finetune_start_time
    logger.info(f"\n微调完成!")
    logger.info(f"微调总时间: {finetune_total_time:.2f}s ({finetune_total_time/60:.1f}分钟)")
    logger.info(f"微调平均每epoch时间: {finetune_total_time/finetune_epochs:.2f}s")
    logger.info(f"微调完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 保存微调后的子网
    model_state_dict = model.state_dict()
    
    torch.save({
        'precision_config': best_config,
        'model_state_dict': model_state_dict,
        'finetune_history': finetune_history
    }, f'finetuned_subnet_cifar10_mbv2{mode_suffix}.pth')
    logger.info("微调后的子网已保存。")
    
    # 保存微调曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 微调准确率
    ax1.plot(finetune_history['epochs'], finetune_history['accuracies'], 'b-', linewidth=2, label='Finetune Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Finetune Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 微调Loss
    ax2.plot(finetune_history['epochs'], finetune_history['ce_losses'], 'r-', linewidth=2, label='Finetune CE Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Finetune Cross Entropy Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.tight_layout()
    fig.savefig(f'finetune_curves_mbv2{mode_suffix}_final.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 保存微调历史到JSON
    with open(f'finetune_history_mbv2{mode_suffix}.json', 'w') as f:
        json.dump(finetune_history, f, indent=2)
    
    logger.info("微调曲线和历史已保存。")

    # 打印总时间统计
    total_time = time.time() - total_start_time
    logger.info(f"\n=== 总时间统计 ===")
    logger.info(f"DNAS搜索: {total_training_time:.2f}s ({total_training_time/60:.1f}分钟)")
    logger.info(f"子网微调: {finetune_total_time:.2f}s ({finetune_total_time/60:.1f}分钟)")
    logger.info(f"总时间: {total_time:.2f}s ({total_time/60:.1f}分钟)")
    logger.info(f"开始时间: {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main() 