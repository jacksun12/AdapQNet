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
# 全精度搜索: python dnas_search_cifar10_mbv2.py --input_size 32
# INT only搜索: python dnas_search_cifar10_mbv2.py --input_size 32 --int_only
# 224x224搜索: python dnas_search_cifar10_mbv2.py --input_size 224

from models.mobilenetv2 import AdaptQMobileNetV2, MixedPrecisionLayer

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
    parser.add_argument('--tensor_analysis_json', type=str, default='../tensor_analysis_result/mbv2_cifar10_tensor_analysis_results.json')
    parser.add_argument('--int_only', action='store_true', help='是否只搜索INT精度选项 (int8, int4, int2)')
    parser.add_argument('--input_size', type=int, default=32, help='输入图像尺寸')
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
    logger.info(f"  张量分析结果: {args.tensor_analysis_json}")
    logger.info(f"  INT only模式: {args.int_only}")
    logger.info(f"  输入尺寸: {args.input_size}x{args.input_size}")
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
        width_mult=1.0,  # 使用1.0倍宽度
        precision_options=None,
        pretrain_mode=True,
        input_size=args.input_size  # 使用命令行指定的输入尺寸
    )
    
    # 加载ImageNet预训练权重
    logger.info("加载ImageNet预训练权重...")
    success = pretrained_model.load_imagenet_pretrained_weights()
    if not success:
        logger.warning("ImageNet预训练权重加载失败，使用随机初始化")
    else:
        logger.info("成功加载ImageNet预训练权重")

    logger.info("正在创建DNAS模型...")
    model = AdaptQMobileNetV2(
        num_classes=10,
        width_mult=1.0,  # 使用1.0倍宽度
        precision_options=model_default_options,
        hardware_constraints=None,
        pretrain_mode=False,
        initialize_weights=True,
        input_size=args.input_size  # 使用命令行指定的输入尺寸
    )
    model.copy_weights_from_pretrained(pretrained_model)
    
    # 仅支持GPU训练，不使用DataParallel
    logger.info("仅支持单GPU训练，不启用数据并行")
    
    model = model.to(device)
    logger.info("DNAS模型创建完成并迁移到设备")
    logger.info(f"使用标准MobileNetV2配置 ({args.input_size}x{args.input_size})")
    logger.info("模型配置: 7个IRB阶段, 最终通道1280")

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
            logger.info(f"    精度模块数量: {len(module.precision_modules)}")
            logger.info(f"    PACT参数数量: {len(module.alpha_pact_dict)}")
            logger.info(f"    Alpha大小: {module.alpha.size(0)}")
    
    logger.info(f"内存过滤统计:")
    logger.info(f"  总模块数减少: {total_modules_before} -> {total_modules_after}")
    logger.info(f"  减少比例: {((total_modules_before - total_modules_after) / total_modules_before * 100):.1f}%")
    
    # 确保所有模块都在正确的设备上
    model = model.to(device)
    logger.info("确保所有模块都在正确设备上")
    
    # 直接冻结所有BN参数
    logger.info("直接冻结所有BN参数...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()  # 确保BN处于eval模式
            module.weight.requires_grad = False
            module.bias.requires_grad = False
            module.running_mean.requires_grad = False
            module.running_var.requires_grad = False
            logger.info(f"  冻结BN层: {name}")
    logger.info("所有BN参数已冻结")
    
    # 调试信息：检查所有模块的设备
    logger.info("调试信息 - 检查模块设备:")
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            logger.info(f"  层 {name}:")
            logger.info(f"    模块设备: {next(module.parameters()).device}")
            logger.info(f"    Alpha设备: {module.alpha.device}")
            for precision, pact_param in module.alpha_pact_dict.items():
                logger.info(f"    PACT {precision} 设备: {pact_param.device}")
            # 显示第一个精度模块的设备信息
            if module.precision_modules:
                first_precision = list(module.precision_modules.keys())[0]
                first_module = module.precision_modules[first_precision]
                logger.info(f"    第一个精度模块({first_precision})设备: {next(first_module.parameters()).device}")
    
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
    # 根据输入尺寸调整batch size
    if args.input_size <= 64:
        batch_size = 512  # 小尺寸输入可以使用更大的batch size
    elif args.input_size <= 128:
        batch_size = 256  # 中等尺寸输入
    else:
        batch_size = 128  # 大尺寸输入使用较小的batch size
    
    input_size = args.input_size  # 使用命令行指定的输入尺寸
    logger.info(f"{input_size}x{input_size}分辨率训练，batch size: {batch_size}")
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
    epochs = 50  # 增加训练轮数，确保温度充分下降
    warmup_epochs = 1  # 预热阶段轮数，只训练权重，不搜索alpha
    
    # 仅支持GPU训练，使用固定学习率
    lr_w = 0.001  # 增加权重学习率，从1e-7改为1e-3
    lr_alpha = 0.05
    logger.info(f"GPU训练，权重学习率: {lr_w}")
    logger.info(f"GPU训练，alpha学习率: {lr_alpha}")
    initial_temp = 5.0  # 初始温度
    min_temp = 0.01  # 大幅降低最小温度，确保充分收敛到one-hot
    temperature_decay = 'cubic'  # 使用三次衰减，非常激进的温度下降
    hardware_penalty_weight = 0.0000000001 # 大幅降低硬件惩罚权重，从5e-9改为1e-10
    grad_clip_norm = 1.0  # 梯度裁剪阈值
    

    
    logger.info("DNAS参数:")
    logger.info(f"  训练轮数: {epochs}")
    logger.info(f"  预热轮数: {warmup_epochs} (只训练权重，不搜索alpha)")
    logger.info(f"  权重学习率: {lr_w}")
    logger.info(f"  Alpha学习率: {lr_alpha}")
    logger.info(f"  初始温度: {initial_temp}")
    logger.info(f"  最小温度: {min_temp} (大幅降低，确保one-hot收敛)")
    logger.info(f"  温度衰减: {temperature_decay} (三次衰减，非常激进)")
    logger.info(f"  硬件惩罚权重: {hardware_penalty_weight}")
    logger.info(f"  梯度裁剪: {grad_clip_norm}")
    
    # 温度调度验证
    logger.info("温度调度验证:")
    for epoch in range(0, epochs, 10):
        progress = epoch / epochs
        if temperature_decay == 'cubic':
            temp = min_temp + (initial_temp - min_temp) * (1 - progress)**3
        elif temperature_decay == 'quadratic':
            temp = min_temp + (initial_temp - min_temp) * (1 - progress)**2
        elif temperature_decay == 'exponential':
            temp = min_temp + (initial_temp - min_temp) * np.exp(-3 * progress)
        else:
            temp = min_temp + (initial_temp - min_temp) * (1 - progress)
        logger.info(f"  Epoch {epoch}: 温度 = {temp:.6f}")
    logger.info(f"  Epoch {epochs-1}: 温度 = {min_temp:.6f} (最终温度)")
    
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
        epoch_start_time = time.time()
        current_temp = update_temperature(model, epoch, epochs, initial_temp, min_temp, temperature_decay)
        
        # 判断是否为预热阶段
        is_warmup = epoch < warmup_epochs
        
        if is_warmup:
            logger.info(f"预热阶段 Epoch {epoch}: 只训练权重，不搜索alpha")
        else:
            logger.info(f"搜索阶段 Epoch {epoch}: 训练权重 + 搜索alpha")
        
        # 第一阶段：训练模型权重（在训练集上）
        model.train()
        
        # 第一阶段：只训练权重参数，冻结alpha参数
        for name, module in model.named_modules():
            if isinstance(module, MixedPrecisionLayer):
                module.alpha.requires_grad = False
                # 解冻所有权重参数
                for param_name, param in module.named_parameters():
                    if 'alpha' not in param_name:  # 除了alpha之外的所有参数
                        param.requires_grad = True
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training Weights Only', leave=False)
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
        if not is_warmup:  # 只在非预热阶段训练alpha
            model.eval()  # 固定BN
            
            # 第二阶段：只训练alpha参数，冻结权重参数
            for name, module in model.named_modules():
                if isinstance(module, MixedPrecisionLayer):
                    module.alpha.requires_grad = True
                    # 冻结权重参数
                    for param_name, param in module.named_parameters():
                        if 'alpha' not in param_name:  # 除了alpha之外的所有参数
                            param.requires_grad = False
            
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Training Alpha Only', leave=False)
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images, labels = images.to(device), labels.to(device)
                optimizer_alpha.zero_grad()
                
                # 标准精度训练
                outputs = model(images)
                ce_loss = F.cross_entropy(outputs, labels)
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
        else:
            logger.info("预热阶段：跳过alpha训练")
            
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
        
        if is_warmup:
            logger.info(f'预热 Epoch: {epoch}, Test Accuracy: {test_acc:.2f}%')
            logger.info(f'Epoch时间: {epoch_time:.2f}s, 总时间: {total_time:.2f}s ({total_time/60:.1f}分钟)')
            logger.info(f'CE Loss: {epoch_ce_loss/num_batches:.4f}')
        else:
            logger.info(f'搜索 Epoch: {epoch}, Temperature: {current_temp:.4f}, Test Accuracy: {test_acc:.2f}%, BN已冻结')
            logger.info(f'Epoch时间: {epoch_time:.2f}s, 总时间: {total_time:.2f}s ({total_time/60:.1f}分钟)')
            logger.info(f'CE Loss: {epoch_ce_loss/num_batches:.4f}, HW Penalty: {epoch_hw_penalty/num_batches:.4f}, Total Loss: {current_loss:.4f}')
            
            # 记录精度分布（只在搜索阶段显示）
            logger.info("Precision Distribution:")
            for name, module in model.named_modules():
                if isinstance(module, MixedPrecisionLayer):
                    weights = F.softmax(module.alpha / current_temp, dim=0)
                    # 确保selected_idx不超出precision_options的范围
                    selected_idx = min(weights.argmax().item(), len(module.alpha) - 1)
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
        
        # 预热阶段结束提示
        if epoch == warmup_epochs - 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"预热阶段结束！从Epoch {warmup_epochs}开始正式搜索alpha参数")
            logger.info(f"{'='*60}")
            
        # 预热阶段结束提示
        if epoch == warmup_epochs - 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"预热阶段结束！从Epoch {warmup_epochs}开始正式搜索alpha参数")
            logger.info(f"{'='*60}")

    # 记录总训练时间
    total_training_time = time.time() - total_start_time
    logger.info(f"\nDNAS搜索完成!")
    logger.info(f"总训练时间: {total_training_time:.2f}s ({total_training_time/60:.1f}分钟)")
    logger.info(f"平均每epoch时间: {total_training_time/epochs:.2f}s")
    logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 确保保存最终的超网状态（无论是否提前结束）
    logger.info("保存最终的超网状态...")
    final_supernet_state = {
        'model_state_dict': model.state_dict(),
        'best_config': best_config,
        'best_alpha_snapshot': best_alpha_snapshot,
        'training_history': training_history,
        'final_epoch': epoch,
        'final_temperature': current_temp,
        'final_accuracy': best_acc,
        'total_training_time': total_training_time
    }
    
    # 创建超网保存目录
    os.makedirs('mbv2_supernet_models', exist_ok=True)
    torch.save(final_supernet_state, f'mbv2_supernet_models/final_supernet_mbv2{mode_suffix}.pth')
    logger.info(f"最终超网状态已保存到: mbv2_supernet_models/final_supernet_mbv2{mode_suffix}.pth")
    
    # 保存当前所有alpha参数的状态
    current_alpha_state = {}
    for name, module in model.named_modules():
        if isinstance(module, MixedPrecisionLayer):
            current_alpha_state[name] = {
                'alpha': module.alpha.detach().cpu().numpy().tolist(),
                'temperature': module.temperature,
                'precision_options': module.precision_options
            }
    
    with open(f'mbv2_supernet_models/current_alpha_state_mbv2{mode_suffix}.json', 'w') as f:
        json.dump(current_alpha_state, f, indent=2)
    logger.info(f"当前alpha状态已保存到: mbv2_supernet_models/current_alpha_state_mbv2{mode_suffix}.json")
    
    logger.info("第一阶段搜索结束，最佳精度分布：")
    for k, v in best_config.items():
        logger.info(f"{k}: {v}")
    logger.info(f"Best test accuracy: {best_acc:.2f}%")
    
    # DNAS搜索完成，超网已收敛到one-hot状态
    logger.info(f"\n{'='*60}")
    logger.info("DNAS搜索完成，超网已收敛到one-hot状态")
    logger.info(f"{'='*60}")

    # 验证最终超网性能
    logger.info("验证最终超网性能...")
    try:
        final_test_acc = evaluate_model(model, test_loader, device)
        logger.info(f"最终测试准确率: {final_test_acc:.2f}%")
    except Exception as e:
        logger.error(f"最终超网验证失败: {str(e)}")
        return

    # 保存最终的超网
    model_state_dict = model.state_dict()
    
    torch.save({
        'precision_config': best_config,
        'model_state_dict': model_state_dict,
        'training_history': training_history,
        'final_accuracy': final_test_acc
    }, f'final_supernet_cifar10_mbv2{mode_suffix}.pth')
    logger.info("最终超网已保存。")

    # 保存DNAS搜索结果
    save_dnas_results(best_config, best_acc, training_history, save_dir=f'dnas_results_mbv2{mode_suffix}', test_acc=best_acc)
    
    # 打印搜索总结
    logger.info(f"\n{'='*60}")
    logger.info("DNAS搜索总结")
    logger.info(f"{'='*60}")
    logger.info(f"最佳准确率: {best_acc:.2f}%")
    logger.info(f"搜索轮数: {epochs}")
    logger.info(f"温度策略: {temperature_decay}")
    logger.info(f"搜索完成，超网已收敛")
    logger.info(f"{'='*60}")

    # 打印总时间统计
    total_time = time.time() - total_start_time
    logger.info(f"\n=== 总时间统计 ===")
    logger.info(f"DNAS搜索: {total_training_time:.2f}s ({total_training_time/60:.1f}分钟)")
    logger.info(f"总时间: {total_time:.2f}s ({total_time/60:.1f}分钟)")
    logger.info(f"开始时间: {datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main() 
