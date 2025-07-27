import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from tqdm import tqdm
from .visualization_utils import plot_training_curves, plot_quick_finetune_results
from .training_utils import evaluate_model
from .model_utils import apply_precision_config


def save_dnas_results(best_config, best_acc, training_history, save_dir='dnas_results', test_acc=0.0):
    """保存DNAS搜索结果"""
    logger = logging.getLogger(__name__)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存最佳配置
    config_path = os.path.join(save_dir, 'best_precision_config.json')
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # 保存训练历史
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 保存训练曲线
    plot_training_curves(
        training_history['epochs'],
        training_history['accuracies'],
        training_history['ce_losses'],
        training_history['hw_penalties'],
        training_history['total_losses'],
        os.path.join(save_dir, 'training_curves.png')
    )
    
    # 保存快速微调结果曲线
    plot_quick_finetune_results(training_history, os.path.join(save_dir, 'quick_finetune_curves.png'))
    
    # 保存结果摘要
    summary_path = os.path.join(save_dir, 'dnas_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"DNAS Search Results Summary\n")
        f.write(f"==========================\n")
        f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Total Epochs: {len(training_history['epochs'])}\n")
        f.write(f"Final Temperature: {training_history['temperatures'][-1]:.4f}\n")
        f.write(f"\nPrecision Distribution:\n")
        for layer, precision in best_config.items():
            f.write(f"  {layer}: {precision}\n")
        
        # 添加快速微调结果
        if 'quick_finetune_results' in training_history and training_history['quick_finetune_results']:
            f.write(f"\nQuick Finetune Results:\n")
            f.write(f"  Total Quick Finetune Sessions: {len(training_history['quick_finetune_results'])}\n")
            
            improvements = [r['test_acc_after_finetune'] - r['validation_acc'] 
                           for r in training_history['quick_finetune_results']]
            avg_improvement = sum(improvements) / len(improvements)
            max_improvement = max(improvements)
            min_improvement = min(improvements)
            
            f.write(f"  Average Improvement: {avg_improvement:.2f}%\n")
            f.write(f"  Max Improvement: {max_improvement:.2f}%\n")
            f.write(f"  Min Improvement: {min_improvement:.2f}%\n")
            
            f.write(f"\n  Detailed Results:\n")
            for result in training_history['quick_finetune_results']:
                f.write(f"    Epoch {result['epoch']}: Val={result['validation_acc']:.2f}%, "
                       f"Test={result['test_acc_after_finetune']:.2f}%, "
                       f"Improvement={result['test_acc_after_finetune'] - result['validation_acc']:.2f}%\n")
    
    logger.info(f"DNAS结果已保存到 {save_dir}/ 目录")


def finetune_subnet(model, train_loader, val_loader, test_loader, device, epochs=25):
    """微调子网"""
    logger = logging.getLogger(__name__)

    # 冻结所有alpha参数
    for name, param in model.named_parameters():
        if 'alpha' in name:
            param.requires_grad = False
    
    logger.info(f"开始微调子网 (epochs={epochs})")
    
    # 设置微调参数
    finetune_lr = 0.001
    optimizer_finetune = optim.Adam(model.parameters(), lr=finetune_lr, betas=(0.9, 0.999), weight_decay=4e-5)
    scheduler_finetune = optim.lr_scheduler.CosineAnnealingLR(optimizer_finetune, epochs)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # 训练
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer_finetune.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer_finetune.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler_finetune.step()
        
        # 评估
        acc = evaluate_model(model, val_loader, device)
        
        if acc > best_acc:
            best_acc = acc
        
        if epoch % 10 == 0:
            logger.info(f'[Finetune] Epoch: {epoch}, Val Acc: {acc:.2f}%, Loss: {epoch_loss/num_batches:.4f}')
    
    # 最终在测试集上评估
    final_test_acc = evaluate_model(model, test_loader, device)
    logger.info(f"微调完成，最终测试准确率: {final_test_acc:.2f}%")
    
    return final_test_acc 