import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from adaptqnet.models.mobilenetv2 import AdaptQMobileNetV2

def create_cifar10_loaders(batch_size=64, num_workers=4):
    """创建CIFAR10数据加载器，上采样到112x112"""
    transform_train = transforms.Compose([
        transforms.Resize(128),  # 先resize到128
        transforms.RandomCrop(112),  # 随机裁剪到112x112
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(128),  # 先resize到128
        transforms.CenterCrop(112),  # 中心裁剪到112x112
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def train_mobilenetv2(model, train_loader, val_loader, save_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 训练参数（针对112x112调整）
    num_epochs = 200  # 保持epoch数
    base_lr = 0.001  # 提高学习率，因为输入更小
    weight_decay = 0.05
    grad_clip = 5.0
    
    # 使用标签平滑的交叉熵损失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(),
                         lr=base_lr,
                         weight_decay=weight_decay)
    
    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    best_acc = 0
    save_path = Path(save_dir) / 'mobilenetv2_cifar_112x112_0.5_fp32_pretrained.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"开始训练 MobileNetV2 (112x112, width_mult=0.5)")
    print(f"设备: {device}")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {base_lr}")
    print(f"批次大小: {train_loader.batch_size}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:  # 减少打印频率，因为224x224训练更慢
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {train_loss/(batch_idx+1):.3f} | '
                      f'Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f'\nEpoch: {epoch} | Train Loss: {train_loss/len(train_loader):.3f} | '
              f'Val Loss: {val_loss/len(val_loader):.3f} | Val Acc: {acc:.2f}%')
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': acc,
                'best_accuracy': best_acc,
                'input_size': '112x112',  # 记录输入尺寸
                'width_mult': 0.5,  # 记录宽度倍数
            }, save_path)
            print(f'Best model saved! Accuracy: {acc:.2f}%')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='batch size (adjusted for 112x112)')
    args = parser.parse_args()
    
    # 创建模型，设置pretrain_mode=True确保使用FP32
    model = AdaptQMobileNetV2(
        num_classes=10,  # CIFAR10的类别数
        width_mult=0.5,  # 使用0.5倍宽度
        precision_options=["fp32"],  # 预训练时只使用FP32
        hardware_constraints=None,
        pretrain_mode=True,  # 设置预训练模式
        initialize_weights=True
    )
    
    print(f"模型创建完成: MobileNetV2")
    print(f"输入尺寸: 112x112")
    print(f"宽度倍数: 0.5")
    print(f"批次大小: {args.batch_size}")
    
    # 创建数据加载器
    train_loader, val_loader = create_cifar10_loaders(batch_size=args.batch_size)
    
    # 开始训练
    train_mobilenetv2(model, train_loader, val_loader, args.save_dir)

if __name__ == '__main__':
    main()