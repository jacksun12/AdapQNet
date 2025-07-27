import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import logging


def create_train_val_test_loaders(batch_size=64, num_workers=8, input_size=112):
    """创建训练集、验证集和测试集加载器（支持自定义分辨率）
    Args:
        batch_size: 批次大小（单卡96优化）
        num_workers: 数据加载器工作进程数
        input_size: 输入图像尺寸，默认112x112
    """
    logger = logging.getLogger(__name__)
    
    # 训练集变换：包含数据增强
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # 调整到目标尺寸
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # 添加随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色增强
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 验证集和测试集变换：只包含resize和标准化
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # 调整到目标尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 训练集：使用原始的train=True数据集（50000张）
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    
    # 从测试集（train=False，10000张）中划分验证集和测试集
    test_dataset_full = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform)
    
    # 从测试集中取6000张作为验证集，4000张作为测试集
    val_size = 6000
    test_size = 4000
    
    val_dataset, test_dataset = random_split(
        test_dataset_full, [val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    logger.info(f"数据集划分 ({input_size}x{input_size}分辨率):")
    logger.info(f"  训练集: {len(train_dataset)} 张 (权重训练)")
    logger.info(f"  验证集: {len(val_dataset)} 张 (alpha训练)")
    logger.info(f"  测试集: {len(test_dataset)} 张 (性能评估)")
    logger.info(f"  批次大小: {batch_size}")
    logger.info(f"  输入尺寸: {input_size}x{input_size}")
    
    return train_loader, val_loader, test_loader 