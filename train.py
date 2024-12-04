import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3
from models.quantization import QuantizedConvBNReLU
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from utils.config import default_cfg
import random
import numpy as np
from utils.parser import get_parser
from utils.memory_checker import MemoryChecker

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_data_loaders(cfg):
    """创建数据加载器"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(cfg.data.crop_size, padding=cfg.data.padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std),
    ])
    
    if cfg.data.dataset.lower() == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=cfg.data.data_path, train=True,
            download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=cfg.data.data_path, train=False,
            download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")
    
    train_loader = DataLoader(
        trainset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        testset,
        batch_size=cfg.training.test_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_optimizer(cfg, model):
    """创建优化器"""
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay
    )
    
    if cfg.training.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs)
    elif cfg.training.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.training.lr_step_size,
            gamma=cfg.training.lr_gamma
        )
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.training.lr_scheduler}")
        
    return optimizer, scheduler

def compute_cost(model):
    """计算架构的总成本 Cost(a)"""
    total_cost = 0
    for module in model.modules():
        if isinstance(module, QuantizedConvBNReLU):
            # 根据公式(11)或(12)计算每层的成本
            num_params = module.conv.weight.numel()  # #PARAM
            weight_bits = module.get_weight_bits()   # weight-bit
            act_bits = module.get_activation_bits()  # act-bit
            flops = module.get_flops()              # #FLOP
            
            # 使用公式(12): m_ij_k × #FLOP × weight-bit × act-bit
            layer_cost = flops * weight_bits * act_bits
            total_cost += layer_cost
            
    return total_cost

def compute_cost_weight(cost, beta=1.0, gamma=0.5):
    """计算成本权重 C(Cost(a)) = β(log(Cost(a)))^γ"""
    log_cost = torch.log(torch.tensor(cost) + 1e-7)  # 添加小值避免log(0)
    return beta * (log_cost ** gamma)

def compute_loss(cfg, model, output, target):
    """统一返回格式"""
    criterion = torch.nn.CrossEntropyLoss()
    
    ce_loss = criterion(output, target)
    cost = compute_cost(model)
    cost_weight = compute_cost_weight(cost, cfg.training.cost_beta, cfg.training.cost_gamma)
    total_loss = ce_loss * cost_weight
    
    return total_loss, {
        'ce_loss': ce_loss.item(),
        'cost': cost,
        'cost_weight': cost_weight.item(),
        'total_loss': total_loss.item()
    }

def train_epoch(cfg, model, train_loader, optimizer, epoch):
    """训练一个epoch，只更新模型参数"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Modify the gradient settings
    for name, param in model.named_parameters():
        if 'alpha' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    criterion = torch.nn.CrossEntropyLoss()
    
    end = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(cfg.device), target.to(cfg.device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Calculate loss and metrics
        loss, metrics = compute_loss(cfg, model, output, target)
        
        # 测量准确率
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))
        
        # 反向传播，只更新模型参数
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % cfg.print_freq == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
    
    return {
        'loss': losses.avg,
        'top1': top1.avg,
        'top5': top5.avg
    }

def validate(cfg, model, val_loader):
    """验证模型"""
    model.eval()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        end = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(cfg.device), target.to(cfg.device)
            
            output = model(data)
            loss, metrics = compute_loss(cfg, model, output, target)
            
            # 测���准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            # 添加详细的日志，使用print替代logger
            if batch_idx % cfg.print_freq == 0:
                print(
                    f'Validate: [{batch_idx}/{len(val_loader)}]\t'
                    f'Loss {metrics["total_loss"]:.4f}\t'
                    f'CE Loss {metrics["ce_loss"]:.4f}\t'
                    f'Cost {metrics["cost"]:.4f}\t'
                    f'Cost Weight {metrics["cost_weight"]:.4f}\t'
                    f'Acc@1 {acc1[0]:.3f}\t'
                    f'Acc@5 {acc5[0]:.3f}'
                )
            
            # 更新统计
            losses.update(metrics['total_loss'], data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
            
            # 测量时间
            batch_time.update(time.time() - end)
            end = time.time()
    
    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    
    return {
        'loss': losses.avg,
        'top1': top1.avg,
        'top5': top5.avg
    }

def save_checkpoint(cfg, epoch, model, optimizer, scheduler, is_best, metrics):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'metrics': metrics
    }
    
    # 保存最新的检查点
    filename = os.path.join(cfg.save_path, 'checkpoint.pth.tar')
    torch.save(checkpoint, filename)
    
    # 如果是最佳模型，复制一份
    if is_best:
        best_filename = os.path.join(cfg.save_path, 'model_best.pth.tar')
        from shutil import copyfile
        copyfile(filename, best_filename)

def get_model(args):
    """根据参数创建模型"""
    if args.model == 'mbv2':
        return MobileNetV2(
            num_classes=args.num_classes,
            width_mult=args.width_mult
        )
    elif args.model == 'mbv3-large':
        return MobileNetV3(
            num_classes=args.num_classes,
            mode='large'
        )
    elif args.model == 'mbv3-small':
        return MobileNetV3(
            num_classes=args.num_classes,
            mode='small'
        )
    elif args.model.startswith('efficientnet-b'):
        model_index = int(args.model[-1])
        model_func = globals()[f'efficientnet_b{model_index}']
        return model_func(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

def update_precision_parameters(cfg, model, test_loader, epoch):
    # Initialize metrics dictionary with running totals
    running_metrics = {
        'loss': 0.0,
        'acc1': 0.0,
        'acc5': 0.0,
        'memory_cost': 0.0
    }
    
    # 只允许精度参数的梯度
    for name, param in model.named_parameters():
        if 'alpha' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 创建精度参数优化器
    precision_optimizer = optim.Adam(
        [p for n, p in model.named_parameters() if 'alpha' in n],
        lr=0.01  # 可以调整学习率
    )
    
    for data, target in test_loader:
        data, target = data.to(cfg.device), target.to(cfg.device)
        
        precision_optimizer.zero_grad()
        output = model(data)
        
        # 计算损失：分类损失 + 资源约束
        loss, batch_metrics = compute_loss(cfg, model, output, target)
        
        # 反向传播，更新精度参数
        loss.backward()
        precision_optimizer.step()
        
        running_metrics['loss'] += loss.item()
        running_metrics['acc1'] += accuracy(output, target, topk=(1,))[0]
        running_metrics['acc5'] += accuracy(output, target, topk=(5,))[0]
        running_metrics['memory_cost'] += compute_cost(model)
    
    # Average the metrics
    for key in running_metrics:
        running_metrics[key] /= len(test_loader)
    
    return running_metrics

def main():
    # 解析参数
    args = get_parser().parse_args()
    cfg = default_cfg.from_args(args)
    
    # 设置随机种子
    set_seed(cfg.seed)
    
    # 保存配置
    cfg.save(os.path.join(cfg.save_path, 'config.json'))
    
    # 创建模型
    model = get_model(args)
    model = model.to(cfg.device)
    
    # 在训练开始前进行一次内存分析
    memory_checker = MemoryChecker(
        model, 
        input_shape=(cfg.training.batch_size, 3, 32, 32)
    )
    
    # 打印次内存分析并筛选可行精度
    memory_checker.print_memory_analysis(
        flash_threshold_mb=cfg.training.flash_threshold,
        ram_threshold_mb=cfg.training.memory_threshold
    )
    
    # 根据内存约束筛选可行精度
    flash_ok, flash_info = memory_checker.check_flash_constraints(cfg.training.flash_threshold)
    ram_ok, ram_info = memory_checker.check_ram_constraints(cfg.training.memory_threshold)
    
    if not flash_ok or not ram_ok:
        print("Warning: Some precision configurations may be filtered due to memory constraints")
    
    # 创建数据加载器
    train_loader, test_loader = get_data_loaders(cfg)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    
    # 如果只是评估模型
    if args.evaluate:
        validate(cfg, model, test_loader)
        return
    
    # 创建优化器和学习率调度器
    optimizer, scheduler = get_optimizer(cfg, model)
    
    # 训练循环
    best_acc = 0
    for epoch in range(cfg.training.epochs):
        # 第一步：使用训练集更新模型参数
        train_metrics = train_epoch(cfg, model, train_loader, optimizer, epoch)
        
        # 更新温度参数
        temperature = cfg.quant.initial_temperature * torch.exp(
            torch.tensor(-cfg.quant.temperature_decay * epoch, dtype=torch.float32))
        temperature = max(temperature, cfg.quant.min_temperature)
        
        # 更新所有量化层的温度
        for module in model.modules():
            if isinstance(module, QuantizedConvBNReLU):
                module.temperature = temperature
        
        # 第二步：使用训练集（而不是验证集）更新精度参数
        precision_metrics = update_precision_parameters(cfg, model, train_loader, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 在验证集上评估，但不更新任何参数
        val_metrics = validate(cfg, model, test_loader)
        
        # 保存检查点
        is_best = val_metrics['top1'] > best_acc
        best_acc = max(val_metrics['top1'], best_acc)
        
        if (epoch + 1) % cfg.save_freq == 0 or is_best:
            save_checkpoint(
                cfg, epoch, model, optimizer, scheduler,
                is_best, {
                    'train': train_metrics,
                    'precision': precision_metrics,
                    'val': val_metrics
                }
            )
    
    # 训练结束，冻结模型
    model.freeze()
    
    # 终评估
    final_metrics = validate(cfg, model, test_loader)
    print(f"Final accuracy: {final_metrics['top1']:.2f}%")

if __name__ == '__main__':
    main()