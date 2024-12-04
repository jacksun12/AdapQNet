import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import json
from pathlib import Path

from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3
from utils.config import Config
from utils.memory_checker import MemoryChecker
from utils.parser import get_parser

def evaluate(model, test_loader, device, criterion):
    """评估模型性能"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{test_loss/total:.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    return {
        'loss': test_loss / total,
        'accuracy': 100. * correct / total
    }

def analyze_model(model, input_shape, flash_threshold, ram_threshold):
    """分析模型的内存使用情况"""
    memory_checker = MemoryChecker(model, input_shape)
    memory_checker.print_memory_analysis(flash_threshold, ram_threshold)
    
    # 返回详细的内存分析结果
    flash_ok, flash_info = memory_checker.check_flash_constraints(flash_threshold)
    ram_ok, ram_info = memory_checker.check_ram_constraints(ram_threshold)
    
    return {
        'flash': {
            'total': flash_info['total_flash'],
            'satisfied': flash_ok,
            'layer_precisions': flash_info['layer_precisions']
        },
        'ram': {
            'max': ram_info['max_ram'],
            'satisfied': ram_ok,
            'layer_ram': ram_info['layer_ram']
        }
    }

def main():
    # 解析命令行参数
    parser = get_parser()
    # 添加评估特定的参数
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='path to model checkpoint')
    parser.add_argument('--save-analysis', action='store_true',
                      help='save memory analysis to file')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # 加载配置
    if os.path.exists(os.path.join(Path(args.checkpoint).parent, 'config.json')):
        cfg = Config.load(os.path.join(Path(args.checkpoint).parent, 'config.json'))
    else:
        cfg = Config.from_args(args)
    
    # 准备数据
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.data.mean, cfg.data.std),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root=cfg.data.data_path, train=False,
        download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.training.test_batch_size,
        shuffle=False, num_workers=cfg.data.num_workers)
    
    # 创建模型
    if args.model == 'mbv2':
        model = MobileNetV2(num_classes=cfg.model.num_classes)
    elif args.model == 'mbv3-large':
        model = MobileNetV3(num_classes=cfg.model.num_classes, mode='large')
    elif args.model == 'mbv3-small':
        model = MobileNetV3(num_classes=cfg.model.num_classes, mode='small')
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # 加载检查点
    if os.path.isfile(args.checkpoint):
        print(f"=> loading checkpoint '{args.checkpoint}'")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded checkpoint '{args.checkpoint}' "
              f"(epoch {checkpoint['epoch']})")
    else:
        raise ValueError(f"No checkpoint found at '{args.checkpoint}'")
    
    model = model.to(device)
    model.eval()
    
    # 分析模型
    print("\nAnalyzing model memory usage...")
    memory_analysis = analyze_model(
        model,
        input_shape=(1, 3, 32, 32),
        flash_threshold=cfg.training.flash_threshold,
        ram_threshold=cfg.training.ram_threshold
    )
    
    # 评估模型
    print("\nEvaluating model performance...")
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, device, criterion)
    
    # 打印结果
    print("\n=== Evaluation Results ===")
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    
    # 保存分析结果
    if args.save_analysis:
        results = {
            'metrics': metrics,
            'memory_analysis': memory_analysis,
            'model': args.model,
            'checkpoint': args.checkpoint
        }
        
        output_file = os.path.join(
            Path(args.checkpoint).parent,
            'evaluation_results.json'
        )
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()