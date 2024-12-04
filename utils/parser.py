import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Mixed Precision Training')
    
    # 基础配置
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use')
    
    # 模型配置
    parser.add_argument('--model', type=str, default='mbv2',
                        choices=['mbv2', 'mbv3-large', 'mbv3-small', 
                                'efficientnet-b0', 'efficientnet-b1', 
                                'efficientnet-b2', 'efficientnet-b3',
                                'efficientnet-b4', 'efficientnet-b5',
                                'efficientnet-b6', 'efficientnet-b7'],
                        help='model architecture')
    parser.add_argument('--width-mult', type=float, default=1.0,
                        help='width multiplier for mobilenet')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of classes')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='batch size for testing')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    
    # 量化配置
    parser.add_argument('--precision-options', type=str, nargs='+',
                        default=['fp32', 'fp16', 'int8', 'int4', 'int2', 'int1'],
                        help='precision options for quantization')
    parser.add_argument('--memory-threshold', type=float, default=1.0,
                        help='memory threshold for penalty')
    
    # 路径配置
    parser.add_argument('--data-path', type=str, default='./data',
                        help='path to dataset')
    parser.add_argument('--save-path', type=str, default='./checkpoints',
                        help='path to save checkpoints')
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='path to save logs')
    
    # 其他配置
    parser.add_argument('--resume', type=str, default='',
                        help='path to latest checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='save frequency')
    
    return parser
