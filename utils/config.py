from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import torch

@dataclass
class QuantConfig:
    """量化配置"""
    precision_options: List[str] = field(default_factory=lambda: [
        'fp32', 'fp16', 'int8', 'int4', 'int2', 'int1'
    ])
    
    # 不同精度的内存成本(bits)
    memory_bits: dict = field(default_factory=dict)
    
    # Gumbel-Softmax参数
    initial_temperature: float = 1.0
    min_temperature: float = 0.1
    temperature_decay: float = 0.01

@dataclass
class ModelConfig:
    """模型配置"""
    # 基础模型参数
    num_classes: int = 10
    width_mult: float = 1.0
    
    # MobileNetV2配置
    input_channel: int = 32
    last_channel: int = 1280
    
    # 网络结构配置 [t, c, n, s]
    # t: expansion factor
    # c: output channels
    # n: number of blocks
    # s: stride
    inverted_residual_setting: List[List[int]] = (
        [1, 16, 1, 1],
        [6, 24, 2, 1],  # stride 2 -> 1 for CIFAR
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    )

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    epochs: int = 200
    batch_size: int = 128
    test_batch_size: int = 100
    
    # 优化器参数
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # 学习率调度
    lr_scheduler: str = 'cosine'  # ['step', 'cosine']
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    
    # 损失函数权重
    memory_threshold: float = 0.25
    flash_threshold: float = 2.0
    
    # 精度参数优化器配置
    precision_lr: float = 0.01
    precision_optimizer: str = 'adam'
    
    # 成本函数参数
    cost_beta: float = 1.0   # β 系数，用于调整初始成本权重
    cost_gamma: float = 0.5  # γ 系数，用于调整成本项的重要性

@dataclass
class DataConfig:
    """数据配置"""
    # 数据集参数
    dataset: str = 'cifar10'
    data_path: str = './data'
    num_workers: int = 2
    
    # 数据增强参数
    crop_size: int = 32
    padding: int = 4
    
    # 标准化参数
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)

@dataclass
class Config:
    """总配置"""
    # 运行配置
    seed: int = 42
    gpu: Optional[int] = None  # None表示使用所有可用GPU
    distributed: bool = False
    
    # 路径配置
    save_path: str = './checkpoints'
    log_path: str = './logs'
    
    # 模型保存配置
    save_freq: int = 10  # 每N个epoch保存一次
    
    # 打印配置
    print_freq: int = 100  # 每N个batch打印一次
    
    # 子配置
    quant: QuantConfig = QuantConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    
    def __post_init__(self):
        """初始化后的处理"""
        # 设置设备
        self.device = torch.device(
            f"cuda:{self.gpu}" if torch.cuda.is_available() and self.gpu is not None
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        
        # 创建必要的目录
        import os
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
    
    def from_args(cls, args):
        """统一从命令行参数创建配置"""
        config = cls
        # 使用反射自动设置所有匹配的属性
        for k, v in vars(args).items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config
    
    def save(self, path):
        """保存配置到文件"""
        import json
        
        # Convert dataclass objects to dictionaries
        config_dict = asdict(self)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path):
        """从文件加载配置"""
        import json
        with open(path, 'r') as f:
            config = cls()
            config.__dict__.update(json.load(f))
            return config

# 默认配置
default_cfg = Config()
default_cfg.quant = QuantConfig()
default_cfg.quant.initial_temperature = 5.0   # T0: 初始温度
default_cfg.quant.temperature_decay = 0.1     # η: 衰减率
default_cfg.quant.min_temperature = 0.1       # 最小温度阈值