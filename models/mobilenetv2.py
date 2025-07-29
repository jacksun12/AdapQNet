import torch
import torch.nn as nn
import math
from typing import List, Dict
from .base import BaseModel, MixedPrecisionLayer
import torch.nn.functional as F
import os

class ConvBNReLU(nn.Module):
    """Convolution layer with BatchNorm and ReLU6"""
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 groups: int = 1,
                 precision_options: List[str] = None,
                 pretrain_mode: bool = False):
        super().__init__()
        if pretrain_mode:
            # Use standard Sequential in pretrain mode
            self.conv_bn_relu = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                         padding=kernel_size//2, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            # Use MixedPrecisionLayer in quantization mode
            self.conv_bn_relu = MixedPrecisionLayer(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding=kernel_size//2, groups=groups, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU6(inplace=True)
                ),
                precision_options or ["fp32", "fp16", "int8", "int4", "int2"]
            )

    def forward(self, x):
        return self.conv_bn_relu(x)

class ConvBN(nn.Module):
    """Convolution layer with BatchNorm (no ReLU)"""
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 groups: int = 1,
                 precision_options: List[str] = None,
                 pretrain_mode: bool = False):
        super().__init__()
        if pretrain_mode:
            # Use standard Sequential in pretrain mode
            self.conv_bn = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                         padding=kernel_size//2, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # Use MixedPrecisionLayer in quantization mode
            self.conv_bn = MixedPrecisionLayer(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding=kernel_size//2, groups=groups, bias=False),
                    nn.BatchNorm2d(out_channels)
                ),
                precision_options or ["fp32", "fp16", "int8", "int4", "int2"]
            )

    def forward(self, x):
        return self.conv_bn(x)

class InvertedResidual(nn.Module):
    """Inverted Residual Block for MobileNetV2"""
    def __init__(self, 
                 in_planes: int, 
                 out_planes: int, 
                 stride: int, 
                 expand_ratio: int,
                 precision_options: List[str] = None,
                 pretrain_mode: bool = False):
        super().__init__()
        self.stride = stride
        self.use_shortcut = stride == 1 and in_planes == out_planes
        hidden_dim = int(in_planes * expand_ratio)
        if expand_ratio != 1:
            # Expansion layer: Conv + BN + ReLU
            self.expand = ConvBNReLU(in_planes, hidden_dim, 1, 
                                   precision_options=precision_options,
                                   pretrain_mode=pretrain_mode)
        else:
            self.expand = None
            hidden_dim = in_planes
        # Depthwise convolution: Conv + BN + ReLU
        self.depthwise = ConvBNReLU(hidden_dim, hidden_dim, 3, stride, 
                                   groups=hidden_dim,
                                   precision_options=precision_options,
                                   pretrain_mode=pretrain_mode)
        # Projection layer: Conv + BN (no ReLU)
        self.project = ConvBN(hidden_dim, out_planes, 1,
                            precision_options=precision_options,
                            pretrain_mode=pretrain_mode)

    def forward(self, x):
        identity = x
        if self.expand is not None:
            x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_shortcut:
            x = x + identity
        return x

class AdaptQMobileNetV2(BaseModel):
    """AdaptQNet version of MobileNetV2"""
    # Standard MobileNetV2 configuration
    cfgs = [
        # t, c, n, s
        [1,  16, 1, 1],
        [6,  24, 2, 2],
        [6,  32, 3, 2],
        [6,  64, 4, 2],
        [6,  96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    
    def __init__(self, 
                 num_classes=1000, 
                 width_mult=1.0,
                 precision_options=None,
                 hardware_constraints=None,
                 pretrain_mode=False,
                 initialize_weights=True,
                 input_size=224):
        super().__init__(precision_options, hardware_constraints)
        self.input_size = input_size
        self.cfg = self.cfgs
        self.last_channel = int(1280 * max(1.0, width_mult))
            
        # Build feature extractor
        self.features = self._build_mixed_precision_features(width_mult, pretrain_mode)
        # Classifier
        if pretrain_mode:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                MixedPrecisionLayer(
                    nn.Linear(self.last_channel, num_classes),
                    precision_options=self.precision_options
                )
            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if initialize_weights:
            self._initialize_weights()

    def _build_mixed_precision_features(self, width_mult, pretrain_mode=False):
        """Build feature extractor"""
        input_channel = int(32 * width_mult)
        features = []
        
        # First convolution layer - adjust stride based on input size
        if self.input_size <= 64:
            # For small inputs (32x32, 64x64), use stride=1 to preserve spatial dimensions
            first_conv = ConvBNReLU(3, input_channel, 3, stride=1,
                                   precision_options=self.precision_options,
                                   pretrain_mode=pretrain_mode)
        else:
            # For larger inputs, use stride=2
            first_conv = ConvBNReLU(3, input_channel, 3, stride=2,
                                   precision_options=self.precision_options,
                                   pretrain_mode=pretrain_mode)
        features.append(first_conv)
        
        # Build inverted residual blocks
        for t, c, n, s in self.cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(
                    input_channel, output_channel, stride, t,
                    precision_options=self.precision_options,
                    pretrain_mode=pretrain_mode
                ))
                input_channel = output_channel
        # Final 1x1 convolution
        features.append(ConvBNReLU(
            input_channel, self.last_channel, 1,
            precision_options=self.precision_options,
            pretrain_mode=pretrain_mode
        ))
        return nn.Sequential(*features)

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def load_imagenet_pretrained_weights(self, pretrained_path=None):
        """加载ImageNet预训练权重"""
        import torchvision.models as models
        
        try:
            if pretrained_path and os.path.exists(pretrained_path):
                # 从本地文件加载
                print(f"从本地文件加载预训练权重: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                # 从torchvision加载官方预训练模型
                print("从torchvision加载ImageNet预训练权重...")
                pretrained_model = models.mobilenet_v2(pretrained=True)
                state_dict = pretrained_model.state_dict()
            
            # 创建预训练模式的模型来加载权重
            pretrained_model = AdaptQMobileNetV2(
                num_classes=1000,
                width_mult=1.0,
                precision_options=None,
                pretrain_mode=True,
                initialize_weights=False
            )
            
            # 加载权重
            pretrained_model.load_state_dict(state_dict, strict=False)
            print("ImageNet预训练权重加载成功")
            
            # 将权重复制到当前模型，跳过分类器层
            self.copy_weights_from_pretrained(pretrained_model, skip_classifier=True)
            print("权重已成功迁移到当前模型（跳过分类器层）")
            
            return True
            
        except Exception as e:
            print(f"加载ImageNet预训练权重失败: {e}")
            return False

    def load_imagenet_pretrained_from_url(self, url=None):
        """从URL加载ImageNet预训练权重"""
        import urllib.request
        import os
        
        if url is None:
            # 使用默认的预训练权重URL（这里需要根据实际情况设置）
            url = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
        
        try:
            # 创建缓存目录
            cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 下载文件名
            filename = os.path.basename(url)
            filepath = os.path.join(cache_dir, filename)
            
            # 如果文件不存在，则下载
            if not os.path.exists(filepath):
                print(f"正在下载预训练权重: {url}")
                urllib.request.urlretrieve(url, filepath)
                print(f"下载完成: {filepath}")
            else:
                print(f"使用缓存的预训练权重: {filepath}")
            
            # 加载权重
            return self.load_imagenet_pretrained_weights(filepath)
            
        except Exception as e:
            print(f"从URL加载预训练权重失败: {e}")
            return False

# 使用示例:
# 
# 1. 创建模型并加载ImageNet预训练权重
# model = AdaptQMobileNetV2(num_classes=10, input_size=224)
# model.load_imagenet_pretrained_from_url()
# 
# 2. 从本地文件加载预训练权重
# model = AdaptQMobileNetV2(num_classes=10, input_size=224)
# model.load_imagenet_pretrained_weights("path/to/pretrained.pth")
# 
# 3. 使用torchvision预训练模型
# model = AdaptQMobileNetV2(num_classes=10, input_size=224)
# model.load_imagenet_pretrained_weights()
