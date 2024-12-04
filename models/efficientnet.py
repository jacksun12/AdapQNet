import torch
import torch.nn as nn
import math
from .quantization import QuantizedConvBNReLU

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_rate = drop_rate
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = int(in_channels * expand_ratio)

        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                QuantizedConvBNReLU(in_channels, hidden_dim, 1, with_relu=True)
            ])

        # Depthwise
        layers.extend([
            QuantizedConvBNReLU(hidden_dim, hidden_dim, kernel_size, 
                               stride=stride, padding=kernel_size//2, 
                               groups=hidden_dim, with_relu=True)
        ])

        # SE
        if se_ratio:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.append(SELayer(hidden_dim, reduction=hidden_dim // se_channels))

        # Projection
        layers.extend([
            QuantizedConvBNReLU(hidden_dim, out_channels, 1, with_relu=False)
        ])

        self.conv = nn.Sequential(*layers)
        
        if self.use_residual:
            self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        if self.use_residual:
            return x + self.dropout(self.conv(x))
        return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=1000):
        super(EfficientNet, self).__init__()
        
        # 基础配置 (EfficientNet-B0)
        settings = [
            # t,  c,  n, s, k
            [1,  16,  1, 1, 3],  # MBConv1, 3x3
            [6,  24,  2, 2, 3],  # MBConv6, 3x3
            [6,  40,  2, 2, 5],  # MBConv6, 5x5
            [6,  80,  3, 2, 3],  # MBConv6, 3x3
            [6, 112,  3, 1, 5],  # MBConv6, 5x5
            [6, 192,  4, 2, 5],  # MBConv6, 5x5
            [6, 320,  1, 1, 3]   # MBConv6, 3x3
        ]

        # 调整通道数和层数
        in_channels = self._round_filters(32, width_mult)
        last_channels = self._round_filters(1280, width_mult)
        
        # 第一层卷积
        layers = [QuantizedConvBNReLU(3, in_channels, 3, stride=2, 
                                    padding=1, with_relu=True)]

        # 构建 MBConv blocks
        for t, c, n, s, k in settings:
            out_channels = self._round_filters(c, width_mult)
            repeats = self._round_repeats(n, depth_mult)
            
            for i in range(repeats):
                stride = s if i == 0 else 1
                layers.append(MBConvBlock(in_channels, out_channels, 
                                        kernel_size=k, stride=stride, 
                                        expand_ratio=t))
                in_channels = out_channels

        # 最后一层卷积
        layers.append(QuantizedConvBNReLU(in_channels, last_channels, 
                                        kernel_size=1, with_relu=True))
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channels, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _round_filters(self, filters, width_mult):
        """Round number of filters based on width multiplier."""
        multiplier = width_mult
        divisor = 8
        filters *= multiplier
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats, depth_mult):
        """Round number of layers based on depth multiplier."""
        return int(math.ceil(depth_mult * repeats))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def freeze(self):
        """训练后冻结所有量化层"""
        for module in self.modules():
            if isinstance(module, QuantizedConvBNReLU):
                module.freeze()

def efficientnet_b0(num_classes=1000):
    return EfficientNet(width_mult=1.0, depth_mult=1.0, num_classes=num_classes)

def efficientnet_b1(num_classes=1000):
    return EfficientNet(width_mult=1.0, depth_mult=1.1, num_classes=num_classes)

def efficientnet_b2(num_classes=1000):
    return EfficientNet(width_mult=1.1, depth_mult=1.2, num_classes=num_classes)

def efficientnet_b3(num_classes=1000):
    return EfficientNet(width_mult=1.2, depth_mult=1.4, num_classes=num_classes)

def efficientnet_b4(num_classes=1000):
    return EfficientNet(width_mult=1.4, depth_mult=1.8, num_classes=num_classes)

def efficientnet_b5(num_classes=1000):
    return EfficientNet(width_mult=1.6, depth_mult=2.2, num_classes=num_classes)

def efficientnet_b6(num_classes=1000):
    return EfficientNet(width_mult=1.8, depth_mult=2.6, num_classes=num_classes)

def efficientnet_b7(num_classes=1000):
    return EfficientNet(width_mult=2.0, depth_mult=3.1, num_classes=num_classes)