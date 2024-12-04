import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization import QuantizedConvBNReLU

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se, activation):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        hidden_dim = round(in_channels * expand_ratio)
        
        layers = []
        if expand_ratio != 1:
            # Pointwise
            layers.append(QuantizedConvBNReLU(in_channels, hidden_dim, kernel_size=1, with_relu=True))
        
        # Depthwise
        layers.append(QuantizedConvBNReLU(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, with_relu=True))
        
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))
        
        # Pointwise-linear
        layers.append(QuantizedConvBNReLU(hidden_dim, out_channels, kernel_size=1, with_relu=False))
        
        self.conv = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, mode='large'):
        super(MobileNetV3, self).__init__()
        self.mode = mode
        if mode == 'large':
            self.cfgs = [
                # k, exp, c, se, nl, s
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1],
            ]
            last_channel = 1280
        elif mode == 'small':
            self.cfgs = [
                # k, exp, c, se, nl, s
                [3, 16, 16, True, 'RE', 2],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],
            ]
            last_channel = 1024
        else:
            raise ValueError("Unsupported mode: {}".format(mode))

        input_channel = 16
        layers = [QuantizedConvBNReLU(3, input_channel, kernel_size=3, stride=2, padding=1, with_relu=True)]
        
        for k, exp, c, se, nl, s in self.cfgs:
            output_channel = c
            activation = nn.ReLU(inplace=True) if nl == 'RE' else HSwish()
            layers.append(InvertedResidual(input_channel, output_channel, k, s, exp, se, activation))
            input_channel = output_channel
        
        self.features = nn.Sequential(*layers)
        self.conv = QuantizedConvBNReLU(input_channel, last_channel, kernel_size=1, with_relu=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x