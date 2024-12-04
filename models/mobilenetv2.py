import torch
import torch.nn as nn
from .quantization import QuantizedConvBNReLU

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(
                QuantizedConvBNReLU(
                    in_channels, 
                    hidden_dim, 
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                    with_relu=True,
                    relu6=True
                )
            )
        
        # Depthwise
        layers.append(
            QuantizedConvBNReLU(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
                with_relu=True,
                relu6=True
            )
        )
        
        # Pointwise projection
        layers.append(
            QuantizedConvBNReLU(
                hidden_dim,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                with_relu=False
            )
        )
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        
        # CIFAR-10 specific settings
        input_channel = int(input_channel * width_mult)
        if width_mult > 1.0:
            last_channel = int(last_channel * width_mult)

        # Building blocks configuration
        # t: expansion factor, c: output channels, n: number of blocks, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # Changed stride 2 -> 1 for CIFAR-10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First layer
        self.features = [
            QuantizedConvBNReLU(
                3, input_channel, 
                kernel_size=3,
                stride=1,  # Changed stride 2 -> 1 for CIFAR-10
                padding=1,
                bias=False,
                with_relu=True,
                relu6=True
            )
        ]

        # Inverted Residual Blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        # Last convolution
        self.features.append(
            QuantizedConvBNReLU(
                input_channel,
                last_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                with_relu=True,
                relu6=True
            )
        )
        
        self.features = nn.Sequential(*self.features)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channel, num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
                
    def freeze(self):
        """Freeze all quantized layers after training"""
        for module in self.modules():
            if isinstance(module, QuantizedConvBNReLU):
                module.freeze()
                