import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

class QuantizedConvBNReLU(nn.Module):
    """Quantized Conv2d + BatchNorm2d + ReLU module"""
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        precision_options: List[str] = ['fp32', 'fp16', 'int8', 'int4', 'int2', 'int1'],
        with_relu: bool = True,
        relu6: bool = False
    ):
        super(QuantizedConvBNReLU, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Layers
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # ReLU configuration
        self.with_relu = with_relu
        self.relu6 = relu6
        if with_relu:
            self.relu = nn.ReLU6() if relu6 else nn.ReLU()
            
        # Quantization parameters
        self.precision_options = precision_options
        self.alpha = nn.Parameter(torch.ones(len(precision_options)))
        self.temperature = 1.0
        
        # Buffers for quantization scales
        self.register_buffer('weight_scales', torch.ones(len(precision_options)))
        self.register_buffer('act_scales', torch.ones(len(precision_options)))
        
        # Memory tracking
        self.register_buffer('memory_cost', torch.zeros(1))
        
    def fold_bn(self, mean: torch.Tensor, var: torch.Tensor, 
                weight: Optional[torch.Tensor] = None, 
                bias: Optional[torch.Tensor] = None, 
                eps: float = 1e-5):
        """Fold BatchNorm parameters into convolution weights"""
        if weight is None:
            weight = torch.ones_like(mean)
        if bias is None:
            bias = torch.zeros_like(mean)
            
        denom = torch.sqrt(var + eps)
        b = weight / denom
        a = bias - mean * b
        
        # Fold into conv weights
        self.conv.weight.data *= b.view(-1, 1, 1, 1)
        
        # Update bias
        if self.conv.bias is None:
            self.conv.bias = nn.Parameter(a)
        else:
            self.conv.bias.data += a
            
    def quantize_weights(self, precision: str) -> torch.Tensor:
        """Quantize weights based on precision"""
        if precision == 'fp32':
            return self.conv.weight
        elif precision == 'fp16':
            return self.conv.weight.half().float()
            
        # INT8: -128 to 127
        elif precision == 'int8':
            scale = self.conv.weight.abs().max() / 127.
            self.weight_scales[self.precision_options.index('int8')] = scale
            return torch.clamp(torch.round(self.conv.weight / scale), -128, 127) * scale
            
        # INT4: -8 to 7
        elif precision == 'int4':
            scale = self.conv.weight.abs().max() / 7.
            self.weight_scales[self.precision_options.index('int4')] = scale
            return torch.clamp(torch.round(self.conv.weight / scale), -8, 7) * scale
            
        # INT2: -2 to 1
        elif precision == 'int2':
            scale = self.conv.weight.abs().max() / 1.
            self.weight_scales[self.precision_options.index('int2')] = scale
            return torch.clamp(torch.round(self.conv.weight / scale), -2, 1) * scale
            
        # INT1: -1 to 1 (Binary)
        elif precision == 'int1':
            scale = self.conv.weight.abs().mean()
            self.weight_scales[self.precision_options.index('int1')] = scale
            return torch.sign(self.conv.weight) * scale
            
        return self.conv.weight
        
    def quantize_activations(self, x: torch.Tensor, precision: str) -> torch.Tensor:
        """Quantize activations based on precision"""
        if precision == 'fp32':
            return x
        elif precision == 'fp16':
            return x.half().float()
            
        # INT8
        elif precision == 'int8':
            scale = x.abs().max() / 127.
            self.act_scales[self.precision_options.index('int8')] = scale
            return torch.clamp(torch.round(x / scale), -128, 127) * scale
            
        # INT4
        elif precision == 'int4':
            scale = x.abs().max() / 7.
            self.act_scales[self.precision_options.index('int4')] = scale
            return torch.clamp(torch.round(x / scale), -8, 7) * scale
            
        # INT2
        elif precision == 'int2':
            scale = x.abs().max() / 1.
            self.act_scales[self.precision_options.index('int2')] = scale
            return torch.clamp(torch.round(x / scale), -2, 1) * scale
            
        # INT1
        elif precision == 'int1':
            scale = x.abs().mean()
            self.act_scales[self.precision_options.index('int1')] = scale
            return torch.sign(x) * scale
            
        return x
        
    def get_memory_cost(self, precision: str) -> float:
        """Calculate memory cost for different precisions"""
        memory_bits = {
            'fp32': 32,
            'fp16': 16,
            'int8': 8,
            'int4': 4,
            'int2': 2,
            'int1': 1
        }
        return memory_bits[precision]
    
    def get_weight_bits(self):
        """Get the number of bits used for weights"""
        # Map each precision option to its bit width and move to same device as alpha
        bits_mapping = torch.tensor([32, 16, 8, 4, 2, 1], device=self.alpha.device)  # Corresponds to fp32, fp16, int8, int4, int2, int1
        return torch.sum(self.alpha * bits_mapping)
    
    def get_activation_bits(self):
        """Get the number of bits used for activations"""
        # Map each precision option to its bit width and move to same device as alpha
        bits_mapping = torch.tensor([32, 16, 8, 4, 2, 1], device=self.alpha.device)  # Corresponds to fp32, fp16, int8, int4, int2, int1
        return torch.sum(self.alpha * bits_mapping)
    
    def get_flops(self):
        """Calculate the number of FLOPs for this layer"""
        # For a conv layer: out_h * out_w * out_channels * kernel_size^2 * in_channels
        out_h = self.output_size[2]
        out_w = self.output_size[3]
        kernel_size = self.kernel_size[0]
        in_channels = self.in_channels
        out_channels = self.out_channels
        
        return out_h * out_w * out_channels * kernel_size * kernel_size * in_channels
        
    def compute_output_size(self, input_size):
        """Compute the output size of the convolutional layer."""
        h_in, w_in = input_size
        h_out = (h_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        self.output_size = (1, self.out_channels, h_out, w_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization"""
        # Compute output size
        self.compute_output_size(x.size()[2:])

        if self.training:
            # Use Gumbel-Softmax for precision selection
            probs = F.gumbel_softmax(self.alpha, tau=self.temperature, hard=False)
            
            # Initialize output and costs
            quantized_out = 0
            memory_cost = 0
            
            for i, precision in enumerate(self.precision_options):
                # Quantize weights and activations
                w = self.quantize_weights(precision)
                x_quant = self.quantize_activations(x, precision)
                
                # Compute output
                temp_out = F.conv2d(
                    x_quant, w, self.conv.bias, 
                    self.stride, self.padding,
                    self.dilation, self.groups
                )
                temp_out = self.bn(temp_out)
                
                if self.with_relu:
                    temp_out = self.relu(temp_out)
                    
                # Accumulate weighted output and costs
                quantized_out += probs[i] * temp_out
                memory_cost += probs[i] * self.get_memory_cost(precision)
            
            # Store costs for loss computation
            self.memory_cost = memory_cost
            
            return quantized_out
            
        else:
            # Inference mode: use best precision
            precision_idx = torch.argmax(self.alpha)
            precision = self.precision_options[precision_idx]
            
            w = self.quantize_weights(precision)
            x_quant = self.quantize_activations(x, precision)
            
            out = F.conv2d(
                x_quant, w, self.conv.bias,
                self.stride, self.padding,
                self.dilation, self.groups
            )
            out = self.bn(out)
                          
            if self.with_relu:
                out = self.relu(out)
                
            return out
            
    def freeze(self):
        """Freeze the layer after training"""
        if self.training:
            self.eval()
            
        # Fold BN parameters
        self.fold_bn(
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias
        )
        
        # Select final precision
        precision_idx = torch.argmax(self.alpha)
        self.current_precision = self.precision_options[precision_idx]
        
        # Quantize weights to final precision
        self.conv.weight.data = self.quantize_weights(self.current_precision)

class BaseQuantizedModel(nn.Module):
    """创建基础量化模型类"""
    def freeze(self):
        for module in self.modules():
            if isinstance(module, QuantizedConvBNReLU):
                module.freeze()
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)