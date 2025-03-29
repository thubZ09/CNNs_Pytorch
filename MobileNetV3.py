import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional, Union

class ConvBlock(nn.Module):
    """
    Configurable convolution block with optional activation and normalization.
    
    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        kernel_size: Convolution kernel size
        stride: Convolution stride
        activation: Activation function (default: ReLU)
        groups: Number of groups for grouped convolution
        use_bn: Whether to use batch normalization
        bias: Whether to use bias in convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: Optional[nn.Module] = nn.ReLU(),
        groups: int = 1,
        use_bn: bool = True,
        bias: bool = False
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.bn(self.conv(x))


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block with channel-wise attention.
    
    Args:
        channels: Number of input/output channels
        reduction_ratio: Channel reduction ratio (default: 4)
    """
    def __init__(self, channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class MobileNetV3Block(nn.Module):
    """
    MobileNetV3 Bottleneck Block with optional SE and residual connection.
    
    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        kernel_size: Convolution kernel size
        expansion_ratio: Channel expansion ratio
        use_se: Whether to use Squeeze-and-Excitation
        activation: Activation function
        stride: Block stride
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        expansion_ratio: float,
        use_se: bool,
        activation: nn.Module,
        stride: int
    ) -> None:
        super().__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        expanded_channels = int(in_channels * expansion_ratio)
        
        layers = []
        # Expansion layer
        if expansion_ratio != 1:
            layers.append(ConvBlock(
                in_channels, expanded_channels, 1, 1,
                activation=activation
            ))
            
        # Depthwise convolution
        layers.append(ConvBlock(
            expanded_channels, expanded_channels, kernel_size, stride,
            groups=expanded_channels, activation=activation
        ))
        
        # Squeeze-and-excitation
        if use_se:
            layers.append(SqueezeExcitation(expanded_channels))
            
        # Projection layer
        layers.append(ConvBlock(
            expanded_channels, out_channels, 1, 1,
            activation=None, use_bn=True
        ))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.block(x)
        return x + residual if self.use_residual else x


class MobileNetV3(nn.Module):
    """
    MobileNetV3 implementation with configurable architecture.
    
    Args:
        model_type: Model variant ('large' or 'small')
        in_channels: Input channels (default: 3)
        num_classes: Output classes (default: 1000)
        dropout_rate: Final dropout rate (default: 0.8)
    """
    def __init__(
        self,
        model_type: str = "large",
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout_rate: float = 0.8
    ) -> None:
        super().__init__()
        self.config = self._get_config(model_type)
        
        # Initial convolution
        self.features = nn.Sequential(
            ConvBlock(in_channels, 16, 3, 2, activation=nn.Hardswish())
        )
        
        # Build bottleneck blocks
        current_channels = 16
        for params in self.config:
            kernel_size, exp_ratio, _, out_channels, use_se, act, stride = params
            self.features.append(MobileNetV3Block(
                current_channels, out_channels,
                kernel_size=kernel_size,
                expansion_ratio=exp_ratio,
                use_se=use_se,
                activation=act(),
                stride=stride
            ))
            current_channels = out_channels
        
        # Classifier
        final_expansion = 960 if model_type == "large" else 576
        final_features = 1280 if model_type == "large" else 1024
        self.classifier = nn.Sequential(
            ConvBlock(current_channels, final_expansion, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(final_expansion, final_features, 1, 1, nn.Hardswish(), use_bn=False, bias=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(final_features, num_classes, 1)
        )
        
        self._initialize_weights()

    def _get_config(self, model_type: str) -> List[Tuple]:
        """Get architecture configuration for specified model type."""
        ReLU = partial(nn.ReLU, inplace=True)
        HSwish = partial(nn.Hardswish, inplace=True)
        
        configurations = {
            "large": [
                (3, 1.0, 16, 16, False, ReLU, 1),
                (3, 4.0, 16, 24, False, ReLU, 2),
                (3, 3.0, 24, 24, False, ReLU, 1),
                (5, 3.0, 24, 40, True, ReLU, 2),
                (5, 3.0, 40, 40, True, ReLU, 1),
                (5, 3.0, 40, 40, True, ReLU, 1),
                (3, 6.0, 40, 80, False, HSwish, 2),
                (3, 2.5, 80, 80, False, HSwish, 1),
                (3, 2.3, 80, 80, False, HSwish, 1),
                (3, 2.3, 80, 80, False, HSwish, 1),
                (3, 6.0, 80, 112, True, HSwish, 1),
                (3, 6.0, 112, 112, True, HSwish, 1),
                (5, 6.0, 112, 160, True, HSwish, 2),
                (5, 6.0, 160, 160, True, HSwish, 1),
                (5, 6.0, 160, 160, True, HSwish, 1)
            ],
            "small": [
                (3, 1.0, 16, 16, True, ReLU, 2),
                (3, 4.5, 16, 24, False, ReLU, 2),
                (3, 3.7, 24, 24, False, ReLU, 1),
                (5, 4.0, 24, 40, True, HSwish, 2),
                (5, 6.0, 40, 40, True, HSwish, 1),
                (5, 6.0, 40, 40, True, HSwish, 1),
                (5, 3.0, 40, 48, True, HSwish, 1),
                (5, 3.0, 48, 48, True, HSwish, 1),
                (5, 6.0, 48, 96, True, HSwish, 2),
                (5, 6.0, 96, 96, True, HSwish, 1),
                (5, 6.0, 96, 96, True, HSwish, 1)
            ]
        }
        
        if model_type not in configurations:
            raise ValueError(f"Invalid model_type: {model_type}. Choose from {list(configurations.keys())}")
            
        return configurations[model_type]

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


if __name__ == "__main__":
    # Example usage
    model = MobileNetV3(model_type="large")
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")  # Should be (1, 1000)