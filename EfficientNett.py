import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

class ConvBlock(nn.Module):
    """
    Basic convolution block with optional activation.
    
    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        kernel_size: Convolution kernel size
        stride: Convolution stride
        groups: Number of groups for grouped convolution
        activation: Whether to apply SiLU activation
        bias: Whether to use bias in convolution
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        activation: bool = True,
        bias: bool = False
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # Auto-calculate padding
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    
    Args:
        channels: Number of input channels
        reduction_ratio: Channel reduction ratio for bottleneck
    """
    def __init__(self, channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.SiLU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights

class MBConv(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block with optional SE and stochastic depth.
    
    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        kernel_size: Convolution kernel size
        stride: Block stride
        expansion_ratio: Channel expansion ratio
        reduction_ratio: SE reduction ratio
        survival_prob: Stochastic depth survival probability
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expansion_ratio: float,
        reduction_ratio: int = 4,
        survival_prob: float = 0.5
    ) -> None:
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = int(in_channels * expansion_ratio)
        
        # Expansion phase
        self.expand = nn.Identity()
        if expansion_ratio != 1:
            self.expand = ConvBlock(in_channels, hidden_dim, 1, 1)

        # Depthwise convolution
        self.dw_conv = ConvBlock(
            hidden_dim, hidden_dim, kernel_size, stride,
            groups=hidden_dim, activation=True
        )

        # Squeeze-and-excitation
        self.se = SqueezeExcitation(hidden_dim, reduction_ratio)

        # Projection
        self.project = ConvBlock(hidden_dim, out_channels, 1, 1, activation=False)

        # Stochastic depth
        self.stochastic_depth = StochasticDepth(survival_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.expand(x)
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.project(x)
        
        if self.use_residual:
            x = self.stochastic_depth(x)
            x += residual
            
        return x

class EfficientNet(nn.Module):
    """
    EfficientNet model with compound scaling.
    
    Args:
        model_name: Model variant (B0-B7)
        in_channels: Input channels (default: 3)
        num_classes: Output classes (default: 1000)
        verbose: Whether to print layer dimensions (default: False)
    """
    def __init__(
        self,
        model_name: str = "B0",
        in_channels: int = 3,
        num_classes: int = 1000,
        verbose: bool = False
    ) -> None:
        super().__init__()
        self.verbose = verbose
        
        # Base configuration
        self.config = self._get_base_config()
        phi_params = self._get_phi_params(model_name)
        
        # Scaling coefficients
        depth_coef, width_coef = 1.2, 1.1
        self.depth_factor = depth_coef ** phi_params.phi
        self.width_factor = width_coef ** phi_params.phi
        
        # Network construction
        self.features = self._build_stages(in_channels, phi_params.resolution)
        self.classifier = self._build_classifier(num_classes, phi_params.dropout)
        
        self._initialize_weights()

    def _get_base_config(self) -> Dict:
        """Return base network configuration."""
        return {
            "stages": [
                # [operator, channels, layers, kernel, stride, expansion]
                [ConvBlock, 32, 1, 3, 2, 1],
                [MBConv, 16, 1, 3, 1, 1],
                [MBConv, 24, 2, 3, 2, 6],
                [MBConv, 40, 2, 5, 2, 6],
                [MBConv, 80, 3, 3, 2, 6],
                [MBConv, 112, 3, 5, 1, 6],
                [MBConv, 192, 4, 5, 2, 6],
                [MBConv, 320, 1, 3, 1, 6],
                [ConvBlock, 1280, 1, 1, 1, 1]
            ],
            "phi_config": {
                "B0": (0, 224, 0.2),
                "B1": (0.5, 240, 0.2),
                "B2": (1, 260, 0.3),
                "B3": (2, 300, 0.3),
                "B4": (3, 380, 0.4),
                "B5": (4, 456, 0.4),
                "B6": (5, 528, 0.5),
                "B7": (6, 600, 0.5)
            }
        }

    def _get_phi_params(self, model_name: str):
        """Validate and return scaling parameters."""
        if model_name not in self.config["phi_config"]:
            raise ValueError(f"Invalid model name: {model_name}. "
                             f"Choose from {list(self.config['phi_config'].keys())}")
        
        from collections import namedtuple
        PhiParams = namedtuple("PhiParams", ["phi", "resolution", "dropout"])
        return PhiParams(*self.config["phi_config"][model_name])

    def _build_stages(self, in_channels: int, input_size: int) -> nn.Sequential:
        """Construct network stages with scaled parameters."""
        stages = []
        current_channels = in_channels
        
        for i, stage_config in enumerate(self.config["stages"]):
            op, base_channels, base_layers, kernel, stride, expansion = stage_config
            
            # Scale channels and layers
            channels = self._scale_channels(base_channels)
            layers = self._scale_layers(base_layers)
            
            # Build stage
            stage = []
            for layer_idx in range(layers):
                layer_stride = stride if layer_idx == layers - 1 else 1
                
                if op == MBConv:
                    stage.append(
                        op(current_channels, channels, kernel, layer_stride, expansion)
                else:
                    stage.append(op(current_channels, channels, kernel, layer_stride))
                
                current_channels = channels
            
            stages.extend(stage)
            
            if self.verbose:
                print(f"Stage {i+1}: {channels} channels, {layers} layers")
        
        return nn.Sequential(*stages)

    def _build_classifier(self, num_classes: int, dropout: float) -> nn.Sequential:
        """Build final classification head."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(self.config["stages"][-1][1], num_classes)
        )

    def _scale_channels(self, channels: int) -> int:
        """Apply width scaling coefficient."""
        return int(channels * self.width_factor // 8 * 8)  # Round to multiple of 8

    def _scale_layers(self, layers: int) -> int:
        """Apply depth scaling coefficient."""
        return int(layers * self.depth_factor)

    def _initialize_weights(self) -> None:
        """Initialize weights using truncated normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional dimension logging."""
        if self.verbose:
            print(f"Input shape: {x.shape}")
        
        x = self.features(x)
        x = self.classifier(x)
        
        if self.verbose:
            print(f"Output shape: {x.shape}")
        return x

if __name__ == "__main__":
    # Example usage
    model = EfficientNet(model_name="B0", verbose=True)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(f"Final output shape: {output.shape}")