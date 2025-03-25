import torch
import torch.nn as nn
from typing import List, Dict

class ConvBlock(nn.Module):
    """
    A convolutional block consisting of a 2D convolution followed by batch normalization.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride for the convolution
        padding (int): Padding for the convolution
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, padding: int) -> None:
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.c(x))

class ResidualBlock(nn.Module):
    """
    Bottleneck residual block from the ResNet architecture.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        is_first_block (bool): Whether this is the first block in a stage (modifies stride and channels)
    """
    def __init__(self, in_channels: int, out_channels: int, is_first_block: bool = False) -> None:
        super().__init__()
        self.projection = in_channels != out_channels
        stride = 1
        res_channels = in_channels // 4

        # Adjust parameters if this is the first block in a stage
        if self.projection:
            if is_first_block:
                # First block of the first stage doesn't downsample
                self.p = ConvBlock(in_channels, out_channels, 1, 1, 0)
                res_channels = in_channels  
            else:
                # Subsequent stages' first blocks downsample
                self.p = ConvBlock(in_channels, out_channels, 1, 2, 0)
                stride = 2
                res_channels = in_channels // 2

        # Bottleneck layers
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual path
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)

        # Shortcut connection
        if self.projection:
            x = self.p(x)
            
        return self.relu(f + x)

class ResNet(nn.Module):
    """
    ResNet model with bottleneck blocks.
    
    Args:
        config_name (int): Depth of ResNet (50, 101, or 152)
        in_channels (int): Number of input channels (default: 3)
        num_classes (int): Number of output classes (default: 1000)
    """
    def __init__(self, config_name: int, in_channels: int = 3, num_classes: int = 1000) -> None:
        super().__init__()
        
        configurations: Dict[int, List[int]] = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        
        if config_name not in configurations:
            raise ValueError(f"Invalid config_name: {config_name}. Must be one of {list(configurations.keys())}")
        
        num_blocks = configurations[config_name]
        stage_channels = [256, 512, 1024, 2048]

        # Initial layers
        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Create residual blocks
        self.blocks = nn.ModuleList()
        # First stage
        self.blocks.append(ResidualBlock(64, 256, is_first_block=True))
        for _ in range(num_blocks[0] - 1):
            self.blocks.append(ResidualBlock(256, 256))
        
        # Subsequent stages
        for i in range(1, len(stage_channels)):
            # First block of the stage downsamples
            self.blocks.append(ResidualBlock(stage_channels[i-1], stage_channels[i]))
            for _ in range(num_blocks[i] - 1):
                self.blocks.append(ResidualBlock(stage_channels[i], stage_channels[i]))
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.relu = nn.ReLU()
        
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Example usage
    model = ResNet(50)
    image = torch.randn(1, 3, 224, 224)
    output = model(image)
    print(f"Output shape: {output.shape}")  # Should be (1, 1000)