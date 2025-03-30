import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class VGGBlock(nn.Module):
    """
    multiple convolutional layers followed by max pooling
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_layers: Number of convolutional layers in the block
    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        
        # First convolution (input -> output channels)
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        
        # Additional convolutions (maintaining output channels)
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ]
            
        # Final max pooling
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class VGG(nn.Module):
    """
    network architecture
    
    Args:
        model_size: Model variant (VGG11, VGG13, VGG16, VGG19)
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 1000)
    """
    configs: Dict[str, List[int]] = {
        "VGG11": [1, 1, 2, 2, 2],
        "VGG13": [2, 2, 2, 2, 2],
        "VGG16": [2, 2, 3, 3, 3],
        "VGG19": [2, 2, 4, 4, 4]
    }
    
    channel_sequence: List[int] = [64, 128, 256, 512, 512]
    
    def __init__(
        self, 
        model_size: str,
        in_channels: int = 3, 
        num_classes: int = 1000
    ) -> None:
        super().__init__()
        
        # Validate model configuration
        if model_size not in self.configs:
            raise ValueError(f"Invalid model_size: {model_size}. "
                             f"Choose from {list(self.configs.keys())}")
        
        # Build feature extractor
        features: List[nn.Module] = []
        current_channels = in_channels
        config = self.configs[model_size]
        
        for i, (block_layers, out_channels) in enumerate(zip(config, self.channel_sequence)):
            features.append(
                VGGBlock(current_channels, out_channels, num_layers=block_layers)
            )
            current_channels = out_channels
            
        self.features = nn.Sequential(*features)
        
        # Build classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming normal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expects input tensor of shape [batch, 3, 224, 224]"""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Example usage
    model = VGG("VGG11")
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Should be (1, 1000)