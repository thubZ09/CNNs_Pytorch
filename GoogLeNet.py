import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

class InceptionBlock(nn.Module):
    """
    Inception block with multiple parallel convolutional paths.
    
    Args:
        in_channels: Number of input channels
        branch_channels: Tuple of output channels for each branch:
            (1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj)
    """
    def __init__(self, in_channels: int, branch_channels: Tuple[int, ...]) -> None:
        super().__init__()
        if len(branch_channels) != 6:
            raise ValueError("branch_channels must contain 6 values for all branches")
            
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels[0], kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels[1], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels[1], branch_channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels[3], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels[3], branch_channels[4], kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: 3x3 pool -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels[5], kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

class AuxiliaryClassifier(nn.Module):
    """
    Auxiliary classifier
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        dropout_rate: Dropout probability (default: 0.7)
    """
    def __init__(self, in_channels: int, num_classes: int, dropout_rate: float = 0.7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) implementation with auxiliary classifiers.
    
    Args:
        num_classes: Number of output classes (default: 1000)
        in_channels: Number of input channels (default: 3)
        dropout_rate: Main classifier dropout (default: 0.4)
    """
    inception_config: List[Tuple[int, Tuple[int, ...]]] = [
        # (in_channels, (branch1, branch2_red, branch2, branch3_red, branch3, branch4))
        (192, (64, 96, 128, 16, 32, 32)),
        (256, (128, 128, 192, 32, 96, 64)),
        (480, (192, 96, 208, 16, 48, 64)),
        (512, (160, 112, 224, 24, 64, 64)),
        (512, (128, 128, 256, 24, 64, 64)),
        (512, (112, 144, 288, 32, 64, 64)),
        (528, (256, 160, 320, 32, 128, 128)),
        (832, (256, 160, 320, 32, 128, 128)),
        (832, (384, 192, 384, 48, 128, 128))
    ]

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        dropout_rate: float = 0.4
    ) -> None:
        super().__init__()
        
        # Initial layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception blocks
        self.inception_blocks = nn.ModuleList()
        for config in self.inception_config:
            self.inception_blocks.append(InceptionBlock(*config))
        
        # Auxiliary classifiers
        self.aux1 = AuxiliaryClassifier(512, num_classes)
        self.aux2 = AuxiliaryClassifier(528, num_classes)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Dropout(p=dropout_rate),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using truncated normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returns list of outputs:
        [auxiliary1_output, auxiliary2_output, main_output]
        """
        outputs = []
        
        # Initial features
        x = self.features(x)
        
        # Inception blocks
        for i, block in enumerate(self.inception_blocks):
            x = block(x)
            
            # Add intermediate pooling
            if i in {2, 7}:
                x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
            
            # Auxiliary classifiers
            if i == 3:
                outputs.append(self.aux1(x))
            elif i == 6:
                outputs.append(self.aux2(x))
        
        # Final classification
        outputs.append(self.classifier(x))
        return outputs

if __name__ == "__main__":
    # Example usage
    model = GoogLeNet()
    input_tensor = torch.randn(1, 3, 224, 224)
    outputs = model(input_tensor)
    
    print("Output shapes:")
    for i, out in enumerate(outputs):
        print(f"Classifier {i+1}: {out.shape}")  # Should be (1, 1000) x3