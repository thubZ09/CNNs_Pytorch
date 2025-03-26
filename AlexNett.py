import torch
import torch.nn as nn
from typing import Optional

class AlexNet(nn.Module):
    """
    With original architecture specifications.
    
    Args:
        in_channels (int): Number of input channels (default: 3)
        num_classes (int): Number of output classes (default: 1000)
        dropout_prob (float): Dropout probability for classifier (default: 0.5)
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        num_classes: int = 1000,
        dropout_prob: float = 0.5
    ) -> None:
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )

        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize weights according to original paper:
        - Convolutional and Linear layers: Gaussian with std=0.01
        - Biases: 
          - 1 for 2nd/4th/5th conv layers and 1st/2nd fc layers
          - 0 otherwise
        """
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                
        # Initialize specific biases to 1
        nn.init.constant_(self.features[4].bias, 1)   # 2nd conv
        nn.init.constant_(self.features[8].bias, 1)   # 4th conv
        nn.init.constant_(self.features[10].bias, 1)  # 5th conv
        nn.init.constant_(self.classifier[1].bias, 1) # 1st fc
        nn.init.constant_(self.classifier[4].bias, 1) # 2nd fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass expects input tensor of shape [batch, 3, 227, 227]
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Example usage
    net = AlexNet()
    input_tensor = torch.randn(1, 3, 227, 227)
    output = net(input_tensor)
    print(f"Output shape: {output.shape}")  # Should be (1, 1000)