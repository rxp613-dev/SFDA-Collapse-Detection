import torch
import torch.nn as nn

class WDCNNBackbone(nn.Module):
    """
    WDCNN: Wide Deep Convolutional Neural Network for Bearing Fault Diagnosis
    
    Reference: Wei et al., 2016, "A New Deep Learning Model for Fault Diagnosis with Good Anti-Noise and Domain Adaptation Ability on Raw Vibration Signals"
    
    Architecture:
    - Conv1: 1D Conv (1 -> 16, kernel_size=64) + BN + ReLU + MaxPool(2)
    - Conv2: 1D Conv (16 -> 32, kernel_size=3) + BN + ReLU + MaxPool(2)
    - Conv3: 1D Conv (32 -> 64, kernel_size=3) + BN + ReLU + MaxPool(2)
    - Conv4: 1D Conv (64 -> 64, kernel_size=3) + BN + ReLU + MaxPool(2)
    - Conv5: 1D Conv (64 -> 64, kernel_size=3) + BN + ReLU + MaxPool(2)
    - AdaptiveAvgPool -> FC -> feature_dim
    """
    
    def __init__(self, feature_dim=256):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=1, padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input signal [B, 1, L]
            
        Returns:
            features: Feature vector [B, feature_dim]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.adaptive_pool(x)
        features = self.fc(x)
        return features


class ResNet1DBackbone(nn.Module):
    """
    ResNet-1D for Bearing Fault Diagnosis
    
    Architecture:
    - Conv1: 1D Conv (1 -> 64, kernel_size=7) + BN + ReLU + MaxPool(3)
    - Layer1: 2 Residual Blocks (64 -> 64)
    - Layer2: 2 Residual Blocks (64 -> 128, stride=2)
    - Layer3: 2 Residual Blocks (128 -> 256, stride=2)
    - AdaptiveAvgPool -> FC -> feature_dim
    """
    
    def __init__(self, feature_dim=256):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input signal [B, 1, L]
            
        Returns:
            features: Feature vector [B, feature_dim]
        """
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.adaptive_pool(x)
        features = self.fc(x)
        return features


class ResidualBlock1D(nn.Module):
    """1D Residual Block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


if __name__ == '__main__':
    # Test WDCNN
    model_wdcnn = WDCNNBackbone(feature_dim=256)
    x = torch.randn(32, 1, 2048)
    features_wdcnn = model_wdcnn(x)
    print(f"WDCNN: Input {x.shape} -> Output {features_wdcnn.shape}")
    
    # Test ResNet-1D
    model_resnet = ResNet1DBackbone(feature_dim=256)
    features_resnet = model_resnet(x)
    print(f"ResNet-1D: Input {x.shape} -> Output {features_resnet.shape}")
