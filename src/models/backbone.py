import torch
import torch.nn as nn

class BearingFaultBackbone(nn.Module):
    """
    轴承故障诊断Backbone (鲁棒版)
    
    架构: 1D-CNN (3层卷积)
    输入: [B, 1, L] (L可以是1024、2048等任意长度)
    输出: [B, 256] 特征
    
    核心改进: 使用自适应池化层，避免硬编码维度
    """
    
    def __init__(self, feature_dim=256):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=32, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入信号 [B, 1, L] (自适应池化，任意长度均可)
        
        Returns:
            features: 提取的特征 [B, 256]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        features = self.fc(x)
        
        return features

if __name__ == '__main__':
    model = BearingFaultBackbone()
    
    for length in [1024, 2048, 512]:
        x = torch.randn(16, 1, length)
        features = model(x)
        print(f"输入长度 {length}: shape {x.shape} → 输出 {features.shape}")
    
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")