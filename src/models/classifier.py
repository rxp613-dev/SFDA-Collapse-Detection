import torch
import torch.nn as nn

class FaultClassifier(nn.Module):
    """
    故障分类头
    
    输入: 256维特征
    输出: 类别概率
    
    Args:
        feature_dim: 特征维度 (默认256)
        num_classes: 类别数量
    """
    
    def __init__(self, feature_dim=256, num_classes=4):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, features):
        """
        分类前向传播
        
        Args:
            features: Backbone提取的特征 [B, 256]
        
        Returns:
            logits: 分类输出 [B, num_classes]
            probs: Softmax概率 [B, num_classes]
        """
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)
        
        return logits, probs

class CompleteModel(nn.Module):
    """
    完整模型: Backbone + Classifier
    
    用于源域预训练和目标域适应
    """
    
    def __init__(self, feature_dim=256, num_classes=4):
        super().__init__()
        
        from models.backbone import BearingFaultBackbone
        
        self.backbone = BearingFaultBackbone(feature_dim)
        self.classifier = FaultClassifier(feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        """
        完整前向传播
        
        Args:
            x: 输入信号 [B, 1, L]
            return_features: 是否返回特征（用于原型提取）
        
        Returns:
            logits: 分类输出
            probs: Softmax概率
            features: 特征（可选）
        """
        features = self.backbone(x)
        logits, probs = self.classifier(features)
        
        if return_features:
            return logits, probs, features
        else:
            return logits, probs
    
    def get_features(self, x):
        """只提取特征，不分类"""
        return self.backbone(x)
    
    def freeze_backbone_partial(self):
        """
        冻结Backbone的前两层卷积
        
        用于Phase 3目标域适应时冻结底层特征提取器
        """
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.conv2.parameters():
            param.requires_grad = False
        print("Backbone前两层已冻结")

if __name__ == '__main__':
    model = CompleteModel(num_classes=4)
    
    for length in [1024, 2048]:
        x = torch.randn(32, 1, length)
        logits, probs = model(x)
        print(f"输入长度 {length}: 输出 {logits.shape}")
    
    print(f"概率sum: {probs.sum(dim=1)[0]}")