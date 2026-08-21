"""
任务4: 模型结构改进 - Attention层
添加注意力机制提升特征表达能力，预期准确率60-65%

Attention机制:
1. Self-Attention: 特征图内部注意力
2. Channel Attention: 通道注意力（类似SENet）
3. Spatial Attention: 空间注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力模块（SENet风格）"""
    
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, length]
        
        Returns:
            attended: [batch, channels, length]
        """
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))  # [batch, channels]
        max_out = self.fc(self.max_pool(x).squeeze(-1))  # [batch, channels]
        
        # 组合
        channel_attention = self.sigmoid(avg_out + max_out).unsqueeze(-1)  # [batch, channels, 1]
        
        # 应用注意力
        return x * channel_attention

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, length]
        
        Returns:
            attended: [batch, channels, length]
        """
        # 平均和最大特征图
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, length]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [batch, 1, length]
        
        # 拼接
        concat = torch.cat([avg_out, max_out], dim=1)  # [batch, 2, length]
        
        # 卷积生成空间注意力
        spatial_attention = self.sigmoid(self.conv(concat))  # [batch, 1, length]
        
        # 应用注意力
        return x * spatial_attention

class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""
    
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, length]
        
        Returns:
            attended: [batch, channels, length]
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class SelfAttention(nn.Module):
    """自注意力模块（简化版）"""
    
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        Args:
            x: [batch, length, embed_dim]
        
        Returns:
            attended: [batch, length, embed_dim]
        """
        batch_size, length, embed_dim = x.shape
        
        # 计算Q, K, V
        qkv = self.qkv(x).reshape(batch_size, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, length, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, heads, length, length]
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力到V
        out = (attn @ v).transpose(1, 2).reshape(batch_size, length, embed_dim)
        
        # 最终投影
        out = self.fc(out)
        
        return out

class AttentionBackbone(nn.Module):
    """带Attention的Backbone"""
    
    def __init__(self, in_channels=1, feature_dim=256, use_cbam=True, use_self_attention=True):
        super().__init__()
        
        self.use_cbam = use_cbam
        self.use_self_attention = use_self_attention
        
        # 卷积层（与原backbone相同）
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Conv1后添加CBAM
        if use_cbam:
            self.cbam1 = CBAM(64, reduction_ratio=16, kernel_size=7)
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Conv2后添加CBAM
        if use_cbam:
            self.cbam2 = CBAM(128, reduction_ratio=16, kernel_size=7)
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Self-Attention（在特征层）
        if use_self_attention:
            self.self_attention = SelfAttention(embed_dim=256, num_heads=8)
    
    def forward(self, x):
        """
        Args:
            x: [batch, 1, length]
        
        Returns:
            features: [batch, feature_dim]
        """
        # Conv1
        x = self.conv1(x)
        
        # CBAM1
        if self.use_cbam:
            x = self.cbam1(x)
        
        # Conv2
        x = self.conv2(x)
        
        # CBAM2
        if self.use_cbam:
            x = self.cbam2(x)
        
        # Conv3
        x = self.conv3(x)  # [batch, 256, 1]
        
        # Flatten
        x = x.squeeze(-1)  # [batch, 256]
        
        # Self-Attention
        if self.use_self_attention:
            x = x.unsqueeze(1)  # [batch, 1, 256]
            x = self.self_attention(x)  # [batch, 1, 256]
            x = x.squeeze(1)  # [batch, 256]
        
        return x

if __name__ == '__main__':
    print("Attention模块测试")
    print("="*60)
    
    # 测试CBAM
    cbam = CBAM(in_channels=64)
    x = torch.randn(10, 64, 1024)
    out = cbam(x)
    print(f"CBAM: input={x.shape}, output={out.shape}")
    
    # 测试Self-Attention
    self_attn = SelfAttention(embed_dim=256, num_heads=8)
    x = torch.randn(10, 32, 256)
    out = self_attn(x)
    print(f"Self-Attention: input={x.shape}, output={out.shape}")
    
    # 测试完整AttentionBackbone
    backbone = AttentionBackbone(use_cbam=True, use_self_attention=True)
    x = torch.randn(10, 1, 2048)
    out = backbone(x)
    print(f"AttentionBackbone: input={x.shape}, output={out.shape}")
    
    print("\n✓ Attention模块测试完成")