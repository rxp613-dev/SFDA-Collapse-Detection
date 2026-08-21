"""
Tent算法实现 (Test-time Entropy Minimization)

参考: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization" (ICLR 2021)

核心思想:
1. 冻结骨干网络，只调整Batch Normalization层
2. 通过最小化预测熵进行自适应
3. 使用梯度下降优化BN参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.classifier import CompleteModel
from src.models.prototype_manager import PrototypeManager


class Tent:
    """Tent算法实现"""

    def __init__(self, model, prototype_manager, lr=1e-3):
        """
        Args:
            model: 源模型（包含backbone和classifier）
            prototype_manager: 原型管理器
            lr: 学习率
        """
        self.model = model
        self.prototype_manager = prototype_manager
        self.lr = lr

        # 冻结backbone的特征提取层
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # 只优化BN层参数
        bn_params = []
        for module in self.model.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                bn_params.extend(list(module.parameters()))

        self.optimizer = torch.optim.Adam(bn_params, lr=lr)

    def compute_entropy_loss(self, logits):
        """
        计算预测熵损失

        H(p) = -sum(p * log(p))
        其中 p = softmax(logits)
        """
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.mean()

    def adapt_epoch(self, dataloader, epoch):
        """
        执行一个epoch的自适应

        Args:
            dataloader: 目标域数据加载器
            epoch: 当前epoch

        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, labels, _) in enumerate(dataloader):
            data = data.cuda()
            labels = labels.cuda()

            # 前向传播
            features = self.model.backbone(data)
            logits, _ = self.model.classifier(features)

            # 计算熵损失
            loss = self.compute_entropy_loss(logits)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def predict(self, dataloader):
        """
        预测目标域标签

        Args:
            dataloader: 数据加载器

        Returns:
            predictions: 预测标签
            true_labels: 真实标签
        """
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for data, labels, _ in dataloader:
                data = data.cuda()
                labels = labels.cuda()

                features = self.model.backbone(data)
                logits, _ = self.model.classifier(features)
                preds = logits.argmax(dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        return predictions, true_labels
