"""
SHOT算法实现 (Source HypOthesis Transfer)

参考: Liang et al., "Do We Really Need to Access the Source Data?
       Source Hypothesis Transfer for Unsupervised Domain Adaptation" (ICML 2020)

核心思想:
1. 冻结骨干网络，只调整分类器
2. 通过信息最大化（Information Maximization）进行自适应
   - 最小化互信息（MI）：鼓励模型做出 confident 的预测
   - 最大化多样性（Diversity）：鼓励模型在不同类别上均匀分布
3. 使用伪标签进行自训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.classifier import CompleteModel
from src.models.prototype_manager import PrototypeManager


class SHOT:
    """SHOT算法实现"""

    def __init__(self, model, prototype_manager, lambda_mi=1.0, lambda_div=1.0, lr=1e-3):
        """
        Args:
            model: 源模型（包含backbone和classifier）
            prototype_manager: 原型管理器
            lambda_mi: 互信息损失权重
            lambda_div: 多样性损失权重
            lr: 学习率
        """
        self.model = model
        self.prototype_manager = prototype_manager
        self.lambda_mi = lambda_mi
        self.lambda_div = lambda_div
        self.lr = lr

        # 冻结backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # 只优化分类器
        self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=lr)

    def compute_information_loss(self, logits):
        """
        计算信息损失 = 互信息损失 + 多样性损失

        互信息损失: -H(y|x) = sum(p * log(p))
            鼓励模型做出confident的预测

        多样性损失: H(E[y]) = -sum(p_avg * log(p_avg))
            鼓励模型在不同类别上均匀分布
        """
        probs = F.softmax(logits, dim=1)

        # 互信息损失 (minimize conditional entropy)
        mi_loss = torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

        # 多样性损失 (maximize marginal entropy)
        avg_probs = probs.mean(dim=0)
        div_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        return mi_loss, div_loss

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

            # 计算信息损失
            mi_loss, div_loss = self.compute_information_loss(logits)
            loss = self.lambda_mi * mi_loss + self.lambda_div * div_loss

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
