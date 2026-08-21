"""
XJTU-SY轴承故障诊断数据集

数据格式：
- data.npy: (N, 2048) - N个样本，每个样本2048个时间点
- labels.npy: (N,) - 对应的故障标签
- 4个类别：0=正常, 1=内圈故障, 2=外圈故障, 3=滚动体故障

工况条件：
- 35Hz12kN: 转速35Hz，载荷12kN
- 37.5Hz11kN: 转速37.5Hz，载荷11kN
- 40Hz10kN: 转速40Hz，载荷10kN
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class XJTUSYDataset(Dataset):
    """XJTU-SY轴承故障诊断数据集"""

    def __init__(self, condition="35Hz12kN", transform=None):
        """
        初始化XJTU-SY数据集

        Args:
            condition: 工况条件 ("35Hz12kN", "37.5Hz11kN", "40Hz10kN")
            transform: 数据增强变换
        """
        self.condition = condition
        self.transform = transform

        # 加载数据
        data_path = f"data/processed/XJTU-SY/{condition}/data.npy"
        labels_path = f"data/processed/XJTU-SY/{condition}/labels.npy"

        self.data = np.load(data_path)  # (N, 2048)
        self.labels = np.load(labels_path)  # (N,)

        # 添加通道维度 (N, 1, 2048)
        self.data = self.data[:, np.newaxis, :]

        print(f"Loaded XJTU-SY dataset: {condition}")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Unique labels: {np.unique(self.labels)}")

        # 统计每个类别的样本数
        for label in np.unique(self.labels):
            count = np.sum(self.labels == label)
            print(f"  Class {label}: {count} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本

        Returns:
            data: (1, 2048) - 振动信号
            label: int - 故障标签
            domain_flag: int - 域标识（固定为0）
        """
        data = self.data[idx]
        label = self.labels[idx]

        # 转换为Tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)

        # 应用变换
        if self.transform is not None:
            data = self.transform(data)

        # 域标识（XJTU-SY作为源域或目标域）
        domain_flag = 0

        return data, label, domain_flag


class XJTUSYDatasetPair(Dataset):
    """
    XJTU-SY数据集对（用于跨工况迁移）

    源域：35Hz12kN
    目标域：37.5Hz11kN 或 40Hz10kN
    """

    def __init__(self, source_condition="35Hz12kN", target_condition="37.5Hz11kN"):
        """
        初始化数据集对

        Args:
            source_condition: 源域工况条件
            target_condition: 目标域工况条件
        """
        self.source_condition = source_condition
        self.target_condition = target_condition

        # 加载源域数据
        source_data_path = f"data/processed/XJTU-SY/{source_condition}/data.npy"
        source_labels_path = f"data/processed/XJTU-SY/{source_condition}/labels.npy"

        self.source_data = np.load(source_data_path)[:, np.newaxis, :]
        self.source_labels = np.load(source_labels_path)

        # 加载目标域数据
        target_data_path = f"data/processed/XJTU-SY/{target_condition}/data.npy"
        target_labels_path = f"data/processed/XJTU-SY/{target_condition}/labels.npy"

        self.target_data = np.load(target_data_path)[:, np.newaxis, :]
        self.target_labels = np.load(target_labels_path)

        print(f"\nXJTU-SY Dataset Pair:")
        print(f"  Source: {source_condition} ({len(self.source_data)} samples)")
        print(f"  Target: {target_condition} ({len(self.target_data)} samples)")

    def get_source_dataset(self):
        """获取源域数据集"""
        return XJTUSYDataset(self.source_condition)

    def get_target_dataset(self):
        """获取目标域数据集"""
        return XJTUSYDataset(self.target_condition)


if __name__ == "__main__":
    # 测试数据集加载
    print("=" * 60)
    print("Testing XJTU-SY Dataset")
    print("=" * 60)

    # 测试单个数据集
    dataset = XJTUSYDataset(condition="35Hz12kN")

    # 测试数据加载
    data, label, domain_flag = dataset[0]
    print(f"\nSample 0:")
    print(f"  Data shape: {data.shape}")
    print(f"  Label: {label}")
    print(f"  Domain flag: {domain_flag}")

    # 测试数据集对
    print("\n" + "=" * 60)
    print("Testing XJTU-SY Dataset Pair")
    print("=" * 60)

    dataset_pair = XJTUSYDatasetPair(
        source_condition="35Hz12kN",
        target_condition="37.5Hz11kN"
    )

    source_dataset = dataset_pair.get_source_dataset()
    target_dataset = dataset_pair.get_target_dataset()

    print(f"\nSource dataset size: {len(source_dataset)}")
    print(f"Target dataset size: {len(target_dataset)}")
