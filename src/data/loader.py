import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class BearingFaultDataset(Dataset):
    """
    轴承故障诊断数据集

    支持源域和目标域数据加载，通过domain_flag区分

    Args:
        data_path: .pt文件路径
        domain_flag: 0表示源域，1表示目标域
        transform: 数据增强（可选）
    """

    def __init__(self, data_path, domain_flag=0, transform=None):
        data_dict = torch.load(data_path)

        self.samples = data_dict['samples']
        self.labels = data_dict['labels']
        self.domain_flag = domain_flag
        self.transform = transform

        print(f"加载数据集: {data_path}")
        print(f"样本数: {len(self.samples)}, domain_flag: {domain_flag}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回单个样本

        Returns:
            data: 振动信号 [1, 1024]
            label: 故障类别标签
            domain_flag: 域标识（0=源域，1=目标域）
            idx: 样本唯一索引（用于SHOT Stage 2伪标签对齐）
        """
        sample = self.samples[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label, self.domain_flag, idx

    def get_num_classes(self):
        """返回类别数量"""
        return len(torch.unique(self.labels))

    def get_class_distribution(self):
        """返回类别分布"""
        return torch.bincount(self.labels)


def load_cwru(hp, data_dir=None):
    """
    加载CWRU数据集

    Args:
        hp: 马力数，如 '0hp', '1hp', '2hp', '3hp'
        data_dir: 数据目录路径，默认为项目根目录下的 data/processed

    Returns:
        BearingFaultDataset: 数据集对象
    """
    if data_dir is None:
        # 默认路径
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / 'data' / 'processed'

    data_path = Path(data_dir) / f'cwru_{hp}.pt'

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    return BearingFaultDataset(data_path, domain_flag=0)


if __name__ == '__main__':
    pass
