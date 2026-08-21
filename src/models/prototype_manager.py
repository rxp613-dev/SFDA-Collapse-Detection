"""
创新点B: 动态原型演化

核心思路:
1. 初始化: 加载Phase 2的初始源域原型
2. 每个Batch/EPOCH: 利用可靠样本计算目标域类均值T_mean
3. 动量更新: Current_Prototypes = m*P + (1-m)*T, m=0.99
4. 演化后的原型用于计算对比损失和排斥损失

关键参数: 动量系数m=0.99（防止原型跑偏）
"""

import torch

class PrototypeManager:
    """
    原型动态演化管理器
    
    核心功能:
    1. 加载初始源域原型
    2. 根据可靠样本动态更新原型
    3. 记录演化轨迹（用于可视化）
    """
    
    def __init__(self, num_classes, feature_dim=256, momentum=0.99, initial_prototypes=None, verbose=True):
        """
        Args:
            num_classes: 类别数量
            feature_dim: 特征维度（默认256）
            momentum: 动量系数，必须>=0.99
            initial_prototypes: Optional外部原型 [num_classes, feature_dim]
            verbose: 是否打印详细信息
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.verbose = verbose
        
        self.current_prototypes = None
        self.initial_prototypes = None
        self.trajectory_history = []
        
        if momentum < 0.99 and self.verbose:
            print(f"警告: momentum={momentum} < 0.99, 原型可能剧烈变化")
        
        if initial_prototypes is not None:
            self.load_initial_prototypes(initial_prototypes)
        
        if self.verbose:
            print(f"原型管理器初始化:")
            print(f"  类别数: {num_classes}")
            print(f"  动量系数: {momentum}")
            if initial_prototypes is not None:
                print(f"  外部原型: 已加载")
    
    def load_initial_prototypes(self, init_prototypes):
        """
        加载初始源域原型
        
        Args:
            init_prototypes: Phase 2提取的原型 [num_classes, 256]
        """
        self.initial_prototypes = init_prototypes.clone()
        self.current_prototypes = init_prototypes.clone()
        
        print(f"初始原型加载: shape={init_prototypes.shape}")
        
        self.trajectory_history.append({
            'epoch': 0,
            'prototypes': self.current_prototypes.clone()
        })
    
    def update_prototypes(self, reliable_features, reliable_pseudo_labels, verbose=None):
        """
        动态更新原型
        
        Args:
            reliable_features: 可靠样本特征 [N, 256]
            reliable_pseudo_labels: 可靠样本伪标签 [N]
            verbose: 是否打印详细信息（None则使用self.verbose）
        """
        if verbose is None:
            verbose = self.verbose
            
        if len(reliable_features) == 0:
            if verbose:
                print("无可靠样本，原型不更新")
            return
        
        target_means = []
        
        for class_id in range(self.num_classes):
            class_mask = reliable_pseudo_labels == class_id
            class_features = reliable_features[class_mask]
            
            if len(class_features) > 0:
                target_mean = class_features.mean(dim=0)
            else:
                target_mean = self.current_prototypes[class_id]
            
            target_means.append(target_mean)
        
        target_means = torch.stack(target_means, dim=0)
        
        self.current_prototypes = (
            self.momentum * self.current_prototypes +
            (1 - self.momentum) * target_means
        )
        
        distance_from_initial = torch.norm(
            self.current_prototypes - self.initial_prototypes,
            dim=1
        ).mean()
        
        if self.verbose:
            print(f"原型更新:")
            print(f"  可靠样本: {len(reliable_features)}")
            print(f"  平均距离初始原型: {distance_from_initial:.4f}")
    
    def get_current_prototypes(self):
        """返回当前原型"""
        return self.current_prototypes
    
    def record_trajectory(self, epoch):
        """记录原型演化轨迹"""
        self.trajectory_history.append({
            'epoch': epoch,
            'prototypes': self.current_prototypes.clone()
        })
    
    def get_trajectory_history(self):
        """返回完整的演化轨迹"""
        return self.trajectory_history

    def compute_prototypes(self, model, dataloader):
        """
        从源域数据计算初始原型

        Args:
            model: 源域模型
            dataloader: 源域数据加载器
        """
        device = next(model.parameters()).device
        model.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch_data, batch_labels, _ in dataloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                features = model.backbone(batch_data)
                all_features.append(features)
                all_labels.append(batch_labels)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算每个类别的原型
        prototypes = []
        for class_id in range(self.num_classes):
            class_mask = all_labels == class_id
            class_features = all_features[class_mask]

            if len(class_features) > 0:
                class_prototype = class_features.mean(dim=0)
            else:
                # 如果某个类别没有样本，使用零向量
                class_prototype = torch.zeros(self.feature_dim, device=device)

            prototypes.append(class_prototype)

        prototypes = torch.stack(prototypes)

        # 加载计算得到的原型
        self.load_initial_prototypes(prototypes)

        return prototypes

    def save_trajectory_to_json(self, filepath, strategy='auto'):
        """保存轨迹历史到JSON文件用于可视化"""
        import json
        
        if strategy == 'auto':
            strategy = 'fixed' if self.momentum > 0.998 else 'momentum'
        
        trajectory_data = {
            'strategy': strategy,
            'trajectory': []
        }
        
        for record in self.trajectory_history:
            trajectory_data['trajectory'].append({
                'epoch': record['epoch'],
                'prototypes': record['prototypes'].cpu().numpy().tolist() if record['prototypes'].device.type == 'cuda' else record['prototypes'].numpy().tolist(),
                'accuracy': record.get('accuracy', None),
                'prototype_distance': record.get('prototype_distance', None)
            })
        
        with open(filepath, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"✓ Trajectory saved to {filepath} ({len(trajectory_data['trajectory'])} epochs)")

if __name__ == '__main__':
    manager = PrototypeManager(num_classes=4, momentum=0.99)
    
    init_prototypes = torch.randn(4, 256)
    manager.load_initial_prototypes(init_prototypes)
    
    reliable_features = torch.randn(50, 256)
    reliable_pseudo_labels = torch.randint(0, 4, (50,))
    
    manager.update_prototypes(reliable_features, reliable_pseudo_labels)
    
    print(f"\n原型管理器测试完成")