"""
目标域适应脚本（Phase 3 - 核心创新点）

流程:
1. 加载源域预训练模型和初始原型
2. 加载目标域无标签数据
3. 使用三大创新点进行域适应训练:
   - A: 流形去噪
   - B: 动态原型演化
   - C: 边界排斥
4. 保存适应后的模型

输出:
- experiments/checkpoints/target_adapt.pt
"""

import sys
sys.path.append('src')

import torch
from torch.utils.data import DataLoader
from models.classifier import CompleteModel
from models.prototype_manager import PrototypeManager
from adaptation.manifold_filter import ManifoldFilter
from adaptation.losses import CombinedSFDALoss
from data.loader import BearingFaultDataset
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
import os

class TargetAdaptor:
    """目标域适应器"""
    
    def __init__(self, config):
        self.config = config
        
        set_seed(42)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = CompleteModel(
            feature_dim=config['feature_dim'],
            num_classes=config['num_classes']
        ).to(self.device)
        
        checkpoint = torch.load(config['source_checkpoint'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.freeze_backbone_partial()
        
        init_prototypes = torch.load(config['init_prototypes_path'])
        
        self.prototype_manager = PrototypeManager(
            num_classes=config['num_classes'],
            feature_dim=config['feature_dim'],
            momentum=config['momentum']
        )
        self.prototype_manager.load_initial_prototypes(init_prototypes)
        
        self.target_dataset = BearingFaultDataset(
            config['target_data_path'],
            domain_flag=1
        )
        
        self.loader = DataLoader(
            self.target_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        self.manifold_filter = ManifoldFilter(
            epsilon=config['epsilon'],
            confidence_threshold=config['confidence_threshold']
        )
        
        self.combined_loss = CombinedSFDALoss(
            cls_weight=config['cls_weight'],
            con_weight=config['con_weight'],
            repel_weight=config['repel_weight']
        )
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['lr']
        )
        
        print(f"\n目标域适应初始化:")
        print(f"  设备: {self.device}")
        print(f"  目标域样本: {len(self.target_dataset)}")
        print(f"  类别数: {config['num_classes']}")
        print(f"  epsilon: {config['epsilon']}")
        print(f"  momentum: {config['momentum']}")
    
    def adapt_epoch(self, epoch):
        """目标域适应一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, _, _) in enumerate(self.loader):
            data = data.to(self.device)
            
            logits, probs, features = self.model(data, return_features=True)
            
            current_prototypes = self.prototype_manager.get_current_prototypes()
            
            reliable_indices, unreliable_indices, pseudo_labels = \
                self.manifold_filter.filter_samples(features, probs, current_prototypes)
            
            if len(reliable_indices) > 0:
                reliable_features = features[reliable_indices]
                reliable_logits = logits[reliable_indices]
                reliable_pseudo_labels = pseudo_labels[reliable_indices]
                
                self.prototype_manager.update_prototypes(
                    reliable_features.detach(),
                    reliable_pseudo_labels
                )
            else:
                reliable_features = torch.tensor([]).to(self.device)
                reliable_logits = torch.tensor([]).to(self.device)
                reliable_pseudo_labels = torch.tensor([]).to(self.device)
            
            if len(unreliable_indices) > 0:
                unreliable_features = features[unreliable_indices]
                unreliable_probs = probs[unreliable_indices]
            else:
                unreliable_features = torch.tensor([]).to(self.device)
                unreliable_probs = torch.tensor([]).to(self.device)
            
            loss, loss_dict = self.combined_loss(
                reliable_logits, reliable_pseudo_labels, reliable_features,
                unreliable_probs, unreliable_features,
                current_prototypes
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if len(reliable_indices) > 0:
                pred = reliable_logits.argmax(dim=1)
                total_correct += (pred == reliable_pseudo_labels).sum().item()
                total_samples += len(reliable_indices)
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}/{len(self.loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Reliable: {len(reliable_indices)}, "
                      f"Pseudo-Acc: {100*total_correct/max(total_samples,1):.2f}%")
        
        avg_loss = total_loss / len(self.loader)
        pseudo_acc = 100 * total_correct / max(total_samples, 1)
        
        # Calculate prototype distance from initial
        prototype_distance = torch.norm(
            self.prototype_manager.current_prototypes - self.prototype_manager.initial_prototypes
        ).item() if self.prototype_manager.initial_prototypes is not None else 0.0
        
        # Record trajectory with accuracy and distance
        self.prototype_manager.record_trajectory(epoch)
        # Update last record with accuracy/distance
        if len(self.prototype_manager.trajectory_history) > 0:
            self.prototype_manager.trajectory_history[-1]['accuracy'] = pseudo_acc
            self.prototype_manager.trajectory_history[-1]['prototype_distance'] = prototype_distance
        
        return avg_loss, pseudo_acc
    
    def adapt(self, num_epochs=100):
        """完整目标域适应流程"""
        print(f"\n开始目标域适应 ({num_epochs} epochs):")
        
        best_acc = 0
        
        for epoch in range(1, num_epochs + 1):
            loss, acc = self.adapt_epoch(epoch)
            
            print(f"Epoch {epoch}/{num_epochs}: Loss={loss:.4f}, Pseudo-Acc={acc:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    loss,
                    f'experiments/checkpoints/{self.config["task_name"]}_adapt_best.pt',
                    prototypes=self.prototype_manager.get_current_prototypes()
                )
        
        print(f"\n目标域适应完成! 最佳伪标签准确率: {best_acc:.2f}%")
        
        save_checkpoint(
            self.model,
            self.optimizer,
            num_epochs,
            loss,
            f'experiments/checkpoints/{self.config["task_name"]}_adapt.pt',
            prototypes=self.prototype_manager.get_current_prototypes()
        )
    
    def evaluate_on_target(self, target_data_path):
        """在目标域上评估真实准确率"""
        print(f"\n评估目标域真实准确率...")
        
        test_dataset = BearingFaultDataset(target_data_path, domain_flag=1)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels, _ in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                logits, probs = self.model(data)
                pred = logits.argmax(dim=1)
                
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * correct / total
        print(f"目标域真实准确率: {accuracy:.2f}%")
        
        return accuracy

if __name__ == '__main__':
    config = {
        'task_name': 'task3_cwru2pu',
        'source_checkpoint': 'experiments/checkpoints/source_pretrain.pt',
        'init_prototypes_path': 'experiments/checkpoints/init_prototypes.pt',
        'target_data_path': 'data/processed/pu_k001_k004.pt',
        'num_classes': 4,
        'feature_dim': 256,
        'batch_size': 64,
        'lr': 1e-4,
        'epsilon': 0.15,
        'confidence_threshold': 0.8,
        'momentum': 0.99,
        'cls_weight': 1.0,
        'con_weight': 0.1,
        'repel_weight': 0.05,
    }
    
    adaptor = TargetAdaptor(config)
    
    adaptor.adapt(num_epochs=100)
    
    adaptor.evaluate_on_target('data/processed/pu_k001_k004.pt')
    
    print("\nPhase 3完成!")