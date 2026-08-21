"""
源域预训练脚本（Phase 2）

流程:
1. 加载源域数据
2. 训练模型100 epochs
3. 提取并保存初始原型
4. 保存模型权重

输出:
- experiments/checkpoints/source_pretrain.pt
- experiments/checkpoints/init_prototypes.pt
"""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.classifier import CompleteModel
from data.loader import BearingFaultDataset
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, extract_prototypes
import os

class SourcePreTrainer:
    """源域预训练器"""
    
    def __init__(self, config):
        self.config = config
        
        set_seed(42)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.source_dataset = BearingFaultDataset(
            config['source_data_path'],
            domain_flag=0
        )
        
        self.num_classes = self.source_dataset.get_num_classes()
        
        self.model = CompleteModel(
            feature_dim=config['feature_dim'],
            num_classes=self.num_classes
        ).to(self.device)
        
        self.loader = DataLoader(
            self.source_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['lr']
        )
        
        print(f"\n源域预训练初始化:")
        print(f"  设备: {self.device}")
        print(f"  源域样本: {len(self.source_dataset)}")
        print(f"  类别数: {self.num_classes}")
        print(f"  Batch size: {config['batch_size']}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels, domain_flags) in enumerate(self.loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            logits, probs = self.model(data)
            
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}/{len(self.loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%")
        
        avg_loss = total_loss / len(self.loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs=100):
        """完整训练流程"""
        print(f"\n开始源域预训练 ({num_epochs} epochs):")
        
        best_acc = 0
        
        for epoch in range(1, num_epochs + 1):
            loss, acc = self.train_epoch(epoch)
            
            print(f"Epoch {epoch}/{num_epochs}: Loss={loss:.4f}, Acc={acc:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    loss,
                    'experiments/checkpoints/source_pretrain_best.pt'
                )
        
        print(f"\n训练完成! 最佳准确率: {best_acc:.2f}%")
        
        save_checkpoint(
            self.model,
            self.optimizer,
            num_epochs,
            loss,
            'experiments/checkpoints/source_pretrain.pt'
        )
    
    def extract_and_save_prototypes(self):
        """提取初始源域原型"""
        print("\n提取初始源域原型...")
        
        prototypes = extract_prototypes(
            self.model, self.loader, self.num_classes, self.device
        )
        
        torch.save(prototypes, 'experiments/checkpoints/init_prototypes.pt')
        
        print(f"初始原型已保存: experiments/checkpoints/init_prototypes.pt")
        print(f"原型shape: {prototypes.shape}")
        
        return prototypes

if __name__ == '__main__':
    config = {
        'source_data_path': 'data/processed/cwru_0hp.pt',
        'feature_dim': 256,
        'batch_size': 64,
        'lr': 1e-3,
    }
    
    trainer = SourcePreTrainer(config)
    
    trainer.train(num_epochs=100)
    
    trainer.extract_and_save_prototypes()
    
    print("\nPhase 2完成!")