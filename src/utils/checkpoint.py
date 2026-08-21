import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, save_path, prototypes=None):
    """
    保存训练checkpoint
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前loss
        save_path: 保存路径
        prototypes: 初始原型（可选）
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if prototypes is not None:
        checkpoint['prototypes'] = prototypes
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint保存: {save_path}")

def load_checkpoint(model, optimizer, load_path):
    """
    加载checkpoint
    
    Args:
        model: 模型
        optimizer: 优化器
        load_path: 加载路径
    
    Returns:
        epoch: 起始epoch
        prototypes: 原型（如果有）
    """
    checkpoint = torch.load(load_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    prototypes = checkpoint.get('prototypes', None)
    
    print(f"Checkpoint加载: {load_path}, epoch={epoch}")
    return epoch, prototypes

def extract_prototypes(model, dataloader, num_classes, device):
    """
    高效提取源域初始原型 (GPU批处理版)
    
    Args:
        model: 训练好的模型
        dataloader: 源域DataLoader
        num_classes: 类别数量
        device: 计算设备
    
    Returns:
        prototypes: 类原型 [num_classes, feature_dim]
    """
    model.eval()
    features_dict = {i: [] for i in range(num_classes)}
    
    print("正在提取初始原型 (批处理模式)...")
    
    with torch.no_grad():
        for data, labels, _ in dataloader:
            data = data.to(device)
            
            feats = model.get_features(data)
            
            for feat, label in zip(feats.cpu(), labels):
                features_dict[label.item()].append(feat)
    
    prototypes = []
    for class_id in range(num_classes):
        if len(features_dict[class_id]) > 0:
            class_prototype = torch.mean(torch.stack(features_dict[class_id]), dim=0)
        else:
            class_prototype = torch.zeros(256)
        prototypes.append(class_prototype)
    
    prototypes = torch.stack(prototypes).to(device)
    
    print(f"原型提取完成: Shape {prototypes.shape}")
    for i in range(num_classes):
        count = len(features_dict[i])
        print(f"  类别{i}: {count}个样本")
    
    return prototypes

if __name__ == '__main__':
    pass