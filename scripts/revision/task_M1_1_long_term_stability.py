#!/usr/bin/env python3
"""
任务 M1: 长期稳定性实验
创建时间: 2026-08-13
目标: 验证SHOT在最优学习率下的长期稳定性
方法:
  - 数据集: CWRU 3HP
  - 噪声: 0dB AWGN
  - 方法: SHOT (lr=1e-4)
  - Epochs: 100
  - 随机种子: 15个 (42-56)
  - 总运行次数: 15 runs
GPU: Yes (CUDA enabled)
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

# 导入噪声生成模块
sys.path.insert(0, str(Path(__file__).parent))
from noise_golden import generate_colored_noise

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

RESULTS_DIR = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4


def load_source_model(checkpoint_path):
    """加载源域预训练模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {}
    classifier_state = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            backbone_state[k[len('backbone.'):]] = v
        elif k.startswith('classifier.'):
            classifier_state[k[len('classifier.'):]] = v

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def load_target_data(data_path, snr_db=0, noise_type='awgn'):
    """加载目标域数据并添加噪声"""
    data_dict = torch.load(data_path, map_location=device)
    samples = data_dict['samples']
    labels = data_dict['labels']

    # 添加噪声（使用generate_colored_noise的正确接口）
    if snr_db is not None:
        samples = generate_colored_noise(samples, noise_type, snr_db)

    return samples, labels


def compute_metrics(preds, labels):
    """计算评估指标"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float(accuracy_score(labels, preds) * 100)
    macro_f1 = float(f1_score(labels, preds, average='macro') * 100)
    balanced_acc = float(balanced_accuracy_score(labels, preds) * 100)

    # 计算每个类别的recall
    class_results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())
        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        class_results[name] = {'recall': recall, 'support': true_count}

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc,
        'per_class': class_results
    }


def run_shot_long_term(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42):
    """SHOT长期适应实验（两阶段实现，与task_phase1_1保持一致）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # 阶段1：只更新backbone
    bb.train()
    for param in bb.parameters():
        param.requires_grad = True

    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

    stage1_epochs = num_epochs // 2

    # 记录每个epoch的指标
    epoch_metrics = []

    # 阶段1：最小化熵 + 最大化多样性
    for epoch in range(stage1_epochs):
        bb.train()
        clf.eval()

        epoch_loss = 0
        all_preds = []
        all_labels = []

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            loss = ent_loss + div_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 评估当前epoch
        bb.eval()
        clf.eval()
        with torch.no_grad():
            features = bb(samples.to(device))
            logits, probs = clf(features)
            preds = probs.argmax(dim=1)
            metrics = compute_metrics(preds, labels)
            metrics['epoch'] = epoch + 1
            metrics['loss'] = epoch_loss / len(loader)
            epoch_metrics.append(metrics)

        if (epoch + 1) % 20 == 0:
            print(f"  [Stage1] Epoch {epoch+1}/{stage1_epochs}: Acc={metrics['accuracy']:.2f}%, Macro-F1={metrics['macro_f1']:.2f}%")

    # 阶段2：熵 + 多样性 + 伪标签交叉熵
    for epoch in range(num_epochs - stage1_epochs):
        bb.train()
        clf.eval()

        epoch_loss = 0
        all_preds = []
        all_labels = []

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            features = bb(batch_x)
            logits, probs = clf(features)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity

            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)
            ce_loss = F.cross_entropy(logits, pseudo_labels)

            loss = ent_loss + div_loss + ce_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 评估当前epoch
        bb.eval()
        clf.eval()
        with torch.no_grad():
            features = bb(samples.to(device))
            logits, probs = clf(features)
            preds = probs.argmax(dim=1)
            metrics = compute_metrics(preds, labels)
            metrics['epoch'] = epoch + stage1_epochs + 1
            metrics['loss'] = epoch_loss / len(loader)
            epoch_metrics.append(metrics)

        if (epoch + 1) % 20 == 0:
            print(f"  [Stage2] Epoch {epoch+stage1_epochs+1}/{num_epochs}: Acc={metrics['accuracy']:.2f}%, Macro-F1={metrics['macro_f1']:.2f}%")

    return epoch_metrics


def main():
    print("=" * 60)
    print("任务 M1: 长期稳定性实验")
    print("=" * 60)

    # 实验配置
    config = {
        'dataset': 'CWRU_3HP',
        'noise_type': 'awgn',
        'snr_db': 0,
        'method': 'SHOT',
        'lr': 1e-4,
        'num_epochs': 100,
        'num_seeds': 15,
        'seed_start': 42,
        'timestamp': datetime.now().isoformat()
    }

    print(f"\n实验配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 加载源模型
    source_model_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    print(f"\n加载源模型: {source_model_path}")
    backbone, classifier = load_source_model(source_model_path)

    # 加载目标数据（所有种子使用相同的数据）
    target_data_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'
    print(f"加载目标数据: {target_data_path}")
    samples, labels = load_target_data(target_data_path, snr_db=0, noise_type='awgn')
    print(f"数据形状: {samples.shape}, 标签形状: {labels.shape}")

    # 运行15个种子
    all_results = []

    for seed in range(config['seed_start'], config['seed_start'] + config['num_seeds']):
        print(f"\n{'='*60}")
        print(f"运行种子 {seed} ({seed - config['seed_start'] + 1}/{config['num_seeds']})")
        print(f"{'='*60}")

        epoch_metrics = run_shot_long_term(
            backbone, classifier, samples, labels,
            num_epochs=config['num_epochs'],
            lr=config['lr'],
            seed=seed
        )

        # 获取最终性能
        final_metrics = epoch_metrics[-1]

        result = {
            'seed': seed,
            'final_accuracy': final_metrics['accuracy'],
            'final_macro_f1': final_metrics['macro_f1'],
            'final_balanced_accuracy': final_metrics['balanced_accuracy'],
            'final_per_class': final_metrics['per_class'],
            'epoch_metrics': epoch_metrics
        }

        all_results.append(result)

        print(f"\n最终性能:")
        print(f"  Accuracy: {final_metrics['accuracy']:.2f}%")
        print(f"  Macro-F1: {final_metrics['macro_f1']:.2f}%")
        print(f"  Balanced Acc: {final_metrics['balanced_accuracy']:.2f}%")

    # 计算统计信息
    final_accuracies = [r['final_accuracy'] for r in all_results]
    final_macro_f1s = [r['final_macro_f1'] for r in all_results]
    final_balanced_accs = [r['final_balanced_accuracy'] for r in all_results]

    stats = {
        'accuracy': {
            'mean': np.mean(final_accuracies),
            'std': np.std(final_accuracies),
            'min': np.min(final_accuracies),
            'max': np.max(final_accuracies)
        },
        'macro_f1': {
            'mean': np.mean(final_macro_f1s),
            'std': np.std(final_macro_f1s),
            'min': np.min(final_macro_f1s),
            'max': np.max(final_macro_f1s)
        },
        'balanced_accuracy': {
            'mean': np.mean(final_balanced_accs),
            'std': np.std(final_balanced_accs),
            'min': np.min(final_balanced_accs),
            'max': np.max(final_balanced_accs)
        }
    }

    # 计算每个epoch的平均性能（跨种子）
    avg_epoch_metrics = []
    for epoch_idx in range(config['num_epochs']):
        epoch_accs = [r['epoch_metrics'][epoch_idx]['accuracy'] for r in all_results]
        epoch_f1s = [r['epoch_metrics'][epoch_idx]['macro_f1'] for r in all_results]
        avg_epoch_metrics.append({
            'epoch': epoch_idx + 1,
            'accuracy_mean': np.mean(epoch_accs),
            'accuracy_std': np.std(epoch_accs),
            'macro_f1_mean': np.mean(epoch_f1s),
            'macro_f1_std': np.std(epoch_f1s)
        })

    # 保存结果
    output = {
        'config': config,
        'stats': stats,
        'avg_epoch_metrics': avg_epoch_metrics,
        'individual_results': all_results
    }

    output_path = RESULTS_DIR / 'task_M1_1_long_term_stability.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("实验完成!")
    print(f"{'='*60}")
    print(f"\n统计结果:")
    print(f"  Accuracy: {stats['accuracy']['mean']:.2f}% ± {stats['accuracy']['std']:.2f}%")
    print(f"  Macro-F1: {stats['macro_f1']['mean']:.2f}% ± {stats['macro_f1']['std']:.2f}%")
    print(f"  Balanced Acc: {stats['balanced_accuracy']['mean']:.2f}% ± {stats['balanced_accuracy']['std']:.2f}%")
    print(f"\n结果保存至: {output_path}")


if __name__ == '__main__':
    main()
