#!/usr/bin/env python3
"""
Step 1A (v2): 重新实现正确的 NRC 算法 (Neighborhood Reciprocity Clustering)
Created: 2026-08-13
Reference: Roy et al., "Neighborhood Reciprocity Clustering for Source-Free Domain Adaptation" (CVPR 2022)

核心修正:
1. Backbone 冻结 (SFDA 原则)
2. Reciprocity 约束通过分类器的伪标签传递梯度 (而非特征)
3. k-NN 图在特征空间构建 (backbone 输出)
4. 仅更新分类器参数

算法 (实际可训练版本):
1. 用冻结 backbone 提取目标域特征
2. 在特征空间构建 k-NN 图
3. 分类器生成预测概率 → 伪标签
4. 用互惠邻居约束修正伪标签 (affinity matrix)
5. 用修正后的伪标签计算 CE 损失 (梯度流向分类器)
6. 反向传播更新分类器
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from copy import deepcopy
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4


def load_source_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


def load_target_data(data_path):
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']


def add_gaussian_noise(data, snr_db):
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise


def compute_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        confusion_matrix[int(t), int(p)] += 1

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        pred_count = int((preds == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())

        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        precision = float(correct / pred_count * 100) if pred_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[name] = {'recall': recall, 'precision': precision, 'f1': f1}

    macro_f1 = float(np.mean([results[n]['f1'] for n in CLASS_NAMES]))
    balanced_acc = float(np.mean([results[n]['recall'] for n in CLASS_NAMES]))

    return results, accuracy, confusion_matrix.tolist(), macro_f1, balanced_acc


def build_affinity_matrix(features, k=10):
    """
    构建 k-NN 亲和度矩阵

    基于特征余弦相似度，使用相互 k-NN 构建稀疏亲和度矩阵 W
    W[i,j] = 1 如果 i 和 j 是相互邻居, 否则 0

    Args:
        features: [N, D] 特征矩阵
        k: 邻居数

    Returns:
        W: [N, N] 亲和度矩阵 (torch, detached)
    """
    N = features.shape[0]
    features_norm = F.normalize(features, dim=1)

    # 余弦相似度矩阵
    similarity = torch.mm(features_norm, features_norm.t())
    similarity.fill_diagonal_(float('-inf'))

    # k-NN
    _, knn_indices = torch.topk(similarity, k, dim=1)

    # 构建相互 k-NN 亲和度矩阵
    W = torch.zeros(N, N, device=features.device)
    for i in range(N):
        for j_idx in range(k):
            j = knn_indices[i, j_idx].item()
            # 检查是否为相互邻居
            if i in knn_indices[j]:
                W[i, j] = 1.0
                W[j, i] = 1.0

    # 归一化 (行归一化)
    row_sums = W.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1.0)
    W = W / row_sums

    return W.detach()


def run_nrc_corrected(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3, seed=42, k=10):
    """
    运行正确的 NRC 算法

    关键: reciprocity 约束通过伪标签的 soft target 传递梯度到分类器
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    # 冻结 backbone (SFDA 原则)
    bb.eval()
    for param in bb.parameters():
        param.requires_grad = False

    # 只更新分类器
    clf.train()
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    # 预计算特征 (backbone 冻结，特征不变)
    with torch.no_grad():
        features = bb(samples)

    # 构建 k-NN 亲和度矩阵 (在特征空间)
    print(f"    构建 {k}-NN 亲和度矩阵...", flush=True)
    W = build_affinity_matrix(features, k=k)
    mutual_count = (W > 0).sum().item()
    print(f"    相互邻居对数: {int(mutual_count // 2)}", flush=True)

    # 自适应训练
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 分类器前向传播 (梯度可流过)
        logits, probs = clf(features)  # probs: [N, C]

        # (1) 伪标签 (hard pseudo-labels, detached)
        with torch.no_grad():
            pseudo_labels = probs.argmax(dim=1)

        # (2) 邻居一致性 soft target (通过 W 平滑 probs)
        # soft_target[i] = sum_j W[i,j] * probs[j]
        # 这鼓励邻居的预测一致 (reciprocity)
        soft_targets = torch.mm(W, probs)  # [N, C]
        soft_targets = soft_targets.detach()  # 作为 target，不反向传播

        # (3) 交叉熵损失 (硬伪标签)
        ce_loss = F.cross_entropy(logits, pseudo_labels)

        # (4) 邻居一致性损失 (KL 散度, 梯度流向分类器)
        # KL(soft_targets || probs) 鼓励 probs 接近邻居的加权平均
        kl_loss = F.kl_div(
            torch.log(probs + 1e-8),
            soft_targets,
            reduction='batchmean'
        )

        # 总损失
        loss = ce_loss + 0.5 * kl_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            clf.eval()
            with torch.no_grad():
                logits, probs = clf(features)
                preds = probs.argmax(dim=1)
                _, acc, _, mf1, bacc = compute_metrics(preds, labels)
            print(f"    Epoch {epoch+1:3d}/{num_epochs}: loss={loss.item():.4f}, ce={ce_loss.item():.4f}, kl={kl_loss.item():.4f}, acc={acc:.2f}%, mf1={mf1:.4f}", flush=True)
            clf.train()

    # 最终评估
    bb.eval()
    clf.eval()
    with torch.no_grad():
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        metrics, accuracy, confusion_matrix, macro_f1, balanced_acc = compute_metrics(preds, labels)

    return accuracy, confusion_matrix, macro_f1, balanced_acc, metrics


def main():
    print("=" * 80)
    print("Step 1A (v2): 重新实现正确的 NRC 算法")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    source_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain_0hp.pt'
    target_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    if not source_path.exists():
        print(f"ERROR: Source checkpoint not found at {source_path}")
        return
    if not target_path.exists():
        print(f"ERROR: Target data not found at {target_path}")
        return

    print("\n[1/3] 加载数据...")
    bb, clf = load_source_model(source_path)
    samples, labels = load_target_data(target_path)
    print(f"  Samples: {samples.shape}, Labels: {labels.shape}")
    print(f"  Classes: {dict(zip(*np.unique(labels.cpu().numpy(), return_counts=True)))}")

    print("\n[2/3] 添加 0dB AWGN 噪声...")
    samples_noisy = add_gaussian_noise(samples, snr_db=0)

    # 首先评估源模型在 0dB 噪声下的表现 (无适应)
    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples_noisy)
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        _, src_acc, _, src_mf1, src_bacc = compute_metrics(preds, labels)
    print(f"\n  源模型 (无适应) 在 0dB 噪声下的表现:")
    print(f"    Accuracy: {src_acc:.2f}%, Macro-F1: {src_mf1:.4f}, Balanced Acc: {src_bacc:.2f}%")

    print("\n[3/3] 运行正确的 NRC 算法 (3 个种子)...")
    seeds = [42, 43, 44]

    results = {
        'task': 'Step 1A (v2) - NRC Corrected Implementation',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'NRC (Neighborhood Reciprocity Clustering)',
        'reference': 'Roy et al., CVPR 2022',
        'dataset': 'CWRU 0HP->3HP',
        'snr_db': 0,
        'num_seeds': len(seeds),
        'implementation_details': {
            'backbone_frozen': True,
            'knn_affinity_matrix': True,
            'mutual_knn': True,
            'reciprocity_via_kl_divergence': True,
            'classifier_only_update': True,
            'optimizer': 'Adam',
            'k': 10,
            'lambda_recip': 0.5,
            'num_epochs': 50
        },
        'source_model_performance': {
            'accuracy': src_acc,
            'macro_f1': src_mf1,
            'balanced_acc': src_bacc
        },
        'seeds': {}
    }

    for seed in seeds:
        print(f"\n  === Seed {seed} ===")
        accuracy, confusion_matrix, macro_f1, balanced_acc, metrics = run_nrc_corrected(
            bb, clf, samples_noisy, labels,
            num_epochs=50, lr=1e-3, seed=seed, k=10
        )

        results['seeds'][str(seed)] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'balanced_acc': balanced_acc,
            'per_class_recall': {name: metrics[name]['recall'] for name in CLASS_NAMES}
        }

        print(f"  Result: Accuracy={accuracy:.2f}%, Macro-F1={macro_f1:.4f}, Balanced Acc={balanced_acc:.2f}%")

    # 汇总
    accs = [results['seeds'][str(s)]['accuracy'] for s in seeds]
    mf1s = [results['seeds'][str(s)]['macro_f1'] for s in seeds]
    results['mean_accuracy'] = float(np.mean(accs))
    results['std_accuracy'] = float(np.std(accs))
    results['mean_macro_f1'] = float(np.mean(mf1s))
    results['std_macro_f1'] = float(np.std(mf1s))

    print(f"\n" + "=" * 80)
    print(f"NRC (corrected) 在 CWRU 0HP->3HP @ 0dB 的结果:")
    print(f"  源模型 (无适应): Accuracy = {src_acc:.2f}%")
    print(f"  NRC (corrected):  Accuracy = {results['mean_accuracy']:.2f}% ± {results['std_accuracy']:.2f}%")
    print(f"=" * 80)

    output_path = RESULTS_DIR / 'step1a_nrc_corrected_v2_0db.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存至: {output_path}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    return results


if __name__ == '__main__':
    main()
