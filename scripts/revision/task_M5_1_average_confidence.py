#!/usr/bin/env python3
"""
任务 M5.1: 实现Average Confidence监控信号
创建时间: 2026-08-10
目标: 实现Average Confidence作为崩溃检测的基线信号
方法:
    1. 对每个SNR水平运行SHOT适应
    2. 在适应后的模型上计算Average Confidence
    3. 计算与accuracy的相关性
    4. 计算AUC用于崩溃检测
    5. 保存结果到JSON
    6. 记录到LOG_2026-08-06.md
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
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

RESULTS_DIR = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision'

def load_source_model(checkpoint_path):
    """加载源域模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=4).to(device)

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

def add_gaussian_noise(samples, snr_db):
    """添加高斯噪声"""
    if snr_db == float('inf'):
        return samples

    signal_power = torch.mean(samples ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(samples) * torch.sqrt(noise_power)
    return samples + noise

def run_shot_adaptation(backbone, classifier, samples, labels, num_epochs=50, lr=1e-3):
    """运行SHOT适应"""
    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.eval()

    optimizer = torch.optim.Adam(bb.parameters(), lr=lr)

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, _ = clf(features)

            # 熵最小化
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

            # 多样性损失
            mean_probs = probs.mean(dim=0)
            diversity = torch.sum(mean_probs * torch.log(mean_probs + 1e-8))

            loss = entropy + diversity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return bb, clf

def compute_average_confidence(backbone, classifier, samples):
    """计算Average Confidence"""
    backbone.eval()
    classifier.eval()

    dataset = TensorDataset(samples)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    all_confidences = []

    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x[0].to(device)
            features = backbone(batch_x)
            logits, probs = classifier(features)

            # 计算每个样本的置信度（最大softmax概率）
            confidences = probs.max(dim=1)[0]
            all_confidences.extend(confidences.cpu().numpy())

    # 返回平均置信度
    return float(np.mean(all_confidences))

def compute_accuracy(backbone, classifier, samples, labels):
    """计算accuracy"""
    backbone.eval()
    classifier.eval()

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            features = backbone(batch_x)
            logits, _ = classifier(features)
            preds = logits.argmax(dim=1)

            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return float(correct / total * 100)

def main():
    print("=" * 80)
    print("任务 M5.1: 实现Average Confidence监控信号")
    print("=" * 80)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载源模型
    print("\n1. 加载源模型:")
    source_model_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain.pt'
    backbone, classifier = load_source_model(source_model_path)
    print(f"   ✓ 加载成功")

    # 加载CWRU数据
    print("\n2. 加载CWRU数据:")
    cwru_data_path = PROJECT_ROOT / 'data' / 'processed' / 'cwru_3hp.pt'
    cwru_data = torch.load(cwru_data_path, map_location=device)
    cwru_samples = cwru_data['samples']
    cwru_labels = cwru_data['labels']
    print(f"   ✓ 加载成功: {cwru_samples.shape[0]} 个样本")

    # 定义SNR水平
    snr_levels = ['Clean', '6dB', '3dB', '0dB', '-3dB', '-6dB']
    snr_db_map = {'Clean': float('inf'), '6dB': 6, '3dB': 3, '0dB': 0, '-3dB': -3, '-6dB': -6}

    # 对每个SNR水平运行SHOT适应并计算Average Confidence
    print("\n3. 运行SHOT适应并计算Average Confidence:")
    results = {}

    for snr_key in snr_levels:
        print(f"\n   SNR: {snr_key}")

        # 添加噪声
        snr_db = snr_db_map[snr_key]
        noisy_samples = add_gaussian_noise(cwru_samples, snr_db)

        # 运行SHOT适应
        adapted_bb, adapted_clf = run_shot_adaptation(
            backbone, classifier, noisy_samples, cwru_labels,
            num_epochs=50, lr=1e-3
        )

        # 计算Average Confidence
        avg_conf = compute_average_confidence(adapted_bb, adapted_clf, noisy_samples)

        # 计算accuracy
        acc = compute_accuracy(adapted_bb, adapted_clf, noisy_samples, cwru_labels)

        results[snr_key] = {
            'average_confidence': avg_conf,
            'accuracy': acc
        }

        print(f"      Average Confidence: {avg_conf:.4f}")
        print(f"      Accuracy: {acc:.2f}%")

    # 计算相关性
    print("\n4. 计算与accuracy的相关性:")
    accuracies = [results[snr]['accuracy'] for snr in snr_levels]
    avg_confs = [results[snr]['average_confidence'] for snr in snr_levels]

    accuracies = np.array(accuracies)
    avg_confs = np.array(avg_confs)

    # 计算Spearman相关性
    if np.std(avg_confs) > 0:
        rho, p_value = stats.spearmanr(avg_confs, accuracies)
        print(f"   Spearman ρ: {rho:.4f}")
        print(f"   p-value: {p_value:.4e}")
    else:
        rho, p_value = float('nan'), float('nan')
        print(f"   Spearman ρ: nan (constant input)")
        print(f"   p-value: nan")

    # 计算AUC（将accuracy<70%定义为崩溃）
    print("\n5. 计算AUC用于崩溃检测:")
    collapsed = (accuracies < 70).astype(int)

    if len(np.unique(collapsed)) > 1:
        auc = roc_auc_score(collapsed, avg_confs)
        print(f"   AUC: {auc:.4f}")
    else:
        auc = float('nan')
        print(f"   AUC: nan (no variation in collapse labels)")

    # 保存结果
    output_data = {
        'task': 'M5.1',
        'description': 'Average Confidence监控信号实现',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'CWRU',
        'method': 'SHOT',
        'results': results,
        'correlation': {
            'spearman_rho': float(rho) if not np.isnan(rho) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None
        },
        'auc': float(auc) if not np.isnan(auc) else None
    }

    output_path = RESULTS_DIR / 'task_M5_1_average_confidence.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n6. 结果已保存到: {output_path}")

    # 记录到LOG文件
    log_path = PROJECT_ROOT / 'LOG_2026-08-06.md'

    # 格式化结果用于日志
    rho_str = f"{rho:.4f}" if not np.isnan(rho) else "nan"
    p_value_str = f"{p_value:.4e}" if not np.isnan(p_value) else "nan"
    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "nan"

    log_entry = f"""
### 任务 M5.1: 实现Average Confidence监控信号

**执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**目标**: 实现Average Confidence作为崩溃检测的基线信号

**方法**:
1. 对每个SNR水平运行SHOT适应
2. 在适应后的模型上计算Average Confidence（最大softmax概率的平均值）
3. 计算与accuracy的Spearman相关性
4. 计算AUC用于崩溃检测（accuracy<70%定义为崩溃）

**结果**:
- Average Confidence与accuracy的Spearman ρ: {rho_str}
- p-value: {p_value_str}
- AUC: {auc_str}

**各SNR水平的结果**:
"""

    for snr_key in snr_levels:
        r = results[snr_key]
        log_entry += f"- {snr_key}: AvgConf={r['average_confidence']:.4f}, Acc={r['accuracy']:.2f}%\n"

    log_entry += f"""
**结论**: ✅ M5.1完成 - 成功实现Average Confidence监控信号

---
"""

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)

    print(f"已记录到LOG文件: {log_path}")

    print("\n" + "=" * 80)
    print("✅ 任务 M5.1 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
