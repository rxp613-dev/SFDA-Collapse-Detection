#!/usr/bin/env python3
"""
任务 P2-1: 用V2正典实现重跑Task 3-3消融实验（40 runs）
创建时间: 2026-08-11
目标: 使用正确的RPSWD实现重跑消融实验
方法:
  1. 使用与V2主审计相同的RPSWD实现
  2. 运行4种配置：Full, No_soft_weight, No_repulsion, No_both
  3. 每种配置10个种子，共40次运行
  4. 比较各配置的Accuracy和Macro-F1
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

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

def load_target_data(data_path):
    """加载目标域数据"""
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']

def compute_metrics(preds, labels):
    """计算accuracy和macro-F1"""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    accuracy = float((preds == labels).mean() * 100)

    # Compute macro-F1
    from sklearn.metrics import f1_score
    macro_f1 = float(f1_score(labels, preds, average='macro') * 100)

    return accuracy, macro_f1

def run_rpswd(backbone, classifier, samples, labels, num_epochs=100, lr=1e-4, seed=42,
              use_soft_weighting=True, use_repulsion=True):
    """运行RPSWD适应（V2正典实现）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    clf.train()

    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=lr)

    for epoch in range(num_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            features = bb(batch_x)
            logits, probs = clf(features)
            pseudo_labels = probs.argmax(dim=1)

            # Compute prototypes
            features_norm = F.normalize(features, dim=1)
            prototypes = torch.zeros(4, features.shape[1]).to(device)
            for c in range(4):
                mask = (pseudo_labels == c)
                if mask.sum() > 0:
                    prototypes[c] = features_norm[mask].mean(dim=0)
            prototypes = F.normalize(prototypes, dim=1)

            # Compute boundary scores (KL divergence)
            p_cls = F.softmax(logits, dim=1)
            cos_sim = torch.mm(features_norm, prototypes.t())
            p_proto = F.softmax(cos_sim / 0.1, dim=1)
            boundary_scores = torch.sum(p_cls * torch.log((p_cls + 1e-8) / (p_proto + 1e-8)), dim=1)

            # Compute soft weights
            if use_soft_weighting:
                min_bs = boundary_scores.min()
                max_bs = boundary_scores.max()
                if max_bs - min_bs > 1e-8:
                    omega = 1.0 - (boundary_scores - min_bs) / (max_bs - min_bs + 1e-8)
                else:
                    omega = torch.ones_like(boundary_scores) * 0.5
            else:
                omega = torch.ones_like(boundary_scores)

            # Soft-weighted CE loss
            ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
            weighted_ce = (omega * ce_loss).mean()

            # Repulsion loss
            if use_repulsion:
                cos_sim_target = cos_sim[torch.arange(len(pseudo_labels)), pseudo_labels]
                cos_sim_other = cos_sim.clone()
                cos_sim_other[torch.arange(len(pseudo_labels)), pseudo_labels] = -1e9
                max_cos_sim_other = cos_sim_other.max(dim=1)[0]
                repulsion_loss = torch.relu(0.5 - (cos_sim_target - max_cos_sim_other)).mean()
            else:
                repulsion_loss = 0.0

            loss = weighted_ce + 0.5 * (1 - omega.mean()) * repulsion_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, macro_f1 = compute_metrics(preds, labels)

    return accuracy, macro_f1

def main():
    print("="*80)
    print("任务 P2-1: 用V2正典实现重跑Task 3-3消融实验")
    print("="*80)

    # Load data
    print("\n1. 加载源模型和目标数据...")
    source_model_path = PROJECT_ROOT / 'data/checkpoints/source_pretrain.pt'
    target_data_path = PROJECT_ROOT / 'data/processed/cwru_3hp.pt'

    backbone, classifier = load_source_model(source_model_path)
    samples, labels = load_target_data(target_data_path)

    print(f"   ✓ 源模型加载成功")
    print(f"   ✓ 目标数据加载成功: {samples.shape[0]} 个样本")

    # Run ablation study
    print("\n2. 运行消融实验（4配置 × 10 seeds = 40 runs）...")

    configurations = [
        ('Full_RPSWD', True, True),
        ('No_soft_weight', False, True),
        ('No_repulsion', True, False),
        ('No_both', False, False)
    ]

    results = {}
    seeds = list(range(42, 52))

    for config_name, use_sw, use_rep in configurations:
        print(f"\n   配置: {config_name}")
        config_results = {}

        for seed in seeds:
            accuracy, macro_f1 = run_rpswd(
                backbone, classifier, samples, labels,
                num_epochs=100, lr=1e-4, seed=seed,
                use_soft_weighting=use_sw, use_repulsion=use_rep
            )

            config_results[f'seed_{seed}'] = {
                'accuracy': accuracy,
                'macro_f1': macro_f1
            }

            print(f"      Seed {seed}: Accuracy={accuracy:.2f}%, Macro-F1={macro_f1:.2f}%")

        # Compute statistics
        accuracies = [config_results[f'seed_{s}']['accuracy'] for s in seeds]
        macro_f1s = [config_results[f'seed_{s}']['macro_f1'] for s in seeds]

        results[config_name] = {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'macro_f1_mean': np.mean(macro_f1s),
            'macro_f1_std': np.std(macro_f1s),
            'results': config_results
        }

        print(f"      均值: Accuracy={np.mean(accuracies):.2f}±{np.std(accuracies):.2f}%, "
              f"Macro-F1={np.mean(macro_f1s):.2f}±{np.std(macro_f1s):.2f}%")

    # Save results
    print("\n3. 保存结果...")
    output_data = {
        'task': 'P2-1',
        'description': 'Ablation study with V2 canonical RPSWD implementation',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configurations': results
    }

    output_path = RESULTS_DIR / 'task_P2_1_ablation_study_corrected.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"   ✓ 结果已保存到: {output_path}")

    print("\n" + "="*80)
    print("任务 P2-1 完成")
    print("="*80)

if __name__ == '__main__':
    main()
