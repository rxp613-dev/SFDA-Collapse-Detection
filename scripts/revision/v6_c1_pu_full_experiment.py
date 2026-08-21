#!/usr/bin/env python3
"""
C1: 补全PU数据集完整实验
日期: 2026-08-19
目标: 在PU数据集上运行4种SFDA方法（SHOT, TENT, NRC, SAR），使用10个随机种子
方法:
  1. 加载PU数据集（pu_v4.pt）和源模型（source_pretrain_pu_v4.pt）
  2. 实现4种SFDA方法
  3. 每种方法运行10个种子
  4. 计算accuracy, macro_f1, balanced_acc, ir_recall
  5. 保存结果到JSON
"""

import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, balanced_accuracy_score

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
BATCH_SIZE = 256
LR = 1e-3
NUM_EPOCHS = 5
NUM_SEEDS = 10

print("=" * 80)
print("C1: PU数据集完整实验")
print("=" * 80)
print(f"时间: 2026-08-19")
print(f"设备: {DEVICE}")
print(f"种子数: {NUM_SEEDS}")


def load_source_model(checkpoint_path):
    """加载源模型"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier


@torch.no_grad()
def evaluate(bb, clf, samples, labels, batch_size=BATCH_SIZE):
    """分批评估，避免OOM"""
    bb.eval()
    clf.eval()
    loader = DataLoader(TensorDataset(samples, labels), batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE)
        features = bb(batch_x)
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(batch_y)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).cpu().numpy()

    accuracy = 100.0 * (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average='macro') * 100
    balanced_acc = balanced_accuracy_score(labels, preds) * 100
    mask = labels == 1
    ir_recall = 100.0 * (preds[mask] == 1).mean() if mask.sum() > 0 else 0.0
    return accuracy, macro_f1, balanced_acc, ir_recall


def get_bn_params(bb):
    """获取BN参数，设置train模式"""
    bn_params = []
    for module in bb.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            for p in module.parameters():
                p.requires_grad = True
                bn_params.append(p)
    return bn_params


# ============ SHOT实现 ============
def run_shot(backbone, classifier, samples, labels, seed=42):
    """SHOT: 更新全骨干，熵最小化 + 多样性损失"""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    bb.train(); clf.train()
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=LR)
    loader = DataLoader(TensorDataset(samples, labels), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            mean_prob = probs.mean(dim=0)
            diversity = -torch.sum(mean_prob * torch.log(mean_prob + 1e-8))
            loss = entropy - 0.1 * diversity
            loss.backward()
            optimizer.step()

    return evaluate(bb, clf, samples, labels)


# ============ TENT实现 ============
def run_tent(backbone, classifier, samples, labels, seed=42):
    """TENT: 仅更新BN参数，熵最小化"""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    bb.eval(); clf.eval()
    bn_params = get_bn_params(bb)
    optimizer = torch.optim.Adam(bn_params, lr=LR)
    loader = DataLoader(TensorDataset(samples, labels), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            entropy.backward()
            optimizer.step()

    return evaluate(bb, clf, samples, labels)


# ============ NRC实现（修正版，向量化） ============
def run_nrc(backbone, classifier, samples, labels, seed=42, k=5):
    """NRC: k近邻 + 互惠损失（向量化版本）"""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    bb.train(); clf.train()
    optimizer = torch.optim.Adam(list(bb.parameters()) + list(clf.parameters()), lr=LR)
    loader = DataLoader(TensorDataset(samples, labels), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            features = bb(batch_x)
            logits, probs = clf(features)
            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)

            ce_loss = F.cross_entropy(logits, pseudo_labels)

            # 向量化k近邻互惠损失
            features_norm = F.normalize(features, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
            n = features.shape[0]
            knn_indices = similarity.topk(k+1, dim=1)[1][:, 1:]  # [n, k]

            # 向量化：比较每个样本和其k个邻居的伪标签
            sample_labels = pseudo_labels.unsqueeze(1).expand(n, k)  # [n, k]
            neighbor_labels = pseudo_labels[knn_indices]              # [n, k]
            mismatches = (sample_labels != neighbor_labels).float()
            neighbor_loss = mismatches.mean()

            loss = ce_loss + 0.1 * neighbor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return evaluate(bb, clf, samples, labels)


# ============ SAR实现 ============
def run_sar(backbone, classifier, samples, labels, seed=42):
    """SAR: 熵过滤 + BN参数更新"""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(DEVICE)
    clf = deepcopy(classifier).to(DEVICE)
    bb.eval(); clf.eval()
    bn_params = get_bn_params(bb)
    optimizer = torch.optim.Adam(bn_params, lr=LR)
    loader = DataLoader(TensorDataset(samples, labels), batch_size=BATCH_SIZE, shuffle=True)
    entropy_threshold = np.log(NUM_CLASSES) - 0.01

    for epoch in range(NUM_EPOCHS):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            mask = entropy < entropy_threshold
            if mask.sum() > 0:
                loss = entropy[mask].mean()
            else:
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            loss.backward()
            optimizer.step()

    return evaluate(bb, clf, samples, labels)


# ============ 主实验 ============
print("\n=== 1. 加载PU数据集和源模型 ===")
PU_DATA_PATH = Path('/mnt/data/sfda3/data/processed/pu_v4.pt')
data_dict = torch.load(PU_DATA_PATH, map_location=DEVICE)
samples = data_dict['samples']
labels = data_dict['labels']
print(f"  样本数: {len(samples)}")
print(f"  类别分布: {torch.bincount(labels).tolist()}")

SOURCE_MODEL_PATH = Path('/mnt/data/sfda3/data/checkpoints/source_pretrain_pu_v4.pt')
backbone, classifier = load_source_model(SOURCE_MODEL_PATH)
print(f"  源模型已加载")

# 评估源模型（分批）
src_acc, src_f1, src_bacc, src_ir = evaluate(backbone, classifier, samples, labels)
print(f"  源模型性能: Acc={src_acc:.2f}%, F1={src_f1:.2f}%, BalAcc={src_bacc:.2f}%, IR={src_ir:.2f}%")

# 运行4种方法，每种10个种子
print("\n=== 2. 运行4种SFDA方法（各10个种子） ===")
SEEDS = list(range(42, 42 + NUM_SEEDS))

results = {'SHOT': [], 'TENT': [], 'NRC': [], 'SAR': []}

for method_name, method_func in [('SHOT', run_shot), ('TENT', run_tent), ('NRC', run_nrc), ('SAR', run_sar)]:
    method_start = time.time()
    print(f"\n--- {method_name} ---")
    for seed in SEEDS:
        seed_start = time.time()
        acc, f1, bacc, ir = method_func(backbone, classifier, samples, labels, seed=seed)
        seed_elapsed = time.time() - seed_start
        results[method_name].append({
            'seed': seed,
            'accuracy': acc,
            'macro_f1': f1,
            'balanced_acc': bacc,
            'ir_recall': ir
        })
        print(f"  Seed {seed}: Acc={acc:.2f}%, F1={f1:.2f}%, BalAcc={bacc:.2f}%, IR={ir:.2f}% ({seed_elapsed:.1f}s)")
    method_elapsed = time.time() - method_start
    print(f"  {method_name} 完成，耗时 {method_elapsed:.1f}s ({method_elapsed/60:.1f}min)")

# ============ 统计汇总 ============
print("\n=== 3. 统计汇总 ===")
summary = {}
for method_name in ['SHOT', 'TENT', 'NRC', 'SAR']:
    accs = [r['accuracy'] for r in results[method_name]]
    f1s = [r['macro_f1'] for r in results[method_name]]
    baccs = [r['balanced_acc'] for r in results[method_name]]
    irs = [r['ir_recall'] for r in results[method_name]]

    summary[method_name] = {
        'accuracy': f"{np.mean(accs):.2f} ± {np.std(accs):.2f}%",
        'macro_f1': f"{np.mean(f1s):.2f} ± {np.std(f1s):.2f}%",
        'balanced_acc': f"{np.mean(baccs):.2f} ± {np.std(baccs):.2f}%",
        'ir_recall': f"{np.mean(irs):.2f} ± {np.std(irs):.2f}%",
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'mean_macro_f1': float(np.mean(f1s)),
        'std_macro_f1': float(np.std(f1s)),
        'mean_balanced_acc': float(np.mean(baccs)),
        'std_balanced_acc': float(np.std(baccs)),
        'mean_ir_recall': float(np.mean(irs)),
        'std_ir_recall': float(np.std(irs))
    }

    print(f"\n{method_name}:")
    print(f"  Accuracy:     {summary[method_name]['accuracy']}")
    print(f"  Macro-F1:     {summary[method_name]['macro_f1']}")
    print(f"  Balanced Acc: {summary[method_name]['balanced_acc']}")
    print(f"  IR Recall:    {summary[method_name]['ir_recall']}")

# ============ 保存结果 ============
output_path = Path('/mnt/data/sfda3/prai2026/paper2/experiments/results/revision/v6_c1_pu_full_experiment.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

output_data = {
    'metadata': {
        'date': '2026-08-19',
        'task': 'C1 PU完整实验',
        'dataset': 'PU v4',
        'num_samples': len(samples),
        'num_classes': NUM_CLASSES,
        'num_seeds': NUM_SEEDS,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'seeds': SEEDS,
        'device': str(DEVICE),
        'source_model_accuracy': float(src_acc)
    },
    'results': results,
    'summary': summary
}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ 结果已保存到 {output_path}")
print("\n✓ C1 PU完整实验完成")
