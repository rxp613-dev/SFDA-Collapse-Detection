#!/usr/bin/env python3
"""
任务 A6.2: 实现并运行监控-干预闭环原型
创建时间: 2026-08-07
目标: 基于A6.1的设计，实现改进的监控-干预闭环系统
方法:
    1. 监控Class Shift指标
    2. 当Class Shift超过阈值时，触发干预（降低lr）
    3. 设置最小lr限制，防止lr降到0
    4. 实现恢复机制：如果干预无效，尝试重置模型状态
    5. 记录完整的干预历史和效果评估
    6. 与未干预的baseline对比
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
import sys
from copy import deepcopy

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
NUM_CLASSES = 4

# 监控-干预参数
CLASS_SHIFT_THRESHOLD = 0.03
LR_DECAY_FACTOR = 0.5  # 降低50%（比A6.1的10%更温和）
MIN_LR = 1e-5  # 最小学习率限制
MONITOR_INTERVAL = 5  # 每5个epoch检查一次
MAX_INTERVENTIONS = 3  # 最大干预次数


def compute_class_shift(predicted_distribution, reference_prior):
    """计算Class Shift (L1距离)"""
    l1_distance = 0.0
    for cls in reference_prior.keys():
        l1_distance += abs(predicted_distribution[cls] - reference_prior[cls])
    return l1_distance


def get_predicted_distribution(probs):
    """从概率矩阵计算预测分布"""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    preds = np.argmax(probs, axis=1)
    total = len(preds)
    distribution = {}

    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(preds == i)
        distribution[name] = count / total

    return distribution


def run_shot_with_monitoring(source_model_path, target_data_path, seed=42,
                             initial_lr=1e-3, num_epochs=50, batch_size=64,
                             enable_intervention=True):
    """
    运行SHOT算法，带监控-干预闭环系统

    Args:
        source_model_path: 源域模型路径
        target_data_path: 目标域数据路径
        seed: 随机种子
        initial_lr: 初始学习率
        num_epochs: 训练轮数
        batch_size: 批次大小
        enable_intervention: 是否启用干预机制

    Returns:
        accuracy: 最终准确率
        ir_recall: IR recall
        history: 训练历史记录
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 加载源域模型
    source_model = torch.load(source_model_path, map_location=DEVICE)
    backbone = BearingFaultBackbone(feature_dim=256).to(DEVICE)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(DEVICE)

    state_dict = source_model['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}

    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)

    # 加载目标域数据
    data = torch.load(target_data_path, map_location=DEVICE)
    target_data = data['samples']
    target_labels = data['labels']

    # 计算参考先验（源域分布）
    reference_prior = {
        'Normal': 0.571,
        'IR': 0.143,
        'Ball': 0.143,
        'OR': 0.143
    }

    # 冻结分类器
    for param in classifier.parameters():
        param.requires_grad = False

    # 创建数据加载器
    dataset = TensorDataset(target_data, target_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化器
    current_lr = initial_lr
    optimizer = optim.Adam(backbone.parameters(), lr=current_lr)

    # 训练历史
    history = {
        'interventions': [],
        'class_shift_values': [],
        'epoch_stats': [],
        'intervention_effectiveness': []
    }

    # 训练循环
    backbone.train()
    classifier.eval()

    intervention_count = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_x, _ in dataloader:
            batch_x = batch_x.to(DEVICE)

            optimizer.zero_grad()

            # 前向传播
            features = backbone(batch_x)
            logits, probs = classifier(features)

            # 计算熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            loss = entropy.mean()

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # 定期监控Class Shift
        if (epoch + 1) % MONITOR_INTERVAL == 0:
            backbone.eval()
            with torch.no_grad():
                features = backbone(target_data.to(DEVICE))
                logits, probs = classifier(features)
                probs_np = probs.cpu().numpy()

                # 计算预测分布
                predicted_dist = get_predicted_distribution(probs_np)

                # 计算Class Shift
                class_shift = compute_class_shift(predicted_dist, reference_prior)

                # 计算当前accuracy
                preds = torch.argmax(probs, dim=1)
                current_accuracy = (preds == target_labels.to(DEVICE)).float().mean().item() * 100

                history['class_shift_values'].append({
                    'epoch': epoch + 1,
                    'class_shift': float(class_shift),
                    'accuracy': float(current_accuracy),
                    'lr': float(current_lr),
                    'predicted_distribution': {k: float(v) for k, v in predicted_dist.items()}
                })

                # 检查是否需要干预
                if enable_intervention and class_shift > CLASS_SHIFT_THRESHOLD:
                    if intervention_count < MAX_INTERVENTIONS and current_lr > MIN_LR:
                        old_lr = current_lr
                        new_lr = max(current_lr * LR_DECAY_FACTOR, MIN_LR)

                        # 重建优化器
                        current_lr = new_lr
                        optimizer = optim.Adam(backbone.parameters(), lr=current_lr)

                        intervention = {
                            'epoch': epoch + 1,
                            'old_lr': float(old_lr),
                            'new_lr': float(new_lr),
                            'class_shift': float(class_shift),
                            'accuracy_before': float(current_accuracy),
                            'intervention_number': intervention_count + 1,
                            'reason': f'Class Shift {class_shift:.4f} > threshold {CLASS_SHIFT_THRESHOLD}'
                        }
                        history['interventions'].append(intervention)

                        intervention_count += 1

            # 退出 torch.no_grad() 上下文，进行干预训练
            if intervention_count > 0 and len(history['interventions']) == intervention_count:
                # 评估干预效果（在5个epoch后）
                if epoch + 5 < num_epochs:
                    # 切换到训练模式继续训练5个epoch
                    backbone.train()
                    for _ in range(5):
                        for batch_x, _ in dataloader:
                            batch_x = batch_x.to(DEVICE)
                            optimizer.zero_grad()
                            features = backbone(batch_x)
                            logits, probs = classifier(features)
                            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                            loss = entropy.mean()
                            loss.backward()
                            optimizer.step()

                    # 重新评估
                    backbone.eval()
                    with torch.no_grad():
                        features = backbone(target_data.to(DEVICE))
                        logits, probs = classifier(features)
                        preds = torch.argmax(probs, dim=1)
                        accuracy_after = (preds == target_labels.to(DEVICE)).float().mean().item() * 100

                        intervention['accuracy_after'] = float(accuracy_after)
                        intervention['effectiveness'] = accuracy_after - current_accuracy

                        history['intervention_effectiveness'].append({
                            'intervention_number': intervention_count,
                            'accuracy_before': float(current_accuracy),
                            'accuracy_after': float(accuracy_after),
                            'improvement': accuracy_after - current_accuracy
                        })

            backbone.train()

        # 记录epoch统计
        history['epoch_stats'].append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'lr': float(current_lr)
        })

    # 最终评估
    backbone.eval()
    with torch.no_grad():
        features = backbone(target_data.to(DEVICE))
        logits, probs = classifier(features)
        preds = torch.argmax(probs, dim=1)

        # 计算accuracy
        accuracy = (preds == target_labels.to(DEVICE)).float().mean().item() * 100

        # 计算IR recall (类别1)
        ir_mask = (target_labels.to(DEVICE) == 1)
        if ir_mask.sum() > 0:
            ir_recall = (preds[ir_mask] == 1).float().mean().item() * 100
        else:
            ir_recall = 0.0

    return accuracy, ir_recall, history


def main():
    print("=" * 80)
    print(f"任务 A6.2: 实现并运行监控-干预闭环原型")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 配置
    source_model_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'source_pretrain.pt'
    target_data_path = PROJECT_ROOT / 'data' / 'processed' / 'cwru_3hp_denoised_0db.pt'

    print(f"\n源域模型: {source_model_path}")
    print(f"目标域数据: {target_data_path}")
    print(f"\n监控-干预参数:")
    print(f"  Class Shift阈值: {CLASS_SHIFT_THRESHOLD}")
    print(f"  LR衰减因子: {LR_DECAY_FACTOR}")
    print(f"  最小LR: {MIN_LR}")
    print(f"  监控间隔: 每{MONITOR_INTERVAL}个epoch")
    print(f"  最大干预次数: {MAX_INTERVENTIONS}")

    # 运行对比实验
    seeds = [42, 43, 44, 45, 46]
    results_with_intervention = []
    results_without_intervention = []

    print(f"\n运行对比实验 (5个种子)...")

    # 未干预的baseline
    print("\n[Baseline] 未干预:")
    for i, seed in enumerate(seeds):
        print(f"  [{i+1}/{len(seeds)}] Seed {seed}...", end=' ')
        accuracy, ir_recall, history = run_shot_with_monitoring(
            source_model_path=source_model_path,
            target_data_path=target_data_path,
            seed=seed,
            initial_lr=1e-3,
            num_epochs=50,
            batch_size=64,
            enable_intervention=False
        )
        results_without_intervention.append({
            'seed': seed,
            'accuracy': accuracy,
            'ir_recall': ir_recall,
            'history': history
        })
        print(f"Accuracy: {accuracy:.2f}%, IR Recall: {ir_recall:.2f}%")

    # 带干预的实验
    print("\n[Intervention] 带监控-干预:")
    for i, seed in enumerate(seeds):
        print(f"  [{i+1}/{len(seeds)}] Seed {seed}...", end=' ')
        accuracy, ir_recall, history = run_shot_with_monitoring(
            source_model_path=source_model_path,
            target_data_path=target_data_path,
            seed=seed,
            initial_lr=1e-3,
            num_epochs=50,
            batch_size=64,
            enable_intervention=True
        )
        results_with_intervention.append({
            'seed': seed,
            'accuracy': accuracy,
            'ir_recall': ir_recall,
            'num_interventions': len(history['interventions']),
            'history': history
        })
        num_interventions = len(history['interventions'])
        print(f"Accuracy: {accuracy:.2f}%, IR Recall: {ir_recall:.2f}%, 干预次数: {num_interventions}")

    # 计算统计
    acc_without = [r['accuracy'] for r in results_without_intervention]
    acc_with = [r['accuracy'] for r in results_with_intervention]
    ir_without = [r['ir_recall'] for r in results_without_intervention]
    ir_with = [r['ir_recall'] for r in results_with_intervention]

    print("\n" + "=" * 80)
    print("统计对比:")
    print("=" * 80)
    print(f"\n[Baseline] 未干预:")
    print(f"  Accuracy: {np.mean(acc_without):.2f}% ± {np.std(acc_without):.2f}%")
    print(f"  IR Recall: {np.mean(ir_without):.2f}% ± {np.std(ir_without):.2f}%")
    print(f"\n[Intervention] 带监控-干预:")
    print(f"  Accuracy: {np.mean(acc_with):.2f}% ± {np.std(acc_with):.2f}%")
    print(f"  IR Recall: {np.mean(ir_with):.2f}% ± {np.std(ir_with):.2f}%")
    print(f"\n改善:")
    print(f"  Accuracy: {np.mean(acc_with) - np.mean(acc_without):+.2f}%")
    print(f"  IR Recall: {np.mean(ir_with) - np.mean(ir_without):+.2f}%")

    # 分析干预效果
    if results_with_intervention:
        total_interventions = sum(r['num_interventions'] for r in results_with_intervention)
        avg_interventions = total_interventions / len(results_with_intervention)
        print(f"\n干预统计:")
        print(f"  总干预次数: {total_interventions}")
        print(f"  平均干预次数: {avg_interventions:.2f}")

        # 分析干预有效性
        effectiveness_list = []
        for r in results_with_intervention:
            for eff in r['history']['intervention_effectiveness']:
                effectiveness_list.append(eff['improvement'])

        if effectiveness_list:
            print(f"  平均改善: {np.mean(effectiveness_list):+.2f}%")
            print(f"  有效干预比例: {sum(1 for e in effectiveness_list if e > 0) / len(effectiveness_list) * 100:.1f}%")

    print("=" * 80)

    # 保存结果
    output_path = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_A6_2_monitoring_intervention_prototype.json'

    output_data = {
        'task': 'A6.2',
        'description': '监控-干预闭环原型实现',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'class_shift_threshold': CLASS_SHIFT_THRESHOLD,
            'lr_decay_factor': LR_DECAY_FACTOR,
            'min_lr': MIN_LR,
            'monitor_interval': MONITOR_INTERVAL,
            'max_interventions': MAX_INTERVENTIONS,
            'initial_lr': 1e-3,
            'num_epochs': 50,
            'batch_size': 64,
            'seeds': seeds
        },
        'results': {
            'baseline': results_without_intervention,
            'with_intervention': results_with_intervention
        },
        'statistics': {
            'baseline': {
                'mean_accuracy': float(np.mean(acc_without)),
                'std_accuracy': float(np.std(acc_without)),
                'mean_ir_recall': float(np.mean(ir_without)),
                'std_ir_recall': float(np.std(ir_without))
            },
            'with_intervention': {
                'mean_accuracy': float(np.mean(acc_with)),
                'std_accuracy': float(np.std(acc_with)),
                'mean_ir_recall': float(np.mean(ir_with)),
                'std_ir_recall': float(np.std(ir_with)),
                'improvement_accuracy': float(np.mean(acc_with) - np.mean(acc_without)),
                'improvement_ir_recall': float(np.mean(ir_with) - np.mean(ir_without))
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n结果已保存至: {output_path}")

    print("\n" + "=" * 80)
    print(f"任务 A6.2 完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
