#!/usr/bin/env python3
"""
Step 1 补充: NRC/SAR 实现审核 (详细版)
Created: 2026-08-13
Purpose: 详细审核 NRC 和 SAR 实现的正确性，识别关键缺陷
"""

import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def main():
    print("=" * 80)
    print("Step 1 补充: NRC/SAR 实现详细审核")
    print("=" * 80)

    audit = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'methods': {},
        'critical_issues': [],
        'recommendations': []
    }

    # NRC 审核
    print("\n[1/2] 审核 NRC 实现...")
    nrc_audit = {
        'algorithm': 'NRC (Neighborhood Reciprocity Clustering)',
        'reference': 'Roy et al., "Neighborhood Reciprocity Clustering for Source-Free Domain Adaptation" (CVPR 2022)',
        'implementation_file': '/mnt/data/sfda3/scripts/revision/task_3_1_with_signals.py',
        'key_requirements': [
            'Freeze backbone (only update classifier)',
            'Use pseudo-labels for self-training',
            'Neighborhood reciprocity: encourage similar features to have same predictions',
            'Cluster-based pseudo-label refinement'
        ],
        'actual_implementation': {
            'freezes_backbone': False,  # bb.train() - backbone is trainable
            'uses_pseudo_labels': True,
            'neighborhood_loss': 'Simple feature similarity (not true NRC)',
            'cluster_refinement': False
        },
        'issues': [
            'CRITICAL: Backbone is NOT frozen (should be frozen in SFDA)',
            'MAJOR: Neighbor loss is just -similarity.mean(), not true neighborhood reciprocity',
            'MAJOR: No cluster-based pseudo-label refinement',
            'MAJOR: Missing the "reciprocity" component of NRC'
        ],
        'severity': 'CRITICAL - Implementation does not match NRC algorithm'
    }
    audit['methods']['NRC'] = nrc_audit

    # SAR 审核
    print("[2/2] 审核 SAR 实现...")
    sar_audit = {
        'algorithm': 'SAR (Selective Amplitude Regularization)',
        'reference': 'Niu et al., "Towards Stable Test-Time Adaptation in Dynamic Wild World" (ICLR 2022)',
        'implementation_file': '/mnt/data/sfda3/scripts/revision/task_3_1_with_signals.py',
        'key_requirements': [
            'Freeze backbone (only update BN parameters)',
            'Selective update: only update samples with low entropy',
            'Robust entropy minimization with filter',
            'BN parameter adaptation only'
        ],
        'actual_implementation': {
            'freezes_backbone': True,
            'updates_bn_only': False,  # Updates ALL classifier parameters
            'selective_update': False,  # No entropy-based filtering
            'uses_pseudo_labels': True
        },
        'issues': [
            'CRITICAL: Updates ALL classifier parameters, not just BN (SAR only updates BN)',
            'CRITICAL: No selective update mechanism (core of SAR)',
            'MAJOR: Missing entropy-based sample filtering',
            'MAJOR: Essentially just pseudo-label training, not SAR'
        ],
        'severity': 'CRITICAL - Implementation does not match SAR algorithm'
    }
    audit['methods']['SAR'] = sar_audit

    # 总结
    print("\n" + "=" * 80)
    print("审核结论")
    print("=" * 80)
    print("\nNRC 实现: ❌ 严重错误")
    print("  - Backbone 未冻结 (违反 SFDA 原则)")
    print("  - 邻域损失实现不正确")
    print("  - 缺少聚类优化机制")

    print("\nSAR 实现: ❌ 严重错误")
    print("  - 更新所有分类器参数而非仅 BN")
    print("  - 缺少核心选择性更新机制")
    print("  - 本质上是伪标签训练，不是 SAR")

    print("\n" + "=" * 80)
    print("对实验公平性的影响")
    print("=" * 80)
    print("\n现有 NRC/SAR 实现的问题会导致:")
    print("1. 无法验证论文结论 (NRC=57.17%, SAR=25.55% 是否真实)")
    print("2. 审稿人质疑实验公平性")
    print("3. 需要重新实现或寻找正确实现")

    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)
    print("\n方案 A: 重新实现 NRC/SAR")
    print("  - 优点: 确保算法正确性")
    print("  - 缺点: 耗时较长 (预计 2-3 天)")

    print("\n方案 B: 使用简化版本")
    print("  - 保留现有实现，但明确标注为 'simplified NRC/SAR'")
    print("  - 在论文中说明实现细节")
    print("  - 优点: 快速")
    print("  - 缺点: 可能被审稿人质疑")

    print("\n方案 C: 聚焦已有正确实现的方法")
    print("  - 只报告 SHOT/TENT/RPSWD 的结果")
    print("  - 删除或弱化 NRC/SAR 的讨论")
    print("  - 优点: 避免错误")
    print("  - 缺点: 减少方法覆盖范围")

    audit['recommendations'] = [
        '方案 A (推荐): 重新实现 NRC/SAR，确保算法正确性',
        '在论文中明确说明所有方法的实现细节',
        '为每个方法添加单元测试，验证关键特性',
        '考虑使用开源实现 (如 OpenDomainAdaptation library)'
    ]

    # 保存审核报告
    output_path = RESULTS_DIR / 'step1_supplement_nrc_sar_audit.json'
    with open(output_path, 'w') as f:
        json.dump(audit, f, indent=2)

    print(f"\n审核报告已保存至: {output_path}")
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return audit

if __name__ == '__main__':
    main()
