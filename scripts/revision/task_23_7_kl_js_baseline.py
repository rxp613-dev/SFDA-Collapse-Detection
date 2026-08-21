#!/usr/bin/env python3
"""
任务 23.7: KL散度和JS散度基线对比
创建时间: 2026-08-12
目标: 计算KL散度和Jensen-Shannon散度的AUC，与Class Shift对比
方法:
  1. 加载B2 pooled ROC数据（390 runs，有predicted_distribution）
  2. 对每个run计算KL散度和JS散度
  3. 计算pooled AUC
  4. 与Class Shift对比
数据源: task_B2_pooled_roc_analysis_corrected.json
GPU: 不需要（纯后处理）
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import jensenshannon
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'

def compute_kl_divergence(p, q):
    """计算KL散度 KL(P||Q)"""
    p = np.array(p)
    q = np.array(q)
    # 避免log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    # 重新归一化
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))

def compute_js_divergence(p, q):
    """计算JS散度 JS(P||Q)"""
    p = np.array(p)
    q = np.array(q)
    # 使用scipy的jensenshannon（返回的是距离，需要平方得到散度）
    js_dist = jensenshannon(p, q)
    return js_dist ** 2

def main():
    print("=" * 80)
    print("Task 23.7: KL and JS Divergence Baseline Comparison")
    print("=" * 80)
    
    # 加载B2数据
    print("\n1. Loading B2 pooled ROC data...")
    b2_path = RESULTS_DIR / 'task_B2_pooled_roc_analysis_corrected.json'
    with open(b2_path, 'r') as f:
        b2_data = json.load(f)
    
    runs = b2_data['all_runs']
    print(f"   Loaded {len(runs)} runs")
    
    # 提取per-run数据
    print("\n2. Extracting per-run data...")
    accuracies = []
    macro_f1s = []
    class_shifts = []
    kl_divs = []
    js_divs = []
    
    # CWRU和JNU的参考先验
    cwru_prior = np.array([0.401, 0.20, 0.20, 0.20])
    jnu_prior = np.array([0.50, 0.167, 0.167, 0.166])
    
    for run in runs:
        accuracies.append(run['accuracy'])
        macro_f1s.append(run['macro_f1'])
        class_shifts.append(run['class_shift'])
        
        # 选择参考先验
        if run['dataset'] == 'CWRU':
            ref_prior = cwru_prior
        else:
            ref_prior = jnu_prior
        
        pred_dist = np.array(run['predicted_distribution'])
        
        # 计算KL散度
        kl = compute_kl_divergence(pred_dist, ref_prior)
        kl_divs.append(kl)
        
        # 计算JS散度
        js = compute_js_divergence(pred_dist, ref_prior)
        js_divs.append(js)
    
    accuracies = np.array(accuracies)
    macro_f1s = np.array(macro_f1s)
    class_shifts = np.array(class_shifts)
    kl_divs = np.array(kl_divs)
    js_divs = np.array(js_divs)
    
    print(f"   Class Shift range: [{class_shifts.min():.3f}, {class_shifts.max():.3f}]")
    print(f"   KL divergence range: [{kl_divs.min():.3f}, {kl_divs.max():.3f}]")
    print(f"   JS divergence range: [{js_divs.min():.3f}, {js_divs.max():.3f}]")
    
    # 复合判据
    print("\n3. Computing composite collapse labels...")
    collapsed = (accuracies < 70) | (macro_f1s < 50)
    print(f"   Total collapsed: {collapsed.sum()} / {len(collapsed)}")
    
    # 计算AUC
    print("\n4. Computing AUC for each signal...")
    
    if len(np.unique(collapsed)) > 1:
        cs_auc = roc_auc_score(collapsed, class_shifts)
        kl_auc = roc_auc_score(collapsed, kl_divs)
        js_auc = roc_auc_score(collapsed, js_divs)
        
        print(f"   Class Shift AUC: {cs_auc:.3f}")
        print(f"   KL divergence AUC: {kl_auc:.3f}")
        print(f"   JS divergence AUC: {js_auc:.3f}")
    else:
        cs_auc = kl_auc = js_auc = None
        print("   Cannot compute AUC (only one class present)")
    
    # 保存结果
    print("\n5. Saving results...")
    output = {
        'task': '23.7',
        'description': 'KL and JS divergence baseline comparison',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'composite_criterion': {
            'accuracy_threshold': 70,
            'macro_f1_threshold': 50
        },
        'n_runs': len(runs),
        'n_collapsed': int(collapsed.sum()),
        'auc_results': {
            'class_shift': float(cs_auc) if cs_auc is not None else None,
            'kl_divergence': float(kl_auc) if kl_auc is not None else None,
            'js_divergence': float(js_auc) if js_auc is not None else None
        },
        'note': 'KL divergence and JS divergence are computed from predicted distributions vs reference prior. Class Shift is L1 distance. TV distance = 0.5 * L1, so TV and Class Shift have identical AUC.'
    }
    
    output_path = RESULTS_DIR / 'task_23_7_kl_js_baseline.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"   Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nSignal comparison (pooled AUC):")
    print(f"   Class Shift: {cs_auc:.3f}" if cs_auc else "   Class Shift: N/A")
    print(f"   KL divergence: {kl_auc:.3f}" if kl_auc else "   KL divergence: N/A")
    print(f"   JS divergence: {js_auc:.3f}" if js_auc else "   JS divergence: N/A")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
