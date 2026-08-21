#!/usr/bin/env python3
"""
Phase 5.2: 准备补充材料
Created: 2026-08-05
Purpose: 准备论文的补充材料，包括实验细节、额外结果、验证报告
Method:
  1. 收集所有实验的详细信息
  2. 生成补充材料文档
  3. 整理实验配置和参数
  4. 准备数据可用性声明

Output:
  - docs/supplementary/supplementary_materials.md
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path('/mnt/data/sfda3')
RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
SUPP_DIR = PROJECT_ROOT / 'docs/supplementary'
SUPP_DIR.mkdir(parents=True, exist_ok=True)

def load_json(filename):
    """Load JSON file"""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_supplementary():
    """Generate supplementary materials document"""
    print("=" * 80)
    print("Phase 5.2: Preparing Supplementary Materials")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines = [
        "# Supplementary Materials",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## A. Experimental Configuration",
        "",
        "### A.1 Dataset Details",
        "",
        "- **Source**: CWRU bearing dataset",
        "- **Source domain**: 0 HP load",
        "- **Target domain**: 3 HP load",
        "- **Classes**: Normal (57.19%), IR (14.25%), Ball (14.25%), OR (14.31%)",
        "- **Total samples**: 1,656",
        "- **Feature extraction**: 1D-CNN backbone (265,764 parameters, 256-dim features)",
        "",
        "### A.2 Methods and Hyperparameters",
        "",
        "| Method | Learning Rate | Optimizer | Batch Size | Epochs |",
        "|--------|--------------|-----------|------------|--------|",
        "| SHOT (original) | 1e-3 | SGD | 128 | 100 |",
        "| SHOT (lr=1e-4) | 1e-4 | SGD | 128 | 100 |",
        "| TENT | 1e-3 | Adam | 128 | 100 |",
        "| NRC | 1e-3 | Adam | 128 | 100 |",
        "| SAR | 1e-3 | Adam | 128 | 100 |",
        "| RPSWD | 1e-4 | Adam | 128 | 100 |",
        "",
        "### A.3 Noise Configuration",
        "",
        "- **SNR levels**: Clean, +6dB, +3dB, 0dB, -3dB, -6dB",
        "- **Noise types**: AWGN, Pink, Brown, Blue, Impulsive",
        "- **Noise injection**: After per-sample z-score normalization",
        "- **SNR definition**: Power ratio",
        "",
        "---",
        "",
        "## B. Detailed Results",
        "",
        "### B.1 Phase 0.3: Colored Noise (Golden Pipeline)",
        "",
    ]

    # Load Phase 0.3 data
    data = load_json('task_phase0_3_colored_noise_golden.json')
    if data:
        for noise in ['awgn', 'pink', 'brown', 'blue']:
            lines.append(f"**{noise.upper()} noise @ 0dB:**")
            for method in ['SHOT_lr1e-3', 'SHOT_lr1e-4', 'RPSWD']:
                accs = [v['accuracy'] for v in data['results'][noise][method].values()]
                irs = [v['ir_recall'] for v in data['results'][noise][method].values()]
                lines.append(f"- {method}: Acc={np.mean(accs):.2f}±{np.std(accs):.2f}%, IR={np.mean(irs):.2f}±{np.std(irs):.2f}%")
            lines.append("")

    lines.extend([
        "### B.2 Phase 1.1: LR×SNR Stability",
        "",
    ])

    # Load Phase 1.1 data
    data = load_json('task_phase1_1_lr_snr_stability.json')
    if data:
        for snr in ['0dB', '-3dB']:
            lines.append(f"**{snr}:**")
            for lr in ['lr=1e-02', 'lr=1e-03', 'lr=1e-04', 'lr=1e-05']:
                for method in ['SHOT', 'TENT']:
                    if lr in data['results'][snr]:
                        accs = [v['accuracy'] for v in data['results'][snr][lr][method].values()]
                        irs = [v['ir_recall'] for v in data['results'][snr][lr][method].values()]
                        lines.append(f"- {method} {lr}: Acc={np.mean(accs):.2f}±{np.std(accs):.2f}%, IR={np.mean(irs):.2f}±{np.std(irs):.2f}%")
            lines.append("")

    lines.extend([
        "### B.3 Phase 2.1: OR Recall Bimodality",
        "",
    ])

    # Load expC data
    data = load_json('task_expC_rpswd_or_bimodality.json')
    if data:
        lines.append("| Seed | Normal | IR | Ball | OR |")
        lines.append("|------|--------|-----|------|-----|")
        for seed_key in sorted(data['results'].keys(), key=lambda x: int(x.split('_')[1])):
            r = data['results'][seed_key]['recalls']
            lines.append(f"| {seed_key} | {r['Normal']:.0f}% | {r['IR']:.0f}% | {r['Ball']:.0f}% | **{r['OR']:.0f}%** |")
        lines.append("")
        lines.append(f"**Mahalanobis distances**: OR vs IR = {data['or_distances']['IR']:.2f} (minimum)")
        lines.append("")

    lines.extend([
        "### B.4 Phase 2.2: Majority Voting",
        "",
        "Majority voting across seeds with identical architecture **cannot** resolve OR bimodality:",
        "- 5 seeds learn OR but lose IR",
        "- 2 seeds learn IR but lose OR",
        "- 3 seeds lose both",
        "- Seeds converge to different optimization basins and are not independent",
        "",
        "### B.5 Phase 2.3: TENT Instability",
        "",
        "TENT IR recall distribution at 0dB:",
        "- lr=1e-2: 29.5% ± 45.1% (bimodal coefficient: 1.00)",
        "- lr=1e-3: 63.6% ± 25.5% (bimodal coefficient: 0.30)",
        "- lr=1e-4: 59.7% ± 2.4% (bimodal coefficient: 0.00)",
        "- lr=1e-5: 57.6% ± 0.4% (bimodal coefficient: 0.00)",
        "",
        "Root cause: Entropy minimization in noisy conditions creates rugged landscape with multiple local minima.",
        "",
        "---",
        "",
        "## C. Code and Data Availability",
        "",
        "All experimental scripts and results are available at:",
        "- Scripts: `scripts/revision/`",
        "- Results: `prai2026/paper2/experiments/results/revision/`",
        "- Analysis reports: `docs/analysis/`",
        "",
        "---",
        "",
        f"**Document generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    output_path = SUPP_DIR / 'supplementary_materials.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✓ Generated {output_path}")
    print(f"\n{'=' * 80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    generate_supplementary()
