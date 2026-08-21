#!/usr/bin/env python3
"""
fix_tab4_signal_guide.py
========================
修复Tab.IV (tab4_signal_guide.tex) 的Sens/Spec数据

作者: Claude
日期: 2026-08-09
目标: 从P4 step9 JSON中提取每个方法的最佳信号在0dB和-3dB的平均Sens/Spec

数据源:
  task_P4_step9_calibration_thresholds.json - 包含每个方法每个信号的测试集性能
  task_P4_step10_calibration_update_report.json - 包含best_signals映射

逻辑:
  1. 从step10获取每个方法的最佳信号 (best_signals)
  2. 从step9获取该最佳信号在0dB和-3dB的sensitivity/specificity
  3. 计算两个SNR的平均值填入表格

输出:
  /mnt/data/sfda3/figs/tab4_signal_guide.tex
"""

import json
from pathlib import Path

BASE = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")
OUTPUT = Path("/mnt/data/sfda3/figs/tab4_signal_guide.tex")

def main():
    step9 = json.load(open(BASE / "task_P4_step9_calibration_thresholds.json"))
    step10 = json.load(open(BASE / "task_P4_step10_calibration_update_report.json"))

    best_signals = step10["best_signals"]

    print("=" * 80)
    print("修复Tab.IV - 信号选择指南")
    print("=" * 80)
    print()

    rows = []
    for method in ["SHOT", "TENT", "RPSWD"]:
        best_sig = best_signals[method]

        # 从step9提取该最佳信号在0dB和-3dB的性能
        test_results = step9["test_results"][method]

        sens_vals = []
        spec_vals = []
        for snr in ["0dB", "-3dB"]:
            if snr in test_results and best_sig in test_results[snr]:
                sig_data = test_results[snr][best_sig]
                sens_vals.append(sig_data["sensitivity"])
                spec_vals.append(sig_data["specificity"])
                print(f"  {method}/{best_sig} @ {snr}: Sens={sig_data['sensitivity']:.3f}, Spec={sig_data['specificity']:.3f}")

        avg_sens = sum(sens_vals) / len(sens_vals) if sens_vals else 0
        avg_spec = sum(spec_vals) / len(spec_vals) if spec_vals else 0
        print(f"  {method}/{best_sig} average: Sens={avg_sens:.3f}, Spec={avg_spec:.3f}")
        print()

        # Note列
        notes = {
            "SHOT": "Class Shift AUC=0 on CWRU",
            "TENT": "Best on JNU calibrated",
            "RPSWD": "Calibration sample-size sensitive",
        }

        rows.append((method, best_sig.replace("_", " ").title(), f"{avg_sens:.3f}", f"{avg_spec:.3f}", notes[method]))

    # 生成LaTeX表格
    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\caption{Method-specific monitoring signal selection guide (JNU calibrated test).}")
    lines.append(r"\label{tab:signal}")
    lines.append(r"\begin{tabular}{llrrl}")
    lines.append(r"\toprule")
    lines.append(r"Method & Best signal & Sens & Spec & Note \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 已生成: {OUTPUT}")
    print()
    print("验证:")
    print(OUTPUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
