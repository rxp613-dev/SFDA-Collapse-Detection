#!/usr/bin/env python3
"""
Task: Rebuild task_3_1 JSON from Table 1
Created: 2026-08-04
Purpose: 从Table 1 (table1_main_gradient.tex) 反向重建 task_3_1 JSON文件
Method: 解析LaTeX表格，提取所有数值，生成标准JSON格式
Note: 该JSON文件原本损坏(0字节)，需要从已发表的表格数据恢复
"""

import json
import re
from pathlib import Path
from datetime import datetime

# 输出路径
OUTPUT_DIR = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")
OUTPUT_FILE = OUTPUT_DIR / "task_3_1_snr_comparison_label_free.json"

# Table 1 数据 (从 table1_main_gradient.tex 提取)
# 格式: method -> {snr_level: {acc_mean, acc_std, ir_mean, ir_std}}
table1_data = {
    "SHOT_original": {
        "Clean": {"acc_mean": 99.93, "acc_std": 0.05, "ir_mean": 100.00, "ir_std": 0.00},
        "6dB": {"acc_mean": 99.61, "acc_std": 0.41, "ir_mean": 99.28, "ir_std": 1.38},
        "3dB": {"acc_mean": 98.19, "acc_std": 0.82, "ir_mean": 96.31, "ir_std": 2.46},
        "0dB": {"acc_mean": 58.38, "acc_std": 1.12, "ir_mean": 0.04, "ir_std": 0.13},
        "-3dB": {"acc_mean": 57.85, "acc_std": 1.42, "ir_mean": 0.21, "ir_std": 0.28},
        "-6dB": {"acc_mean": 60.16, "acc_std": 1.64, "ir_mean": 3.52, "ir_std": 2.67},
    },
    "TENT": {
        "Clean": {"acc_mean": 97.79, "acc_std": 0.46, "ir_mean": 100.00, "ir_std": 0.00},
        "6dB": {"acc_mean": 96.68, "acc_std": 0.19, "ir_mean": 96.02, "ir_std": 1.84},
        "3dB": {"acc_mean": 96.26, "acc_std": 0.36, "ir_mean": 91.65, "ir_std": 1.48},
        "0dB": {"acc_mean": 89.93, "acc_std": 2.30, "ir_mean": 35.72, "ir_std": 20.38},
        "-3dB": {"acc_mean": 85.62, "acc_std": 0.28, "ir_mean": 1.36, "ir_std": 3.31},
        "-6dB": {"acc_mean": 83.96, "acc_std": 0.11, "ir_mean": 0.38, "ir_std": 0.13},
    },
    "NRC": {
        "Clean": {"acc_mean": 40.01, "acc_std": 26.21, "ir_mean": 30.00, "ir_std": 45.83},
        "6dB": {"acc_mean": 60.04, "acc_std": 16.68, "ir_mean": 10.00, "ir_std": 30.00},
        "3dB": {"acc_mean": 47.16, "acc_std": 22.21, "ir_mean": 10.00, "ir_std": 30.00},
        "0dB": {"acc_mean": 60.04, "acc_std": 16.66, "ir_mean": 30.00, "ir_std": 45.83},
        "-3dB": {"acc_mean": 48.58, "acc_std": 23.24, "ir_mean": 20.00, "ir_std": 40.00},
        "-6dB": {"acc_mean": 37.14, "acc_std": 23.22, "ir_mean": 30.00, "ir_std": 45.83},
    },
    "SAR": {
        "Clean": {"acc_mean": 85.75, "acc_std": 0.00, "ir_mean": 0.00, "ir_std": 0.00},
        "6dB": {"acc_mean": 82.08, "acc_std": 7.73, "ir_mean": 68.09, "ir_std": 44.51},
        "3dB": {"acc_mean": 39.84, "acc_std": 22.89, "ir_mean": 99.41, "ir_std": 0.74},
        "0dB": {"acc_mean": 25.55, "acc_std": 5.65, "ir_mean": 99.53, "ir_std": 0.58},
        "-3dB": {"acc_mean": 14.25, "acc_std": 0.00, "ir_mean": 100.00, "ir_std": 0.00},
        "-6dB": {"acc_mean": 14.25, "acc_std": 0.00, "ir_mean": 100.00, "ir_std": 0.00},
    },
    "RPSWD_unfrozen": {
        "Clean": {"acc_mean": 91.00, "acc_std": 6.60, "ir_mean": 97.08, "ir_std": 8.77},
        "6dB": {"acc_mean": 88.51, "acc_std": 8.47, "ir_mean": 70.04, "ir_std": 45.76},
        "3dB": {"acc_mean": 91.27, "acc_std": 6.84, "ir_mean": 89.96, "ir_std": 29.99},
        "0dB": {"acc_mean": 86.88, "acc_std": 3.69, "ir_mean": 49.11, "ir_std": 49.15},
        "-3dB": {"acc_mean": 84.96, "acc_std": 5.74, "ir_mean": 59.62, "ir_std": 48.11},
        "-6dB": {"acc_mean": 82.20, "acc_std": 3.94, "ir_mean": 27.63, "ir_std": 26.16},
    },
}

# SNR levels in order
snr_levels = ["Clean", "6dB", "3dB", "0dB", "-3dB", "-6dB"]

def build_json():
    """构建标准JSON格式"""
    json_data = {
        "task": "3-1",
        "description": "Main gradient audit: 5 methods × 6 SNR levels",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Rebuilt from Table 1 (table1_main_gradient.tex) on 2026-08-04. Original JSON was corrupted.",
        "source": "table1_main_gradient.tex",
        "seeds": "42-51 (10 seeds)",
        "snr_levels": {}
    }

    for snr in snr_levels:
        snr_key = snr  # Keep as-is (Clean, 6dB, etc.)
        json_data["snr_levels"][snr_key] = {
            "methods": {}
        }

        for method, data in table1_data.items():
            snr_data = data[snr]
            json_data["snr_levels"][snr_key]["methods"][method] = {
                "mean_accuracy": snr_data["acc_mean"],
                "std_accuracy": snr_data["acc_std"],
                "mean_ir_recall": snr_data["ir_mean"],
                "std_ir_recall": snr_data["ir_std"],
                "n_seeds": 10
            }

    return json_data

def main():
    print("="*80)
    print("Task: Rebuild task_3_1 JSON from Table 1")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {OUTPUT_FILE}")

    # Build JSON
    json_data = build_json()

    # Verify structure
    print(f"\nJSON structure:")
    print(f"  Tasks: {json_data['task']}")
    print(f"  SNR levels: {list(json_data['snr_levels'].keys())}")
    print(f"  Methods: {list(json_data['snr_levels']['Clean']['methods'].keys())}")

    # Print key values for verification
    print(f"\nKey values (0dB):")
    for method in json_data['snr_levels']['0dB']['methods']:
        data = json_data['snr_levels']['0dB']['methods'][method]
        print(f"  {method}: Acc={data['mean_accuracy']:.2f}±{data['std_accuracy']:.2f}%, IR={data['mean_ir_recall']:.2f}±{data['std_ir_recall']:.2f}%")

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✓ JSON saved to: {OUTPUT_FILE}")
    print(f"✓ File size: {OUTPUT_FILE.stat().st_size} bytes")
    print(f"✓ Task completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
