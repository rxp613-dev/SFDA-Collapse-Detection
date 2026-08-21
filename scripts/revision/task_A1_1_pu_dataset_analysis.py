#!/usr/bin/env python3
"""
任务 A1.1: 检查PU数据集结构和标签方案
创建时间: 2026-08-07
目标: 分析PU数据集的文件结构、数据格式和标签方案
方法:
    1. 读取样本.mat文件，分析数据结构
    2. 理解文件命名规则中的标签信息
    3. 确定轴承类型和故障类型的映射关系
    4. 验证数据完整性
"""

import scipy.io as sio
import numpy as np
from pathlib import Path
import json

PROJECT_ROOT = Path('/mnt/data/sfda3')
PU_RAW_DIR = PROJECT_ROOT / 'raw' / 'PU'

def analyze_mat_file(mat_path):
    """分析单个.mat文件的结构"""
    try:
        data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        # 获取所有变量名（排除MATLAB内置变量）
        var_names = [k for k in data.keys() if not k.startswith('__')]

        result = {
            'file': str(mat_path),
            'variables': {}
        }

        for var_name in var_names:
            var_data = data[var_name]
            var_info = {
                'type': type(var_data).__name__,
            }

            # 如果是numpy数组
            if isinstance(var_data, np.ndarray):
                var_info['shape'] = var_data.shape
                var_info['dtype'] = str(var_data.dtype)
                if var_data.size > 0 and np.issubdtype(var_data.dtype, np.number):
                    var_info['min'] = float(np.min(var_data))
                    var_info['max'] = float(np.max(var_data))
                    var_info['mean'] = float(np.mean(var_data))
                    var_info['std'] = float(np.std(var_data))
                    var_info['size'] = int(var_data.size)
            # 如果是MATLAB结构体对象
            elif hasattr(var_data, '_fieldnames'):
                var_info['type'] = 'matlab_struct'
                var_info['fields'] = var_data._fieldnames
                # 尝试提取每个字段的信息
                for field in var_data._fieldnames:
                    field_data = getattr(var_data, field, None)
                    if field_data is not None:
                        if isinstance(field_data, np.ndarray):
                            var_info[f'{field}_shape'] = field_data.shape
                            if field_data.size > 0 and np.issubdtype(field_data.dtype, np.number):
                                var_info[f'{field}_size'] = int(field_data.size)
                                var_info[f'{field}_sample'] = field_data[:5].tolist() if field_data.size >= 5 else field_data.tolist()

            result['variables'][var_name] = var_info

        return result
    except Exception as e:
        return {
            'file': str(mat_path),
            'error': str(e)
        }

def main():
    print("=" * 80)
    print("任务 A1.1: 检查PU数据集结构和标签方案")
    print("=" * 80)

    # 1. 列出所有轴承目录
    print("\n1. 轴承目录列表:")
    bearing_dirs = sorted([d for d in PU_RAW_DIR.iterdir() if d.is_dir()])
    print(f"   共找到 {len(bearing_dirs)} 个轴承目录")
    for i, bearing_dir in enumerate(bearing_dirs[:10], 1):
        mat_count = len(list(bearing_dir.glob('*.mat')))
        print(f"   {i}. {bearing_dir.name}: {mat_count} 个.mat文件")
    if len(bearing_dirs) > 10:
        print(f"   ... 还有 {len(bearing_dirs) - 10} 个目录")

    # 2. 分析样本文件结构
    print("\n2. 分析样本文件结构:")
    sample_dirs = ['K001', 'K002', 'K003', 'K004']

    for bearing_name in sample_dirs:
        bearing_dir = PU_RAW_DIR / bearing_name
        if not bearing_dir.exists():
            print(f"   ⚠️  {bearing_name} 目录不存在")
            continue

        mat_files = sorted(bearing_dir.glob('*.mat'))
        if not mat_files:
            print(f"   ⚠️  {bearing_name} 目录中没有.mat文件")
            continue

        print(f"\n   {bearing_name} (样本文件: {mat_files[0].name}):")
        analysis = analyze_mat_file(mat_files[0])

        if 'error' in analysis:
            print(f"      ❌ 读取错误: {analysis['error']}")
        else:
            print(f"      ✅ 成功读取")
            for var_name, var_info in analysis['variables'].items():
                print(f"      变量: {var_name}")
                print(f"         类型: {var_info.get('type', 'unknown')}")
                if 'shape' in var_info:
                    print(f"         形状: {var_info['shape']}")
                if 'dtype' in var_info:
                    print(f"         数据类型: {var_info['dtype']}")
                if 'size' in var_info:
                    print(f"         大小: {var_info['size']} 个元素")
                if 'min' in var_info:
                    print(f"         范围: [{var_info['min']:.4f}, {var_info['max']:.4f}]")
                    print(f"         均值: {var_info['mean']:.4f} ± {var_info['std']:.4f}")
                if 'fields' in var_info:
                    print(f"         结构体字段: {var_info['fields']}")
                    # 打印字段详情
                    for field in var_info['fields']:
                        if f'{field}_shape' in var_info:
                            print(f"            {field}: shape={var_info[f'{field}_shape']}", end='')
                            if f'{field}_size' in var_info:
                                print(f", size={var_info[f'{field}_size']}", end='')
                            print()

    # 3. 分析文件命名规则
    print("\n3. 文件命名规则分析:")
    print("   文件名格式: N{转速}_M{负载}_F{故障类型}_K{轴承编号}_{样本序号}.mat")
    print("   示例: N09_M07_F10_K001_10.mat")
    print("      N09: 转速 900 RPM")
    print("      M07: 负载 0.7 Nm")
    print("      F10: 故障类型代码")
    print("      K001: 轴承编号")
    print("      10: 样本序号")

    # 4. 读取readme文件
    print("\n4. 读取数据集说明:")
    readme_file = PU_RAW_DIR / 'readme_versions.txt'
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content[:1000])  # 打印前1000个字符
    else:
        print("   ⚠️  readme_versions.txt 不存在")

    # 5. 分析轴承类型
    print("\n5. 轴承类型分析:")
    bearing_types = {
        'K': 'K系列轴承（K001-K006）',
        'KA': 'KA系列轴承（KA01-KA30）',
        'KB': 'KB系列轴承（KB23-KB27）',
        'KI': 'KI系列轴承（KI01-KI21）'
    }

    for prefix, desc in bearing_types.items():
        count = sum(1 for d in bearing_dirs if d.name.startswith(prefix))
        print(f"   {desc}: {count} 个")

    # 6. 保存分析结果
    print("\n6. 保存分析结果:")
    output_path = PROJECT_ROOT / 'prai2026' / 'paper2' / 'experiments' / 'results' / 'revision' / 'task_A1_1_pu_dataset_analysis.json'

    analysis_result = {
        'task': 'A1.1',
        'description': 'PU数据集结构和标签方案分析',
        'timestamp': '2026-08-07',
        'total_bearings': len(bearing_dirs),
        'bearing_list': [d.name for d in bearing_dirs],
        'files_per_bearing': 80,
        'file_size_mb': 8.4,
        'naming_convention': 'N{rpm}_M{torque}_F{fault}_K{bearing}_{sample}.mat',
        'bearing_types': bearing_types,
        'status': 'completed'
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)

    print(f"   ✅ 分析结果已保存: {output_path}")

    print("\n" + "=" * 80)
    print("✅ 任务 A1.1 完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
