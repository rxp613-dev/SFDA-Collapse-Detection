#!/usr/bin/env python3
"""
诊断PU数据结构
"""

import scipy.io as sio
import numpy as np
from pathlib import Path

PU_RAW_DIR = Path('/mnt/data/sfda3/raw/PU')

def deep_inspect(obj, indent=0, max_depth=5):
    """深度检查对象结构"""
    prefix = "  " * indent

    if indent > max_depth:
        print(f"{prefix}... (max depth reached)")
        return

    if isinstance(obj, np.ndarray):
        print(f"{prefix}ndarray: shape={obj.shape}, dtype={obj.dtype}")
        if obj.size > 0 and obj.size <= 20 and np.issubdtype(obj.dtype, np.number):
            print(f"{prefix}  values: {obj}")
        elif obj.size > 20:
            print(f"{prefix}  size: {obj.size}")
    elif hasattr(obj, '_fieldnames'):
        print(f"{prefix}matlab_struct with fields: {obj._fieldnames}")
        for field in obj._fieldnames:
            print(f"{prefix}  field '{field}':")
            field_val = getattr(obj, field, None)
            if field_val is not None:
                deep_inspect(field_val, indent + 2, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__}: len={len(obj)}")
        if len(obj) > 0 and len(obj) <= 5:
            for i, item in enumerate(obj):
                print(f"{prefix}  [{i}]:")
                deep_inspect(item, indent + 2, max_depth)
    elif isinstance(obj, dict):
        print(f"{prefix}dict: keys={list(obj.keys())}")
        for key, val in list(obj.items())[:5]:
            print(f"{prefix}  '{key}':")
            deep_inspect(val, indent + 2, max_depth)
    else:
        print(f"{prefix}{type(obj).__name__}: {obj}")

def main():
    # 检查一个样本文件
    sample_file = PU_RAW_DIR / 'K001' / 'N09_M07_F10_K001_1.mat'

    print("=" * 80)
    print(f"深度检查: {sample_file}")
    print("=" * 80)

    data = sio.loadmat(sample_file, squeeze_me=True, struct_as_record=False)

    print("\n顶层变量:")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"\n变量 '{key}':")
            deep_inspect(data[key], indent=1, max_depth=5)

    # 检查另一个文件，看看是否有不同的结构
    print("\n" + "=" * 80)
    sample_file2 = PU_RAW_DIR / 'K001' / 'N09_M07_F10_K001_10.mat'
    print(f"深度检查: {sample_file2}")
    print("=" * 80)

    data2 = sio.loadmat(sample_file2, squeeze_me=True, struct_as_record=False)

    print("\n顶层变量:")
    for key in data2.keys():
        if not key.startswith('__'):
            print(f"\n变量 '{key}':")
            deep_inspect(data2[key], indent=1, max_depth=5)

    # 检查不同轴承类型的文件
    print("\n" + "=" * 80)
    sample_file3 = PU_RAW_DIR / 'KA01' / 'N09_M07_F10_KA01_1.mat'
    if sample_file3.exists():
        print(f"深度检查: {sample_file3}")
        print("=" * 80)

        data3 = sio.loadmat(sample_file3, squeeze_me=True, struct_as_record=False)

        print("\n顶层变量:")
        for key in data3.keys():
            if not key.startswith('__'):
                print(f"\n变量 '{key}':")
                deep_inspect(data3[key], indent=1, max_depth=5)

if __name__ == '__main__':
    main()
