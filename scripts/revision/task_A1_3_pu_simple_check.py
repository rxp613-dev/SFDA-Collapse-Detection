#!/usr/bin/env python3
"""
简单检查PU数据结构
"""

import scipy.io as sio
import numpy as np
from pathlib import Path

PU_RAW_DIR = Path('/mnt/data/sfda3/raw/PU')

# 检查K001的一个样本
sample_file = PU_RAW_DIR / 'K001' / 'N09_M07_F10_K001_1.mat'

print("=" * 80)
print(f"文件: {sample_file}")
print("=" * 80)

# 使用不同的加载方式
data = sio.loadmat(sample_file, squeeze_me=False, struct_as_record=True)

print("\n变量名:")
for key in data.keys():
    if not key.startswith('__'):
        print(f"  {key}: {type(data[key])}")

# 获取数据变量
var_name = [k for k in data.keys() if not k.startswith('__')][0]
struct_data = data[var_name]

print(f"\n变量 '{var_name}':")
print(f"  类型: {type(struct_data)}")
print(f"  形状: {struct_data.shape if hasattr(struct_data, 'shape') else 'N/A'}")

# 如果是structured array，检查dtype
if hasattr(struct_data, 'dtype'):
    print(f"  dtype: {struct_data.dtype}")
    print(f"  dtype names: {struct_data.dtype.names}")

    # 尝试访问每个字段
    if struct_data.dtype.names:
        for field_name in struct_data.dtype.names:
            field_data = struct_data[field_name][0, 0]
            print(f"\n  字段 '{field_name}':")
            print(f"    类型: {type(field_data)}")

            if isinstance(field_data, np.ndarray):
                print(f"    形状: {field_data.shape}")
                print(f"    dtype: {field_data.dtype}")

                # 如果是嵌套的structured array
                if field_data.dtype.names:
                    print(f"    嵌套字段: {field_data.dtype.names}")
                    for nested_name in field_data.dtype.names:
                        nested_data = field_data[nested_name][0, 0]
                        print(f"      {nested_name}: {type(nested_data)}")
                        if isinstance(nested_data, np.ndarray):
                            print(f"        形状: {nested_data.shape}")
                            if nested_data.size > 0 and nested_data.size <= 20:
                                print(f"        值: {nested_data}")
                else:
                    # 普通数组
                    if field_data.size > 0:
                        print(f"    大小: {field_data.size}")
                        if field_data.size <= 10:
                            print(f"    值: {field_data}")
                        else:
                            print(f"    前5个值: {field_data[:5]}")
                            print(f"    后5个值: {field_data[-5:]}")
