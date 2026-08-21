#!/usr/bin/env python3
"""
调试PU数据提取 - 版本2
"""

import scipy.io as sio
import numpy as np
from pathlib import Path

PU_RAW_DIR = Path('/mnt/data/sfda3/raw/PU')

# 检查K001的一个样本
sample_file = PU_RAW_DIR / 'K001' / 'N09_M07_F10_K001_1.mat'

print("=" * 80)
print(f"调试数据提取 V2: {sample_file.name}")
print("=" * 80)

# 读取MATLAB文件
data = sio.loadmat(sample_file, squeeze_me=False, struct_as_record=True)
var_name = sample_file.stem
struct_data = data[var_name]

# 提取X字段
x_field = struct_data['X'][0, 0]

# 遍历所有通道
print(f"\n通道数: {x_field.shape[1]}")
for i in range(x_field.shape[1]):
    print(f"\n通道 {i}:")
    channel = x_field[0, i]

    # 提取Data字段
    if 'Data' in channel.dtype.names:
        data_field = channel['Data']
        print(f"  Data field shape: {data_field.shape}")
        print(f"  Data field dtype: {data_field.dtype}")

        # 尝试不同的访问方式
        print(f"\n  访问方式 1: data_field[0]")
        try:
            signal1 = data_field[0]
            print(f"    Shape: {signal1.shape}")
            print(f"    Size: {signal1.size}")
            if signal1.size > 0:
                print(f"    Min: {signal1.min()}, Max: {signal1.max()}")
        except Exception as e:
            print(f"    Error: {e}")

        print(f"\n  访问方式 2: data_field.flatten()")
        try:
            signal2 = data_field.flatten()
            print(f"    Shape: {signal2.shape}")
            print(f"    Size: {signal2.size}")
            if signal2.size > 0:
                print(f"    Min: {signal2.min()}, Max: {signal2.max()}")
        except Exception as e:
            print(f"    Error: {e}")

        print(f"\n  访问方式 3: data_field.ravel()")
        try:
            signal3 = data_field.ravel()
            print(f"    Shape: {signal3.shape}")
            print(f"    Size: {signal3.size}")
            if signal3.size > 0:
                print(f"    Min: {signal3.min()}, Max: {signal3.max()}")
        except Exception as e:
            print(f"    Error: {e}")

        print(f"\n  访问方式 4: data_field.reshape(-1)")
        try:
            signal4 = data_field.reshape(-1)
            print(f"    Shape: {signal4.shape}")
            print(f"    Size: {signal4.size}")
            if signal4.size > 0:
                print(f"    Min: {signal4.min()}, Max: {signal4.max()}")
        except Exception as e:
            print(f"    Error: {e}")
