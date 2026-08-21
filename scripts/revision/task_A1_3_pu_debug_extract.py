#!/usr/bin/env python3
"""
调试PU数据提取
"""

import scipy.io as sio
import numpy as np
from pathlib import Path

PU_RAW_DIR = Path('/mnt/data/sfda3/raw/PU')

# 检查K001的一个样本
sample_file = PU_RAW_DIR / 'K001' / 'N09_M07_F10_K001_1.mat'

print("=" * 80)
print(f"调试数据提取: {sample_file.name}")
print("=" * 80)

# 读取MATLAB文件
data = sio.loadmat(sample_file, squeeze_me=False, struct_as_record=True)
var_name = sample_file.stem
struct_data = data[var_name]

print(f"\nstruct_data shape: {struct_data.shape}")
print(f"struct_data dtype names: {struct_data.dtype.names}")

# 提取X字段
x_field = struct_data['X'][0, 0]
print(f"\nX field shape: {x_field.shape}")
print(f"X field dtype names: {x_field.dtype.names}")

# 遍历所有通道
print(f"\n通道数: {x_field.shape[1]}")
for i in range(x_field.shape[1]):
    print(f"\n通道 {i}:")
    channel = x_field[0, i]
    print(f"  类型: {type(channel)}")
    print(f"  dtype names: {channel.dtype.names}")

    # 提取Data字段
    if 'Data' in channel.dtype.names:
        data_field = channel['Data']
        print(f"  Data field shape: {data_field.shape}")
        print(f"  Data field dtype: {data_field.dtype}")

        # 提取实际数据
        signal_data = data_field[0, 0]
        print(f"  Signal data shape: {signal_data.shape}")
        print(f"  Signal data dtype: {signal_data.dtype}")
        print(f"  Signal data size: {signal_data.size}")

        if signal_data.size > 0:
            print(f"  Signal data min: {signal_data.min()}")
            print(f"  Signal data max: {signal_data.max()}")
            print(f"  Signal data mean: {signal_data.mean()}")
            print(f"  Signal data std: {signal_data.std()}")

            # 尝试flatten
            flattened = signal_data.flatten()
            print(f"  Flattened shape: {flattened.shape}")
            print(f"  Flattened size: {flattened.size}")
