#!/usr/bin/env python3
"""
深度检查PU数据的X和Y字段内容
"""

import scipy.io as sio
import numpy as np
from pathlib import Path

PU_RAW_DIR = Path('/mnt/data/sfda3/raw/PU')

def inspect_xy_fields(obj, indent=0):
    """检查X和Y字段的内容"""
    prefix = "  " * indent

    if hasattr(obj, '_fieldnames'):
        for field in obj._fieldnames:
            field_val = getattr(obj, field, None)
            if field_val is not None:
                print(f"{prefix}{field}:")

                if field in ['X', 'Y'] and isinstance(field_val, np.ndarray):
                    print(f"{prefix}  shape: {field_val.shape}, dtype: {field_val.dtype}")

                    # 检查每个元素
                    for i, elem in enumerate(field_val):
                        print(f"{prefix}  [{i}]:")
                        if isinstance(elem, np.ndarray):
                            print(f"{prefix}    ndarray: shape={elem.shape}, dtype={elem.dtype}")
                            if elem.size > 0:
                                if np.issubdtype(elem.dtype, np.number):
                                    print(f"{prefix}    min={elem.min():.6f}, max={elem.max():.6f}, mean={elem.mean():.6f}")
                                    if elem.size <= 10:
                                        print(f"{prefix}    values: {elem}")
                                    else:
                                        print(f"{prefix}    first 5: {elem[:5]}")
                                        print(f"{prefix}    last 5: {elem[-5:]}")
                        else:
                            print(f"{prefix}    {type(elem).__name__}: {elem}")

                elif field == 'Description':
                    inspect_xy_fields(field_val, indent + 1)
                elif field == 'Info':
                    inspect_xy_fields(field_val, indent + 1)
                else:
                    if isinstance(field_val, np.ndarray):
                        print(f"{prefix}  ndarray: shape={field_val.shape}, dtype={field_val.dtype}")
                    else:
                        print(f"{prefix}  {type(field_val).__name__}")

def main():
    # 检查K001的一个样本
    sample_file = PU_RAW_DIR / 'K001' / 'N09_M07_F10_K001_1.mat'

    print("=" * 80)
    print(f"检查X和Y字段: {sample_file.name}")
    print("=" * 80)

    data = sio.loadmat(sample_file, squeeze_me=True, struct_as_record=False)
    var_name = list(data.keys())[0]
    struct_data = data[var_name]

    inspect_xy_fields(struct_data)

    # 检查Measurement.Length字段，这应该包含信号长度信息
    print("\n" + "=" * 80)
    print("检查Description.Measurement.Length:")
    print("=" * 80)

    if hasattr(struct_data, 'Description') and hasattr(struct_data.Description, 'Measurement'):
        measurement = struct_data.Description.Measurement
        if hasattr(measurement, 'Length'):
            length_info = measurement.Length
            print(f"Length type: {type(length_info)}")
            if isinstance(length_info, np.ndarray):
                print(f"Length shape: {length_info.shape}")
                print(f"Length values: {length_info}")
            else:
                print(f"Length value: {length_info}")

    # 检查Info字段
    print("\n" + "=" * 80)
    print("检查Info字段:")
    print("=" * 80)

    if hasattr(struct_data, 'Info'):
        info = struct_data.Info
        for field in info._fieldnames:
            field_val = getattr(info, field, None)
            print(f"{field}: {field_val}")

if __name__ == '__main__':
    main()
