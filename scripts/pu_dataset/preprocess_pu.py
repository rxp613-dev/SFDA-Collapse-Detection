#!/usr/bin/env python3
"""
PU Dataset Preprocessing for SFDA Experiments
Paderborn University Bearing Dataset
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import resample
from pathlib import Path
import json
from tqdm import tqdm

# PU Dataset structure
PU_CATEGORIES = {
    'Healthy': ['K001', 'K002', 'K003', 'K004', 'K005', 'K006'],
    'Outer_Race': ['KA01', 'KA03', 'KA04', 'KA05', 'KA06', 'KA07',
                   'KA08', 'KA09', 'KA15', 'KA16', 'KA22', 'KA30'],
    'Inner_Race': ['KI01', 'KI03', 'KI04', 'KI05', 'KI07', 'KI08',
                   'KI14', 'KI16', 'KI17', 'KI18', 'KI21'],
    'Ball': ['KB23', 'KB24', 'KB27']
}

# Load conditions
LOAD_CONDITIONS = {
    'N09_M07': {'speed': 900, 'torque': 0.7},    # 900 RPM, 0.7 Nm
    'N15_M01': {'speed': 1500, 'torque': 0.1},   # 1500 RPM, 0.1 Nm
    'N15_M07': {'speed': 1500, 'torque': 0.7}    # 1500 RPM, 0.7 Nm
}

def load_pu_signal(mat_path, target_length=2048):
    """Load vibration signal from PU .mat file"""
    mat = sio.loadmat(mat_path)
    data_key = [k for k in mat.keys() if not k.startswith('__')][0]
    data = mat[data_key]

    # Extract vibration signal (index 6)
    Y = data[0, 0]['Y']
    vib = Y[0, 6]  # vibration_1
    signal = vib['Data'][0]

    # Resample to target length if needed
    if len(signal) > target_length:
        signal = resample(signal, target_length)
    elif len(signal) < target_length:
        # Pad with zeros if too short
        signal = np.pad(signal, (0, target_length - len(signal)), 'constant')

    return signal

def build_pu_dataset(pu_dir, output_dir, target_length=2048, load_condition='N09_M07'):
    """Build PU dataset for SFDA experiments"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_all = []
    y_all = []
    metadata = []

    print(f"Building PU dataset for load condition: {load_condition}")

    for category, bearings in PU_CATEGORIES.items():
        print(f"\nProcessing {category} ({len(bearings)} bearings)...")

        for bearing in bearings:
            bearing_dir = Path(pu_dir) / bearing
            if not bearing_dir.exists():
                print(f"  WARNING: {bearing} not found, skipping...")
                continue

            # Find .mat files for this load condition
            mat_files = [f for f in bearing_dir.glob('*.mat')
                        if f.name.startswith(load_condition)]

            if not mat_files:
                print(f"  WARNING: No files for {load_condition} in {bearing}")
                continue

            print(f"  {bearing}: {len(mat_files)} files")

            for mat_file in mat_files:
                try:
                    signal = load_pu_signal(mat_file, target_length)
                    X_all.append(signal)
                    y_all.append(category)
                    metadata.append({
                        'file': str(mat_file),
                        'bearing': bearing,
                        'category': category,
                        'load_condition': load_condition
                    })
                except Exception as e:
                    print(f"  ERROR loading {mat_file}: {e}")
                    continue

    # Convert to numpy arrays
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all)

    # Normalize each signal (Z-score)
    for i in range(len(X)):
        X[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-8)

    # Save dataset
    np.save(output_dir / f'X_{load_condition}.npy', X)
    np.save(output_dir / f'y_{load_condition}.npy', y)

    # Save metadata
    with open(output_dir / f'metadata_{load_condition}.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print(f"Total samples: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Class distribution:")
    for cat in PU_CATEGORIES.keys():
        count = np.sum(y == cat)
        print(f"  {cat}: {count} ({count/len(y)*100:.1f}%)")

    return X, y

def create_train_test_split(X, y, test_size=0.3, random_state=42):
    """Create train/test split with stratification"""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess PU dataset')
    parser.add_argument('--pu_dir', type=str, default='/mnt/data/sfda3/raw/PU',
                       help='Path to PU dataset')
    parser.add_argument('--output_dir', type=str, default='/mnt/data/sfda3/data/PU',
                       help='Output directory')
    parser.add_argument('--target_length', type=int, default=2048,
                       help='Target signal length')

    args = parser.parse_args()

    # Build dataset for each load condition
    for load_cond in LOAD_CONDITIONS.keys():
        print(f"\n{'='*60}")
        print(f"Processing load condition: {load_cond}")
        print(f"{'='*60}\n")

        X, y = build_pu_dataset(
            args.pu_dir,
            args.output_dir,
            target_length=args.target_length,
            load_condition=load_cond
        )

        # Create train/test split
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)

        # Save splits
        output_dir = Path(args.output_dir)
        np.save(output_dir / f'X_train_{load_cond}.npy', X_train)
        np.save(output_dir / f'X_test_{load_cond}.npy', X_test)
        np.save(output_dir / f'y_train_{load_cond}.npy', y_train)
        np.save(output_dir / f'y_test_{load_cond}.npy', y_test)

        print(f"\nTrain/Test split saved:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
