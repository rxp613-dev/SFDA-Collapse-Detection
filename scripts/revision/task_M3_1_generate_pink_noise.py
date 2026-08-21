#!/usr/bin/env python3
"""
Task M3.1: Generate pink noise (1/f) dataset for CWRU at 0dB
Date: 2026-08-10
Objective: Generate pink noise using Voss-McCartney algorithm and create noisy CWRU dataset
Method:
  1. Load clean CWRU 3HP data
  2. Generate pink noise (1/f spectrum) using Voss-McCartney algorithm
  3. Scale noise to achieve 0dB SNR
  4. Add noise to clean signals
  5. Save noisy dataset
"""

import numpy as np
import torch
import json
from pathlib import Path

# Paths
DATA_DIR = Path("/mnt/data/sfda3/data/processed")
RESULTS_DIR = Path("/mnt/data/sfda3/prai2026/paper2/experiments/results/revision")

def generate_pink_noise_voss(length, num_rows=5):
    """
    Generate pink noise using Voss-McCartney algorithm.
    
    Args:
        length: Length of noise signal
        num_rows: Number of rows for averaging (controls pinkness)
    
    Returns:
        Pink noise signal
    """
    # Initialize rows with random values
    rows = [np.random.randn(length) for _ in range(num_rows)]
    
    # Create pink noise by averaging rows with different update rates
    pink_noise = np.zeros(length)
    for i in range(num_rows):
        # Each row updates at a different rate (powers of 2)
        update_rate = 2 ** i
        for j in range(length):
            if j % update_rate == 0:
                pink_noise[j:] += rows[i][j]
    
    # Normalize
    pink_noise = pink_noise / np.std(pink_noise)
    
    return pink_noise

def generate_pink_noise_spectral(length):
    """
    Generate pink noise using spectral method (more accurate).
    
    Args:
        length: Length of noise signal
    
    Returns:
        Pink noise signal with 1/f spectrum
    """
    # Generate white noise
    white_noise = np.random.randn(length)
    
    # Compute FFT
    fft_white = np.fft.rfft(white_noise)
    
    # Create 1/f filter
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1.0  # Avoid division by zero at DC
    
    # Apply 1/f filter (pink noise has 1/f power spectrum)
    filter_1f = 1.0 / np.sqrt(freqs)
    filter_1f[0] = 0.0  # Remove DC component
    
    # Apply filter
    fft_pink = fft_white * filter_1f
    
    # Inverse FFT
    pink_noise = np.fft.irfft(fft_pink, n=length)
    
    # Normalize
    pink_noise = pink_noise / np.std(pink_noise)
    
    return pink_noise

def compute_snr(signal, noise):
    """Compute SNR in dB"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def add_noise_at_snr(clean_signal, noise, target_snr_db):
    """
    Add noise to clean signal at target SNR.
    
    Args:
        clean_signal: Clean signal
        noise: Noise signal (will be scaled)
        target_snr_db: Target SNR in dB
    
    Returns:
        Noisy signal
    """
    # Compute signal power
    signal_power = np.mean(clean_signal ** 2)
    
    # Compute required noise power for target SNR
    target_snr_linear = 10 ** (target_snr_db / 10)
    required_noise_power = signal_power / target_snr_linear
    
    # Scale noise
    current_noise_power = np.mean(noise ** 2)
    noise_scale = np.sqrt(required_noise_power / current_noise_power)
    scaled_noise = noise * noise_scale
    
    # Add noise
    noisy_signal = clean_signal + scaled_noise
    
    # Verify SNR
    actual_snr = compute_snr(clean_signal, scaled_noise)
    
    return noisy_signal, actual_snr

def main():
    print("Task M3.1: Generate pink noise dataset for CWRU at 0dB")
    print("="*80)
    
    # Load clean CWRU 3HP data
    print("\n1. Loading clean CWRU 3HP data...")
    clean_data_path = DATA_DIR / "cwru_3hp.pt"
    clean_data = torch.load(clean_data_path, weights_only=True)
    
    # Extract signals and labels
    if isinstance(clean_data, dict):
        signals = clean_data['samples'].numpy()  # [N, 1, 1024]
        labels = clean_data['labels'].numpy()
    else:
        signals = clean_data[:, 0, :].numpy()  # [N, 1024]
        labels = None
    
    print(f"   Loaded {len(signals)} samples")
    print(f"   Signal shape: {signals.shape}")
    
    # Generate pink noise for each sample
    print("\n2. Generating pink noise...")
    noisy_signals = []
    snr_values = []
    
    for i, signal in enumerate(signals):
        # Generate pink noise
        pink_noise = generate_pink_noise_spectral(len(signal))
        
        # Add noise at 0dB SNR
        noisy_signal, actual_snr = add_noise_at_snr(signal, pink_noise, target_snr_db=0.0)
        
        noisy_signals.append(noisy_signal)
        snr_values.append(actual_snr)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(signals)} samples")
    
    noisy_signals = np.array(noisy_signals)
    snr_values = np.array(snr_values)
    
    print(f"\n3. SNR statistics:")
    print(f"   Mean SNR: {np.mean(snr_values):.2f} dB")
    print(f"   Std SNR: {np.std(snr_values):.2f} dB")
    print(f"   Min SNR: {np.min(snr_values):.2f} dB")
    print(f"   Max SNR: {np.max(snr_values):.2f} dB")
    
    # Save noisy dataset
    print("\n4. Saving noisy dataset...")
    if labels is not None:
        noisy_data = {
            'signals': torch.tensor(noisy_signals, dtype=torch.float32).unsqueeze(1),  # [N, 1, 1024]
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    else:
        noisy_data = torch.tensor(noisy_signals, dtype=torch.float32).unsqueeze(1)
    
    output_path = DATA_DIR / "cwru_3hp_pink_noise_0db.pt"
    torch.save(noisy_data, output_path)
    print(f"   Saved to: {output_path}")
    
    # Save metadata
    metadata = {
        "task": "M3.1",
        "description": "Pink noise dataset generation",
        "noise_type": "pink (1/f)",
        "target_snr_db": 0.0,
        "actual_snr_mean_db": float(np.mean(snr_values)),
        "actual_snr_std_db": float(np.std(snr_values)),
        "num_samples": len(signals),
        "signal_length": len(signals[0]),
        "output_file": str(output_path)
    }
    
    metadata_path = RESULTS_DIR / "task_M3_1_pink_noise_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n5. Metadata saved to: {metadata_path}")
    
    print("\n" + "="*80)
    print("Task M3.1 completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
