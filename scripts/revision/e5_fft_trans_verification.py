#!/usr/bin/env python3
"""
E5: FFT-Trans Zero-Variance Verification
Created: 2026-08-15
Purpose: Debug why FFT-Trans produces identical results (71.44%±0.00%) across all 10 seeds
Input: Source model (0HP pretrained), target domain (3HP, 0dB SNR)
Output: Detailed analysis of FFT-Trans adaptation behavior
Method: Run 10 seeds with detailed logging of adaptation trajectory
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from copy import deepcopy
from datetime import datetime

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier
from src.data.loaders import load_cwru_data
from src.utils.noise import add_awgn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

RESULTS_DIR = PROJECT_ROOT / 'prai2026/paper2/experiments/results/revision'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SEEDS = list(range(42, 52))  # 10 seeds
SNR_DB = 0.0
NOISE_SEED = 2026
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4


class FFTTransformer(nn.Module):
    """Simplified FFT-Trans implementation"""
    def __init__(self, input_dim=1024, num_classes=4):
        super().__init__()
        self.fft_dim = input_dim // 2 + 1
        self.transformer = nn.TransformerEncoderLayer(
            d_model=self.fft_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.fft_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Apply FFT
        x_fft = torch.fft.rfft(x.squeeze(1), dim=-1)
        x_fft = torch.abs(x_fft)
        
        # Transformer encoding
        x_encoded = self.transformer(x_fft.unsqueeze(1))
        
        # Classification
        logits = self.classifier(x_encoded.squeeze(1))
        return logits


def evaluate_model(model, dataloader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)
            outputs = model(samples)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total


def run_fft_trans_detailed(seed):
    """Run FFT-Trans with detailed logging"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n  Seed {seed}:", flush=True)
    
    # Load data
    source_data = load_cwru_data('0HP', device=device)
    target_data = load_cwru_data('3HP', device=device)
    
    # Add noise to target
    target_noisy = add_awgn(target_data['samples'], SNR_DB, NOISE_SEED + seed)
    
    # Create dataloaders
    source_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(source_data['samples'], source_data['labels']),
        batch_size=BATCH_SIZE, shuffle=True
    )
    target_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(target_noisy, target_data['labels']),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Initialize model
    model = FFTTransformer(num_classes=4).to(device)
    
    # Pretrain on source (50 epochs)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print(f"    Pretraining...", flush=True)
    for epoch in range(50):
        model.train()
        epoch_loss = 0.0
        for samples, labels in source_loader:
            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            src_acc = evaluate_model(model, source_loader)
            print(f"      Epoch {epoch+1}: loss={epoch_loss/len(source_loader):.4f}, src_acc={src_acc:.2f}%", flush=True)
    
    # Adapt on target (30 epochs) with detailed logging
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"    Adapting...", flush=True)
    trajectory = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_entropy = 0.0
        
        for samples, _ in target_loader:
            optimizer.zero_grad()
            outputs = model(samples)
            
            # Entropy minimization
            probs = torch.softmax(outputs, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            entropy.backward()
            optimizer.step()
            epoch_entropy += entropy.item()
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            tgt_acc = evaluate_model(model, target_loader)
            avg_entropy = epoch_entropy / len(target_loader)
            
            # Compute prediction entropy distribution
            model.eval()
            all_probs = []
            with torch.no_grad():
                for samples, _ in target_loader:
                    samples = samples.to(device)
                    outputs = model(samples)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy())
            
            all_probs = np.concatenate(all_probs, axis=0)
            pred_entropy = -(all_probs * np.log(all_probs + 1e-8)).sum(axis=1)
            
            trajectory.append({
                'epoch': epoch + 1,
                'accuracy': float(tgt_acc),
                'avg_entropy': float(avg_entropy),
                'mean_pred_entropy': float(np.mean(pred_entropy)),
                'std_pred_entropy': float(np.std(pred_entropy)),
                'max_confidence': float(np.max(all_probs)),
                'min_confidence': float(np.min(all_probs)),
            })
            
            print(f"      Epoch {epoch+1}: acc={tgt_acc:.2f}%, entropy={avg_entropy:.4f}, mean_pred_ent={np.mean(pred_entropy):.4f}", flush=True)
    
    # Final evaluation
    final_acc = evaluate_model(model, target_loader)
    print(f"    Final accuracy: {final_acc:.2f}%", flush=True)
    
    return {
        'final_accuracy': float(final_acc),
        'trajectory': trajectory,
    }


def main():
    print("=" * 80, flush=True)
    print("E5: FFT-Trans Zero-Variance Verification")
    print("=" * 80, flush=True)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    results = {}
    
    for seed in SEEDS:
        result = run_fft_trans_detailed(seed)
        results[f"seed_{seed}"] = result
    
    # Analyze variance
    accuracies = [results[f"seed_{seed}"]['final_accuracy'] for seed in SEEDS]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\n{'=' * 80}", flush=True)
    print(f"Summary: {mean_acc:.2f}% ± {std_acc:.2f}%", flush=True)
    print(f"Range: [{min(accuracies):.2f}%, {max(accuracies):.2f}%]", flush=True)
    
    # Save results
    output = {
        'task': 'E5: FFT-Trans Zero-Variance Verification',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'seeds': SEEDS,
            'snr_db': SNR_DB,
            'noise_seed': NOISE_SEED,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
        },
        'results': results,
        'summary': {
            'mean': float(mean_acc),
            'std': float(std_acc),
            'min': float(min(accuracies)),
            'max': float(max(accuracies)),
        },
    }
    
    output_path = RESULTS_DIR / 'e5_fft_trans_verification.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'=' * 80}", flush=True)
    print(f"Results saved to: {output_path}", flush=True)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
