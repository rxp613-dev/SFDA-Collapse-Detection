#!/usr/bin/env python3
"""Quick experiment for 3HP → 0HP migration direction"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path('/mnt/data/sfda3')
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 4
NOISE_SEED = 2026

def load_source_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = BearingFaultBackbone(feature_dim=256).to(device)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES).to(device)
    state_dict = checkpoint['model_state_dict']
    backbone_state = {k[len('backbone.'):]: v for k, v in state_dict.items() if k.startswith('backbone.')}
    classifier_state = {k[len('classifier.'):]: v for k, v in state_dict.items() if k.startswith('classifier.')}
    backbone.load_state_dict(backbone_state)
    classifier.load_state_dict(classifier_state)
    return backbone, classifier

def load_target_data(data_path):
    data_dict = torch.load(data_path, map_location=device)
    return data_dict['samples'], data_dict['labels']

def add_gaussian_noise(data, snr_db):
    if snr_db == float('inf'):
        return data
    signal_power = torch.mean(data ** 2, dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(data) * torch.sqrt(noise_power)
    return data + noise

def compute_metrics(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    accuracy = float((preds == labels).mean() * 100)

    CLASS_NAMES = ['Normal', 'IR', 'Ball', 'OR']
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        confusion_matrix[int(t), int(p)] += 1

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        true_count = int((labels == i).sum())
        pred_count = int((preds == i).sum())
        correct = int(((preds == i) & (labels == i)).sum())
        recall = float(correct / true_count * 100) if true_count > 0 else 0.0
        precision = float(correct / pred_count * 100) if pred_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[name] = {'recall': recall, 'precision': precision, 'f1': f1}

    macro_f1 = float(np.mean([results[n]['f1'] for n in CLASS_NAMES]))
    balanced_acc = float(np.mean([results[n]['recall'] for n in CLASS_NAMES]))
    ir_recall = results['IR']['recall']

    return accuracy, macro_f1, balanced_acc, ir_recall

def run_shot_corrected(backbone, classifier, samples, labels, num_epochs=30, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    bb = deepcopy(backbone).to(device)
    clf = deepcopy(classifier).to(device)

    bb.train()
    for param in bb.parameters():
        param.requires_grad = True
    clf.eval()
    for param in clf.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(bb.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    dataset = TensorDataset(samples, labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    stage1_epochs = num_epochs // 2

    for epoch in range(stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            loss = ent_loss + div_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for epoch in range(num_epochs - stage1_epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            features = bb(batch_x)
            logits, probs = clf(features)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_loss = entropy.mean()
            mean_probs = probs.mean(dim=0)
            diversity = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8))
            div_loss = -diversity
            with torch.no_grad():
                pseudo_labels = probs.argmax(dim=1)
            ce_loss = F.cross_entropy(logits, pseudo_labels)
            loss = ent_loss + div_loss + ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    bb.eval()
    clf.eval()
    with torch.no_grad():
        features = bb(samples.to(device))
        logits, probs = clf(features)
        preds = probs.argmax(dim=1)
        accuracy, macro_f1, balanced_acc, ir_recall = compute_metrics(preds, labels)

    return accuracy, macro_f1, balanced_acc, ir_recall

# Main
print("Testing 3HP → 0HP migration at 0dB")
source_backbone, source_classifier = load_source_model(PROJECT_ROOT / 'data/checkpoints/source_pretrain_3hp.pt')
target_samples, target_labels = load_target_data(PROJECT_ROOT / 'data/processed/cwru_0hp.pt')

torch.manual_seed(NOISE_SEED)
np.random.seed(NOISE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(NOISE_SEED)
    torch.cuda.manual_seed_all(NOISE_SEED)

noisy_samples = add_gaussian_noise(target_samples, 0)

seeds = [42, 43, 44]
accuracies, macro_f1s, balanced_accs, ir_recalls = [], [], [], []

for seed in seeds:
    acc, mf1, bacc, ir = run_shot_corrected(
        source_backbone, source_classifier,
        noisy_samples, target_labels,
        num_epochs=30, lr=1e-3, seed=seed
    )
    accuracies.append(acc)
    macro_f1s.append(mf1)
    balanced_accs.append(bacc)
    ir_recalls.append(ir)
    print(f"  Seed {seed}: Acc={acc:.2f}%, Macro-F1={mf1:.2f}%, BalAcc={bacc:.2f}%, IR={ir:.2f}%")

print(f"\nResults:")
print(f"  Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}%")
print(f"  Macro-F1: {np.mean(macro_f1s):.2f} ± {np.std(macro_f1s):.2f}%")
print(f"  Balanced Acc: {np.mean(balanced_accs):.2f} ± {np.std(balanced_accs):.2f}%")
print(f"  IR Recall: {np.mean(ir_recalls):.2f} ± {np.std(ir_recalls):.2f}%")
