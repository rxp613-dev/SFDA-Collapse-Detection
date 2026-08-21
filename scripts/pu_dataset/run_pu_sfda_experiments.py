#!/usr/bin/env python3
"""
PU Dataset SFDA Experiments for IEEE Access M7
Cross-load, cross-speed, and cross-dataset experiments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/mnt/data/sfda3/results/pu_experiments')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PU_DATA_DIR = Path('/mnt/data/sfda3/data/PU')
CLASS_NAMES = ['Healthy', 'Outer_Race', 'Inner_Race', 'Ball']
NUM_CLASSES = 4
NUM_EPOCHS = 50
BATCH_SIZE = 128

# 30 seeds for statistical significance
SEEDS = list(range(42, 72))  # 30 seeds [42-71]
METHODS = ['Source_Only', 'SHOT', 'TENT', 'NRC', 'SAR']

print("=" * 80)
print("PU Dataset SFDA Experiments (M7)")
print("=" * 80)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"Methods: {METHODS}")
print(f"Seeds: {len(SEEDS)} seeds")
print(f"Total experiments: {len(METHODS)} x {len(SEEDS)} x 3 scenarios")


def load_pu_data(load_condition, split='train'):
    """Load PU dataset for a specific load condition"""
    X = np.load(PU_DATA_DIR / f'X_{split}_{load_condition}.npy')
    y = np.load(PU_DATA_DIR / f'y_{split}_{load_condition}.npy', allow_pickle=True)

    # Convert string labels to integers
    label_map = {'Healthy': 0, 'Outer_Race': 1, 'Inner_Race': 2, 'Ball': 3}
    y_int = np.array([label_map[label] for label in y])

    # Convert to torch tensors
    X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension [N, 1, 2048]
    y = torch.LongTensor(y_int)

    return X, y


def create_model():
    """Create a fresh model"""
    backbone = BearingFaultBackbone(feature_dim=256)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES)

    class CompleteModel(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier

        def forward(self, x):
            features = self.backbone(x)
            logits, _ = self.classifier(features)  # FaultClassifier returns (logits, probs)
            return logits

    return CompleteModel(backbone, classifier)


def train_source_model(X_train, y_train, X_test, y_test, seed=42):
    """Train source model on PU dataset"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = create_model().to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(DEVICE))
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        accuracy = accuracy_score(y_test.numpy(), test_preds)

    return model, accuracy


def sota_shot(model, X_target, lr=1e-3, epochs=50):
    """SHOT: Minimize prediction entropy"""
    model = deepcopy(model).to(DEVICE)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    dataset = TensorDataset(X_target)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)

            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)

            # Entropy minimization
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

            # Diversity regularization
            avg_probs = probs.mean(dim=0)
            diversity = torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

            loss = entropy - 0.1 * diversity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def sota_tent(model, X_target, lr=1e-3, epochs=50):
    """TENT: Adapt BatchNorm parameters"""
    model = deepcopy(model).to(DEVICE)

    # Freeze backbone except BN
    bn_params = []
    for name, param in model.backbone.named_parameters():
        if 'bn' in name or 'norm' in name or 'BatchNorm' in name:
            bn_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False

    # If no BN params found, adapt all backbone params
    if len(bn_params) == 0:
        bn_params = list(model.backbone.parameters())
        for param in bn_params:
            param.requires_grad = True

    optimizer = torch.optim.Adam(bn_params, lr=lr)

    dataset = TensorDataset(X_target)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)

            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()

    return model


def sota_nrc(model, X_target, lr=1e-3, epochs=50):
    """NRC: Neighborhood reciprocity"""
    model = deepcopy(model).to(DEVICE)

    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    dataset = TensorDataset(X_target)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)

            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)

            # Pseudo-labels
            pseudo_labels = probs.argmax(dim=1)

            # Classification loss with pseudo-labels
            loss = F.cross_entropy(logits, pseudo_labels)

            # Neighborhood consistency (simplified)
            features = model.backbone(batch_x)
            features_norm = F.normalize(features, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
            consistency_loss = -similarity.mean()

            total_loss = loss + 0.01 * consistency_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return model


def sota_sar(model, X_target, lr=1e-3, epochs=50):
    """SAR: Selective entropy minimization"""
    model = deepcopy(model).to(DEVICE)

    # Only adapt BN parameters
    bn_params = []
    for name, param in model.backbone.named_parameters():
        if 'bn' in name or 'norm' in name or 'BatchNorm' in name:
            bn_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False

    # If no BN params found, adapt all backbone params
    if len(bn_params) == 0:
        bn_params = list(model.backbone.parameters())
        for param in bn_params:
            param.requires_grad = True

    optimizer = torch.optim.Adam(bn_params, lr=lr)
    dataset = TensorDataset(X_target)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)

            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)

            # Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            # Selective: only update if entropy is above threshold
            mask = entropy > 0.5
            if mask.sum() > 0:
                loss = entropy[mask].mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(DEVICE))
        preds = logits.argmax(dim=1).cpu().numpy()

        accuracy = accuracy_score(y_test.numpy(), preds)
        macro_f1 = f1_score(y_test.numpy(), preds, average='macro')

    return accuracy, macro_f1


def run_experiment(scenario_name, source_condition, target_condition, seed):
    """Run a single experiment"""
    # Load data
    X_source_train, y_source_train = load_pu_data(source_condition, 'train')
    X_source_test, y_source_test = load_pu_data(source_condition, 'test')
    X_target_train, y_target_train = load_pu_data(target_condition, 'train')
    X_target_test, y_target_test = load_pu_data(target_condition, 'test')

    # Train source model
    source_model, source_acc = train_source_model(
        X_source_train, y_source_train,
        X_source_test, y_source_test,
        seed=seed
    )

    results = {
        'Source_Only': evaluate_model(source_model, X_target_test, y_target_test)
    }

    # Run SFDA methods
    for method_name in ['SHOT', 'TENT', 'NRC', 'SAR']:
        if method_name == 'SHOT':
            adapted_model = sota_shot(source_model, X_target_train, lr=1e-3, epochs=NUM_EPOCHS)
        elif method_name == 'TENT':
            adapted_model = sota_tent(source_model, X_target_train, lr=1e-3, epochs=NUM_EPOCHS)
        elif method_name == 'NRC':
            adapted_model = sota_nrc(source_model, X_target_train, lr=1e-3, epochs=NUM_EPOCHS)
        elif method_name == 'SAR':
            adapted_model = sota_sar(source_model, X_target_train, lr=1e-3, epochs=NUM_EPOCHS)

        results[method_name] = evaluate_model(adapted_model, X_target_test, y_target_test)

    return results


def run_scenario(scenario_name, source_condition, target_condition):
    """Run full scenario with all seeds"""
    print(f"\n{'='*80}")
    print(f"Scenario: {scenario_name}")
    print(f"Source: {source_condition} → Target: {target_condition}")
    print(f"{'='*80}")

    all_results = {method: {'accuracy': [], 'f1': []} for method in METHODS}

    for i, seed in enumerate(SEEDS):
        print(f"  Seed {i+1}/{len(SEEDS)}: {seed}...", end=' ')

        results = run_experiment(scenario_name, source_condition, target_condition, seed)

        for method in METHODS:
            acc, f1 = results[method]
            all_results[method]['accuracy'].append(acc)
            all_results[method]['f1'].append(f1)

        print(f"SHOT: {results['SHOT'][0]:.2%}")

    # Compute statistics
    summary = {}
    for method in METHODS:
        acc_mean = np.mean(all_results[method]['accuracy'])
        acc_std = np.std(all_results[method]['accuracy'])
        f1_mean = np.mean(all_results[method]['f1'])
        f1_std = np.std(all_results[method]['f1'])

        summary[method] = {
            'accuracy': f"{acc_mean:.2%} ± {acc_std:.2%}",
            'f1': f"{f1_mean:.2%} ± {f1_std:.2%}",
            'acc_mean': float(acc_mean),
            'acc_std': float(acc_std),
            'f1_mean': float(f1_mean),
            'f1_std': float(f1_std)
        }

        print(f"\n{method}: {summary[method]['accuracy']}")

    return summary


def main():
    """Run all PU dataset experiments"""

    # Scenario 1: Cross-Load (N09_M07 → N15_M07)
    # 900 RPM, 0.7 Nm → 1500 RPM, 0.7 Nm
    scenario1_results = run_scenario(
        "Cross-Load",
        "N09_M07",
        "N15_M07"
    )

    # Scenario 2: Cross-Speed (N15_M07 → N09_M07)
    # 1500 RPM, 0.7 Nm → 900 RPM, 0.7 Nm
    scenario2_results = run_scenario(
        "Cross-Speed",
        "N15_M07",
        "N09_M07"
    )

    # Scenario 3: Cross-Torque (N15_M07 → N15_M01)
    # 1500 RPM, 0.7 Nm → 1500 RPM, 0.1 Nm
    scenario3_results = run_scenario(
        "Cross-Torque",
        "N15_M07",
        "N15_M01"
    )

    # Save results
    all_results = {
        'cross_load': scenario1_results,
        'cross_speed': scenario2_results,
        'cross_torque': scenario3_results,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'device': str(DEVICE),
            'num_seeds': len(SEEDS),
            'num_epochs': NUM_EPOCHS,
            'methods': METHODS,
            'scenarios': {
                'cross_load': {'source': 'N09_M07', 'target': 'N15_M07'},
                'cross_speed': {'source': 'N15_M07', 'target': 'N09_M07'},
                'cross_torque': {'source': 'N15_M07', 'target': 'N15_M01'}
            }
        }
    }

    output_file = RESULTS_DIR / 'pu_dataset_experiments.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    return all_results


if __name__ == '__main__':
    main()
