#!/usr/bin/env python3
"""
Quick test of PU dataset SFDA experiment (3 seeds only)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from src.models.backbone import BearingFaultBackbone
from src.models.classifier import FaultClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PU_DATA_DIR = Path('/mnt/data/sfda3/data/PU')
NUM_CLASSES = 4
NUM_EPOCHS = 50
BATCH_SIZE = 128

print("=" * 60)
print("PU Dataset SFDA Test (3 seeds)")
print("=" * 60)
print(f"Device: {DEVICE}")


def load_pu_data(load_condition, split='train'):
    """Load PU dataset"""
    X = np.load(PU_DATA_DIR / f'X_{split}_{load_condition}.npy')
    y = np.load(PU_DATA_DIR / f'y_{split}_{load_condition}.npy', allow_pickle=True)

    label_map = {'Healthy': 0, 'Outer_Race': 1, 'Inner_Race': 2, 'Ball': 3}
    y_int = np.array([label_map[label] for label in y])

    X = torch.FloatTensor(X).unsqueeze(1)
    y = torch.LongTensor(y_int)

    return X, y


def create_model():
    """Create model"""
    backbone = BearingFaultBackbone(feature_dim=256)
    classifier = FaultClassifier(feature_dim=256, num_classes=NUM_CLASSES)

    class CompleteModel(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier

        def forward(self, x):
            features = self.backbone(x)
            logits, _ = self.classifier(features)
            return logits

    return CompleteModel(backbone, classifier)


def train_source_model(X_train, y_train, X_test, y_test, seed=42):
    """Train source model"""
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
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(DEVICE))
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        accuracy = accuracy_score(y_test.numpy(), test_preds)

    print(f"  Source model accuracy: {accuracy:.2%}")
    return model, accuracy


def sota_shot(model, X_target, lr=1e-3, epochs=50):
    """SHOT"""
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

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            avg_probs = probs.mean(dim=0)
            diversity = torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

            loss = entropy - 0.1 * diversity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def sota_tent(model, X_target, lr=1e-3, epochs=50):
    """TENT"""
    model = deepcopy(model).to(DEVICE)

    bn_params = []
    for name, param in model.backbone.named_parameters():
        if 'bn' in name or 'norm' in name or 'BatchNorm' in name:
            bn_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False

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
    """NRC"""
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
            pseudo_labels = probs.argmax(dim=1)

            loss = F.cross_entropy(logits, pseudo_labels)

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
    """SAR"""
    model = deepcopy(model).to(DEVICE)

    bn_params = []
    for name, param in model.backbone.named_parameters():
        if 'bn' in name or 'norm' in name or 'BatchNorm' in name:
            bn_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False

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

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            mask = entropy > 0.5
            if mask.sum() > 0:
                loss = entropy[mask].mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model"""
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(DEVICE))
        preds = logits.argmax(dim=1).cpu().numpy()
        accuracy = accuracy_score(y_test.numpy(), preds)

    return accuracy


def main():
    """Run test experiment"""

    # Load data for Cross-Load scenario
    print("\nLoading data (Cross-Load: N09_M07 → N15_M07)...")
    X_source_train, y_source_train = load_pu_data('N09_M07', 'train')
    X_source_test, y_source_test = load_pu_data('N09_M07', 'test')
    X_target_train, y_target_train = load_pu_data('N15_M07', 'train')
    X_target_test, y_target_test = load_pu_data('N15_M07', 'test')

    print(f"  Source train: {X_source_train.shape[0]} samples")
    print(f"  Source test: {X_source_test.shape[0]} samples")
    print(f"  Target train: {X_target_train.shape[0]} samples")
    print(f"  Target test: {X_target_test.shape[0]} samples")

    # Test with 3 seeds
    seeds = [42, 43, 44]
    results = {'Source_Only': [], 'SHOT': [], 'TENT': [], 'NRC': [], 'SAR': []}

    for seed in seeds:
        print(f"\nSeed {seed}:")

        # Train source model
        source_model, source_acc = train_source_model(
            X_source_train, y_source_train,
            X_source_test, y_source_test,
            seed=seed
        )

        # Evaluate on target
        source_only_acc = evaluate_model(source_model, X_target_test, y_target_test)
        results['Source_Only'].append(source_only_acc)
        print(f"  Source Only: {source_only_acc:.2%}")

        # SHOT
        shot_model = sota_shot(source_model, X_target_train, lr=1e-3, epochs=NUM_EPOCHS)
        shot_acc = evaluate_model(shot_model, X_target_test, y_target_test)
        results['SHOT'].append(shot_acc)
        print(f"  SHOT: {shot_acc:.2%}")

        # TENT
        tent_model = sota_tent(source_model, X_target_train, lr=1e-3, epochs=NUM_EPOCHS)
        tent_acc = evaluate_model(tent_model, X_target_test, y_target_test)
        results['TENT'].append(tent_acc)
        print(f"  TENT: {tent_acc:.2%}")

        # NRC
        nrc_model = sota_nrc(source_model, X_target_train, lr=1e-3, epochs=NUM_EPOCHS)
        nrc_acc = evaluate_model(nrc_model, X_target_test, y_target_test)
        results['NRC'].append(nrc_acc)
        print(f"  NRC: {nrc_acc:.2%}")

        # SAR
        sar_model = sota_sar(source_model, X_target_train, lr=1e-3, epochs=NUM_EPOCHS)
        sar_acc = evaluate_model(sar_model, X_target_test, y_target_test)
        results['SAR'].append(sar_acc)
        print(f"  SAR: {sar_acc:.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary (3 seeds)")
    print("=" * 60)

    for method in results:
        accs = results[method]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{method}: {mean_acc:.2%} ± {std_acc:.2%}")

    print("\n✓ Test completed successfully!")
    print("Ready to run full experiment with 30 seeds.")


if __name__ == '__main__':
    main()
