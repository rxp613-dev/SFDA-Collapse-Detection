"""
Component 3: Source-Softmax Cold-Start Bootstrapping

Three-phase pseudo-label schedule for minority class recovery:
- Epochs 1-20 (Soft phase): Source classifier softmax as soft pseudo-labels
- Epochs 21-50 (Transition): Linear interpolation soft → hard
- Epochs 51-100 (Hard phase): Standard hard pseudo-labels from prototype distance

SFDA-compliant: No source data accessed — only source model output on target data.

Paper 2: SFDA-BoundaryRepel
"""

import torch
import torch.nn.functional as F
import numpy as np


class ColdStartScheduler:
    """
    Three-phase pseudo-label scheduler for minority class bootstrapping.
    """

    def __init__(self, soft_epochs=20, transition_epochs=30, epsilon=0.15,
                 source_model=None, mode='soft_labels'):
        """
        Args:
            soft_epochs: Number of soft-label epochs (default 20)
            transition_epochs: Number of transition epochs (default 30)
            epsilon: Geometric distance threshold for hard phase
            source_model: Source-pretrained model (for extracting softmax)
                          Set at training time via set_source_model()
            mode: 'soft_labels' (default), 'none' (no cold-start, ablation)
        """
        self.soft_epochs = soft_epochs
        self.transition_epochs = transition_epochs
        self.epsilon = epsilon
        self.source_model = source_model
        self.mode = mode

    def set_source_model(self, model):
        """Set the source-pretrained model for softmax extraction."""
        self.source_model = model

    def get_phase(self, epoch):
        """
        Get the current phase based on epoch number (1-indexed).

        Returns: 'soft', 'transition', or 'hard'
        """
        if self.mode == 'none':
            return 'hard'

        if epoch <= self.soft_epochs:
            return 'soft'
        elif epoch <= self.soft_epochs + self.transition_epochs:
            return 'transition'
        else:
            return 'hard'

    def get_interpolation_weight(self, epoch):
        """
        Get the soft→hard interpolation weight alpha.

        alpha = 1.0 → fully soft
        alpha = 0.0 → fully hard

        Args:
            epoch: 1-indexed epoch number

        Returns:
            alpha: float in [0, 1]
        """
        phase = self.get_phase(epoch)

        if phase == 'soft':
            return 1.0
        elif phase == 'hard':
            return 0.0
        else:  # transition
            # Linear decay from 1.0 at end of soft phase to 0.0 at start of hard phase
            progress = (epoch - self.soft_epochs) / self.transition_epochs
            return 1.0 - progress

    def get_source_softmax(self, features):
        """
        Get source classifier softmax probabilities for target features.

        Args:
            features: [N, D] L2-normalized target features

        Returns:
            softmax_probs: [N, C] source classifier probability distribution
        """
        if self.source_model is None:
            raise RuntimeError("Source model not set. Call set_source_model() first.")

        self.source_model.eval()
        with torch.no_grad():
            logits, probs = self.source_model.classifier(features)
        return probs

    def assign_pseudo_labels(self, epoch, features, prototypes, epsilon_override=None):
        """
        Assign pseudo-labels according to the current phase schedule.

        Args:
            epoch: 1-indexed epoch number
            features: [N, D] L2-normalized target features
            prototypes: [C, D] current class prototypes (L2-normalized)
            epsilon_override: Optional epsilon override (for sensitivity testing)

        Returns:
            pseudo_labels: [N] hard pseudo-labels (class indices)
            soft_targets: [N, C] soft targets for prototype update weighting
                          (or None in hard phase)
            phase: str — current phase ('soft', 'transition', 'hard')
        """
        eps = epsilon_override if epsilon_override is not None else self.epsilon
        phase = self.get_phase(epoch)

        # Hard pseudo-labels from prototype distance
        cos_sims = torch.mm(features, prototypes.t())  # [N, C]
        distances = 1.0 - cos_sims
        hard_labels = distances.argmin(dim=1)  # [N]

        if phase == 'hard' or self.mode == 'none':
            # Standard epsilon-based selection
            min_dists = distances.min(dim=1)[0]  # [N]
            hard_labels[min_dists > eps] = -1  # Mark unreliable as -1
            return hard_labels, None, 'hard'

        # Get source softmax
        source_probs = self.get_source_softmax(features)  # [N, C]

        if phase == 'soft':
            # Soft phase: use source softmax for both labels and weighting
            soft_labels = source_probs.argmax(dim=1)
            return soft_labels, source_probs, 'soft'

        else:  # transition
            alpha = self.get_interpolation_weight(epoch)

            # Interpolated targets for prototype update
            hard_onehot = F.one_hot(hard_labels, num_classes=prototypes.shape[0]).float()
            soft_targets = alpha * source_probs + (1.0 - alpha) * hard_onehot

            # Labels: use argmax of interpolated distribution
            transition_labels = soft_targets.argmax(dim=1)

            return transition_labels, soft_targets, 'transition'

    def get_per_class_selection_stats(self, pseudo_labels, true_labels=None):
        """
        Compute per-class pseudo-label selection statistics (diagnostic).

        Args:
            pseudo_labels: [N] assigned pseudo-labels (-1 for unlabeled)
            true_labels: [N] optional ground-truth labels for recall computation

        Returns:
            dict with per-class selection counts and rates
        """
        n = len(pseudo_labels)
        labeled_mask = pseudo_labels >= 0
        n_labeled = labeled_mask.sum().item()

        stats = {
            'n_total': n,
            'n_labeled': n_labeled,
            'selection_rate': n_labeled / n if n > 0 else 0.0,
        }

        # Per-class counts
        n_classes = pseudo_labels.max().item() + 1
        per_class = {}
        for c in range(n_classes):
            c_count = (pseudo_labels == c).sum().item()
            per_class[c] = {
                'count': c_count,
                'rate': c_count / n if n > 0 else 0.0,
            }
        stats['per_class'] = per_class

        # Per-class recall (if true labels available)
        if true_labels is not None:
            for c in range(n_classes):
                c_true_mask = true_labels == c
                c_true_count = c_true_mask.sum().item()
                if c_true_count > 0:
                    c_correct = ((pseudo_labels == c) & c_true_mask).sum().item()
                    per_class[c]['true_count'] = c_true_count
                    per_class[c]['recall'] = c_correct / c_true_count

        return stats


if __name__ == '__main__':
    # Test without source model (hard phase only)
    scheduler = ColdStartScheduler(soft_epochs=20, transition_epochs=30, epsilon=0.15)

    print("ColdStart Scheduler Test:")
    for epoch in [1, 10, 20, 35, 50, 51, 75, 100]:
        phase = scheduler.get_phase(epoch)
        alpha = scheduler.get_interpolation_weight(epoch)
        print(f"  Epoch {epoch:3d}: phase={phase:<10} alpha={alpha:.3f}")

    # Test with features
    features = F.normalize(torch.randn(64, 256), dim=1)
    prototypes = F.normalize(torch.randn(4, 256), dim=1)

    # Without source model, hard phase should still work
    labels, targets, phase = scheduler.assign_pseudo_labels(100, features, prototypes)
    print(f"\n  Epoch 100 (no source model, hard phase):")
    print(f"    Phase: {phase}")
    print(f"    Labels shape: {labels.shape}")
    print(f"    Labeled samples: {(labels >= 0).sum().item()}/{len(labels)}")
