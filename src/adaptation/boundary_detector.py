"""
Component 1: Classifier-Prototype Disagreement Boundary Detector

Identifies class-boundary samples via KL divergence between the source
classifier's probability distribution and the prototype-based probability
distribution. Key insight: when IR samples are classified as Ball by prototype
distance but the classifier retains IR discrimination (77.7% recall, 82.6% in
top-2), the disagreement signals boundary ambiguity.

Boundary score = KL(p_classifier || p_prototype)

Paper 2: SFDA-BoundaryRepel
"""

import torch
import torch.nn.functional as F


class BoundaryDetector:
    """
    Boundary sample detector using classifier-prototype disagreement.

    Two modes:
    - 'kl': KL divergence (default, more sensitive to distribution differences)
    - 'margin': Prototype ambiguity margin (simpler, geometric only)
    """

    def __init__(self, mode='kl', temperature=0.10, percentile=70):
        """
        Args:
            mode: 'kl' for KL divergence, 'margin' for prototype ambiguity
            temperature: Temperature for prototype softmax (default 0.10)
            percentile: Boundary threshold percentile (default 70)
        """
        self.mode = mode
        self.temperature = temperature
        self.percentile = percentile
        self.cached_threshold = None  # Cache for frozen-boundary ablation

    def compute_boundary_scores(self, features, classifier_logits, prototypes):
        """
        Compute boundary scores for all target samples.

        Args:
            features: L2-normalized features [N, D]
            classifier_logits: Raw classifier logits [N, C]
            prototypes: Current class prototypes [C, D] (L2-normalized)

        Returns:
            boundary_scores: [N] — higher = more boundary-ambiguous
        """
        # Classifier probability distribution
        p_cls = F.softmax(classifier_logits, dim=1)  # [N, C]

        # Prototype probability distribution
        cos_sims = torch.mm(features, prototypes.t())  # [N, C]
        p_proto = F.softmax(cos_sims / self.temperature, dim=1)  # [N, C]

        if self.mode == 'kl':
            # KL(p_cls || p_proto) = sum_c p_cls(c) * log(p_cls(c) / p_proto(c))
            # Use numerically stable log computation
            log_p_cls = F.log_softmax(classifier_logits, dim=1)
            log_p_proto = F.log_softmax(cos_sims / self.temperature, dim=1)

            # KL = sum(p_cls * (log_p_cls - log_p_proto))
            kl_div = (p_cls * (log_p_cls - log_p_proto)).sum(dim=1)

            # Clamp to prevent extreme values
            kl_div = torch.clamp(kl_div, min=0.0, max=10.0)
            boundary_scores = kl_div  # [N]

        elif self.mode == 'margin':
            # Prototype ambiguity: 1 - |cos(top1) - cos(top2)|
            top2_cos, _ = cos_sims.topk(2, dim=1)
            ambiguity = 1.0 - (top2_cos[:, 0] - top2_cos[:, 1]).abs()
            boundary_scores = ambiguity  # [N]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return boundary_scores, p_cls, p_proto

    def partition_samples(self, boundary_scores, freeze_threshold=False):
        """
        Partition samples into boundary set B and core set C.

        Args:
            boundary_scores: [N]
            freeze_threshold: If True, use cached threshold (frozen-boundary ablation)

        Returns:
            boundary_mask: bool [N] — True for boundary samples
            core_mask: bool [N] — True for core (non-boundary) samples
            threshold: float — the threshold value used
        """
        if freeze_threshold and self.cached_threshold is not None:
            threshold = self.cached_threshold
        else:
            threshold = torch.quantile(boundary_scores, self.percentile / 100.0)
            self.cached_threshold = threshold

        boundary_mask = boundary_scores > threshold
        core_mask = ~boundary_mask

        return boundary_mask, core_mask, threshold.item()

    def get_stats(self, boundary_scores, labels=None):
        """
        Compute per-class boundary statistics (for diagnostics).

        Args:
            boundary_scores: [N]
            labels: [N] (optional, for per-class breakdown)

        Returns:
            dict with per-class statistics
        """
        stats = {
            'mean': boundary_scores.mean().item(),
            'std': boundary_scores.std().item(),
            'max': boundary_scores.max().item(),
            'min': boundary_scores.min().item(),
            'threshold_50': torch.quantile(boundary_scores, 0.50).item(),
            'threshold_70': torch.quantile(boundary_scores, 0.70).item(),
            'threshold_90': torch.quantile(boundary_scores, 0.90).item(),
        }

        if labels is not None:
            per_class = {}
            for c in range(labels.max().item() + 1):
                c_mask = labels == c
                c_scores = boundary_scores[c_mask]
                per_class[int(c)] = {
                    'mean': c_scores.mean().item(),
                    'std': c_scores.std().item(),
                    'n': c_mask.sum().item(),
                }
            stats['per_class'] = per_class

        return stats


if __name__ == '__main__':
    # Test with synthetic data
    detector_kl = BoundaryDetector(mode='kl')
    detector_margin = BoundaryDetector(mode='margin')

    N, D, C = 64, 256, 4
    features = F.normalize(torch.randn(N, D), dim=1)
    logits = torch.randn(N, C)
    prototypes = F.normalize(torch.randn(C, D), dim=1)
    labels = torch.randint(0, C, (N,))

    scores_kl, _, _ = detector_kl.compute_boundary_scores(features, logits, prototypes)
    scores_margin, _, _ = detector_margin.compute_boundary_scores(features, logits, prototypes)

    b_mask_kl, c_mask_kl, thresh_kl = detector_kl.partition_samples(scores_kl)
    b_mask_margin, c_mask_margin, thresh_margin = detector_margin.partition_samples(scores_margin)

    print(f"Boundary Detector Test:")
    print(f"  KL mode: threshold={thresh_kl:.4f}, boundary={b_mask_kl.sum()}/{N}")
    print(f"  Margin mode: threshold={thresh_margin:.4f}, boundary={b_mask_margin.sum()}/{N}")

    stats = detector_kl.get_stats(scores_kl, labels)
    print(f"  KL stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
