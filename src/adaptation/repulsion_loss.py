"""
Component 2: Prototype-Level Contrastive Repulsion Loss

NONL-inspired NT-Xent formulation operating at the prototype level.
Normalizes over negatives only (excluding the pseudo-labeled class),
eliminating intra-class repulsion by construction.

Two modes:
- 'nt_xent': Softmax contrastive normalized over negatives (Eq.7 in paper)
- 'hinge': Directed margin hinge (simpler baseline, push from top-1 confused class)

Paper 2: SFDA-BoundaryRepel
"""

import torch
import torch.nn.functional as F


class PrototypeRepulsionLoss:
    """
    Prototype-level contrastive repulsion loss.

    Key properties:
    - Operates at prototype level (not instance-instance) → no intra-class repulsion
    - Normalizes over negatives only (NONL principle)
    - Hinge guardrail prevents unbounded repulsion for already-separated classes
    """

    def __init__(self, mode='nt_xent', temperature=0.10, hinge_margin=0.5):
        """
        Args:
            mode: 'nt_xent' (softmax contrastive) or 'hinge' (directed margin)
            temperature: Softmax temperature (default 0.10, sharper near boundary)
            hinge_margin: Cosine similarity margin guardrail (default 0.5)
        """
        self.mode = mode
        self.temperature = temperature
        self.hinge_margin = hinge_margin

    def compute(self, boundary_features, pseudo_labels, prototypes):
        """
        Compute repulsion loss for boundary samples.

        Args:
            boundary_features: L2-normalized features of boundary samples [B, D]
            pseudo_labels: Pseudo-labels of boundary samples [B] (may be WRONG)
            prototypes: Current class prototypes [C, D] (L2-normalized)

        Returns:
            loss: scalar repulsion loss
        """
        if len(boundary_features) == 0:
            return torch.tensor(0.0, device=boundary_features.device)

        B, D = boundary_features.shape
        C = prototypes.shape[0]

        if self.mode == 'nt_xent':
            loss = self._nt_xent_repel(boundary_features, pseudo_labels, prototypes)
        elif self.mode == 'hinge':
            loss = self._hinge_repel(boundary_features, pseudo_labels, prototypes)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return loss

    def _nt_xent_repel(self, features, pseudo_labels, prototypes):
        """
        NT-Xent repulsion normalized over negatives only.

        L = -(1/|B|) * sum_i log(
            exp(cos(z_i, mu_yi) / tau) /
            sum_{c != yi} exp(cos(z_i, mu_c) / tau)
        )

        This simultaneously:
        - Pulls z_i toward mu_yi (numerator)
        - Pushes z_i from ALL competing prototypes (denominator sum)
        """
        B, D = features.shape
        C = prototypes.shape[0]

        # Cosine similarities: [B, C]
        cos_sims = torch.mm(features, prototypes.t())

        # Apply hinge guardrail: clip low cosine similarities
        cos_sims_clipped = torch.clamp(cos_sims - self.hinge_margin, min=0.0)

        # Positive term: cos(z_i, mu_{y_hat_i})
        pos_sims = cos_sims[torch.arange(B), pseudo_labels]  # [B]

        # Negative sum: sum_{c != y_hat_i} exp(cos(z_i, mu_c) / tau)
        # Create mask to exclude the pseudo-labeled class
        neg_mask = torch.ones(B, C, device=features.device)
        neg_mask[torch.arange(B), pseudo_labels] = 0.0

        # Apply hinge guardrail before exponentiation
        neg_exp = torch.exp(cos_sims_clipped / self.temperature) * neg_mask
        neg_sum = neg_exp.sum(dim=1)  # [B]

        # Loss per sample
        pos_exp = torch.exp(pos_sims / self.temperature)
        loss_per_sample = -torch.log(pos_exp / (pos_exp + neg_sum + 1e-10))

        return loss_per_sample.mean()

    def _hinge_repel(self, features, pseudo_labels, prototypes):
        """
        Directed margin hinge: push only from the SINGLE most-confused class.

        For each boundary sample i:
        - Find the class c* != y_hat_i with highest cos(z_i, mu_c)
        - Apply hinge: max(0, cos(z_i, mu_c*) - margin)

        Simpler baseline vs. NT-Xent (which pushes from ALL classes).
        """
        B, D = features.shape
        C = prototypes.shape[0]

        # Cosine similarities: [B, C]
        cos_sims = torch.mm(features, prototypes.t())

        # Mask out the pseudo-labeled class
        mask = torch.ones(B, C, device=features.device)
        mask[torch.arange(B), pseudo_labels] = 0.0

        # Find the most-confused class for each sample
        masked_sims = cos_sims * mask - (1 - mask) * 1e10  # Set own class to -inf
        most_confused_sims, _ = masked_sims.max(dim=1)  # [B]

        # Hinge loss: repel from the most-confused class
        repulsion = torch.clamp(most_confused_sims - self.hinge_margin, min=0.0)

        return repulsion.mean()

    def compute_ir_ball_cos_sim(self, prototypes):
        """
        Compute IR-Ball prototype cosine similarity (diagnostic).

        Args:
            prototypes: [C, D] L2-normalized

        Returns:
            cos_sim: scalar — IR-Ball cosine similarity
        """
        # Assume IR = index 1, Ball = index 2 (CWRU class ordering)
        ir_proto = prototypes[1:2]  # [1, D]
        ball_proto = prototypes[2:3]  # [1, D]
        return F.cosine_similarity(ir_proto, ball_proto).item()


if __name__ == '__main__':
    # Test both modes
    features = F.normalize(torch.randn(32, 256), dim=1)
    pseudo_labels = torch.randint(0, 4, (32,))
    prototypes = F.normalize(torch.randn(4, 256), dim=1)

    nt_xent = PrototypeRepulsionLoss(mode='nt_xent', temperature=0.10, hinge_margin=0.5)
    hinge = PrototypeRepulsionLoss(mode='hinge', hinge_margin=0.5)

    loss_nx = nt_xent.compute(features, pseudo_labels, prototypes)
    loss_h = hinge.compute(features, pseudo_labels, prototypes)
    cos_sim = nt_xent.compute_ir_ball_cos_sim(prototypes)

    print(f"Repulsion Loss Test:")
    print(f"  NT-Xent loss: {loss_nx.item():.4f}")
    print(f"  Hinge loss:   {loss_h.item():.4f}")
    print(f"  IR-Ball cos_sim: {cos_sim:.4f}")

    # Test with empty boundary set
    empty_loss = nt_xent.compute(
        torch.randn(0, 256), torch.randint(0, 4, (0,)), prototypes
    )
    print(f"  Empty boundary loss: {empty_loss.item():.4f} (should be 0.0)")
