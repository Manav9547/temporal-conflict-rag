"""
Unit tests for the Margin-Penalized InfoNCE loss (src/retriever/losses.py).

The hand-computed test below is the important one: it independently
recomputes the expected loss using plain `math.exp`/`math.log` (not by
re-deriving the same tensor formula) so it actually catches sign/placement
bugs like "gamma added after dividing by tau" instead of before, which would
silently produce a plausible-looking but wrong loss value.

Run with: python -m pytest tests/test_retriever_losses.py -v
"""

import math

import torch

from src.retriever.losses import alignment, margin_penalized_infonce, uniformity


def test_margin_penalized_infonce_matches_hand_computation():
    tau, gamma = 0.05, 0.2

    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])       # identical to q -> sim = 1
    hard_neg = torch.tensor([[0.6, 0.8], [0.8, 0.6]])  # sim(q_i, hard_i) = 0.6

    # Independent hand computation, not reusing the module's tensor ops.
    # Row 0: sim_pos = [1, 0] (self, other), sim_hard = 0.6
    # logits = [1/tau, 0/tau, (0.6+gamma)/tau] = [20, 0, 16], target index 0
    # Row 1: sim_pos = [0, 1], sim_hard = 0.6 -> logits = [0, 20, 16], target index 1
    def hand_loss(logits: list[float], target_idx: int) -> float:
        denom = sum(math.exp(z) for z in logits)
        return -math.log(math.exp(logits[target_idx]) / denom)

    expected_loss_0 = hand_loss([1 / tau, 0 / tau, (0.6 + gamma) / tau], target_idx=0)
    expected_loss_1 = hand_loss([0 / tau, 1 / tau, (0.6 + gamma) / tau], target_idx=1)
    expected_mean = (expected_loss_0 + expected_loss_1) / 2

    actual = margin_penalized_infonce(q, pos, hard_neg, tau=tau, gamma=gamma).item()

    assert math.isclose(actual, expected_mean, rel_tol=1e-4), (
        f"Loss mismatch: got {actual}, hand-computed {expected_mean}. "
        f"Check gamma/tau placement -- gamma must be added to the similarity "
        f"BEFORE dividing by tau, i.e. (sim + gamma) / tau."
    )


def test_gamma_increases_loss_monotonically():
    """A larger margin makes the hard negative harder to satisfy, so loss
    should strictly increase with gamma (all else fixed)."""
    q = torch.tensor([[1.0, 0.0]])
    pos = torch.tensor([[1.0, 0.0]])
    hard_neg = torch.tensor([[0.9, 0.436]])  # high similarity, deliberately close to q

    loss_small_gamma = margin_penalized_infonce(q, pos, hard_neg, tau=0.05, gamma=0.0).item()
    loss_large_gamma = margin_penalized_infonce(q, pos, hard_neg, tau=0.05, gamma=0.5).item()

    assert loss_large_gamma > loss_small_gamma


def test_higher_temperature_increases_loss_when_positive_dominates():
    """When sim(q,pos) >> sim(q,neg), a flatter (higher-tau) distribution
    pulls probability mass away from the correct answer, raising the loss."""
    q = torch.tensor([[1.0, 0.0]])
    pos = torch.tensor([[1.0, 0.0]])
    hard_neg = torch.tensor([[0.0, 1.0]])

    loss_sharp = margin_penalized_infonce(q, pos, hard_neg, tau=0.05, gamma=0.2).item()
    loss_flat = margin_penalized_infonce(q, pos, hard_neg, tau=0.5, gamma=0.2).item()

    assert loss_flat > loss_sharp


def test_alignment_zero_for_identical_embeddings():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pos = q.clone()
    assert alignment(q, pos).item() == 0.0


def test_alignment_matches_squared_distance_for_orthogonal_vectors():
    q = torch.tensor([[1.0, 0.0]])
    pos = torch.tensor([[0.0, 1.0]])
    # ||(1,0) - (0,1)||^2 = 1^2 + (-1)^2 = 2
    assert math.isclose(alignment(q, pos).item(), 2.0, rel_tol=1e-6)


def test_uniformity_lower_for_spread_out_embeddings():
    """More isotropic (spread out) embeddings should have a lower (more
    negative) uniformity score than tightly clustered embeddings."""
    clustered = torch.tensor([[1.0, 0.0], [1.0, 0.001], [1.0, -0.001], [0.999, 0.0]])
    spread = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

    u_clustered = uniformity(clustered).item()
    u_spread = uniformity(spread).item()

    assert u_spread < u_clustered
