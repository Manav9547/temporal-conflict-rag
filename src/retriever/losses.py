"""
Margin-Penalized InfoNCE loss (Stage 1), plus the alignment/uniformity latent
geometry diagnostics from the proposal (Wang & Isola, 2020).

The loss, from Proposal.pdf §3:

    L = -log [ exp(e_q . e_d+ / tau)
               / ( exp(e_q . e_d+ / tau)
                   + sum_j exp(e_q . e_dj- / tau)
                   + sum_k exp((e_q . e_dk_r + gamma) / tau) ) ]

where e_dj- are random in-batch negatives and e_dk_r are mined temporally
stale/conflicting hard negatives, with margin gamma added *inside* the exp,
before the /tau division — i.e. (sim + gamma) / tau, not sim/tau + gamma.
This is what makes hard negatives strictly harder to satisfy than a plain
InfoNCE negative: to drive the loss down, the model must push
sim(q, hard_neg) down far enough that even after adding gamma it still loses
to sim(q, pos) by a full-margin's worth of logit distance.

Implementation: for a batch of B (query, positive, hard_negative) triples,
- the B x B matrix of query-vs-all-positives similarities supplies both the
  true positive (diagonal) and the random in-batch negatives (off-diagonal,
  i.e. e_dj- = other examples' positives) in one matmul;
- the per-example hard negative similarity is a single extra column with
  margin gamma baked in;
- softmax cross-entropy over that row, with the diagonal index as the
  target, is exactly the formula above (this is the standard way InfoNCE is
  implemented — cross-entropy IS -log(softmax) at the target index).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

TAU = 0.05    # fixed temperature — see docs/DECISIONS.md #2
GAMMA = 0.2   # margin penalty for mined hard negatives


def margin_penalized_infonce(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    hard_neg_emb: torch.Tensor,
    tau: float = TAU,
    gamma: float = GAMMA,
) -> torch.Tensor:
    """
    q_emb, pos_emb, hard_neg_emb: [B, dim], L2-normalized.
    Row i's hard_neg_emb[i] is that example's mined stale/conflicting negative;
    row i's random negatives are pos_emb[j] for all j != i (in-batch).

    Returns: scalar loss (mean over the batch).
    """
    B = q_emb.size(0)
    assert pos_emb.size(0) == B and hard_neg_emb.size(0) == B

    sim_pos_matrix = q_emb @ pos_emb.t()              # [B, B]; diag = positive, off-diag = random negs
    sim_hard = (q_emb * hard_neg_emb).sum(dim=1)       # [B]    ; one mined hard negative per row

    logits_main = sim_pos_matrix / tau                 # [B, B]
    logits_hard = ((sim_hard + gamma) / tau).unsqueeze(1)  # [B, 1] -- margin added BEFORE /tau

    logits = torch.cat([logits_main, logits_hard], dim=1)  # [B, B+1]
    targets = torch.arange(B, device=q_emb.device)          # diagonal index i is the positive for row i

    return F.cross_entropy(logits, targets)


def alignment(q_emb: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
    """E_(q,d+)[ ||e_q - e_d+||^2 ] -- lower means positives are pulled closer."""
    return (q_emb - pos_emb).pow(2).sum(dim=1).mean()


def uniformity(emb: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """log E_{x,y}[ exp(-t ||e_x - e_y||^2) ] over all pairs in `emb`
    -- more negative means embeddings are more spread out (higher isotropy),
    which is the proposal's target direction (avoids representation collapse)."""
    sq_dists = torch.pdist(emb, p=2).pow(2)
    return sq_dists.mul(-t).exp().mean().log()
