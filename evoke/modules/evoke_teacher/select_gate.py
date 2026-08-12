"""
Select-gate: relevance gating (keep/mask) over the frames recalled by importance-select.

Phase 1, z-score hard gate:
  - Ranking is untouched: topk still selects on the raw-dot scores and this module only decides
    whether a selected frame participates in attention. Hence `gate ON + kappa=-1e9 +
    cos_floor=-1` is bit-identical to gate OFF -- the regression invariant.
  - Thresholds come from outlier calibration, not an absolute value: query/key are normalised to
    cosine, median/MAD over the candidates give the null centre and spread, and
    z = (cos - median) / (MAD + eps). If nothing stands out, nothing is recalled; the z-score
    also cancels any global scaling from layer, timestep or noise level.
  - Purely local tensor ops, no collectives, so SP-safe: communication still fetches a fixed
    K_max frames and the gate only adds a bias inside attention.

Phase 2, learned gate: one SelectGateHead per DiTBlock (hard-concrete / L0, Louizos et al. 2018)
mapping [cos, soft-clipped z, normalised timestep] to a gate logit; sampled while training,
deterministic at eval, with g in [0,1] becoming the additive bias log(g+eps) so gradients reach
the gate through the attention softmax. Warm-start (zero final weight, bias=+4) gives g == 1
exactly at eval, so loading an older checkpoint starts at baseline. Ranking is untouched here
too. compute_select_gate_features is the differentiable variant; compute_select_keep keeps its
no_grad.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def compute_select_keep(
    chunk_q: torch.Tensor,
    cand_keys: torch.Tensor,
    top_indices_rel: torch.Tensor,
    kappa: float = 2.0,
    cos_floor: float = 0.0,
    min_candidates: int = 8,
    mad_floor: float = 1e-3,
    min_keep: int = 0,
) -> torch.Tensor:
    """
    Keep mask for the frames already chosen by raw-dot topk (True = keep, False = mask out in
    attention).

    Args:
        chunk_q: (B, d) chunk query vector (chunk_to_state_proj output, not normalised)
        cand_keys: (B, N, d) keys of all candidate frames (frame_keys[:, 1:nearby_boundary_g])
        top_indices_rel: (B, K) indices of the selected frames within cand_keys (0-based)
        kappa: z-score threshold; larger is more conservative
        cos_floor: absolute cosine floor, so the best of a uniformly bad set is not kept
        min_candidates: below this the null distribution is unreliable -> keep everything
        mad_floor: a MAD under this means the scores are indistinguishable (e.g. all cos ~ 0 at
            high noise) -> keep that row whole
        min_keep: always keep the min_keep highest-z frames (0 allows masking all)

    Returns:
        keep_mask: (B, K) bool
    """
    B, K = top_indices_rel.shape
    N = cand_keys.shape[1]
    device = chunk_q.device

    # Too few candidates for a reliable null distribution -> keep everything, as plain top-k does.
    if N < min_candidates:
        return torch.ones(B, K, dtype=torch.bool, device=device)

    # Cosine similarity, computed in float32 to keep median/MAD numerically stable.
    q = torch.nn.functional.normalize(chunk_q.float(), dim=-1)        # (B, d)
    keys = torch.nn.functional.normalize(cand_keys.float(), dim=-1)   # (B, N, d)
    cos = torch.einsum('bd,bnd->bn', q, keys)                          # (B, N)

    # Robust null calibration by median/MAD: most of the candidate pool is irrelevant frames.
    med = cos.median(dim=1, keepdim=True).values                       # (B, 1)
    mad = (cos - med).abs().median(dim=1, keepdim=True).values         # (B, 1)
    z = (cos - med) / (mad + 1e-6)                                     # (B, N)

    cos_sel = torch.gather(cos, 1, top_indices_rel)                    # (B, K)
    z_sel = torch.gather(z, 1, top_indices_rel)                        # (B, K)

    keep = (z_sel >= kappa) & (cos_sel >= cos_floor)                   # (B, K)

    # MAD too small -> scores indistinguishable (e.g. a high-noise step) -> keep this row whole.
    indistinct = (mad < mad_floor).squeeze(1)                          # (B,)
    keep[indistinct] = True

    # Floor: always keep the min_keep highest-z frames.
    if min_keep > 0:
        k_floor = min(min_keep, K)
        _, top_z = z_sel.topk(k_floor, dim=1)                          # (B, k_floor)
        keep.scatter_(1, top_z, True)

    return keep


def compute_select_gate_features(
    chunk_q: torch.Tensor,
    cand_keys: torch.Tensor,
    top_indices_rel: torch.Tensor,
    t_frac_row: torch.Tensor,
) -> torch.Tensor:
    """
    Input features for the Phase 2 learned gate head. Differentiable, unlike compute_select_keep:
    gradients flow back through cos/z into chunk_to_state_proj / linear_attn.

    Args:
        chunk_q: (B, d) chunk query vector (chunk_to_state_proj output, not normalised)
        cand_keys: (B, N, d) keys of all candidate frames (frame_keys[:, 1:nearby_boundary_g])
        top_indices_rel: (B, K) indices of the selected frames within cand_keys (0-based)
        t_frac_row: (B,) normalised timestep (t/1000, in [0,1]) -- the noise-aware feature

    Returns:
        feats: (B, K, 3) float32 = [cos_sel, soft-clipped z_sel, t_frac]
    """
    B, K = top_indices_rel.shape

    # Cosine similarity, computed in float32 to keep median/MAD numerically stable.
    q = F.normalize(chunk_q.float(), dim=-1)                           # (B, d)
    keys = F.normalize(cand_keys.float(), dim=-1)                      # (B, N, d)
    cos = torch.einsum('bd,bnd->bn', q, keys)                          # (B, N)

    # Robust null calibration by median/MAD: the Phase 1 formula, but differentiable.
    med = cos.median(dim=1, keepdim=True).values                       # (B, 1)
    mad = (cos - med).abs().median(dim=1, keepdim=True).values         # (B, 1)
    z = (cos - med) / (mad + 1e-6)                                     # (B, N)

    cos_sel = torch.gather(cos, 1, top_indices_rel)                    # (B, K)
    z_sel = torch.gather(z, 1, top_indices_rel)                        # (B, K)
    # As MAD -> 0 the z magnitude can reach 1e6: soft-clip to (-8, 8) to bound the gate MLP input.
    z_sel = torch.tanh(z_sel / 8.0) * 8.0                              # (B, K)

    t_col = t_frac_row.float().view(B, 1).expand(B, K)                 # (B, K)
    feats = torch.stack([cos_sel, z_sel, t_col], dim=-1)               # (B, K, 3)
    return feats


class SelectGateHead(nn.Module):
    """
    Hard-concrete / L0 learnable gate head (Louizos et al. 2018), one per DiTBlock.

    Linear(feat_dim, hidden_dim) -> SiLU -> Linear(hidden_dim, 1) -> gate logit alpha. About 80
    parameters per block, ~3K over 40 blocks, negligible against 14B.

    The gate transform always computes in float32, so it is stable under bf16 weights:
      train: u ~ U(1e-6, 1-1e-6), s = sigmoid((log u - log(1-u) + alpha)/beta)
      eval:  s = sigmoid(alpha)
      g = clamp(s*(zeta-gamma)+gamma, 0, 1),  gamma=-0.1, zeta=1.1, so g reaches exactly 0 or 1

    Warm-start is what keeps older checkpoints usable: net[-1].weight=0 and
    net[-1].bias=logit_bias_init(+4.0) give alpha == 4, hence eval
    g = clamp(1.2*sigmoid(4)-0.1, 0, 1) = 1.0 exactly. Loading a checkpoint that has no gate
    parameters (strict=False) therefore behaves like baseline, and training closes the gates
    from there. Same pattern as the zero-initialised history_encoder.output_gate and cam modules.
    """

    def __init__(self, feat_dim: int = 3, hidden_dim: int = 16,
                 temperature: float = 0.6667, gamma: float = -0.1, zeta: float = 1.1,
                 logit_bias_init: float = 4.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.logit_bias_init = float(logit_bias_init)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Warm-start: zero output weights plus a positive bias, so the gate is fully open (g == 1)."""
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, self.logit_bias_init)

    def compute_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (..., feat_dim) -> alpha (...,) float32, matching the weight dtype automatically."""
        w_dtype = self.net[0].weight.dtype
        alpha = self.net(feats.to(w_dtype)).squeeze(-1)
        return alpha.float()

    def forward(self, feats: torch.Tensor, training: bool) -> tuple:
        """
        Args:
            feats: (B, K, feat_dim) gate input features
            training: True -> hard-concrete sampling; False -> deterministic eval gate

        Returns:
            g: (B, K) float32 gate value in [0, 1]
            alpha: (B, K) float32 gate logit
        """
        alpha = self.compute_logits(feats)                             # (B, K) fp32
        if training:
            u = torch.rand_like(alpha).clamp_(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + alpha) / self.temperature)
        else:
            s = torch.sigmoid(alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        g = s_bar.clamp(0.0, 1.0)                                      # (B, K)
        return g, alpha

    def open_prob(self, alpha: torch.Tensor) -> torch.Tensor:
        """Expected L0: P(g > 0) = sigmoid(alpha - beta*log(-gamma/zeta)); used by the sparsity budget."""
        return torch.sigmoid(alpha - self.temperature * math.log(-self.gamma / self.zeta))


def gate_to_bias(g: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Gate value -> additive attention bias (float32 in and out; the caller casts to x.dtype):
      g = 1     -> 0           (exactly open: bit-identical to no bias, which warm-start and the
                                eval fast path rely on)
      0 < g < 1 -> log(g+eps)  (differentiable; eps=1e-4 caps the log gradient at 1e4)
      g = 0     -> -1e9        (hard off, the value Phase 1 masks with)
    g = 0 and g = 1 fall in the flat clamp region of hard-concrete, where the gradient is already
    zero, so substituting a constant through torch.where does not break it.
    """
    g32 = g.float()
    return torch.where(
        g32 >= 1.0,
        torch.zeros_like(g32),
        torch.where(g32 > 0.0, torch.log(g32 + eps), torch.full_like(g32, -1e9)),
    )
