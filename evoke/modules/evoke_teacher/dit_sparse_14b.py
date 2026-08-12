"""
Wan2.2 I2V A14B Sparse Attention DiT — V10

Core changes in V10 vs V8mt:
  - drop patch_embedding_2x/4x/8x (static Conv3d compression)
  - nearby/select now spatially downsample the current hidden state on the fly (F.interpolate)
  - multi_scale_x / multi_scale_freqs are no longer needed
  - no need to replicate multi_scale_x under SP (saves ~10GB)
  - downsampled K gets scale-back RoPE (positions mapped back to the original grid coords, relative position encoding preserved)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from einops import rearrange


_gpu_cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = _gpu_cap[0] >= 9  # FA3 requires Hopper (sm_90+)
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = _gpu_cap[0] >= 8  # FA2 requires Ampere (sm_80+)
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_2_AVAILABLE = False

print(f"[Attention Backend] FA3={FLASH_ATTN_3_AVAILABLE}, FA2={FLASH_ATTN_2_AVAILABLE}, GPU={_gpu_cap}")

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    SAGE_ATTN_AVAILABLE = False

from .select_gate import (
    compute_select_keep,            # Phase 1: zscore hard gate (no_grad, kept as-is)
    compute_select_gate_features,   # Phase 2: learned gate input features (differentiable)
    SelectGateHead,                 # Phase 2: hard-concrete learnable gate head
    gate_to_bias,                   # Phase 2: gate value g -> additive bias (log-gate)
)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False, attn_bias=None):
    # attn_bias: additive float bias, shape broadcastable to (B, num_heads, Lq, Lk), e.g. (B, 1, 1, Lk).
    # select-gate uses it to set the K columns of gated-off frames to -1e9 (softmax weight -> 0).
    # FA2/FA3/Sage do not support arbitrary additive bias -> force SDPA when attn_bias is not None;
    # when attn_bias is None every branch behaves exactly as the original implementation.
    if attn_bias is not None:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def _scale_to_tokens(scale_factor: int, per_frame_tokens: int, spatial_hw=(30, 52)) -> int:
    """number of tokens left after spatial downsampling by the given scale factor."""
    H, W = spatial_hw
    return (H // scale_factor) * (W // scale_factor)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024*8, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class LinearAttention(nn.Module):
    """
    Linear Attention module: compresses sequence information into a state representation at O(N) cost.

    Core idea:
    - use the feature map phi(x) = elu(x) + 1 instead of softmax
    - compute: out = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T @ 1)
    - phi(K)^T @ V can be seen as the compressed state representation

    14B optimization: inner_dim lets linear attention run in a lower-dim space, cutting params and memory.
    Q/K/V: dim → inner_dim, O: inner_dim → dim
    """
    def __init__(self, dim: int, num_heads: int, state_dim: int = None,
                 inner_dim: int = None, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        # inner_dim: internal dim of linear attention (default = dim, 1024 recommended for 14B)
        self.inner_dim = inner_dim if inner_dim is not None else dim
        self.num_heads = num_heads if inner_dim is None else max(1, self.inner_dim // (dim // num_heads))
        self.head_dim = self.inner_dim // self.num_heads
        self.state_dim = state_dim if state_dim is not None else self.inner_dim
        self.eps = eps

        # Q, K, V projections: dim -> inner_dim
        self.q = nn.Linear(dim, self.inner_dim)
        self.k = nn.Linear(dim, self.inner_dim)
        self.v = nn.Linear(dim, self.inner_dim)
        # O projection: inner_dim -> dim (map back to the original dim)
        self.o = nn.Linear(self.inner_dim, dim)

        # optional: state projection layer
        self.state_proj = nn.Linear(self.head_dim * self.head_dim, self.state_dim) if state_dim is not None else None

        # Normalization (in inner_dim space)
        self.norm_q = RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = RMSNorm(self.inner_dim, eps=eps)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """feature map function: phi(x) = elu(x) + 1"""
        return F.elu(x) + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor [B, N, D]

        Returns:
            out: linear attention output [B, N, D]
            state: compressed state representation [B, num_heads, head_dim, head_dim] or [B, num_heads, state_dim]
            z: normalization factor [B, num_heads, head_dim]
        """
        B, N, D = x.shape

        # project Q, K, V: [B, N, dim] -> [B, N, inner_dim]
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        # reshape to multi-head layout: [B, num_heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # apply feature map
        q = self.feature_map(q)  # [B, num_heads, N, head_dim]
        k = self.feature_map(k)  # [B, num_heads, N, head_dim]

        # compute state: S = K^T @ V  [B, num_heads, head_dim, head_dim]
        state = torch.einsum('bhnd,bhnv->bhdv', k, v)

        # compute normalization factor: z = K^T @ 1  [B, num_heads, head_dim]
        z = k.sum(dim=2)  # [B, num_heads, head_dim]

        # compute output: out = Q @ S / (Q @ z)
        # Q @ S: [B, num_heads, N, head_dim]
        qkv = torch.einsum('bhnd,bhdv->bhnv', q, state)

        # Q @ z: [B, num_heads, N]
        qk_sum = torch.einsum('bhnd,bhd->bhn', q, z).unsqueeze(-1) + self.eps

        # normalize
        out = qkv / qk_sum  # [B, num_heads, N, head_dim]

        # reshape back to inner_dim, then O-project back to dim
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_dim)
        out = self.o(out)  # [B, N, inner_dim] → [B, N, dim]

        # Apply state projection if defined
        if self.state_proj is not None:
            state = self.state_proj(state.flatten(2))

        # cache phi(Q) (multi-head layout) so the output can be recomputed after the SP all-reduce, avoiding duplicate work
        self._cached_q_mapped = q  # [B, num_heads, N, head_dim]

        return out, state, z


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6,
                 sparse_attn: bool = True, chunk_size: int = 256, overlap_size: int = 32, num_global_tokens: int = 8,
                 per_frame_tokens: int = 30 * 52,  # 1560, for 480×832 with Wan2.1 VAE (stride 4×8×8)
                 num_retained_tokens: int = 1024,
                 # Frame-level select config
                 num_select_frames: int = 4,           # number of importance-selected frames
                 num_nearby_frames: int = 3,            # number of nearby framepack frames
                 # long-video memory optimizations
                 chunk_batch_size: int = None,           # chunks processed per batch (None=all at once, 30 ~= 10s)
                 inner_checkpoint: bool = False,         # checkpoint inside FFN/linear_attn
                 lazy_qkv: bool = False,                 # compute Q/K/V on demand (not over the full sequence, saves ~13.5GB)
                 select_scales: list = None,             # per-frame compression ratio for importance select, e.g. ['1x','2x','4x','8x']
                 # Select-gate: relevance gating of importance-select recalled frames ('none' = exactly the original behaviour)
                 select_gate_mode: str = 'none',         # 'none' | 'zscore'(Phase 1) | 'learned'(Phase 2)
                 select_gate_kappa: float = 2.0,         # [zscore] z-score threshold (larger = more conservative)
                 select_gate_cos_floor: float = -1.0,    # [zscore] absolute cosine floor (-1=disabled; under collapsed geometry an absolute threshold drifts across layers and kills good frames, see measured calibration)
                 select_gate_min_candidates: int = 8,    # [zscore] fewer candidates than this -> keep all (null distribution unreliable)
                 select_gate_mad_floor: float = 1e-6,    # [zscore] MAD below this -> keep all (scores indistinguishable); measured MAD is ~1e-5, the old 1e-3 fired the fallback 100% of the time making the gate a no-op
                 select_gate_min_keep: int = 0,          # [zscore] force-keep the top-N frames by z (0 = allow recalling 0 frames)
                 # Select-gate Phase 2 (learned) specific params
                 select_gate_temp: float = 0.6667,       # [learned] hard-concrete temperature beta
                 select_gate_budget_target: float = 0.5, # [learned] sparsity budget target: expected mean open probability
                 select_gate_budget_weight: float = 0.0, # [learned] budget regularizer lambda (0=off, read by loss.py)
                 # Sink Distance-Decay (SDF): mitigates long-video "first-frame lock-in / rewind"
                 sink_decay_mode: str = 'none',          # 'none'(=bit-identical to baseline) | 'downsample'(far chunks use a downsampled sink)
                 sink_decay_onset: int = 40,             # [downsample] global start frame < onset uses the full-res sink (opening identity anchor); >= onset uses the downsampled sink
                 sink_decay_factor: int = 2,             # [downsample] spatial downsample factor of the far sink (2/4/8), keeps low-freq identity, removes the copyable high-fidelity frame0
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.head_dim = dim // num_heads

        # Sparse Attention Config
        self.sparse_attn = sparse_attn
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.num_global_tokens = num_global_tokens
        self.num_retained_tokens = num_retained_tokens

        # Frame-level select config
        self.num_select_frames = num_select_frames
        self.num_nearby_frames = num_nearby_frames

        # long-video memory optimizations
        self.chunk_batch_size = chunk_batch_size
        self.inner_checkpoint = inner_checkpoint
        self.lazy_qkv = lazy_qkv
        # select_scales: '1x'=full resolution, '2x'/'4x'/'8x'=spatial downsample (F.interpolate)
        self.select_scales = select_scales or ['1x', '2x', '4x', '8x']

        # Select-gate: relevance gating of importance-select recalled frames ('none' = exactly the original behaviour)
        self.select_gate_mode = select_gate_mode
        self.select_gate_kappa = select_gate_kappa
        self.select_gate_cos_floor = select_gate_cos_floor
        self.select_gate_min_candidates = select_gate_min_candidates
        self.select_gate_mad_floor = select_gate_mad_floor
        self.select_gate_min_keep = select_gate_min_keep
        # Select-gate Phase 2 (learned) hyperparams
        self.select_gate_temp = select_gate_temp
        self.select_gate_budget_target = select_gate_budget_target
        self.select_gate_budget_weight = select_gate_budget_weight
        # Sink Distance-Decay (SDF): config-only (no new weights/buffers -> no extra state_dict key)
        self.sink_decay_mode = sink_decay_mode
        self.sink_decay_onset = sink_decay_onset
        self.sink_decay_factor = sink_decay_factor
        # debug/stats: (kept, total) per chunk from the last forward, filled only when the gate is on
        self._select_gate_last_stats = []
        # [learned] gate input features stashed on every forward (detached fp32), read by the loss.py budget regularizer.
        # : forward attributes are not differentiable under gradient checkpointing,
        # so we stash the features and let loss.py re-run the light gate head to get exact gate-param grads for the regularizer.
        self._select_gate_reg_feats = []

        # Linear Attention module (inner_dim=1024 to cut params and memory)
        self.linear_attn = LinearAttention(dim, num_heads, inner_dim=1024, eps=eps)
        self.linear_attn_norm = nn.LayerNorm(dim, eps=eps)

        # Sparse Attention modules
        if self.sparse_attn:
            # importance_head: frame-level scoring (token scores averaged per frame)
            self.importance_head = nn.Linear(dim, 1)
            # init moved into reinit_sparse_modules() so ZeRO-3 param sharding does not break xavier

            # State query module (projects into the linear attention inner dim)
            _la_inner = self.linear_attn.inner_dim  # 1024
            _la_heads = self.linear_attn.num_heads   # 8
            self.chunk_to_state_proj = nn.Linear(dim, _la_inner)
            self.chunk_to_state_norm = RMSNorm(_la_inner, eps=eps)

            # global attention output projection (inner_dim back to dim)
            self.global_attn_out_proj = nn.Linear(_la_inner, dim)

            # per-head gate (uses the linear attention head count)
            self.global_attn_gate = nn.Parameter(torch.zeros(_la_heads))

            # each layer queries the global context with its own state (no cross-layer accumulation)

            # learnable overlap blend sharpness factor
            self.blend_sharpness = nn.Parameter(torch.zeros(1))

        # Select-gate Phase 2: learnable gate head - built only in learned mode,
        # so none/zscore models keep a bit-identical state_dict key set (old-ckpt compatible).
        # the name 'select_gate_head' is registered in trainable_groups._SPARSE_KEYWORDS /
        # logger.SPARSE_KEYWORDS / runner resume / load_sparse_checkpoint - four keyword lists.
        self.select_gate_head = None
        if self.sparse_attn and select_gate_mode == 'learned':
            self.select_gate_head = SelectGateHead(
                feat_dim=3, hidden_dim=16, temperature=select_gate_temp)

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.per_frame_tokens = per_frame_tokens

    def _query_global_state(self, chunk_x: torch.Tensor, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        query this layer's global state with the chunk hidden (Linear Attention style).
        uses only the current layer's state, no cross-layer accumulation.

        Args:
            chunk_x: [B, L, D] - raw hidden of the chunk (D = model dim, e.g. 5120)
            state: [B, la_heads, la_head_dim, la_head_dim] - linear attention state
            z: [B, la_heads, la_head_dim] - linear attention normalization factor

        Returns:
            out: [B, L, D] - global context output
        """
        B, L, D = chunk_x.shape
        la_heads = self.linear_attn.num_heads
        la_head_dim = self.linear_attn.head_dim
        la_inner = self.linear_attn.inner_dim

        # map into the linear attention inner space and apply the feature map
        q = self.chunk_to_state_norm(self.chunk_to_state_proj(chunk_x))  # [B, L, la_inner]
        q = q.view(B, L, la_heads, la_head_dim).transpose(1, 2)
        q = F.elu(q) + 1  # feature map: φ(x) = elu(x) + 1

        # Linear Attention: out = φ(Q) @ state / (φ(Q) @ z)
        qkv = torch.einsum('bhnd,bhdv->bhnv', q, state)
        qk_sum = torch.einsum('bhnd,bhd->bhn', q, z).unsqueeze(-1) + 1e-6

        out = qkv / qk_sum
        out = out.transpose(1, 2).contiguous().view(B, L, la_inner)
        out = self.global_attn_out_proj(out)  # [B, L, la_inner] → [B, L, D]
        return out

    def sparse_self_attention(self, x, freqs, frame_keys, sink_len, spatial_hw, state, z,
                              sp_frame_offset=0, sp_num_frames_global=None, freqs_full=None,
                              freqs_3d=None, select_gate_t_frac=None):
        """
        v8 sparse self-attention: per-chunk importance select + state query.

        Args:
            x: [B, S, D] - input sequence (under SP, the tokens of the local frames)
            freqs: RoPE frequencies (under SP, the freqs of the local frames)
            frame_keys: [B, F_global, 1024] - global frame keys (1024-dim, already all-gathered under SP)
            sink_len: number of sink tokens
            spatial_hw: tuple (H, W) spatial grid size, used for F.interpolate downsampling
            state: [B, la_heads, la_head_dim, la_head_dim] - linear attention state (already all-reduced under SP)
            z: [B, la_heads, la_head_dim] - normalization factor (already all-reduced under SP)
            sp_frame_offset: global start frame index of this card's frames under SP
            sp_num_frames_global: total global frame count under SP (None = not SP)
            freqs_3d: tuple (f_cis, h_cis, w_cis) 3D RoPE components, used to build scale-back RoPE for compressed K
            select_gate_t_frac: [learned select-gate only] normalized timestep (t/1000, in [0,1]),
                shape (B,) or a scalar tensor. none/zscore modes ignore this argument entirely.
        """
        assert freqs_3d is not None, (
            "freqs_3d must be provided for scale-back RoPE on compressed K. "
            "Check that WanModel.forward() passes freqs_3d=self.freqs in _block_extra_kw."
        )
        B, S, D = x.shape
        pf = self.per_frame_tokens
        num_frames = S // pf  # local frame count
        num_frames_global = sp_num_frames_global if sp_num_frames_global else num_frames
        lazy_qkv = getattr(self, 'lazy_qkv', False)
        _sp = sp_num_frames_global is not None  # whether SP is active

        # === Q/K/V: lazy mode computes on demand, otherwise the full sequence is computed at once ===
        if not lazy_qkv:
            # standard mode: compute full-sequence Q/K/V at once (short video / inference)
            q_full = self.self_attn.norm_q(self.self_attn.q(x))
            k_full = self.self_attn.norm_k(self.self_attn.k(x))
            v_full = self.self_attn.v(x)
            q_full = rope_apply(q_full, freqs, self.num_heads)
            k_full = rope_apply(k_full, freqs, self.num_heads)

        else:
            # Lazy mode
            q_full = k_full = v_full = None

        # Sink K/V: global frame 0; under SP card 0 computes it and broadcasts to the other cards
        x_sink = x[:, :pf]
        freqs_sink = freqs[:pf]
        k_sink = rope_apply(self.self_attn.norm_k(self.self_attn.k(x_sink)), freqs_sink, self.num_heads)
        v_sink = self.self_attn.v(x_sink)
        if _sp:
            from .sp_runtime import broadcast_with_grad, halo_exchange
            k_sink = broadcast_with_grad(k_sink.contiguous(), src_rank=0)
            v_sink = broadcast_with_grad(v_sink.contiguous(), src_rank=0)
        # WARNING SDF SP safety: in downsample mode far chunks switch to k_sink_ds(no_grad), so the full-res k_sink is used only by "near chunks".
        #   -> a rank that owns only far frames (f_start >= onset) no longer uses the full k_sink -> its broadcast backward collective never fires -> NCCL
        #   misalignment deadlock against the near ranks (measured on a smoke run: forward completed at loss=0.04, backward hung on the first step).
        #   fix: in SDF mode treat the whole sink (near full + far downsample) uniformly as no_grad identity context -> no sink-broadcast
        #   backward collective at all -> independent of "which card uses which sink", dangling fully eliminated. none mode does not detach, bit-identical to baseline.
        # under 2D SP the sink must be detached too (not only in downsample mode):
        #   the broadcast_with_grad backward of k_sink/v_sink is an SP-group **group collective** (dist.reduce), which _nccl_anchor cannot
        #   rescue (group collectives are matched by order, cards fire backward at different moments -> order misalignment, see the hard-won note at L606). after mpu first got 2D backward
        # working we measured: step0 passed on all 3 engines (the deadlocks are solved), but later backward hung -- in every SP group
        #   sp_rank1 was stuck together in one SP group collective waiting for sp_rank0, which is exactly this sink-broadcast backward order misalignment. detach -> does not register
        #   the sink-broadcast backward group collective -> sp_rank0/1 stay symmetric. cost: sink k/v (incl. critic-LoRA) get no gradient from the frame0-sink attn path
        #   (they still train as usual through local/nearby/full-attn), the same trade-off as downsample SDF (L544, L608-609).
        #   G=1: _sp=False and sink_decay_mode='none' -> no detach -> bit-identical to baseline.
        if self.sink_decay_mode == 'downsample' or _sp:
            k_sink = k_sink.detach()
            v_sink = v_sink.detach()

        # Halo exchange: under SP fetch the nearby boundary frames from the previous card
        halo_prev = None
        if _sp and self.num_nearby_frames > 0:
            halo_prev = halo_exchange(x, self.num_nearby_frames, pf)  # [B, nearby*pf, D] or [B, 0, D]

        # Helper: in lazy mode compute K/V on demand and apply RoPE
        def _lazy_kv(x_slice, freqs_slice):
            k_s = rope_apply(self.self_attn.norm_k(self.self_attn.k(x_slice)), freqs_slice, self.num_heads)
            v_s = self.self_attn.v(x_slice)
            return k_s, v_s

        # take K/V by spatially downsampling the current hidden state
        _scale_map = {'1x': 1, '2x': 2, '4x': 4, '8x': 8}
        H_sp, W_sp = spatial_hw if spatial_hw else (30, 52)

        def _downsample_kv(frame_tokens, scale_factor, global_frame_idx=None):
            """spatially downsample one frame of tokens, then compute K/V with scale-back RoPE.
            frame_tokens: [B, H*W, D] = [B, pf, D]
            scale_factor: 2/4/8
            global_frame_idx: int, global frame index (used to build the 3D RoPE)
            returns: (k, v), each [B, H'*W', D]
            """
            ft = frame_tokens.view(B, H_sp, W_sp, D).permute(0, 3, 1, 2)  # [B, D, H, W]
            H_out, W_out = H_sp // scale_factor, W_sp // scale_factor
            ft = F.interpolate(ft, size=(H_out, W_out), mode='bilinear', align_corners=False)
            ft = ft.permute(0, 2, 3, 1).reshape(B, H_out * W_out, D)  # [B, t_ds, D]
            k_s = self.self_attn.norm_k(self.self_attn.k(ft))
            v_s = self.self_attn.v(ft)
            # Scale-back RoPE: map the downsampled positions back to the original grid coords
            # h_indices = [0, sf, 2*sf, ...], w_indices = [0, sf, 2*sf, ...]
            # keeps the relative positions between Q (original coords) and K (scale-back coords) correct
            if global_frame_idx is not None and freqs_3d is not None:
                # debug switch: set DISABLE_SCALE_BACK_ROPE=1 to skip RoPE on the downsampled K, for chasing grid artifacts
                if os.environ.get('DISABLE_SCALE_BACK_ROPE', '0') == '1':
                    return k_s, v_s
                f_cis, h_cis, w_cis = freqs_3d
                h_idx = torch.arange(H_out, device=h_cis.device) * scale_factor  # scale-back to original grid
                w_idx = torch.arange(W_out, device=h_cis.device) * scale_factor
                ds_freqs = torch.cat([
                    f_cis[global_frame_idx].view(1, 1, -1).expand(H_out, W_out, -1),
                    h_cis[h_idx].view(H_out, 1, -1).expand(H_out, W_out, -1),
                    w_cis[w_idx].view(1, W_out, -1).expand(H_out, W_out, -1),
                ], dim=-1).reshape(H_out * W_out, 1, -1).to(k_s.device)
                k_s = rope_apply(k_s, ds_freqs, self.num_heads)
            return k_s, v_s

        # Sink Distance-Decay (SDF): in downsample mode precompute one "downsampled sink" K/V (global frame 0, scale-back RoPE),
        # for far chunks (ext_f_start_g >= onset) to use instead of the full-res sink -> keeps low-freq identity, removes the pixel-copyable sharp frame0.
        # under SP same as the full sink: each card computes it locally, then broadcast from rank0 (x_sink on rank != 0 is not the real frame 0 and gets overwritten).
        # none mode never enters this block -> the sink path is bit-identical to baseline.
        k_sink_ds = v_sink_ds = None
        if self.sink_decay_mode == 'downsample':
            # WARNING SP safety (hard-won): the downsampled sink is computed+broadcast under no_grad -> the broadcast registers no backward collective op.
            #   otherwise the backward of broadcast_with_grad is a collective, while "which rank uses k_sink_ds" depends on whether that card's local
            #   chunks are >= onset -- the card with f_start=0 has only near chunks (< onset) -> never uses k_sink_ds -> its backward
            #   collective never fires, misaligning the NCCL calls against the cards that do use it -> deadlock (a smoke run measured hanging on the first backward step).
            #   note: a zero-contribution _nccl_anchor cannot rescue this -- broadcast is a "group collective" and cards fire backward at different moments -> order misalignment;
            #   the existing halo/exchange anchors are P2P (matched by src/dst, tolerant of reordering), a different class.
            #   the downsampled sink as no_grad context KV: forward content is normal, attn weights/query stay differentiable (the backbone trains through q),
            #   only the sink's own k/v projections get no gradient from this path (they still train as usual via full-sink/local/nearby); inference has no grad and is unaffected.
            with torch.no_grad():
                k_sink_ds, v_sink_ds = _downsample_kv(x_sink, self.sink_decay_factor, global_frame_idx=0)
                if _sp:
                    from .sp_runtime import broadcast_with_grad
                    k_sink_ds = broadcast_with_grad(k_sink_ds.contiguous(), src_rank=0)
                    v_sink_ds = broadcast_with_grad(v_sink_ds.contiguous(), src_rank=0)

        # select-gate: reset the recall stats of this forward
        if self.select_gate_mode != 'none':
            self._select_gate_last_stats = []

        # select-gate(learned): input validation + normalized timestep row vector (B,) fp32
        _sg_t_frac_row = None
        if self.select_gate_mode == 'learned':
            if getattr(self, 'select_gate_head', None) is None:
                raise RuntimeError(
                    "select_gate_mode='learned' but select_gate_head was not built "
                    "(sparse_attn=False, or mode was not 'learned' when the DiTBlock was constructed).")
            if select_gate_t_frac is None:
                raise RuntimeError(
                    "select_gate_mode='learned' requires select_gate_t_frac (normalized timestep, t/1000). "
                    "Check that the caller forwards it: WanModel.forward / model_fn_wan_video (_pass_spatial_hw branch) "
                    "/ WanModelCam.forward via _block_extra_kw['select_gate_t_frac'].")
            _sg_t = select_gate_t_frac.float().reshape(-1)
            if _sg_t.numel() == 1:
                _sg_t = _sg_t.expand(B)
            elif _sg_t.numel() != B:
                raise ValueError(
                    f"select_gate_t_frac numel={_sg_t.numel()} does not match batch B={B} (and is not a scalar).")
            _sg_t_frac_row = _sg_t.to(x.device)
            # reset the stash on every forward (multi-step denoising at inference cannot accumulate leaks; loss.py clears it right after reading)
            self._select_gate_reg_feats = []

        # 3. Chunked attention
        chunk_f = self.chunk_size
        overlap_f = self.overlap_size
        overlap_tokens = overlap_f * pf
        num_chunks = (num_frames + chunk_f - 1) // chunk_f

        # Pre-compute blend weights
        blend_len = 2 * overlap_tokens
        if overlap_tokens > 0 and blend_len > 0:
            positions = torch.arange(blend_len, device=x.device, dtype=x.dtype)
            t_blend = 2.0 * (positions + 1.0) / (blend_len + 1.0) - 1.0
            sharpness = F.softplus(self.blend_sharpness) + 1.0
            alpha = torch.sigmoid(sharpness * t_blend)
            alpha_next = alpha.unsqueeze(0).unsqueeze(-1)
            alpha_prev = 1.0 - alpha_next

        # State query gate
        gate_scalar = torch.sigmoid(self.global_attn_gate).mean() if state is not None else None

        # === Fused chunk Q/K/V computation + batched attention ===
        # instead of first collecting Q/K/V of all chunks and then attending, work in batches: collect a small batch -> attention -> free Q/K/V
        cbs = self.chunk_batch_size if hasattr(self, 'chunk_batch_size') and self.chunk_batch_size else num_chunks

        # Helper: build Q, K_ctx, V_ctx of a single chunk
        # under SP, ci is a local chunk index, converted to a global frame index via sp_frame_offset
        _sp = sp_num_frames_global is not None  # whether SP is active

        # === SP Pre-score: precompute the select frame indices of all chunks, collect the remote frame requests ===
        _scale_map_inv = {'1x': 1, '2x': 2, '4x': 4, '8x': 8}
        precomputed_select = {}  # {chunk_idx: (top_indices [B, num_select], scales [num_select])}
        remote_token_cache = {}  # {global_frame_idx: [B, pf, D]} — autograd-aware raw tokens
        _exchange_received = None  # raw received tensor for autograd anchor
        _dummy_score_anchors = []  # SP symmetrization: chunk_score_q of dummy chunks, wired into attn_output at the end with zero contribution

        if _sp and self.num_select_frames > 0:
            from .sp_runtime import get_sp_rank, get_sp_size, exchange_frame_tokens
            _sp_rank = get_sp_rank()
            _sp_size = get_sp_size()
            _frames_per_rank = (sp_num_frames_global + _sp_size - 1) // _sp_size

            remote_frame_requests = {}  # {source_sp_rank: [global_frame_idx, ...]}
            remote_frame_set = set()    # dedup by gfi (raw tokens are scale-independent)

            # SP deadlock fix v3: sp_frame_offset differs per rank -> num_available differs ->
            # the old code, if num_available<=0: continue, made ranks skip a different number of chunks,
            # so chunk_to_state_proj/topk call counts diverged across SP, triggering ZeRO-2 reduce_partition
            # / autograd-graph backward NCCL op count misalignment -> SeqNum divergence -> ALLGATHER deadlock.
            # fix: every rank walks the full num_chunks loop and calls the same ops on every chunk;
            # num_available<=0 runs topk with dummy keys; topk always asks for _ns_target entries
            # (scores padded with -inf when short), so call counts are fully SP-aligned.
            _ns_target = max(1, self.num_select_frames)

            for ci in range(num_chunks):
                f_start = ci * chunk_f
                ext_f_start = max(0, f_start - overlap_f)
                ext_f_end = min(num_frames, min(f_start + chunk_f, num_frames) + overlap_f)
                ext_t_start = ext_f_start * pf
                ext_t_end = ext_f_end * pf
                ext_f_start_g = ext_f_start + sp_frame_offset

                nearby_boundary_g = max(1, ext_f_start_g - self.num_nearby_frames)
                num_available = nearby_boundary_g - 1
                _is_dummy = (num_available <= 0)

                # 1024-dim scoring (all ranks run it, dummy uses frame_keys[:, 0:1])
                x_ext_ci = x[:, ext_t_start:ext_t_end]
                chunk_score_q = self.chunk_to_state_proj(x_ext_ci.mean(dim=1))  # [B, 1024]
                if _is_dummy:
                    available_keys = frame_keys[:, 0:1]                     # [B, 1, 1024]
                else:
                    available_keys = frame_keys[:, 1:nearby_boundary_g]     # [B, num_available, 1024]
                scores = torch.einsum('bd,bnd->bn', chunk_score_q, available_keys)  # [B, K]

                # topk always asks for _ns_target: pad with -inf when short (never selected, only keeps kernel call counts SP-aligned)
                if scores.shape[1] < _ns_target:
                    _pad_n = _ns_target - scores.shape[1]
                    scores = torch.cat(
                        [scores, scores.new_full((scores.shape[0], _pad_n), float('-inf'))],
                        dim=1)
                _, top_indices_full = scores.topk(_ns_target, dim=1)        # [B, _ns_target]
                top_indices_full = top_indices_full + 1  # global frame index

                if _is_dummy:
                    # dummy chunk: run through the same ops but do not write precomputed_select / do not send remote requests
                    # chunk_score_q is collected into the anchor list and wired into attn_output at the end with zero contribution so backward aligns too
                    _dummy_score_anchors.append(chunk_score_q)
                    # select-gate(learned): dummy chunks also run the gate head once (zero features, deterministic),
                    # alpha is wired into attn_output with zero contribution -> the op sequence per local chunk aligns across ranks
                    # (same pattern as the v3 fix), and the gate params are not dangling on all-dummy ranks either (ZeRO-2 safe).
                    if self.select_gate_mode == 'learned':
                        _sg_dummy_feats = torch.zeros(B, 1, 3, device=x.device, dtype=torch.float32)
                        _, _sg_dummy_alpha = self.select_gate_head(_sg_dummy_feats, training=False)
                        _dummy_score_anchors.append(_sg_dummy_alpha.to(x.dtype))
                    continue

                # real chunk: the actually usable num_select (clamped to num_available, same semantics as before)
                num_select = min(self.num_select_frames, num_available)
                top_indices = top_indices_full[:, :num_select]

                # select-gate: ranking is unchanged (raw-dot topk), the gate only decides how strongly the selected frames take effect.
                # purely local computation, does not change the communicated frame set or the number of collectives -> SP safe.
                # top_indices are global frame indices (already +1); subtract 1 to get the index relative to available_keys.
                # 'zscore'(Phase 1): hard keep/mask (the condition is bit-equivalent to the original `!= 'none'` in a two-mode world);
                # 'learned'(Phase 2): differentiable hard-concrete gate, gate_vals in [0,1].
                keep_mask = None
                gate_vals_pre = None
                if self.select_gate_mode == 'zscore':
                    keep_mask = compute_select_keep(
                        chunk_score_q, available_keys, top_indices - 1,
                        kappa=self.select_gate_kappa,
                        cos_floor=self.select_gate_cos_floor,
                        min_candidates=self.select_gate_min_candidates,
                        mad_floor=self.select_gate_mad_floor,
                        min_keep=self.select_gate_min_keep,
                    )
                elif self.select_gate_mode == 'learned':
                    _sg_feats = compute_select_gate_features(
                        chunk_score_q, available_keys, top_indices - 1, _sg_t_frac_row)  # (B, K, 3)
                    gate_vals_pre, _sg_alpha = self.select_gate_head(
                        _sg_feats, training=self.training)                               # (B, K) fp32
                    # stash detached features for the loss.py budget regularizer
                    self._select_gate_reg_feats.append(_sg_feats.detach())

                # record the scale of each frame
                scales = []
                for si in range(num_select):
                    _ss = self.select_scales[min(si, len(self.select_scales) - 1)]
                    sf = _scale_map_inv.get(_ss, 2)
                    scales.append(sf)

                # learned mode stores a 4-tuple; zscore/none keep the 3-tuple (byte-identical)
                if gate_vals_pre is not None:
                    precomputed_select[ci] = (top_indices, scales, keep_mask, gate_vals_pre)
                else:
                    precomputed_select[ci] = (top_indices, scales, keep_mask)

                # sort out the remote frames (collect gfi only, scale irrelevant)
                for si in range(num_select):
                    gfi = top_indices[0, si].item()
                    _local_start = sp_frame_offset
                    _local_end = _local_start + num_frames
                    if gfi >= _local_start and gfi < _local_end:
                        continue  # local frame, no communication needed

                    src_rank = min(gfi // _frames_per_rank, _sp_size - 1)
                    if gfi not in remote_frame_set:
                        remote_frame_set.add(gfi)
                        remote_frame_requests.setdefault(src_rank, []).append(gfi)

            # batched P2P exchange of remote-frame raw tokens (autograd-aware, the requester computes K/V locally)
            # note: every rank must call this (it contains an all_to_all collective)
            if True:
                remote_token_cache, _exchange_received = exchange_frame_tokens(
                    requests=remote_frame_requests,
                    x=x,
                    per_frame_tokens=pf,
                    sp_frame_offset=sp_frame_offset,
                    num_local_frames=num_frames,
                    frames_per_rank=_frames_per_rank,
                )

        def _build_chunk(ci):
            # local frame indices
            f_start = ci * chunk_f
            f_end = min(f_start + chunk_f, num_frames)
            t_start = f_start * pf
            t_end = f_end * pf
            ext_f_start = max(0, f_start - overlap_f)
            ext_f_end = min(num_frames, f_end + overlap_f)
            ext_t_start = ext_f_start * pf
            ext_t_end = ext_f_end * pf

            # global frame indices (used for the sink / nearby decisions and importance selection)
            f_start_g = f_start + sp_frame_offset
            ext_f_start_g = ext_f_start + sp_frame_offset

            x_ext = x[:, ext_t_start:ext_t_end]

            if not lazy_qkv:
                q_ext = q_full[:, ext_t_start:ext_t_end]
            else:
                q_ext = self.self_attn.norm_q(self.self_attn.q(x_ext))
                q_ext = rope_apply(q_ext, freqs[ext_t_start:ext_t_end], self.num_heads)

            k_parts, v_parts = [], []
            select_bias = None  # select-gate: [B, kv_len] additive bias (None = no gated-off frames)

            # (a) Sink (global frame 0) - both lazy and non-lazy use the broadcast k_sink/v_sink
            # SDF: in downsample mode, far chunks (ext_f_start_g >= onset away from the first frame) use the downsampled sink instead of full-res
            #      -> keeps low-freq identity and breaks the "pixel-copy the first frame" lock-in; none mode takes the else branch, bit-identical to baseline.
            if ext_f_start_g > 0:
                if (self.sink_decay_mode == 'downsample' and k_sink_ds is not None
                        and ext_f_start_g >= self.sink_decay_onset):
                    k_parts.append(k_sink_ds); v_parts.append(v_sink_ds)
                else:
                    k_parts.append(k_sink); v_parts.append(v_sink)

            # (b) Local K/V
            if not lazy_qkv:
                k_parts.append(k_full[:, ext_t_start:ext_t_end]); v_parts.append(v_full[:, ext_t_start:ext_t_end])
            else:
                k_local, v_local = _lazy_kv(x_ext, freqs[ext_t_start:ext_t_end])
                k_parts.append(k_local); v_parts.append(v_local)

            # (c) Nearby framepack - V10: take frames from the current x -> spatial downsample -> K/V
            #     under SP the remote frames come from halo_prev (the previous card's trailing frames)
            nearby_scale_factors = [2, 4, 8]
            for ni in range(self.num_nearby_frames):
                nearby_f_g = ext_f_start_g - 1 - ni  # global frame index
                if nearby_f_g <= 0: break
                sf = nearby_scale_factors[min(ni, len(nearby_scale_factors) - 1)]
                local_fidx = nearby_f_g - sp_frame_offset
                if 0 <= local_fidx < num_frames:
                    # local frame: read straight from x
                    frame_tokens = x[:, local_fidx * pf : (local_fidx + 1) * pf]
                elif _sp and halo_prev is not None and halo_prev.shape[1] > 0:
                    # SP remote frame: read from the halo (halo_prev holds the previous card's trailing nearby frames)
                    # halo_prev = [previous card's nearby-th-from-last frame, ..., last frame]
                    # local_fidx < 0 means it lives on the previous card, offset inside the halo = num_nearby + local_fidx
                    halo_fidx = self.num_nearby_frames + local_fidx
                    if 0 <= halo_fidx < self.num_nearby_frames:
                        frame_tokens = halo_prev[:, halo_fidx * pf : (halo_fidx + 1) * pf]
                    else:
                        continue  # outside the halo range
                else:
                    continue
                k_near, v_near = _downsample_kv(frame_tokens, sf, global_frame_idx=nearby_f_g)
                k_parts.append(k_near); v_parts.append(v_near)

            # (d) Importance select (1024-dim scoring)
            #     under SP: use the pre-score result + remote_kv_cache (K/V precomputed by the source card)
            #     non-SP: score in place + read the frames locally
            nearby_boundary_g = max(1, ext_f_start_g - self.num_nearby_frames)
            num_available = nearby_boundary_g - 1
            if num_available > 0 and self.num_select_frames > 0:
                # fetch top_indices and scales
                keep_mask = None  # select-gate: None = keep all (gate off, or everything passed)
                gate_vals = None  # select-gate(learned): (B, K) fp32 gate values, None = learned not enabled
                if _sp and ci in precomputed_select:
                    _sel_entry = precomputed_select[ci]
                    if len(_sel_entry) == 4:
                        top_indices, scales, keep_mask, gate_vals = _sel_entry
                    elif len(_sel_entry) == 3:
                        top_indices, scales, keep_mask = _sel_entry
                    else:
                        top_indices, scales = _sel_entry
                    num_select = top_indices.shape[1]
                else:
                    # non-SP, or not precomputed: score in place
                    chunk_score_q = self.chunk_to_state_proj(x_ext.mean(dim=1))  # [B, 1024]
                    available_keys = frame_keys[:, 1:nearby_boundary_g]
                    scores = torch.einsum('bd,bnd->bn', chunk_score_q, available_keys)
                    num_select = min(self.num_select_frames, num_available)
                    _, top_indices = scores.topk(num_select, dim=1)
                    # select-gate: ranking is unchanged (raw-dot topk), the gate only decides how strongly the selected frames take effect.
                    # here top_indices are still the indices relative to available_keys (before the +1).
                    # 'zscore'(Phase 1): hard keep/mask (the condition is bit-equivalent to the original `!= 'none'` in a two-mode world);
                    # 'learned'(Phase 2): differentiable hard-concrete gate.
                    if self.select_gate_mode == 'zscore':
                        keep_mask = compute_select_keep(
                            chunk_score_q, available_keys, top_indices,
                            kappa=self.select_gate_kappa,
                            cos_floor=self.select_gate_cos_floor,
                            min_candidates=self.select_gate_min_candidates,
                            mad_floor=self.select_gate_mad_floor,
                            min_keep=self.select_gate_min_keep,
                        )
                    elif self.select_gate_mode == 'learned':
                        _sg_feats = compute_select_gate_features(
                            chunk_score_q, available_keys, top_indices, _sg_t_frac_row)  # (B, K, 3)
                        gate_vals, _sg_alpha = self.select_gate_head(
                            _sg_feats, training=self.training)                           # (B, K) fp32
                        self._select_gate_reg_feats.append(_sg_feats.detach())
                    top_indices = top_indices + 1
                    scales = [_scale_map.get(self.select_scales[min(si, len(self.select_scales) - 1)], 2)
                              for si in range(num_select)]

                # select-gate: record where the select frames sit inside k_parts (used to build the attention bias).
                # the select frames are the last batch of parts appended; prefix length = total token count of the parts present at that moment.
                _num_parts_before_select = len(k_parts)
                _prefix_len = sum(p.shape[1] for p in k_parts)
                _sel_si_order = []  # the si corresponding to the j-th select part (a remote cache miss skips a frame)
                if keep_mask is not None:
                    self._select_gate_last_stats.append(
                        (int(keep_mask.sum().item()), keep_mask.numel()))
                if gate_vals is not None:
                    # learned: stats convention = number of frames with the gate open (g>0) / total frames
                    self._select_gate_last_stats.append(
                        (int((gate_vals > 0).sum().item()), gate_vals.numel()))

                for si in range(num_select):
                    frame_idx = top_indices[:, si]  # global frame index
                    sf = scales[si]
                    gfi = frame_idx[0].item()

                    # check whether it is local
                    _local_start = sp_frame_offset
                    _local_end = _local_start + num_frames
                    _is_local = (gfi >= _local_start and gfi < _local_end)

                    if not _is_local and _sp:
                        # remote frame: take raw tokens from token_cache, compute K/V locally (keeps autograd)
                        if gfi in remote_token_cache:
                            _rtk = remote_token_cache[gfi]  # [B, pf, D]
                            if sf == 1:
                                # full resolution + RoPE (take this frame's freqs from freqs_full)
                                _rf = freqs_full[gfi * pf:(gfi + 1) * pf] if freqs_full is not None else None
                                if _rf is not None:
                                    k_sel = rope_apply(self.self_attn.norm_k(self.self_attn.k(_rtk)),
                                                       _rf, self.num_heads)
                                else:
                                    k_sel = self.self_attn.norm_k(self.self_attn.k(_rtk))
                                v_sel = self.self_attn.v(_rtk)
                            else:
                                k_sel, v_sel = _downsample_kv(_rtk, sf, global_frame_idx=gfi)
                            k_parts.append(k_sel); v_parts.append(v_sel)
                            _sel_si_order.append(si)
                        continue

                    # local frame
                    local_frame_idx = frame_idx - sp_frame_offset if _sp else frame_idx

                    if sf == 1:
                        # full-resolution select: take the whole frame tokens from the local x
                        if not lazy_qkv:
                            token_offsets = torch.arange(pf, device=x.device)
                            gi = local_frame_idx.unsqueeze(-1) * pf + token_offsets.unsqueeze(0)
                            gi = gi.unsqueeze(-1).expand(-1, -1, D)
                            k_parts.append(torch.gather(k_full, 1, gi))
                            v_parts.append(torch.gather(v_full, 1, gi))
                            _sel_si_order.append(si)
                        else:
                            token_offsets = torch.arange(pf, device=x.device)
                            gi = local_frame_idx.unsqueeze(-1) * pf + token_offsets.unsqueeze(0)
                            gi_x = gi.unsqueeze(-1).expand(-1, -1, D)
                            x_sel = torch.gather(x, 1, gi_x)
                            freqs_sel = freqs[gi[0]]
                            sel_k, sel_v = _lazy_kv(x_sel, freqs_sel)
                            k_parts.append(sel_k); v_parts.append(sel_v)
                            _sel_si_order.append(si)
                    else:
                        # compressed select - take the frame from the current x -> spatial downsample -> K/V
                        t_off = local_frame_idx[0].item() * pf
                        frame_tokens = x[:, t_off : t_off + pf]
                        k_sel, v_sel = _downsample_kv(frame_tokens, sf, global_frame_idx=gfi)
                        k_parts.append(k_sel); v_parts.append(v_sel)
                        _sel_si_order.append(si)

                # select-gate(learned, value-gating): multiply the V of a selected frame by the gate value g in [0,1] (instead of adding
                # a log(g) bias to the attention scores) -> stays on FlashAttention throughout, no SDPA / no OOM / no bias.
                # semantics: out = sum_j softmax(s)_ij * (g_j v_j) -- the gate attenuates a selected frame's contribution (g=0 -> contribution zeroed).
                # difference vs log-bias: a gated-off frame still holds softmax weight (no redistribution), i.e. "attenuate" rather than "remove";
                #   the difference is just a scalar scaling that the downstream LayerNorm mostly absorbs, equivalent enough for a learnable gate, and buys full FA speed.
                # warm-start g==1 -> V unchanged (x1.0 bitwise) -> bit-identical to baseline (select_bias always None -> FA branch).
                # gradient: loss -> FA out -> (g*V) -> g -> hard-concrete -> gate params (through feats it also trains chunk_to_state_proj).
                # v_parts[_num_parts_before_select:] are exactly the select-frame V appended by this chunk, aligned one-to-one with _sel_si_order.
                if gate_vals is not None and _sel_si_order:
                    _g_dtype = gate_vals.to(x.dtype)  # (B, K)
                    for _j in range(len(v_parts) - _num_parts_before_select):
                        _pidx = _num_parts_before_select + _j
                        _g_col = _g_dtype[:, _sel_si_order[_j]].view(B, 1, 1)  # (B,1,1) broadcast over (tokens, D)
                        v_parts[_pidx] = v_parts[_pidx] * _g_col
                    # select_bias stays None -> the outer flash_attention takes the FA branch (no SDPA)
                # select-gate(zscore): build this chunk's additive bias - set the columns of gated-off frames to -1e9 (softmax weight -> 0).
                # keep-all (or gate off) -> select_bias stays None -> the outer call still takes the FA branch, same as the original behaviour.
                elif keep_mask is not None and _sel_si_order:
                    _col = _prefix_len
                    _off_spans = []  # [(col_start, col_end, keep_col [B] bool)]
                    for _j, _part in enumerate(k_parts[_num_parts_before_select:]):
                        _keep_col = keep_mask[:, _sel_si_order[_j]]  # [B] bool
                        if not bool(_keep_col.all()):
                            _off_spans.append((_col, _col + _part.shape[1], _keep_col))
                        _col += _part.shape[1]
                    if _off_spans:
                        _total_len = sum(p.shape[1] for p in k_parts)
                        select_bias = torch.zeros(B, _total_len, device=x.device, dtype=x.dtype)
                        for _cs, _ce, _keep_col in _off_spans:
                            select_bias[~_keep_col, _cs:_ce] = -1e9

            k_ctx = torch.cat(k_parts, dim=1)
            v_ctx = torch.cat(v_parts, dim=1)
            return ext_t_start, ext_t_end, t_start, t_end, q_ext, k_ctx, v_ctx, x_ext, select_bias

        # Fused loop: build Q/K/V in batches -> attention -> keep only the output
        chunk_outputs = []  # stores only (et_s, et_e, t_s, t_e, out_ext), not Q/K/V

        for batch_start in range(0, num_chunks, cbs):
            batch_end = min(batch_start + cbs, num_chunks)
            # build the chunk data of the current batch
            batch_data = [_build_chunk(ci) for ci in range(batch_start, batch_end)]
            nb = len(batch_data)

            max_q_len = max(c[4].shape[1] for c in batch_data)
            max_kv_len = max(c[5].shape[1] for c in batch_data)

            q_batch = torch.zeros(nb * B, max_q_len, D, device=x.device, dtype=x.dtype)
            k_batch = torch.zeros(nb * B, max_kv_len, D, device=x.device, dtype=x.dtype)
            v_batch = torch.zeros(nb * B, max_kv_len, D, device=x.device, dtype=x.dtype)

            # select-gate: build a bias only if some chunk in this batch has gated-off frames (otherwise None -> FA branch unchanged)
            bias_batch = None
            if any(c[8] is not None for c in batch_data):
                bias_batch = torch.zeros(nb * B, max_kv_len, device=x.device, dtype=x.dtype)

            for ci, (_, _, _, _, q_ci, k_ci, v_ci, _, bias_ci) in enumerate(batch_data):
                ql, kvl = q_ci.shape[1], k_ci.shape[1]
                q_batch[ci*B:(ci+1)*B, :ql] = q_ci
                k_batch[ci*B:(ci+1)*B, :kvl] = k_ci
                v_batch[ci*B:(ci+1)*B, :kvl] = v_ci
                if bias_batch is not None and bias_ci is not None:
                    bias_batch[ci*B:(ci+1)*B, :kvl] = bias_ci

            # drop the Q/K/V references held by batch_data (keep only metadata + x_ext)
            batch_meta = [(d[0], d[1], d[2], d[3], d[4].shape[1], d[7]) for d in batch_data]
            del batch_data

            # select-gate: when bias is not None, flash_attention internally forces SDPA (FA does not support additive bias)
            _attn_bias = bias_batch.view(nb * B, 1, 1, max_kv_len) if bias_batch is not None else None
            out_batch = flash_attention(q_batch, k_batch, v_batch, num_heads=self.num_heads, attn_bias=_attn_bias)
            del q_batch, k_batch, v_batch, bias_batch, _attn_bias

            for ci, (et_s, et_e, t_s, t_e, ql, x_ext) in enumerate(batch_meta):
                out_local = out_batch[ci*B:(ci+1)*B, :ql]
                if state is not None:
                    out_global = self._query_global_state(x_ext, state, z)
                    out_ext = out_local + gate_scalar * out_global
                else:
                    out_ext = out_local
                chunk_outputs.append((et_s, et_e, t_s, t_e, out_ext))

            del out_batch

        # Phase 2: assemble (exclusive region + blend region)
        attn_output = torch.zeros(B, S, D, device=x.device, dtype=x.dtype)

        for i, (ext_start_i, ext_end_i, cs_i, ce_i, out_i) in enumerate(chunk_outputs):
            excl_start = ext_start_i if i == 0 else cs_i + overlap_tokens
            excl_end = ext_end_i if i == num_chunks - 1 else ce_i - overlap_tokens

            if excl_end > excl_start:
                offset = excl_start - ext_start_i
                attn_output[:, excl_start:excl_end] = out_i[:, offset:offset + (excl_end - excl_start)]

            if i < num_chunks - 1 and overlap_tokens > 0:
                ext_start_next = chunk_outputs[i + 1][0]
                _, _, _, _, out_next = chunk_outputs[i + 1]
                bz_start = ce_i - overlap_tokens
                bz_end = min(ce_i + overlap_tokens, S)
                bz_len = bz_end - bz_start
                out_curr = out_i[:, (bz_start - ext_start_i):(bz_start - ext_start_i + bz_len)]
                out_next_bz = out_next[:, (bz_start - ext_start_next):(bz_start - ext_start_next + bz_len)]
                attn_output[:, bz_start:bz_end] = alpha_prev[:, :bz_len, :] * out_curr + alpha_next[:, :bz_len, :] * out_next_bz

        # SP deadlock fix: make sure halo_prev and exchange_received take part in the autograd graph.
        # rank 0 has an empty halo and an empty remote exchange -> they feed nothing downstream -> autograd never fires
        # the backward of those nodes -> missing P2P NCCL ops -> SeqNum divergence across ranks -> deadlock.
        # adding zero-contribution terms forces autograd backward to traverse every SP communication path.
        # v3: also wire in the chunk_to_state_proj output of dummy chunks (num_available<=0) with zero contribution,
        # so a dummy chunk does not become a dangling subgraph in backward -> ZeRO-2 hook / autograd NCCL call counts misaligned across SP.
        if _sp:
            _nccl_anchor = torch.zeros(1, device=attn_output.device, dtype=attn_output.dtype)
            if halo_prev is not None and isinstance(halo_prev, torch.Tensor) and halo_prev.requires_grad:
                _nccl_anchor = _nccl_anchor + halo_prev.sum() * 0
            if _exchange_received is not None and isinstance(_exchange_received, torch.Tensor) and _exchange_received.requires_grad:
                _nccl_anchor = _nccl_anchor + _exchange_received.sum() * 0
            for _dsa in _dummy_score_anchors:
                if isinstance(_dsa, torch.Tensor) and _dsa.requires_grad:
                    _nccl_anchor = _nccl_anchor + _dsa.sum() * 0
            if _nccl_anchor.requires_grad:
                attn_output = attn_output + _nccl_anchor

        # select-gate(learned): unconditional zero-contribution anchor - even when this rank has 0 select frames at this step
        # (short video / boundary rank / all-dummy chunks), the gate params must still live in the autograd graph,
        # otherwise ZeRO-2's reduce hook / NCCL call counts misalign across ranks -> deadlock
        # (the same hard-won lesson as _nccl_anchor / _dummy_score_anchors above).
        # non-SP DP+ZeRO-2 needs it just as much, so we do not branch on _sp. adds an exact 0.0, numerics bit-identical.
        if self.select_gate_mode == 'learned' and getattr(self, 'select_gate_head', None) is not None:
            _sg_anchor_feats = torch.zeros(1, 1, 3, device=x.device, dtype=torch.float32)
            _, _sg_anchor_alpha = self.select_gate_head(_sg_anchor_feats, training=False)
            if _sg_anchor_alpha.requires_grad:
                attn_output = attn_output + (_sg_anchor_alpha.sum() * 0).to(attn_output.dtype)

        return self.self_attn.o(attn_output)

    def forward(
        self, x, context, t_mod, freqs,
        tokens_per_frame: int = None,
        spatial_hw: tuple = None,
        **kwargs,
    ):
        """
        14B DiTBlock forward - drops the h_linear cross-layer residual, each layer's linear attention takes x directly.
        x already carries all information from earlier layers (via the main residual stream), so no extra residual signal is needed.
        """
        # Linear Attention: global scan -> state/z for global query + frame_keys for per-chunk scoring
        if self.training and self.inner_checkpoint:
            linear_attn_out, state, z = torch.utils.checkpoint.checkpoint(
                self.linear_attn, self.linear_attn_norm(x), use_reentrant=False)
        else:
            linear_attn_out, state, z = self.linear_attn(self.linear_attn_norm(x))

        # SP: all-reduce state/z (LinearAttention's state=K^T@V and z=sum(K) are decomposable sums)
        _sp_active = 'sp_num_frames_global' in kwargs
        if _sp_active:
            from .sp_runtime import allreduce_sum, allgather_frames_no_grad, get_sp_frame_info
            la = self.linear_attn

            # Fix: ghost frames caused state/z to be counted twice (an overlap frame is counted once per card)
            # SP4/61f: 109 tokens entered the sum but only 61 were unique -> conditioning signal diluted 1.8x
            # fix: only the state/z of assigned (non-ghost) frames take part in the all-reduce
            _sp_nfg = kwargs['sp_num_frames_global']
            _sp_offset = kwargs.get('sp_frame_offset', 0)
            _fpr, _orig_f_start, _orig_f_end, _ = get_sp_frame_info(_sp_nfg)
            _at_s = (_orig_f_start - _sp_offset) * self.per_frame_tokens
            _at_e = (_orig_f_end - _sp_offset) * self.per_frame_tokens

            _x_assigned = self.linear_attn_norm(x[:, _at_s:_at_e])
            _B_a, _N_a = _x_assigned.shape[:2]
            _k_a = la.feature_map(
                la.norm_k(la.k(_x_assigned)).view(_B_a, _N_a, la.num_heads, la.head_dim).transpose(1, 2))
            _v_a = la.v(_x_assigned).view(_B_a, _N_a, la.num_heads, la.head_dim).transpose(1, 2)
            state = torch.einsum('bhnd,bhnv->bhdv', _k_a, _v_a)
            z = _k_a.sum(dim=2)
            del _k_a, _v_a, _x_assigned

            state = allreduce_sum(state)
            z = allreduce_sum(z)
            # recompute linear_attn_out with the global state/z so frame_keys matches the single-card result exactly
            # reuse the phi(Q) cached in forward to avoid redoing norm_q -> q -> feature_map
            q_la = la._cached_q_mapped  # [B, num_heads, N, head_dim]
            del la._cached_q_mapped  # Free ~585MB immediately (40 blocks × 585MB = 23GB leak)
            qkv = torch.einsum('bhnd,bhdv->bhnv', q_la, state)
            qk_sum = torch.einsum('bhnd,bhd->bhn', q_la, z).unsqueeze(-1) + la.eps
            del q_la
            la_out = (qkv / qk_sum).transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], la.inner_dim)
            del qkv, qk_sum
            linear_attn_out = la.o(la_out)

        # frame-level pooling: used by per-chunk importance scoring (1024-dim, cuts the SP all-gather traffic)
        B_la = x.shape[0]
        pf_la = self.per_frame_tokens
        num_frames_la = x.shape[1] // pf_la  # under SP this is the local frame count
        _fk_pooled = linear_attn_out.view(B_la, num_frames_la, pf_la, -1).mean(dim=2)  # [B, F, 5120]
        frame_keys_local = self.chunk_to_state_proj(_fk_pooled)  # [B, F, 1024]

        # SP: all-gather frame_keys to obtain the global frame info
        if _sp_active:
            from .sp_runtime import get_sp_size, get_sp_group, get_sp_frame_info
            import torch.distributed as dist
            sp_size = get_sp_size()
            _num_frames_global = kwargs.get('sp_num_frames_global', num_frames_la)
            _sp_frame_offset = kwargs.get('sp_frame_offset', 0)

            # Fix: slice the keys of this card's ASSIGNED frame range out of the ghost-extended frame_keys_local
            # the old code all-gathered the ghost-extended keys directly, so after concatenation frame_keys[i] != the key of global frame i
            # (ghost frames overlap across ranks -> indices scrambled after padding + cat + truncate)
            _fpr, _orig_f_start, _orig_f_end, _ = get_sp_frame_info(_num_frames_global)
            _local_assigned_start = _orig_f_start - _sp_frame_offset  # local start index of the assigned frames
            _local_assigned_end = _orig_f_end - _sp_frame_offset
            frame_keys_assigned = frame_keys_local[:, _local_assigned_start:_local_assigned_end]

            # pad to frames_per_rank (so the all-gather shapes match across cards)
            if frame_keys_assigned.shape[1] < _fpr:
                _pad = torch.zeros(B_la, _fpr - frame_keys_assigned.shape[1], frame_keys_assigned.shape[-1],
                                   device=frame_keys_assigned.device, dtype=frame_keys_assigned.dtype)
                frame_keys_padded = torch.cat([frame_keys_assigned, _pad], dim=1)
            else:
                frame_keys_padded = frame_keys_assigned

            # all-gather: concatenate in rank order -> frame_keys[i] == the key of global frame i
            gathered = [torch.zeros_like(frame_keys_padded) for _ in range(sp_size)]
            dist.all_gather(gathered, frame_keys_padded.contiguous(), group=get_sp_group())
            frame_keys = torch.cat(gathered, dim=1)
            frame_keys = frame_keys[:, :_num_frames_global]
        else:
            frame_keys = frame_keys_local

        # Timestep modulation
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        # Sparse self-attention with importance scores + global state
        if self.sparse_attn:
            attn_out = self.sparse_self_attention(
                input_x, freqs, frame_keys, tokens_per_frame,
                spatial_hw, state, z,
                sp_frame_offset=kwargs.get('sp_frame_offset', 0),
                sp_num_frames_global=kwargs.get('sp_num_frames_global', None),
                freqs_full=kwargs.get('freqs_full', None),
                freqs_3d=kwargs.get('freqs_3d', None),
                select_gate_t_frac=kwargs.get('select_gate_t_frac', None))
            x = self.gate(x, gate_msa, attn_out)
        else:
            x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))

        # Cross-attention (per-chunk prompt switching)
        # chunk_batch_size (cbs) controls how many chunks are merged per batch: one cross_attn handles cbs chunks
        # cuts the num_chunks Python loop + kernel launch overhead; cbs=None merges everything by default; cbs=1 degenerates to the old behaviour
        segment_contexts_encoded = kwargs.get('segment_contexts_encoded', None)
        chunk_context_map = kwargs.get('chunk_context_map', None)
        if segment_contexts_encoded is not None and chunk_context_map is not None:
            pf = self.per_frame_tokens
            num_frames = x.shape[1] // pf
            chunk_f = self.chunk_size
            num_chunks = (num_frames + chunk_f - 1) // chunk_f
            cbs = getattr(self, 'chunk_batch_size', None) or num_chunks
            x_normed = self.norm3(x)
            cross_out = torch.zeros_like(x)
            B, _, D = x.shape
            for batch_start in range(0, num_chunks, cbs):
                batch_end = min(batch_start + cbs, num_chunks)
                nb = batch_end - batch_start
                # collect q + range + seg_idx of the chunks in this batch
                q_list, ranges, seg_indices = [], [], []
                for ci in range(batch_start, batch_end):
                    t_s = ci * chunk_f * pf
                    t_e = min((ci + 1) * chunk_f, num_frames) * pf
                    q_list.append(x_normed[:, t_s:t_e])
                    ranges.append((t_s, t_e))
                    seg_indices.append(int(chunk_context_map[ci]))
                # Pad Q to max_q_len (the last chunk may be shorter than chunk_f)
                max_q_len = max(q.shape[1] for q in q_list)
                q_batch = torch.zeros(nb * B, max_q_len, D, device=x.device, dtype=x.dtype)
                q_lens = []
                for i, q in enumerate(q_list):
                    ql = q.shape[1]
                    q_batch[i*B:(i+1)*B, :ql] = q
                    q_lens.append(ql)
                # KV: gather the segment context belonging to each chunk
                seg_idx_t = torch.tensor(seg_indices, device=x.device, dtype=torch.long)
                ctx_batch = segment_contexts_encoded[:, seg_idx_t]      # [B, nb, L_text, D]
                L_text = ctx_batch.shape[2]
                kv_batch = ctx_batch.permute(1, 0, 2, 3).reshape(nb * B, L_text, D)
                # one cross_attn handles nb chunks
                out_batch = self.cross_attn(q_batch, kv_batch)
                # write back into cross_out (restore the per-chunk slices)
                for i, (t_s, t_e) in enumerate(ranges):
                    ql = q_lens[i]
                    cross_out[:, t_s:t_e] = out_batch[i*B:(i+1)*B, :ql]
            x = x + cross_out
        else:
            # Cross-attention long-sequence chunking (same idea as the FFN: split Q along the token dim, K/V come from context and stay whole)
            _CHUNK_THRESHOLD = 100000
            _chunk_size = 20000
            if x.shape[1] > _CHUNK_THRESHOLD:
                x_normed = self.norm3(x)
                cross_out = torch.empty_like(x)
                for _i, _c in enumerate(x_normed.split(_chunk_size, dim=1)):
                    _s = _i * _chunk_size
                    cross_out[:, _s:_s + _c.shape[1]] = self.cross_attn(_c, context)
                x = x + cross_out
            else:
                x = x + self.cross_attn(self.norm3(x), context)

        # FFN (long-sequence chunking: split norm+modulate+FFN+gate as a whole along the token dim, avoiding full-sequence intermediate OOM)
        _FFN_CHUNK_THRESHOLD = 100000
        _ffn_chunk_size = 20000
        if x.shape[1] > _FFN_CHUNK_THRESHOLD:
            for _s in range(0, x.shape[1], _ffn_chunk_size):
                _e = min(_s + _ffn_chunk_size, x.shape[1])
                _chunk_in = modulate(self.norm2(x[:, _s:_e]), shift_mlp, scale_mlp)
                x[:, _s:_e] = x[:, _s:_e] + gate_mlp * self.ffn(_chunk_in)
        else:
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            if self.training and self.inner_checkpoint:
                ffn_out = torch.utils.checkpoint.checkpoint(self.ffn, input_x, use_reentrant=False)
            else:
                ffn_out = self.ffn(input_x)
            x = self.gate(x, gate_mlp, ffn_out)
        return x



class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        # Sparse Attention config params
        sparse_attn: bool = True,
        chunk_size: int = 8,
        overlap_size: int = 1,
        num_global_tokens: int = 8,
        per_frame_tokens: int = 30 * 52,  # 1560, for 480×832 with Wan2.1 VAE (stride 4×8×8)
        num_retained_tokens: int = 1024,
        # Frame-level select config
        num_select_frames: int = 4,
        num_nearby_frames: int = 3,
        # multi-teacher representation supervision config
        # teacher_config: dict, format {"dino": {"dim": 1024, "enabled": True}, ...}
        # defaults to a single DINO teacher when not passed
        teacher_dim: int = 1024,
        teacher_config: dict = None,
        # long-video memory optimizations
        chunk_batch_size: int = None,   # chunks processed per batch (None=all at once)
        inner_checkpoint: bool = False, # checkpoint inside FFN/linear_attn
        lazy_qkv: bool = False,         # compute Q/K/V on demand (no full-sequence compute during training)
        select_scales: list = None,     # importance select compression ratios, e.g. ['1x','2x','4x','8x']
        # Select-gate: relevance gating of importance-select recalled frames ('none' = exactly the original behaviour)
        select_gate_mode: str = 'none',
        select_gate_kappa: float = 2.0,
        select_gate_cos_floor: float = -1.0,    # -1=disabled (under collapsed cosine geometry an absolute threshold drifts across layers and kills good frames)
        select_gate_min_candidates: int = 8,
        select_gate_mad_floor: float = 1e-6,    # measured MAD is ~1e-5; the old 1e-3 fired the fallback 100% of the time making the gate a no-op
        select_gate_min_keep: int = 0,
        # Select-gate Phase 2 (learned) specific params
        select_gate_temp: float = 0.6667,
        select_gate_budget_target: float = 0.5,
        select_gate_budget_weight: float = 0.0,
        # Sink Distance-Decay (SDF):
        sink_decay_mode: str = 'none',
        sink_decay_onset: int = 40,
        sink_decay_factor: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.num_heads = num_heads

        # Patch Embedding (1× base resolution)
        # multi-scale downsampling is computed at runtime with F.interpolate (no patch_embedding_2x/4x/8x submodules)
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        # long-video memory optimization log
        if chunk_batch_size or inner_checkpoint or lazy_qkv or select_scales:
            print(f"[MemOpt] chunk_batch_size={chunk_batch_size}, inner_checkpoint={inner_checkpoint}, lazy_qkv={lazy_qkv}, select_scales={select_scales or ['1x','2x','4x','8x']}")
        # SDF activation log (printed only in downsample mode, silent for none -> does not disturb the baseline)
        if sink_decay_mode != 'none':
            print(f"[SinkDecay] mode={sink_decay_mode}, onset={sink_decay_onset} latent-frames, factor={sink_decay_factor}x (far chunks use a downsampled sink to break first-frame lock-in)")

        # chunk_size and overlap_size are counted in frames
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps,
                     sparse_attn=sparse_attn, chunk_size=chunk_size,
                     overlap_size=overlap_size, num_global_tokens=num_global_tokens,
                     per_frame_tokens=per_frame_tokens, num_retained_tokens=num_retained_tokens,
                     num_select_frames=num_select_frames, num_nearby_frames=num_nearby_frames,
                     chunk_batch_size=chunk_batch_size, inner_checkpoint=inner_checkpoint,
                     lazy_qkv=lazy_qkv, select_scales=select_scales,
                     select_gate_mode=select_gate_mode,
                     select_gate_kappa=select_gate_kappa,
                     select_gate_cos_floor=select_gate_cos_floor,
                     select_gate_min_candidates=select_gate_min_candidates,
                     select_gate_mad_floor=select_gate_mad_floor,
                     select_gate_min_keep=select_gate_min_keep,
                     select_gate_temp=select_gate_temp,
                     select_gate_budget_target=select_gate_budget_target,
                     select_gate_budget_weight=select_gate_budget_weight,
                     sink_decay_mode=sink_decay_mode,
                     sink_decay_onset=sink_decay_onset,
                     sink_decay_factor=sink_decay_factor)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        self.has_image_pos_emb = has_image_pos_emb

        # multi-teacher representation supervision projection MLPs (configurable)
        # projection heads created dynamically from teacher_config
        if teacher_config is None:
            # default: a single DINO teacher
            teacher_config = {"dino": {"dim": teacher_dim, "enabled": True}}
        self.teacher_config = teacher_config
        self.repr_projs = nn.ModuleDict()
        for name, cfg in teacher_config.items():
            if cfg.get("enabled", False):
                t_dim = cfg["dim"]
                self.repr_projs[name] = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, t_dim),
                )

    def _sparse_weights_already_loaded(self):
        """check whether the sparse params already hold valid values (not all-zero = the checkpoint already contains sparse weights)."""
        for block in self.blocks:
            if hasattr(block, 'linear_attn') and hasattr(block.linear_attn, 'q'):
                if block.linear_attn.q.weight.abs().max() > 1e-8:
                    return True
                return False
        return False

    def load_sparse_checkpoint(self, ckpt_path):
        """
        load sparse params from a checkpoint, overriding the current values.

        supports a merged checkpoint (base+sparse) or a sparse-only checkpoint.
        only sparse-related keys are extracted (linear_attn, importance_head, blend, global_attn, ...).
        the _orig_mod. prefix is handled automatically.
        """
        from safetensors.torch import load_file
        st = load_file(ckpt_path)
        sparse_keys = [
            'linear_attn', 'importance_head', 'blend_sharpness',
            'global_attn_gate', 'global_attn_out_proj',
            'chunk_to_state_proj', 'chunk_to_state_norm', 'linear_attn_norm',
            'select_gate_head',  # Phase 2 learned select-gate gate head
        ]
        sparse_sd = {}
        for k, v in st.items():
            # strip the _orig_mod. prefix
            clean_k = k.replace('_orig_mod.', '')
            if any(sk in clean_k for sk in sparse_keys):
                sparse_sd[clean_k] = v
        if not sparse_sd:
            print(f"[Sparse] WARNING: No sparse keys found in {ckpt_path}")
            return
        missing, unexpected = self.load_state_dict(sparse_sd, strict=False)
        loaded = len(sparse_sd) - len(unexpected)
        print(f"[Sparse] Loaded {loaded} sparse keys from {ckpt_path}")
        if unexpected:
            print(f"[Sparse] Skipped {len(unexpected)} unexpected keys")

    def reinit_sparse_modules(self):
        """
        re-initialize the weights of the modules added by sparse attention.

        skips init if the checkpoint already contains valid sparse weights.

        note: under ZeRO-3 params are sharded to 1D, so init must be wrapped in GatheredParameters.
        """
        if self._sparse_weights_already_loaded():
            print("[reinit_sparse_modules] Checkpoint already contains sparse weights, skipping reinit.")
            return
        # ZeRO-3 compatibility: gather every param that needs initializing
        try:
            import deepspeed
            has_deepspeed = True
        except ImportError:
            has_deepspeed = False

        def safe_init(fn, *args, **kwargs):
            """ZeRO-3-safe param init: automatically gathers sharded params"""
            if has_deepspeed and args and hasattr(args[0], 'ds_id'):
                with deepspeed.zero.GatheredParameters(args[0], modifier_rank=0):
                    fn(*args, **kwargs)
            else:
                fn(*args, **kwargs)

        print("[reinit_sparse_modules] Reinitializing sparse attention modules...")
        for block_idx, block in enumerate(self.blocks):
            if hasattr(block, 'sparse_attn') and block.sparse_attn:
                # importance_head: Xavier init
                if hasattr(block, 'importance_head'):
                    safe_init(nn.init.xavier_uniform_, block.importance_head.weight)
                    safe_init(nn.init.zeros_, block.importance_head.bias)

                # State query module
                if hasattr(block, 'chunk_to_state_proj'):
                    safe_init(nn.init.xavier_uniform_, block.chunk_to_state_proj.weight)
                    safe_init(nn.init.zeros_, block.chunk_to_state_proj.bias)

                if hasattr(block, 'chunk_to_state_norm'):
                    safe_init(nn.init.ones_, block.chunk_to_state_norm.weight)

                if hasattr(block, 'global_attn_out_proj'):
                    safe_init(nn.init.zeros_, block.global_attn_out_proj.weight)
                    safe_init(nn.init.zeros_, block.global_attn_out_proj.bias)

                # gate params (initialized to 0)
                if hasattr(block, 'global_attn_gate'):
                    safe_init(nn.init.zeros_, block.global_attn_gate)

                # blend_sharpness
                if hasattr(block, 'blend_sharpness'):
                    safe_init(nn.init.zeros_, block.blend_sharpness)

                # select_gate_head(Phase 2 learned): warm-start init -
                # zero output-layer weights + bias=logit_bias_init -> gate fully open (g==1) ~= baseline
                if getattr(block, 'select_gate_head', None) is not None:
                    _sgh = block.select_gate_head
                    safe_init(nn.init.xavier_uniform_, _sgh.net[0].weight)
                    safe_init(nn.init.zeros_, _sgh.net[0].bias)
                    safe_init(nn.init.zeros_, _sgh.net[-1].weight)
                    safe_init(nn.init.constant_, _sgh.net[-1].bias, _sgh.logit_bias_init)

                # params inside linear_attn
                if hasattr(block, 'linear_attn'):
                    la = block.linear_attn
                    for proj_name in ['q', 'k', 'v', 'o']:
                        if hasattr(la, proj_name):
                            proj = getattr(la, proj_name)
                            if hasattr(proj, 'weight') and proj.weight is not None:
                                safe_init(nn.init.xavier_uniform_, proj.weight)
                            if hasattr(proj, 'bias') and proj.bias is not None:
                                safe_init(nn.init.zeros_, proj.bias)

                    # RMSNorm layers - must be initialized to 1
                    for norm_name in ['norm_q', 'norm_k']:
                        if hasattr(la, norm_name):
                            norm = getattr(la, norm_name)
                            if hasattr(norm, 'weight') and norm.weight is not None:
                                safe_init(nn.init.ones_, norm.weight)

                # linear_attn_norm
                if hasattr(block, 'linear_attn_norm'):
                    norm = block.linear_attn_norm
                    if hasattr(norm, 'weight') and norm.weight is not None:
                        safe_init(nn.init.ones_, norm.weight)
                    if hasattr(norm, 'bias') and norm.bias is not None:
                        safe_init(nn.init.zeros_, norm.bias)

        # multi-teacher projection head init
        if hasattr(self, 'repr_projs'):
            for proj_name, proj in self.repr_projs.items():
                for m in proj.modules():
                    if isinstance(m, nn.Linear):
                        safe_init(nn.init.xavier_uniform_, m.weight)
                        safe_init(nn.init.zeros_, m.bias)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
                print(f"  [repr_proj] Reinitialized '{proj_name}' projection head")

        print(f"[reinit_sparse_modules] Reinitialized sparse modules in {len(self.blocks)} blocks.")

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        return self.patch_embedding(x)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        # keep the raw latent for multi-scale patch embedding
        x_latent = x  # [B, in_dim, T, H_lat, W_lat]

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        # 1× base patchify
        x = self.patchify(x)  # [B, dim, F', H', W']
        f, h, w = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

        # RoPE frequencies (base 1× resolution)
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # === Sequence Parallel: scatter frames ===
        _sp_active = getattr(self, 'sp_enabled', False)
        _sp_total_seq_len = f * h * w
        if _sp_active:
            from .sp_runtime import (
                scatter_frames, get_sp_frame_info, get_sp_rank)
            pf_sp = h * w
            _freqs_full = freqs
            frames_per_rank, _sp_f_start, _sp_f_end, _sp_f_local = get_sp_frame_info(f)

            # Ghost frames: chunk_size frames keep the chunk boundaries identical to the single-card case
            _overlap_f = self.blocks[0].chunk_size if len(self.blocks) > 0 else 8
            _ghost_f_start = max(0, _sp_f_start - _overlap_f)
            _ghost_f_end = min(f, _sp_f_start + frames_per_rank + _overlap_f)
            _ghost_before = _sp_f_start - _ghost_f_start
            _ghost_after = _ghost_f_end - min(f, _sp_f_start + frames_per_rank)

            x = x[:, _ghost_f_start * pf_sp : _ghost_f_end * pf_sp]
            freqs = _freqs_full[_ghost_f_start * pf_sp : _ghost_f_end * pf_sp]
            _sp_f_start = _ghost_f_start

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        _block_extra_kw = dict(
            tokens_per_frame=h*w,
            spatial_hw=(h, w),
            freqs_3d=self.freqs,  # 3D RoPE components (f_cis, h_cis, w_cis), used to build scale-back RoPE for compressed K
            # Select-gate(learned): normalized timestep (t/1000) - the gate head's noise-awareness input.
            # blocks in none/zscore mode ignore this kwarg entirely and change no computation.
            select_gate_t_frac=(timestep.detach().float() / 1000.0).clamp(0.0, 1.0),
        )
        if _sp_active:
            _block_extra_kw['sp_num_frames_global'] = f
            _block_extra_kw['sp_frame_offset'] = _sp_f_start
            _block_extra_kw['freqs_full'] = _freqs_full  # full RoPE, for computing the K/V of remote frames

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                            **_block_extra_kw,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                        **_block_extra_kw,
                    )
            else:
                x = block(x, context, t_mod, freqs, **_block_extra_kw)

        # === Sequence Parallel: trim ghost → head on local → gather ===
        if _sp_active:
            from .sp_runtime import gather_frames, get_sp_frame_info
            _fpr, _orig_f_start, _orig_f_end, _ = get_sp_frame_info(f)
            pf_hw = h * w
            trim_start = _ghost_before * pf_hw
            orig_local_tokens = (_orig_f_end - _orig_f_start) * pf_hw
            x = x[:, trim_start : trim_start + orig_local_tokens].contiguous()
            x = self.head(x, t)
            if x.shape[1] < _fpr * pf_hw:
                x = torch.nn.functional.pad(x, (0, 0, 0, _fpr * pf_hw - x.shape[1]))
            x = gather_frames(x, _sp_total_seq_len)
        else:
            x = self.head(x, t)

        x = self.unpatchify(x, (f, h, w))
        return x