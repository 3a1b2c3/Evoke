# ============================================================================
# VENDORED (byte-exact) from the teacher repo: diffsynth/utils/sequence_parallel.py
#   git commit a13988b, md5 a0024b5145f2581efac556e383156a8c
# evoke's sparse DiT (dit_sparse_14b / _cam) is ported from that repo, and its
# dead SP code depends on the 11 primitives in this module (scatter/gather_frames, allreduce_sum,
# broadcast_with_grad, halo_exchange, exchange_frame_tokens, get_sp_*, init_sequence_parallel, ...).
# diffsynth is not installed and only this self-contained file is needed (torch.distributed only),
# so it is vendored directly instead of pip-installing diffsynth. Keep changes in
# sync with upstream, do not rewrite casually (this includes the heavily debugged
# _build_uniform_p2p_ops v4 fully symmetric P2P deadlock fix).
# ============================================================================
"""
Sequence Parallel utilities for Sparse DiT training.

Scatters the frame sequence across GPUs along the frame dimension; each rank holds only F/sp_size frames.
All communication primitives are implemented as autograd Functions, so gradients flow back correctly.

Usage:
    1. call init_sequence_parallel(sp_size) before training starts
    2. in WanModel.forward, call scatter_frames after patchify and gather_frames before the head
    3. in DiTBlock.forward, call allreduce_sum after linear_attn to aggregate state/z
"""

import os
import contextlib
import torch
import torch.distributed as dist
from torch.autograd import Function
from typing import Optional

# ====== global state ======
_sp_group: Optional[dist.ProcessGroup] = None
_sp_size: int = 1
_sp_rank: int = 0
_world_size: int = 1
# [mpu 2D] DP-stride-G group (the stride-G ranks sharing the same rank%G, excluding the SP-peer ranks), handed to ZeRO-3 for sharding/reduce.
# Only built for true 2D (1 < sp_size < world_size); None when sp==world or SP is off. See sp_zero3.CriticMPU / SP_.
_dp_group: Optional[dist.ProcessGroup] = None
_dp_ranks: Optional[list] = None


def init_sequence_parallel(sp_size: int):
    """
    Initialize the SP process group.
    Ranks inside one SP group share the same data sample and split the per-frame compute between them.
    ranks 0..sp_size-1 form the first SP group, sp_size..2*sp_size-1 the second, and so on.

    The SP group timeout is read from the NCCL_TIMEOUT env var (milliseconds), default 1800000 (30 min).
    PyTorch's default watchdog for dist.new_group() is 600s, so the timeout must be passed explicitly
    for train.py's NCCL_TIMEOUT setting to take effect on the SP group.

    when 1 < sp_size < world_size (true 2D), additionally build the **DP-stride-G group**
    ({r : r % sp_size == j} for j in 0..sp_size-1, e.g. 8 GPUs with G=2 -> {0,2,4,6}/{1,3,5,7}). That group
    **excludes the SP-peer ranks** -> when the critic's ZeRO-3 does parameter sharding/all-gather/grad reduce-scatter on it,
    its rank coverage does not overlap the SP-subgroup all_to_all -> removing the forward/backward interleaving deadlock.
    new_group is a collective: every rank must walk the subgroups in the same order (all SP groups first, then all DP groups).
    """
    global _sp_group, _sp_size, _sp_rank, _world_size, _dp_group, _dp_ranks
    import os, datetime
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size % sp_size == 0, f"world_size={world_size} not divisible by sp_size={sp_size}"

    _sp_size = sp_size
    _sp_rank = rank % sp_size
    _world_size = world_size
    _dp_group = None
    _dp_ranks = None

    _nccl_timeout_ms = int(os.environ.get('NCCL_TIMEOUT', '1800000'))
    _sp_timeout = datetime.timedelta(milliseconds=_nccl_timeout_ms)

    if sp_size == world_size:
        # sp_size == world_size: reuse the default process group, to avoid crossing with the PG
        # DeepSpeed ZeRO-2 creates, which NCCL-deadlocks (two PGs covering the same rank set, so
        # during backward the gradient reduce and the SP collectives block each other)
        _sp_group = dist.group.WORLD
        print(f"[SP] init: world_size={world_size}, sp_size={sp_size}, "
              f"sp_rank={_sp_rank}, dp_size=1 (reusing WORLD group), "
              f"timeout={_nccl_timeout_ms/1000:.0f}s (set via WORLD)")
    else:
        # --- SP groups (G contiguous ranks): every rank walks them in the same order ---
        for i in range(0, world_size, sp_size):
            ranks = list(range(i, i + sp_size))
            group = dist.new_group(ranks, timeout=_sp_timeout)
            if rank in ranks:
                _sp_group = group
        # --- DP-stride-G groups (same rank%G): every rank walks them in the same order (after the SP groups), handed to the critic ZeRO-3 mpu ---
        for j in range(sp_size):
            dp_ranks = list(range(j, world_size, sp_size))
            dp_group = dist.new_group(dp_ranks, timeout=_sp_timeout)
            if rank in dp_ranks:
                _dp_group = dp_group
                _dp_ranks = dp_ranks
        print(f"[SP] init: world_size={world_size}, sp_size={sp_size}, "
              f"sp_rank={_sp_rank}, dp_size={world_size // sp_size}, "
              f"dp_group(stride-{sp_size})={_dp_ranks}, "
              f"timeout={_nccl_timeout_ms/1000:.0f}s")


def sp_diag(msg: str):
    """per-rank stderr trace point (gated on SP + env SP_DIAG=1). SP off / SP_DIAG!=1 -> no-op.
    Locating a 2D hang: every rank prints a 'phase start/done' pair; on a hang the last point with a start but no done = the stuck collective.
    formal long runs leave SP_DIAG unset -> no log spam; diagnostic runs set SP_DIAG=1."""
    if _sp_size <= 1 or os.environ.get("SP_DIAG", "0") != "1":
        return
    try:
        r = dist.get_rank()
    except Exception:
        r = -1
    print(f"[SP-DIAG r{r}] {msg}", flush=True)


def get_sp_group() -> Optional[dist.ProcessGroup]:
    return _sp_group

def get_sp_size() -> int:
    return _sp_size

def get_sp_rank() -> int:
    return _sp_rank

def is_sp_enabled() -> bool:
    return _sp_size > 1

def get_world_size() -> int:
    return _world_size

def is_2d_sp() -> bool:
    """True 2D sequence parallelism (1 < sp_size < world_size) -> take the mpu (DP-stride-G ZeRO-3) path.
    False when sp_size == world_size (the WORLD group is reused) or SP is off."""
    return 1 < _sp_size < _world_size

def get_dp_group() -> Optional[dist.ProcessGroup]:
    """DP-stride-G group used for critic ZeRO-3 sharding/reduce (non-None only in 2D)."""
    return _dp_group

def get_dp_ranks() -> Optional[list]:
    return _dp_ranks


# ====== autograd communication primitives ======

class _ScatterFrames(Function):
    """
    Forward: [B, F*pf, D] -> split by frame across the SP-group ranks -> [B, F_local*pf, D]
    Backward: all-gather the gradients
    """
    @staticmethod
    def forward(ctx, x, per_frame_tokens, group, sp_size, sp_rank):
        ctx.per_frame_tokens = per_frame_tokens
        ctx.group = group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.total_seq_len = x.shape[1]

        B, S, D = x.shape
        num_frames = S // per_frame_tokens
        frames_per_rank = (num_frames + sp_size - 1) // sp_size
        ctx.frames_per_rank = frames_per_rank
        ctx.num_frames = num_frames

        # pad when the frame count is not divisible
        padded_frames = frames_per_rank * sp_size
        if padded_frames > num_frames:
            pad_tokens = (padded_frames - num_frames) * per_frame_tokens
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_tokens))

        start = sp_rank * frames_per_rank * per_frame_tokens
        end = start + frames_per_rank * per_frame_tokens
        return x[:, start:end].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        gathered = [torch.zeros_like(grad_output) for _ in range(ctx.sp_size)]
        dist.all_gather(gathered, grad_output.contiguous(), group=ctx.group)
        grad_full = torch.cat(gathered, dim=1)
        grad_full = grad_full[:, :ctx.total_seq_len]
        return grad_full, None, None, None, None


class _GatherFrames(Function):
    """
    Forward: all-gather [B, F_local*pf, D] -> [B, F*pf, D]
    Backward: take the gradient of the local slice
    """
    @staticmethod
    def forward(ctx, x, total_seq_len, group, sp_size, sp_rank):
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.group = group
        ctx.local_len = x.shape[1]
        ctx.total_seq_len = total_seq_len

        gathered = [torch.zeros_like(x) for _ in range(sp_size)]
        dist.all_gather(gathered, x.contiguous(), group=group)
        result = torch.cat(gathered, dim=1)
        return result[:, :total_seq_len]

    @staticmethod
    def backward(ctx, grad_output):
        padded_len = ctx.local_len * ctx.sp_size
        if grad_output.shape[1] < padded_len:
            grad_output = torch.nn.functional.pad(
                grad_output, (0, 0, 0, padded_len - grad_output.shape[1]))
        start = ctx.sp_rank * ctx.local_len
        end = start + ctx.local_len
        return grad_output[:, start:end].contiguous(), None, None, None, None


class _AllReduceSum(Function):
    """
    Forward: all-reduce sum (aggregates the LinearAttention state/z)
    Backward: all-reduce sum (the gradients must be aggregated too, since d(sum)/dx_i = 1)
    """
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


# ====== user-facing API ======

def scatter_frames(x: torch.Tensor, per_frame_tokens: int) -> torch.Tensor:
    """Scatter along the frame dimension to the SP group. Returned unchanged when SP is disabled."""
    if not is_sp_enabled():
        return x
    return _ScatterFrames.apply(x, per_frame_tokens, _sp_group, _sp_size, _sp_rank)


def gather_frames(x: torch.Tensor, total_seq_len: int) -> torch.Tensor:
    """All-gather the full sequence back from the SP group. Returned unchanged when SP is disabled."""
    if not is_sp_enabled():
        return x
    return _GatherFrames.apply(x, total_seq_len, _sp_group, _sp_size, _sp_rank)


def allreduce_sum(x: torch.Tensor) -> torch.Tensor:
    """All-reduce sum, with autograd support. Returned unchanged when SP is disabled."""
    if not is_sp_enabled():
        return x
    return _AllReduceSum.apply(x, _sp_group)


def allgather_frames_no_grad(x: torch.Tensor) -> torch.Tensor:
    """All-gather without autograd (for data that needs no gradient, e.g. frame_keys)."""
    if not is_sp_enabled():
        return x
    gathered = [torch.zeros_like(x) for _ in range(_sp_size)]
    dist.all_gather(gathered, x.contiguous(), group=_sp_group)
    return torch.cat(gathered, dim=1)


def broadcast_tensor(x: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
    """Broadcast a tensor from the given rank inside the SP group. Used for sink tokens."""
    if not is_sp_enabled():
        return x
    # global rank of src inside the SP group
    global_rank = dist.get_rank()
    sp_group_start = (global_rank // _sp_size) * _sp_size
    src_global = sp_group_start + src_rank
    dist.broadcast(x, src=src_global, group=_sp_group)
    return x


def get_sp_frame_info(num_frames_total: int):
    """
    Compute the SP frame assignment.
    Returns:
        frames_per_rank: frames per rank (including padding)
        f_start: this rank's global start frame index
        f_end: this rank's global end frame index (real frames, excluding padding)
        f_local: this rank's real frame count
    """
    if not is_sp_enabled():
        return num_frames_total, 0, num_frames_total, num_frames_total

    frames_per_rank = (num_frames_total + _sp_size - 1) // _sp_size
    f_start = _sp_rank * frames_per_rank
    f_end = min(f_start + frames_per_rank, num_frames_total)
    f_local = f_end - f_start
    return frames_per_rank, f_start, f_end, f_local


def get_sp_frame_info_for_rank(num_frames_total: int, rank: int, sp_size: int):
    """Compute the frame assignment of an arbitrary rank (not restricted to the current rank)."""
    frames_per_rank = (num_frames_total + sp_size - 1) // sp_size
    f_start = rank * frames_per_rank
    f_end = min(f_start + frames_per_rank, num_frames_total)
    return frames_per_rank, f_start, f_end


def get_ghost_info_for_rank(num_frames_total: int, rank: int, sp_size: int, chunk_size: int):
    """Compute the ghost frame counts of an arbitrary rank (chunk-grid alignment policy of the model_fn_wan_video inference path)."""
    fpr = (num_frames_total + sp_size - 1) // sp_size
    f_start = rank * fpr
    f_end_real = min(f_start + fpr, num_frames_total)
    aligned_start = (f_start // chunk_size) * chunk_size
    ghost_f_start = max(0, aligned_start - chunk_size)
    aligned_end = ((f_end_real + chunk_size - 1) // chunk_size) * chunk_size
    ghost_f_end = min(num_frames_total, aligned_end + chunk_size)
    ghost_before = f_start - ghost_f_start
    ghost_after = ghost_f_end - f_end_real
    return ghost_before, ghost_after


# ====== halo exchange (cross-rank exchange of nearby frames) ======

class _HaloExchange(Function):
    """
    Halo exchange for nearby frames: exchange boundary frames with the adjacent SP ranks.

    Forward:
      - Card k sends its own last num_halo frames to Card k+1
      - Card k receives num_halo frames from Card k-1
      - Card 0 has no predecessor and returns the zero tensor standing in for None

    Backward:
      - propagate gradients in the reverse direction: Card k sends the halo gradient back to Card k-1
    """
    @staticmethod
    def forward(ctx, x, num_halo_frames, per_frame_tokens, group, sp_size, sp_rank):
        ctx.group = group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.num_halo_frames = num_halo_frames
        ctx.per_frame_tokens = per_frame_tokens
        ctx.x_shape = x.shape  # [B, S, D]

        B, S, D = x.shape
        halo_tokens = num_halo_frames * per_frame_tokens

        # global rank computation
        global_rank = dist.get_rank()
        sp_group_start = (global_rank // sp_size) * sp_size
        prev_rank = sp_group_start + sp_rank - 1  # global rank of the previous rank in the SP group
        next_rank = sp_group_start + sp_rank + 1  # global rank of the next rank in the SP group

        # what is sent to the next rank: our own last halo frames
        send_buf = x[:, -halo_tokens:].contiguous() if sp_rank < sp_size - 1 else None
        # the halo to be received from the previous rank
        recv_buf = torch.zeros(B, halo_tokens, D, device=x.device, dtype=x.dtype) if sp_rank > 0 else None

        # SP deadlock fix v4: full-mesh symmetric P2P replaces the ring dummy padding
        # real halo pattern: rank R recvs from R-1, sends to R+1; rank 0 has no recv, rank N-1 has no send
        # the old _pad_p2p_ops_for_sync gave interior ranks 4 ops and edge ranks 3 ops, so the NCCL SeqNum drifted by 1 every time
        recv_bufs = {sp_rank - 1: recv_buf} if sp_rank > 0 else {}
        send_bufs = {sp_rank + 1: send_buf} if sp_rank < sp_size - 1 else {}
        ops = _build_uniform_p2p_ops(recv_bufs, send_bufs, sp_rank, sp_size,
                                     sp_group_start, x.device, group)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        return recv_buf if recv_buf is not None else torch.zeros(B, 0, D, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_halo):
        # reverse direction: send the halo gradient back to the source rank
        group = ctx.group
        sp_rank = ctx.sp_rank
        sp_size = ctx.sp_size
        B, S, D = ctx.x_shape
        halo_tokens = ctx.num_halo_frames * ctx.per_frame_tokens

        global_rank = dist.get_rank()
        sp_group_start = (global_rank // sp_size) * sp_size
        prev_rank = sp_group_start + sp_rank - 1
        next_rank = sp_group_start + sp_rank + 1

        grad_x = torch.zeros(B, S, D, device=grad_halo.device, dtype=grad_halo.dtype)

        # Card k sends the halo gradient to Card k-1 (reverse direction)
        send_buf = grad_halo.contiguous() if sp_rank > 0 and grad_halo.shape[1] > 0 else None
        # Card k receives the gradient from Card k+1 and accumulates it into its last frames
        recv_buf = torch.zeros(B, halo_tokens, D, device=grad_halo.device, dtype=grad_halo.dtype) if sp_rank < sp_size - 1 else None

        # SP deadlock fix v4: the backward halo also uses full-mesh symmetric P2P
        recv_bufs = {sp_rank + 1: recv_buf} if sp_rank < sp_size - 1 else {}
        send_bufs = {sp_rank - 1: send_buf} if (sp_rank > 0 and send_buf is not None) else {}
        ops = _build_uniform_p2p_ops(recv_bufs, send_bufs, sp_rank, sp_size,
                                     sp_group_start, grad_halo.device, group)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        if recv_buf is not None:
            grad_x[:, -halo_tokens:] += recv_buf

        return grad_x, None, None, None, None, None


def halo_exchange(x: torch.Tensor, num_halo_frames: int, per_frame_tokens: int) -> Optional[torch.Tensor]:
    """
    Exchange nearby halo frames with the previous SP rank.

    Args:
        x: [B, S, D] local frame sequence
        num_halo_frames: number of halo frames needed (= num_nearby_frames)
        per_frame_tokens: tokens per frame

    Returns:
        halo_prev: [B, num_halo_frames * pf, D] last-frame tokens of the previous rank
                   Card 0 returns an empty tensor [B, 0, D]
    """
    if not is_sp_enabled():
        return None
    return _HaloExchange.apply(x, num_halo_frames, per_frame_tokens, _sp_group, _sp_size, _sp_rank)




# ====== SP-safe tensor sync (P0: noise/timestep sync) ======

# owner of the clip currently being scored (group-local rank 0.G-1). Default 0 = status quo (broadcast from group rank0).
#   the decouple path temporarily switches it to j via the sp_score_owner context -> every sync_tensor_in_sp_group in the group broadcasts from owner-j,
#   with no need to thread extra parameters through wrapper/cdml. Never changed by default -> byte-identical.
_SP_SCORE_OWNER = [0]
# whether decouple-rollout scoring is active. True only inside the scoring block with sf_decouple_rollout=True and SP on.
#   Always False by default -> sync_tensor_in_sp_group keeps the status quo byte for byte (broadcast from owner), byte-identical.
_SP_DECOUPLE_ACTIVE = [False]
# whether we are currently inside some sp_score_owner(j) block (= the owner-rotated clip of EvokeTeacher SP scoring).
#   under decouple: inside an owner-block -> sync_tensor broadcasts from owner-j (SP frame-shard cooperation needs group-wide agreement);
#   outside an owner-block (= the non-SP Evoke tail-segment / critic path) -> sync_tensor is skipped -> every rank uses its own clip.
_SP_IN_OWNER_BLOCK = [False]


@contextlib.contextmanager
def sp_score_owner(owner_local_rank: int):
    """Make every sync_tensor_in_sp_group inside the block broadcast from group-local `owner_local_rank` (instead of rank0). Default 0 = status quo.
    entering this block marks in-owner-block (only the decouple path calls it): inside the block sync_tensor broadcasts from owner-j."""
    _prev = _SP_SCORE_OWNER[0]
    _prev_blk = _SP_IN_OWNER_BLOCK[0]
    _SP_SCORE_OWNER[0] = int(owner_local_rank)
    _SP_IN_OWNER_BLOCK[0] = True
    try:
        yield
    finally:
        _SP_SCORE_OWNER[0] = _prev
        _SP_IN_OWNER_BLOCK[0] = _prev_blk


@contextlib.contextmanager
def sp_decouple_scope():
    """mark entry into the decouple-rollout scoring block. Inside the block: sync_tensor_in_sp_group outside an owner-block
    skips the broadcast (letting every rank on the non-SP Evoke tail-segment / critic path use its own distinct clip); inside an owner-block
    (sp_score_owner) it still broadcasts from owner-j (EvokeTeacher SP frame-shard cooperation). Outside the block (= status quo / formal) it is never entered -> byte-identical."""
    _prev = _SP_DECOUPLE_ACTIVE[0]
    _SP_DECOUPLE_ACTIVE[0] = True
    try:
        yield
    finally:
        _SP_DECOUPLE_ACTIVE[0] = _prev


def sp_decouple_active() -> bool:
    return bool(_SP_DECOUPLE_ACTIVE[0])


def sp_in_owner_block() -> bool:
    return bool(_SP_IN_OWNER_BLOCK[0])


def sp_current_score_owner() -> int:
    return int(_SP_SCORE_OWNER[0])


def sync_tensor_in_sp_group(x: torch.Tensor) -> torch.Tensor:
    """Broadcast tensor from SP group's owner rank (rank0 by default; changeable via sp_score_owner) to all ranks. No-op if SP disabled.
    when decouple is active and we are not currently inside an owner-block (= the non-SP Evoke tail-segment / critic path) the broadcast is skipped,
    so every rank keeps its own distinct clip. When decouple is inactive this keeps the status quo byte for byte (broadcast from owner) -> byte-identical."""
    if not is_sp_enabled():
        return x
    if _SP_DECOUPLE_ACTIVE[0] and not _SP_IN_OWNER_BLOCK[0]:
        return x
    sp_group_start = (dist.get_rank() // _sp_size) * _sp_size
    dist.broadcast(x, src=sp_group_start + _SP_SCORE_OWNER[0], group=_sp_group)
    return x


def broadcast_from_owner(x: torch.Tensor, owner_local_rank: int) -> torch.Tensor:
    """Broadcast x from SP-group rank `owner_local_rank` (0.G-1) to all group ranks.
    No autograd (used on detached tensors). No-op if SP disabled. Explicit-owner variant of sync_tensor_in_sp_group.
    Requires identical shape/dtype across ranks (fixed-shape clip tensors); use broadcast_varshape_from_owner
    when the shape can differ per clip.
    """
    if not is_sp_enabled():
        return x
    sp_group_start = (dist.get_rank() // _sp_size) * _sp_size
    dist.broadcast(x, src=sp_group_start + int(owner_local_rank), group=_sp_group)
    return x


def broadcast_varshape_from_owner(x, owner_local_rank: int, ref_device, ref_dtype=None):
    """Broadcast a possibly per-rank varying-shape tensor from SP-group owner `owner_local_rank`.
    Negotiates is-None + ndim + shape via a small meta broadcast (so receivers allocate the correct buffer),
    then broadcasts the data. None-safe (owner may pass None -> all ranks get None). No autograd (detached copy).
    ref_device = the local rank's compute device; ref_dtype = target dtype (defaults to owner's dtype).
    Used for clip-dependent tensors whose shape differs across clips (e.g. segment-stacked prompt [1,S,L,D])."""
    if not is_sp_enabled():
        return x
    sp_group_start = (dist.get_rank() // _sp_size) * _sp_size
    src = sp_group_start + int(owner_local_rank)
    is_src = (dist.get_rank() == src)
    # meta: [is_none, ndim, d0..d7]
    meta = torch.zeros(10, dtype=torch.long, device=ref_device)
    if is_src:
        if x is None:
            meta[0] = 1
        else:
            meta[1] = x.dim()
            assert x.dim() <= 8, f"[THROUGHPUT-B] broadcast_varshape ndim>8 not supported: {x.dim()}"
            for _i, _s in enumerate(x.shape):
                meta[2 + _i] = int(_s)
    dist.broadcast(meta, src=src, group=_sp_group)
    if int(meta[0].item()) == 1:
        return None
    _ndim = int(meta[1].item())
    _shape = [int(meta[2 + _i].item()) for _i in range(_ndim)]
    _dt = ref_dtype
    if is_src:
        if _dt is None:
            _dt = x.dtype
        buf = x.to(device=ref_device, dtype=_dt).contiguous()
    else:
        if _dt is None:
            _dt = torch.bfloat16
        buf = torch.empty(_shape, dtype=_dt, device=ref_device)
    dist.broadcast(buf, src=src, group=_sp_group)
    return buf


def broadcast_object_from_owner(obj, owner_local_rank: int):
    """Broadcast an arbitrary picklable Python object (e.g. segment_frame_ranges list) from
    SP-group owner `owner_local_rank` to all group ranks. No-op if SP disabled. Do NOT pass CUDA tensors here
    (device-ordinal pickling is unsafe cross-rank); use broadcast_varshape_from_owner for tensors."""
    if not is_sp_enabled():
        return obj
    sp_group_start = (dist.get_rank() // _sp_size) * _sp_size
    src = sp_group_start + int(owner_local_rank)
    holder = [obj]
    dist.broadcast_object_list(holder, src=src, group=_sp_group)
    return holder[0]


# ====== Autograd-aware broadcast (P2: sink K/V gradient fix) ======

class _BroadcastFromRank0(Function):
    """Forward: broadcast from SP rank 0.  Backward: reduce-sum gradient to rank 0."""
    @staticmethod
    def forward(ctx, x, group, sp_size, sp_rank):
        ctx.group = group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        out = x.clone()
        sp_group_start = (dist.get_rank() // sp_size) * sp_size
        dist.broadcast(out, src=sp_group_start, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.contiguous()
        sp_group_start = (dist.get_rank() // ctx.sp_size) * ctx.sp_size
        dist.reduce(grad, dst=sp_group_start, op=dist.ReduceOp.SUM, group=ctx.group)
        if ctx.sp_rank != 0:
            return torch.zeros_like(grad), None, None, None
        return grad, None, None, None


def broadcast_with_grad(x: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
    """Broadcast from SP rank 0 with autograd support. SP disabled -> identity."""
    if not is_sp_enabled():
        return x
    return _BroadcastFromRank0.apply(x, _sp_group, _sp_size, _sp_rank)


# ====== P2P sync padding (prevents the deadlock caused by NCCL SeqNum divergence) ======

def _pad_p2p_ops_for_sync(ops, sp_rank, sp_size, sp_group_start, device, group):
    """
    [DEPRECATED -- superseded by _build_uniform_p2p_ops]

    Old implementation: append a dummy ring P2P to guarantee ops is non-empty. But that only solved
    "non-empty", not "aligned": the total P2P op count still varies per rank (ghost frames / remote-frame
    requests differ per rank) and every P2POp bumps the NCCL SeqNum -> the drift accumulates -> at
    SeqNum=2888 the watchdog reports an ALLGATHER timeout.
    Kept only so as not to break external callers; no call site uses it any more.
    """
    dummy_send = torch.zeros(1, device=device, dtype=torch.uint8)
    dummy_recv = torch.zeros(1, device=device, dtype=torch.uint8)
    next_rank = sp_group_start + (sp_rank + 1) % sp_size
    prev_rank = sp_group_start + (sp_rank - 1 + sp_size) % sp_size
    ops.append(dist.P2POp(dist.isend, dummy_send, next_rank, group=group))
    ops.append(dist.P2POp(dist.irecv, dummy_recv, prev_rank, group=group))


def _build_uniform_p2p_ops(recv_bufs: dict, send_bufs: dict,
                           sp_rank: int, sp_size: int, sp_group_start: int,
                           device, group):
    """
    SP deadlock fix v4: build a fully symmetric P2P op list, eliminating NCCL SeqNum drift.

    Every rank posts 1 send + 1 recv to each of the other sp_size-1 peers (order: recv first, then send),
    sending a 1-byte uint8 dummy where there is no real data. Total op count = 2*(sp_size-1), exactly equal on all ranks.

    Symmetry guarantee: A has no real data for B (B not in send_bufs) <=> B has no real data to receive from A (A not in recv_bufs),
    because send/recv are the two ends of the same logical communication, so both sides necessarily decide the same way; therefore dummy
    always pairs with dummy and real always pairs with real, and the shapes match.

    Ordering guarantee: every rank walks the peers in for r in range(sp_size) order, recv-then-send within each peer;
    NCCL P2P matches on (src, dst, tag=0), and each pair (A,B) has only 1 op per direction, so there is no ambiguity.

    Args:
        recv_bufs: {peer_local_rank: tensor}, the buffers that really need a recv (peer_rank is 0..sp_size-1 inside the SP group)
        send_bufs: {peer_local_rank: tensor}, the buffers that really need a send
        sp_rank: this rank's rank inside the SP group
        sp_size: SP group size
        sp_group_start: this SP group's start in the global rank space (= node_id * sp_size)

    Returns:
        ops: List[dist.P2POp], length strictly = 2*(sp_size-1)
    """
    ops = []
    for r in range(sp_size):
        if r == sp_rank:
            continue
        target = sp_group_start + r
        # recv from r (real or 1-byte dummy)
        if r in recv_bufs and recv_bufs[r] is not None:
            recv_t = recv_bufs[r]
        else:
            recv_t = torch.empty(1, dtype=torch.uint8, device=device)
        ops.append(dist.P2POp(dist.irecv, recv_t, target, group=group))
        # send to r (real or 1-byte dummy)
        if r in send_bufs and send_bufs[r] is not None:
            send_t = send_bufs[r].contiguous()
        else:
            send_t = torch.zeros(1, dtype=torch.uint8, device=device)
        ops.append(dist.P2POp(dist.isend, send_t, target, group=group))
    return ops


# ====== Autograd-aware frame token exchange (P1: select K/V gradient fix) ======

class _ExchangeFrameTokensGrad(Function):
    """
    P2P exchange of raw frame tokens with autograd support.
    Forward: source extracts frames from local x -> P2P send -> destination receives.
    Backward: reverse P2P to propagate gradients back to source ranks.
    """
    @staticmethod
    def forward(ctx, x, send_map, recv_map, pf, group, sp_size, sp_rank):
        """
        Args:
            x: [B, S, D] local tokens (gradient flows through this)
            send_map: list of (dest_sp_rank, [local_frame_idx, ...])
            recv_map: list of (src_sp_rank, num_frames)
            pf: per_frame_tokens
        Returns:
            received: [B, total_recv_frames * pf, D]
        """
        B, S, D = x.shape
        global_rank = dist.get_rank()
        sp_group_start = (global_rank // sp_size) * sp_size

        # Build send buffers
        send_bufs = {}
        for dest_rank, indices in send_map:
            if indices:
                tokens = torch.cat([x[:, idx * pf:(idx + 1) * pf] for idx in indices], dim=1)
                send_bufs[dest_rank] = tokens.contiguous()

        # Build recv buffers (sorted by src_rank for deterministic concat order)
        recv_bufs = {}
        recv_order = [r for r, _ in recv_map]
        for src_rank, nf in recv_map:
            recv_bufs[src_rank] = torch.zeros(B, nf * pf, D, device=x.device, dtype=x.dtype)

        # SP deadlock fix v4: full-mesh P2P, strictly 2*(sp_size-1) ops per rank
        # the old version only posted to the union of send/recv_bufs.keys(), which differed per rank -> SeqNum drift
        ops = _build_uniform_p2p_ops(recv_bufs, send_bufs, sp_rank, sp_size,
                                     sp_group_start, x.device, group)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        # Concatenate received in recv_order
        parts = [recv_bufs[r] for r in recv_order if r in recv_bufs]
        received = torch.cat(parts, dim=1) if parts else torch.zeros(B, 0, D, device=x.device, dtype=x.dtype)

        # Save for backward
        ctx.send_map = send_map
        ctx.recv_map = recv_map
        ctx.recv_order = recv_order
        ctx.pf = pf
        ctx.group = group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.x_shape = (B, S, D)
        return received

    @staticmethod
    def backward(ctx, grad_received):
        B, S, D = ctx.x_shape
        pf = ctx.pf
        global_rank = dist.get_rank()
        sp_group_start = (global_rank // ctx.sp_size) * ctx.sp_size
        recv_map_dict = dict(ctx.recv_map)

        # Split grad_received by source rank (reverse of forward recv)
        grad_to_send = {}
        offset = 0
        for src_rank in ctx.recv_order:
            nf = recv_map_dict[src_rank]
            n_tokens = nf * pf
            grad_to_send[src_rank] = grad_received[:, offset:offset + n_tokens].contiguous()
            offset += n_tokens

        # Prepare recv buffers for gradient (reverse of forward send)
        grad_to_recv = {}
        for dest_rank, indices in ctx.send_map:
            n_tokens = len(indices) * pf
            grad_to_recv[dest_rank] = torch.zeros(B, n_tokens, D,
                device=grad_received.device, dtype=grad_received.dtype)

        # SP deadlock fix v4: full-mesh P2P, the backward grad is aligned too
        ops = _build_uniform_p2p_ops(grad_to_recv, grad_to_send,
                                     ctx.sp_rank, ctx.sp_size, sp_group_start,
                                     grad_received.device, ctx.group)
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        # Accumulate gradients into grad_x
        grad_x = torch.zeros(B, S, D, device=grad_received.device, dtype=grad_received.dtype)
        for dest_rank, indices in ctx.send_map:
            grad_buf = grad_to_recv[dest_rank]
            buf_offset = 0
            for idx in indices:
                grad_x[:, idx * pf:(idx + 1) * pf] += grad_buf[:, buf_offset:buf_offset + pf]
                buf_offset += pf

        return grad_x, None, None, None, None, None, None


def exchange_frame_tokens(
    requests: dict,
    x: torch.Tensor,
    per_frame_tokens: int,
    sp_frame_offset: int,
    num_local_frames: int,
    frames_per_rank: int,
) -> dict:
    """
    Exchange raw frame tokens between SP ranks with autograd support.
    Unlike exchange_select_kv (detached K/V), this returns raw tokens so the
    requesting rank computes K/V locally -- keeping the full autograd graph.

    Args:
        requests: {source_sp_rank: [global_frame_idx, ...]} deduplicated
        x, per_frame_tokens, sp_frame_offset, num_local_frames, frames_per_rank: same as exchange_select_kv

    Returns:
        token_cache: {global_frame_idx: tensor [B, per_frame_tokens, D]}
    """
    if not is_sp_enabled():
        return {}

    sp_size = _sp_size
    sp_rank = _sp_rank
    group = _sp_group
    global_rank = dist.get_rank()
    sp_group_start = (global_rank // sp_size) * sp_size
    B, S, D = x.shape
    pf = per_frame_tokens

    # === Phase 1: exchange request indices (no autograd, collective) ===
    send_counts = torch.zeros(sp_size, dtype=torch.long, device=x.device)
    for src_rank, frame_list in requests.items():
        send_counts[src_rank] = len(frame_list)

    recv_counts = torch.zeros(sp_size, dtype=torch.long, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts, group=group)

    # P2P exchange frame indices
    send_idx_bufs = {}
    for r in range(sp_size):
        if send_counts[r] > 0:
            send_idx_bufs[r] = torch.tensor(requests[r], dtype=torch.long, device=x.device)

    recv_idx_bufs = {}
    for r in range(sp_size):
        cnt = recv_counts[r].item()
        if cnt > 0:
            recv_idx_bufs[r] = torch.zeros(cnt, dtype=torch.long, device=x.device)

    # SP deadlock fix v4: full-mesh P2P, the index exchange is aligned too
    ops = _build_uniform_p2p_ops(recv_idx_bufs, send_idx_bufs,
                                 sp_rank, sp_size, sp_group_start, x.device, group)
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()

    # Build send_map: [(dest_sp_rank, [local_frame_idx, ...])]
    send_map = []
    for r in range(sp_size):
        if r == sp_rank or r not in recv_idx_bufs:
            continue
        local_indices = [(gfi.item() - sp_frame_offset) for gfi in recv_idx_bufs[r]]
        send_map.append((r, local_indices))

    # Build recv_map: [(src_sp_rank, num_frames)] -- sorted by rank
    recv_map = []
    for r in range(sp_size):
        if r == sp_rank or send_counts[r] == 0:
            continue
        recv_map.append((r, send_counts[r].item()))

    # === Phase 2: exchange frame tokens (autograd-aware) ===
    received = _ExchangeFrameTokensGrad.apply(
        x, send_map, recv_map, pf, group, sp_size, sp_rank)

    # === Phase 3: split received tokens into per-frame cache ===
    token_cache = {}
    offset = 0
    for src_rank, nf in recv_map:
        for gfi in requests[src_rank]:
            token_cache[gfi] = received[:, offset:offset + pf]
            offset += pf

    # return the received tensor for use as an autograd anchor (the cache is empty on a rank with no
    # remote requests, but received still carries a grad_fn, so anchoring it guarantees backward fires the P2P)
    return token_cache, received
