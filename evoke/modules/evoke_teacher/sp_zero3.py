# ============================================================================
# approach (B): the "pre-gather the routed expert" (gather-and-hold) fix
# for the ZeRO-3 x Sequence-Parallel forward deadlock. /.
# ============================================================================
"""
Root cause (measured in an NCCL watchdog dump):
under 2D (sp<world) x ZeRO-3, ZeRO-3's **per-parameter WORLD all-gather** inside the EvokeTeacher
scoring block-loop forward interleaves with the SP-subgroup all_to_all (exchange_frame_tokens) --
sp0 blocks in the SP-subgroup all_to_all waiting for sp1, sp1 blocks in the WORLD parameter
all-gather waiting for sp0 -> cyclic deadlock.

Fix (B):
before the critic/teacher scoring SP forward, pre-all-gather all ZeRO-3 params of the **single
expert currently routed to** (dit_high or dit_low, one 14B ~= 28GB) into a resident (AVAILABLE)
state, and insert a sentinel sub-module id into each param's `ds_active_sub_modules`. The
DeepSpeed parameter coordinator (PartitionedParameterCoordinator) partitions in
`release_sub_module` -> `__release_param` only when
`param.ds_status==AVAILABLE and not param.ds_active_sub_modules`
(partitioned_param_coordinator.py) -> the sentinel is there -> **no partition** for the
whole block-loop, and `fetch_sub_module` sees them already AVAILABLE so it **skips the all-gather**
too (it only gathers NOT_AVAILABLE,). Hence no per-parameter WORLD all-gather inside the
block-loop during the SP forward/backward -> only SP-subgroup collectives remain -> no interleaving
-> no deadlock. The other 3 engines stay ZeRO-3 / world-sharded throughout; only the currently
routed expert is resident for the moment of scoring.

Key correctness points (verified against the deepspeed 0.14.5 source):
- **it works at step0 (RECORD trace) too**: whatever the trace state, release_sub_module ends up in
  __release_param, which only looks at `not ds_active_sub_modules` -> the sentinel does cure the step0 deadlock.
  (`ds_persist` / raising stage3_param_persistence_threshold only takes effect from step1 on, RECORD step0 still releases everything
  -> it cannot cure the step0 deadlock, hence that route was not taken.)
- **unpin uses partition(has_been_updated=False)**: _partition_param hits :1550
  `ds_tensor is not None and not has_been_updated` -> only free_param (drops the full param.data),
  **keeping** the ds_tensor shard the optimizer already updated -> safe for trainable LoRA params too
  (the next gather rebuilds from the updated shard), and even less of an issue for the frozen base, which is never updated.
- **all_gather is a WORLD collective**, so every rank must call it with the same param_list for the
  **same expert**; all ranks share the same seed -> the critic/dmd timestep is globally identical -> the routing agrees
  (a single-expert smoke always uses one expert, so it agrees naturally; for dual-expert the WORLD broadcast of the wrapper-side routing decision is the backstop).
- **the grad mode splits the lifetime**: teacher scoring is entirely no_grad (utils_evoke_post:2642) -> pin -> fwd -> unpin
  is self-contained within one forward; critic scoring is grad-enabled -> the pin spans forward+backward and train_evoke
  cleans it up with unpin_all() after the critic backward. step() starts with partition_all_parameters(), which clears the sentinel
  and partitions -> the explicit unpin is idempotent (once partitioned, partition() hits `not AVAILABLE` and is a no-op).

Everything is a no-op when SP is off (G=1) -> byte-identical. Also a no-op when not ZeRO-3 (no ds_id, e.g. ZeRO-2).
"""

from .sp_runtime import is_sp_enabled

# sentinel sub-module id: real module.id values are assigned by _register_hooks_recursively counting up from 0, so a negative value can never collide.
_SP_PIN_SENTINEL = -0x5350  # 'SP'

# id(module) -> list of pinned params (deduplicated, ascending ds_id). Guards re-entry and feeds the unpin_all backstop cleanup.
_PINNED = {}


def _zero3_params(module):
    """All ZeRO-3-managed params (those with ds_id) under `module`, deduplicated and sorted by ascending ds_id -- same as
    deepspeed.zero.GatheredParameters (identical order on every rank, so the all_gather cannot get misaligned)."""
    params = [p for p in module.parameters() if hasattr(p, "ds_id")]
    return sorted(set(params), key=lambda p: p.ds_id)


def pin_module_params(module):
    """All-gather all ZeRO-3 params of `module` into a resident state and pin them (the sentinel stops the coordinator from partitioning midway).
    SP off / not ZeRO-3 / already pinned -> no-op. Returns whether a pin actually happened (so the caller can decide whether to unpin).

    pin (B) and mpu (A) **must be used together**, they plug different holes:
    - mpu: makes critic ZeRO-3 use the DP-stride-G group (which excludes the SP-peer ranks) -> removes the **cyclic deadlock**
      between the backward reduce-scatter and the SP-subgroup all_to_all.
    - pin: removes the **per-block D-group param all-gather** inside the block-loop -- it interleaves with the SP-subgroup collectives,
      and it requires all D peers (different clips) to reach the same block; the recompute speed differs slightly between clips -> the
      D-group all-gather waits on the slow peer -> which waits on its SP partner -> **stall** (measured on an mpu3 run: without
      pin the D1 group had 10 vs 1-2 outstanding collectives, an asymmetric hang). pin all-gathers the whole expert once into a resident state (zero all-gathers in the block-loop) -> no interleaving.
    With pin, the only D-group collective left is the **single reduce-scatter** of the backward epilogue (LoRA grad 15M < bucket 100M ->
    one epilogue reduce, **after** the recompute SP collectives, no interleaving). So pin+mpu together mean that nowhere in the block-loop
    does a D-group collective interleave with an SP collective. SP off / not ZeRO-3 -> no-op. Under mpu the all_gather lands on the D group
    (a param's ds_process_group *is* the D group), once, before the block-loop, without interleaving."""
    if not is_sp_enabled():
        return False
    if id(module) in _PINNED:
        return True  # already pinned (re-entry guard)
    params = _zero3_params(module)
    if not params:
        return False  # not ZeRO-3 (no ds_id) -> nothing to pin
    from .sp_runtime import sp_diag
    sp_diag(f"pin start (n={len(params)})")
    # one coalesced all-gather to full on the D group (mpu) (only NOT_AVAILABLE params are gathered, already-AVAILABLE ones are skipped).
    params[0].all_gather(param_list=params)
    sp_diag("pin all_gather done")
    # pin: the sentinel stays in ds_active_sub_modules -> __release_param inside release_sub_module will not partition.
    for p in params:
        p.ds_active_sub_modules.add(_SP_PIN_SENTINEL)
    _PINNED[id(module)] = params
    return True


def _release_param_list(params):
    """Remove the sentinel and re-partition (free the full param.data, keep the ds_tensor shard the optimizer updated).
    step() may already have cleared things via partition_all_parameters() -> idempotent on NOT_AVAILABLE params (partition() hits
    `not AVAILABLE` and returns immediately)."""
    if not params:
        return
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    for p in params:
        p.ds_active_sub_modules.discard(_SP_PIN_SENTINEL)
    still_avail = [p for p in params if p.ds_status == ZeroParamStatus.AVAILABLE]
    if still_avail:
        # has_been_updated=False -> only free_param, does not overwrite the ds_tensor shard the optimizer updated.
        still_avail[0].partition(param_list=still_avail, has_been_updated=False)


def unpin_module_params(module):
    """Undo the pin of a single module. Not pinned -> no-op. Called at the end of the forward on the teacher (no_grad) path."""
    _release_param_list(_PINNED.pop(id(module), None))


def unpin_all():
    """Backstop: undo the pin of every module still pinned (train_evoke calls it after the critic backward, to avoid leaking 28GB across steps).
    The pin on the critic (grad-enabled) path is not unpinned at the end of the forward -> this is what cleans it up."""
    for mid in list(_PINNED.keys()):
        _release_param_list(_PINNED.pop(mid, None))


def any_pinned():
    return len(_PINNED) > 0


# ============================================================================
# approach (A) mpu (DeepSpeed-Ulysses done right): the architectural fix for
# the 2D SP x ZeRO-3 forward+backward interleaving deadlock.
# ============================================================================
"""
Root cause recap (measured on two separate runs):
under 2D (sp<world) the critic ZeRO-3 parameter all-gather (forward) and grad reduce-scatter (backward)
land on the **WORLD** group by default, whose rank coverage overlaps the SP-subgroup all_to_all -> circular-wait deadlock. Approach (B) pin only suppresses
the per-parameter all-gather of the forward; the WORLD reduce-scatter of the backward still deadlocks.

Approach (A) mpu, the real fix (verified against the deepspeed 0.14.5 source, see the subagent report):
- pass an mpu to the critic engine so that ZeRO-3's **data-parallel group = the DP-stride-G group**
  ({r: r%G==j}, e.g. 8 GPUs with G=2 -> {0,2,4,6}/{1,3,5,7}). That group **excludes the SP-peer ranks**:
    - the forward parameter all-gather lands on the DP group -> rank0's all-gather only waits for {0,2,4,6}, not for rank1 which is
      blocked in the SP all_to_all -> no cycle -> the forward deadlock is solved (pin becomes redundant, already a no-op).
    - the backward grad reduce-scatter lands on the DP group (stage3.py reduce_scatter_coalesced uses
      self.dp_process_group) -> likewise it does not interleave with the SP all_to_all -> the backward deadlock is solved.
  The sharding degree drops from world to world/G (parameters cost G times more: dual-expert 56G/(world/G)).
- Key correctness point (§14 derivation): the DP-stride-G reduce-scatter only sums inside the DP group -> the partial grad of the SP-peer
  rank (which handles the other half of the frame shards of the same clip) lands in **another** DP group -> the parameter replicas of the two
  DP groups diverge. After the reduce-scatter and before optimizer.step, the **already-reduced grad shards** must be all-reduced (SUM) inside the SP group:
  the geometry lines up -- SP peers (2k,2k+1) both have dp_rank k inside their own DP group -> they hold the **same shard index** ->
  grad_partitions_flat_buffer is element-wise aligned -> a single SUM all-reduce merges the partials of the two frame shards.
  The normalization works out exactly: reduce_scatter already divided by dp_world_size (=world/G), so after the SP-SUM it is (1/(world/G))*sum_clip full_grad
  = the correct batch-mean (effective batch = world/G clips) -> **no xG loss-scale needed** (route-B's xG is void).
  Precondition: the mpu does not implement get_sequence_parallel_world_size() -> deepspeed's self.sequence_parallel_size==1
  -> the reduce_scatter path adds no hidden SP scaling (the one at stage3.py is the all-reduce path, not taken when reduce_scatter=true).
- get_model_parallel_group() = the SP group: used only for grad-**norm** deduplication (stage3.py/1762), it does not reduce the gradients
  themselves (verified in subagent Q4) -> no double correction; after the SP-SUM the two replica shards are identical -> the norm is counted once, correctly.
- global groups.mpu pollution: engine.py `groups.mpu = self.mpu` is **global** and is only set, never reset ->
  groups.mpu must be **saved/restored** around the critic prepare, otherwise a evoke-critic prepared later (mpu=None) would wrongly read
  the critic's stride-G group. During training stage3 uses the captured self.dp_process_group (not the global) -> restoring is safe.
"""


class CriticMPU:
    """DeepSpeed mpu (model-parallel-unit) -- makes critic ZeRO-3 shard/reduce on the DP-stride-G group.

    Only the methods deepspeed 0.14.5 actually calls are implemented (subagent Q1):
      - get_data_parallel_*     : group for ZeRO sharding/all-gather/reduce-scatter = DP-stride-G.
      - get_model_parallel_*    : = the SP group, used only for grad-norm dedup (does not reduce gradients).
    **Deliberately not implemented**: get_sequence_parallel_world_size() (deepspeed would add hidden SP scaling)
    and get_sequence_data_parallel_group() (it would fall back to the combined SP x DP = WORLD group, i.e. the deadlocking WORLD reduce;
    leaving it out -> groups._get_sequence_data_parallel_group() falls back to get_data_parallel_group()=DP-stride-G).
    """

    def __init__(self, dp_group, dp_ranks, sp_group, sp_ranks):
        import torch.distributed as dist
        self._dp_group = dp_group
        self._dp_ranks = list(dp_ranks)
        self._sp_group = sp_group
        self._sp_ranks = list(sp_ranks)
        self._dp_world_size = len(self._dp_ranks)
        self._sp_world_size = len(self._sp_ranks)
        self._dp_rank = dist.get_rank(group=dp_group)
        self._mp_rank = dist.get_rank(group=sp_group)

    # ---- data-parallel = DP-stride-G: ZeRO sharding/all-gather/reduce-scatter land on this group ----
    def get_data_parallel_group(self):
        return self._dp_group

    def get_data_parallel_world_size(self):
        return self._dp_world_size

    def get_data_parallel_rank(self):
        return self._dp_rank

    # ---- model-parallel = the SP group: grad-norm dedup only (stage3 does not reduce the gradients themselves)----
    def get_model_parallel_group(self):
        return self._sp_group

    def get_model_parallel_world_size(self):
        return self._sp_world_size

    def get_model_parallel_rank(self):
        return self._mp_rank


def build_critic_mpu():
    """Build a CriticMPU from sp_runtime's DP-stride-G / SP groups. Not 2D (sp==world or off) -> None."""
    from .sp_runtime import is_2d_sp, get_dp_group, get_dp_ranks, get_sp_group, get_sp_size
    import torch.distributed as dist
    if not is_2d_sp():
        return None
    dp_group = get_dp_group()
    dp_ranks = get_dp_ranks()
    sp_group = get_sp_group()
    assert dp_group is not None and sp_group is not None, "[mpu] 2D SP requires sp_runtime to have built the DP+SP groups"
    # global rank set of this SP group (G contiguous ranks): [g*G, g*G+G)
    g = get_sp_size()
    start = (dist.get_rank() // g) * g
    sp_ranks = list(range(start, start + g))
    return CriticMPU(dp_group, dp_ranks, sp_group, sp_ranks)


def _iter_grad_shard_buffers(engine):
    """Yield the already-reduced grad shard buffers inside the critic engine ZeRO-3 optimizer (for the SP-SUM all-reduce).
    Without CPU-offload (this repo's config): all shards live in a single grad_partitions_flat_buffer (stage3.py).
    With offload (unused): fall back to iterating fp32_partitioned_groups_flat[i].grad."""
    opt = getattr(engine, "optimizer", None)
    if opt is None:
        return
    flat = getattr(opt, "grad_partitions_flat_buffer", None)
    if flat is not None and flat.numel() > 0:
        yield flat
        return
    # offload fallback (not taken in this repo: the config has no offload_optimizer)
    fp32_flat = getattr(opt, "fp32_partitioned_groups_flat", None)
    if fp32_flat is not None:
        for g in fp32_flat:
            if getattr(g, "grad", None) is not None:
                yield g.grad


def sp_allreduce_grad_shards(engine):
    """after the backward reduce-scatter and before optimizer.step, all-reduce (SUM) the
    grad **shards** of every critic rank inside the SP group -- merging the partial grad of the SP-peer rank (the other frame shard of the same clip),
    so the parameter replicas of the two DP groups agree and the gradient = the correct batch-mean (derivation, no xG needed).

    Only active in 2D; SP off / sp==world / no flat buffer -> no-op. All SP groups do it in parallel (disjoint groups do not interfere)."""
    from .sp_runtime import is_2d_sp, get_sp_group, sp_diag
    if not is_2d_sp():
        return
    import torch.distributed as dist
    sp_group = get_sp_group()
    if sp_group is None:
        return
    n = 0
    for buf in _iter_grad_shard_buffers(engine):
        sp_diag(f"sp-sum all_reduce start (numel={buf.numel()})")
        dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=sp_group)
        sp_diag("sp-sum all_reduce done")
        n += 1
    if n == 0:
        # first-step backstop warning: no grad shard buffer found (deepspeed version / config change) -> SP-SUM did not take effect, the numbers will be wrong.
        print("[mpu][WARN] sp_allreduce_grad_shards: no grad shard buffer found (SP-SUM did NOT take effect!) "
              "-- check the deepspeed version / whether the critic is ZeRO-3 / whether optimizer offload is enabled.", flush=True)


def wrap_critic_engine_step(engine):
    """Wrap the critic DeepSpeedEngine.step: run sp_allreduce_grad_shards(engine) before the step.
    accelerate's backward() calls engine.step() internally (utils/deepspeed.py) -> both the reduce-scatter and the
    step happen inside critic_accelerator.backward(), and the training loop cannot get in between them -> engine.step must be wrapped.
    Only active in 2D; idempotent (a repeated wrap warns and is skipped). Returns whether a wrap actually happened."""
    from .sp_runtime import is_2d_sp
    if not is_2d_sp():
        return False
    if getattr(engine, "_sp_step_wrapped", False):
        return True
    _orig_step = engine.step

    def _sp_step(*args, **kwargs):
        from .sp_runtime import sp_diag
        sp_diag("engine.step ENTER (pre sp-sum)")
        sp_allreduce_grad_shards(engine)   # reduce-scatter already done (hook inside backward) -> SP-SUM the shards
        sp_diag("engine.step orig-step begin")
        r = _orig_step(*args, **kwargs)     # original step: grad-norm/clip/update all read the post-SP-SUM shards
        sp_diag("engine.step DONE")
        return r

    engine.step = _sp_step
    engine._sp_step_wrapped = True
    return True
