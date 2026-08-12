"""Share the host-resident copy of the frozen expert base across the ranks of one node.

With per-expert offload each rank keeps its own CPU copy of the expert that is not currently
routed (14B bf16 ~= 28GB), since `_offload_frozen_params_to` allocates a fresh anonymous CPU
tensor every time; eight ranks per node hold ~224GB of which ~196GB is duplicate, enough to get
ranks SIGKILLed by the host OOM killer.

Sharing is safe because the base is read-only -- frozen teacher weights (same filter as
`_offload_frozen_params_to`: names without 'lora_'), scored under no_grad -- and the ranks of a
node are processes in one container sharing a RAM-backed /dev/shm.

Offload becomes zero-copy, but the 28GB H2D on reload then reads from a tmpfs mmap, and a
pageable copy from mmap'd pages costs ~5s per step. `attach(pin=True)` registers the region with
cudaHostRegister so the H2D goes by DMA; EVOKE_SHARED_BASE_PIN=0 disables it.

One .bin holds every frozen parameter in named_parameters order, 64B aligned so a uint8 ->
bf16/fp32 `.view(dtype)` is legal, beside a .json manifest and a .ready marker. The filename
carries sig = sha1(weight dir + all (name, dtype, shape, offset))[:12], so a different checkpoint
or shape picks a different file, and equal sig implies bit-identical content -- an existing file
is safe to reuse. Only LOCAL_RANK 0 writes, via .tmp.<pid> plus an atomic os.replace. Buffers
stay out of the pool; they are small.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Tuple

import torch

_ALIGN = 64                                   # segment alignment; >=8 makes a uint8 view to any dtype legal
_SHM_DIR = os.environ.get("EVOKE_SHARED_BASE_DIR", "/dev/shm")
_READY_TIMEOUT_S = float(os.environ.get("EVOKE_SHARED_BASE_TIMEOUT", "1800"))
_SAMPLE_BLOCKS = 64                           # blocks sampled by verify_integrity
_SAMPLE_BLOCK_BYTES = 4096


def frozen_named_params(module) -> List[Tuple[str, torch.nn.Parameter]]:
    """The same filter as `_offload_frozen_params_to`: names not containing 'lora_'."""
    return [(n, p) for n, p in module.named_parameters() if "lora_" not in n]


def _build_layout(module) -> Tuple[List[dict], int]:
    entries: List[dict] = []
    off = 0
    for name, p in frozen_named_params(module):
        nbytes = p.numel() * p.element_size()
        off = ((off + _ALIGN - 1) // _ALIGN) * _ALIGN
        entries.append({
            "name": name,
            "dtype": str(p.dtype).replace("torch.", ""),
            "shape": list(p.shape),
            "offset": off,
            "nbytes": nbytes,
        })
        off += nbytes
    return entries, off


def _signature(entries: List[dict], extra_id: str) -> str:
    h = hashlib.sha1()
    h.update(str(extra_id).encode())
    for e in entries:
        h.update(f"{e['name']}|{e['dtype']}|{tuple(e['shape'])}|{e['offset']}".encode())
    return h.hexdigest()[:12]


def _paths(tag: str, sig: str) -> Tuple[str, str, str]:
    base = os.path.join(_SHM_DIR, f"evoke_frozen_{tag}_{sig}")
    return base + ".bin", base + ".json", base + ".ready"


def _view_of(flat: torch.Tensor, e: dict) -> torch.Tensor:
    seg = flat[e["offset"]: e["offset"] + e["nbytes"]]
    return seg.view(getattr(torch, e["dtype"])).view(*e["shape"])


def _sample_hash(flat: torch.Tensor) -> str:
    """sha1 over uniformly sampled 4KB blocks; hashing all 28GB would be far too expensive."""
    n = flat.numel()
    if n == 0:
        return ""
    h = hashlib.sha1()
    step = max(1, n // _SAMPLE_BLOCKS)
    for start in range(0, n, step):
        end = min(start + _SAMPLE_BLOCK_BYTES, n)
        h.update(flat[start:end].numpy().tobytes())
    return h.hexdigest()[:16]


def publish(module, tag: str, extra_id: str, verbose: bool = False) -> Tuple[str, int, bool]:
    """Write the module's frozen base to /dev/shm as a single copy. LOCAL_RANK 0 only.

    Returns (sig, total_bytes, wrote); wrote=False means a file with this sig already existed and
    is reused, its content being bit-identical by construction.
    """
    entries, total = _build_layout(module)
    sig = _signature(entries, extra_id)
    bin_p, json_p, ready_p = _paths(tag, sig)
    if os.path.exists(ready_p) and os.path.exists(bin_p):
        if verbose:
            print(f"[SHARED-BASE] publish skipped (already present) tag={tag} sig={sig} "
                  f"{total/2**30:.1f}GiB {bin_p}", flush=True)
        return sig, total, False

    tmp_p = bin_p + f".tmp.{os.getpid()}"
    flat = torch.from_file(tmp_p, shared=True, size=total, dtype=torch.uint8)
    pmap = dict(frozen_named_params(module))
    for e in entries:
        src = pmap[e["name"]].detach()
        if src.device.type != "cpu":
            src = src.to("cpu")
        _view_of(flat, e).copy_(src)
    manifest = {"tag": tag, "sig": sig, "total_bytes": total,
                "sample_hash": _sample_hash(flat), "entries": entries}
    del flat                                   # close the mmap so it lands (tmpfs, i.e. RAM)
    os.replace(tmp_p, bin_p)
    with open(json_p, "w") as f:
        json.dump(manifest, f)
    with open(ready_p, "w") as f:
        f.write(sig)
    if verbose:
        print(f"[SHARED-BASE] published tag={tag} sig={sig} {total/2**30:.1f}GiB "
              f"params={len(entries)} → {bin_p}", flush=True)
    return sig, total, True


def _try_pin(flat: torch.Tensor, tag: str, verbose: bool = False) -> bool:
    """Register the shared mmap region as pinned (cudaHostRegister).

    Once offload is zero-copy the 28GB H2D on reload reads from a tmpfs mmap, and a pageable copy
    from mmap'd pages costs ~5s per step; pinned, the H2D goes by DMA. The price is page-locking
    ~58GiB per node, plus tens of seconds of registration once at startup. Some kernels refuse to
    register a MAP_SHARED region: that returns False and the run continues, only slower.
    EVOKE_SHARED_BASE_PIN=0 disables it.
    """
    if os.environ.get("EVOKE_SHARED_BASE_PIN", "1") != "1":
        return False
    try:
        if not torch.cuda.is_available():
            return False
        torch.cuda.init()
        err = torch.cuda.cudart().cudaHostRegister(flat.data_ptr(), flat.numel(), 0)
        ok = (int(err) == 0)
        if ok and not flat.is_pinned():
            # Registration "succeeded" but the driver does not honour it: treat as failure, and
            #   unregister -- otherwise both costs are paid, a pageable H2D every step *and*
            #   29GiB per expert page-locked for good.
            try:
                torch.cuda.cudart().cudaHostUnregister(flat.data_ptr())
            except Exception:
                pass
            ok = False
        if verbose:
            print(f"[SHARED-BASE] pin tag={tag} {'OK' if ok else f'FAILED(err={int(err)})'} "
                  f"({flat.numel()/2**30:.1f}GiB, cudaHostRegister)", flush=True)
        return ok
    except Exception as _e:
        if verbose:
            print(f"[SHARED-BASE] pin tag={tag} skipped ({type(_e).__name__}: {_e}); "
                  f"H2D stays pageable, about +5s per step", flush=True)
        return False


def attach(module, tag: str, extra_id: str, adopt: bool = True,
           timeout_s: float = _READY_TIMEOUT_S, verbose: bool = False,
           pin: bool = True) -> Tuple[str, int]:
    """Attach to the node-local single copy. adopt=True repoints `p.data` at once, freeing this
    rank's anonymous copy. pin=True registers the region as pinned (see `_try_pin`)."""
    entries, total = _build_layout(module)
    sig = _signature(entries, extra_id)
    bin_p, _json_p, ready_p = _paths(tag, sig)

    t0 = time.time()
    while not (os.path.exists(ready_p) and os.path.exists(bin_p)):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"[SHARED-BASE] waiting for {ready_p} timed out after {timeout_s:.0f}s -- did LOCAL_RANK 0 finish publishing?")
        time.sleep(0.5)

    flat = torch.from_file(bin_p, shared=True, size=total, dtype=torch.uint8)
    shared: Dict[str, torch.Tensor] = {e["name"]: _view_of(flat, e) for e in entries}
    module._shared_host_params = shared        # `_offload_frozen_params_to` keys zero-copy offload off this
    module._shared_host_flat = flat            # keep a reference so the mmap is not collected
    module._shared_host_sig = sig
    module._shared_host_hash = _sample_hash(flat)
    # pin after hashing but before adopt: hashing reads the whole region once, faulting the pages in.
    module._shared_host_pinned = _try_pin(flat, tag, verbose=verbose) if pin else False

    if adopt:
        # Drop this rank's anonymous CPU copy now. Both experts sit on CPU at construction, so this
        #   is the step that actually saves the memory.
        n_adopt = 0
        for name, p in frozen_named_params(module):
            if p.data.device.type == "cpu" and name in shared:
                p.data = shared[name]
                n_adopt += 1
        import gc
        gc.collect()
        if verbose:
            print(f"[SHARED-BASE] attached tag={tag} sig={sig} {total/2**30:.1f}GiB "
                  f"adopt={n_adopt}/{len(entries)} params", flush=True)
    elif verbose:
        print(f"[SHARED-BASE] attached tag={tag} sig={sig} {total/2**30:.1f}GiB (no adopt)", flush=True)
    return sig, total


def publish_and_attach(module, tag: str, extra_id: str, local_rank: int,
                       barrier=None, verbose: bool = False, pin: bool = True) -> Tuple[str, int]:
    """LOCAL_RANK 0 writes, the others wait for .ready, then all attach. `barrier` may be dist.barrier."""
    if int(local_rank) == 0:
        publish(module, tag, extra_id, verbose=verbose)
    if barrier is not None:
        try:
            barrier()
        except Exception as _e:                # e.g. dist not initialised yet -> fall back to polling .ready
            if verbose:
                print(f"[SHARED-BASE] barrier skipped ({type(_e).__name__}), polling .ready instead", flush=True)
    return attach(module, tag, extra_id, adopt=True, verbose=verbose, pin=pin)


def verify_integrity(module, where: str = "") -> bool:
    """Detect a stray write to the shared base, which after sharing corrupts every rank on the node."""
    flat = getattr(module, "_shared_host_flat", None)
    ref = getattr(module, "_shared_host_hash", None)
    if flat is None or not ref:
        return True
    cur = _sample_hash(flat)
    ok = (cur == ref)
    if not ok:
        print(f"[SHARED-BASE][FATAL] the shared frozen base has been corrupted! tag_sig={getattr(module,'_shared_host_sig',None)} "
              f"expect={ref} got={cur} where={where}", flush=True)
    return ok
