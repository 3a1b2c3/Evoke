import copy
import json
import math
import os
from typing import Any, Dict, Iterable, Optional, Union

from huggingface_hub import save_torch_state_dict

from diffusers.utils import (
    deprecate,
    is_torchvision_available,
    is_transformers_available,
)


if is_transformers_available():
    pass

if is_torchvision_available():
    pass

import deepspeed
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


# Adapted from diffusers-style ema https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L263
class EMAModel_Zero3:
    """EMA wrapper compatible with DeepSpeed ZeRO-3."""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        weight_file_prefix: Optional[str] = "",
        **kwargs,
    ):
        """Initialize EMA model with decay schedule parameters."""

        self.model = model

        if kwargs.get("max_value", None) is not None:
            deprecation_message = "The `max_value` argument is deprecated. Please use `decay` instead."
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        if kwargs.get("device", None) is not None:
            deprecation_message = "The `device` argument is deprecated. Please use `to` instead."
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`

        self.model_cls = model_cls
        self.model_config = model_config

        self.weight_file_prefix = weight_file_prefix

    @classmethod
    def extract_ema_kwargs(cls, kwargs):
        """Extract EMA-related keys from a kwargs dict, removing them in-place."""
        ema_kwargs = {}
        for key in [
            "decay",
            "min_decay",
            "optimization_step",
            "update_after_step",
            "use_ema_warmup",
            "inv_gamma",
            "power",
        ]:
            if kwargs.get(key, None) is not None:
                ema_kwargs[key] = kwargs.pop(key)
        return ema_kwargs

    @classmethod
    def from_pretrained(cls, path, model_cls) -> "EMAModel_Zero3":
        config = model_cls.load_config(path)
        ema_kwargs = cls.extract_ema_kwargs(config)
        model = model_cls.from_pretrained(path)

        ema_model = cls(model, model_cls=model_cls, model_config=config)

        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        rank = int(os.getenv("RANK", "0"))
        state_dict = self.state_dict()
        state_dict.pop("model")

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if rank == 0:
                    model_state_dict[k] = vv

        if rank == 0:
            os.makedirs(path, exist_ok=True)
            print(f"state_dict, {state_dict.keys()}")
            import time

            t_start = time.perf_counter()
            print(f"[{t_start:.4f}] save_pretrained start")

            print(type(self.model_config), self.model_config)
            for k, v in state_dict.items():
                if isinstance(self.model_config, dict):
                    self.model.config[k] = v
                else:
                    setattr(self.model_config, k, v)
            t1 = time.perf_counter()
            print(f"[{t1:.4f}] after setattr config ({t1 - t_start:.4f}s)")

            if hasattr(self.model_config, "save_pretrained"):
                self.model_config.save_pretrained(path)
            else:
                with open(os.path.join(path, "config.json"), "w") as f:
                    json.dump(self.model_config, f, indent=2)
            if hasattr(self.model, "generation_config"):
                print(type(self.model.generation_config), self.model.generation_config)
                self.model.generation_config.save_pretrained(path)
                # with open(os.path.join(path, "generation_config.json"), "w") as f:
                #     json.dump(self.model.generation_config, f, indent=2)
            t2 = time.perf_counter()
            print(f"[{t2:.4f}] self.model.save_config(path) ({t2 - t1:.4f}s)")

            if self.weight_file_prefix != "":
                self._save_pretrained_with_prefix(model_state_dict, path, self.weight_file_prefix)
            else:
                torch.save(model_state_dict, os.path.join(path, "pytorch_model.bin"))
            t3 = time.perf_counter()
            print(f"[{t3:.4f}] after save_pretrained ({t3 - t2:.4f}s)")

            print(f"[{t3:.4f}] total {t3 - t_start:.4f}s")
        return model_state_dict

    def _save_pretrained_with_prefix(self, state_dict, save_dir, weight_file_prefix):
        suffix = "{suffix}"
        pattern = f"{weight_file_prefix}{suffix}.safetensors"
        save_torch_state_dict(
            state_dict=state_dict,
            save_directory=save_dir,
            filename_pattern=pattern,
            max_shard_size="5GB",
            safe_serialization=True,
        )

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage.step` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage.step`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

        parameters = list(parameters)

        self.optimization_step += 1

        # Compute decay and update EMA parameters.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay
        for s_param, param in zip(self.model.parameters(), parameters):
            s_tensor, tensor = None, None
            if hasattr(s_param, "ds_tensor"):  # EMA ZeRO-3
                s_tensor = s_param.ds_tensor
                if hasattr(param, "ds_tensor"):  # DiT ZeRO-3
                    tensor = param.ds_tensor
                else:  # DiT ZeRO-2
                    rank, world_size = int(os.getenv("RANK")), int(os.getenv("WORLD_SIZE"))
                    partition_size = math.ceil(param.numel() / world_size)
                    start = partition_size * rank
                    end = start + partition_size

                    one_dim_param = param.data.contiguous().view(-1)
                    if start < param.numel() and end <= param.numel():
                        tensor = one_dim_param.narrow(0, start, partition_size)
                    elif start < param.numel():
                        # raise ValueError(f'start {start}, end {end}, param.numel() {param.numel()}, partition_size {partition_size}')
                        elems_to_copy = param.numel() - start
                        s_tensor = s_param.ds_tensor.narrow(0, 0, elems_to_copy)
                        tensor = one_dim_param.narrow(0, start, elems_to_copy)
                    else:
                        # raise ValueError(f'start {start}, end {end}, param.numel() {param.numel()}, partition_size {partition_size}')
                        continue
            else:  # DiT/EMA ZeRO-2
                s_tensor = s_param.data
                tensor = param.data

            assert s_tensor.shape == tensor.shape, (
                f"mismatch shape, s_tensor: {s_tensor.shape}, tensor: {tensor.shape}"
            )

            if param.requires_grad:
                s_tensor.sub_(one_minus_decay * (s_tensor - tensor.to(s_tensor.dtype)))
            else:
                s_tensor.copy_(tensor)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Copy EMA weights into the given parameters."""
        parameters = list(parameters)
        for s_param, param in zip(self.model.parameters(), parameters):
            param.data.copy_(s_param.to(param.device).data)

    def to(self, device=None, dtype=None) -> None:
        """Move EMA model to the specified device/dtype."""
        self.model = self.model.to(device=device, dtype=dtype)

    def state_dict(self) -> dict:
        """Return EMA state as a dict (tensor references, not copies)."""
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "weight_file_prefix": self.weight_file_prefix,
            "model": self.model.state_dict(),
        }

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Save a snapshot of the given parameters for later restoration."""
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Restore parameters previously saved with `store()`."""
        if self.temp_stored_params is None:
            raise RuntimeError("This ExponentialMovingAverage has no `store()`ed weights to `restore()`")
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data)

        # Better memory-wise.
        self.temp_stored_params = None

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore EMA state from a dict produced by `state_dict()`."""
        # deepcopy to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        self.weight_file_prefix = state_dict.get("weight_file_prefix", self.weight_file_prefix)
        if not isinstance(self.weight_file_prefix, (str)):
            raise ValueError("Invalid weight_file_prefix")

        model_state_dict = state_dict.get("model", None)
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
