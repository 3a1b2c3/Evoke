# VideoAlign / VideoReward (vendored)

The modules in this directory derive from the upstream VideoAlign repository, which releases
**VideoReward** — a Qwen2-VL-based reward model scoring generated video on visual quality, motion
quality and text alignment.

- Upstream: <https://github.com/KlingAIResearch/VideoAlign>
- Paper: *Improving Video Generation with Human Feedback* (NeurIPS 2025)
- License: MIT — see `LICENSE` in this directory. Redistribution must retain the copyright notice
  and the permission notice, which `LICENSE` carries verbatim.
- Copyright: (c) 2025 Kling Team, Kuaishou Technology
- Base model: Qwen2-VL-2B-Instruct. Upstream credits TRL and Qwen2-VL-Finetune for parts of the
  training code.

## What is here, and how it differs from upstream

`inference.py`, `train_reward.py`, `trainer.py`, `data.py`, `utils.py`, `prompt_template.py` and
`vision_process.py` are taken from the upstream tree and adapted to be importable as a package
(`evoke.third_party.videoalign`) rather than run as top-level scripts. Upstream's `checkpoints/`,
`datasets/`, `ds_config/`, `eval_videogen_rewardbench.py`, `train.sh` and `environment.yaml` are not
vendored — nothing here uses them.

## How it is used

Only `VideoVLMRewardInference` from `inference.py` is reachable from this repo, through
`train_evoke.py`, and only when `training_config.is_use_reward_model` is true. **All four released
training recipes set it to false**, so the reward model is never instantiated on the released paths
and its weights are not part of this release. It is kept for the reward-guided experiments the
training code still supports.
