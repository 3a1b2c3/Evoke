#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
# Stage 1 -- camera-controllable long-context training (LoRA)
#   produces: models/train/stage1_camera_control   (release it as models/evoke/stage1_camera_control)
#   init    : models/evoke/stage1_camera_control
#   config  : configs/training/stage1_camera_control.yaml
#   Default scale: 1 node x 8 GPUs
#   Multi-node: the platform injects RANK / MASTER_ADDR / MASTER_PORT. The node and process
#     counts are fixed in the accelerate yaml, because CLI overrides are unreliable under the
#     DEEPSPEED distributed_type -- change ACCELERATE_CONFIG rather than passing
#     --num_machines/--num_processes.
#   Env overrides: ACCELERATE_CONFIG / TRAINING_CONFIG / EVOKE_PYTHON_BIN
# ════════════════════════════════════════════════════════════════════════════
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

# Optional bin directory of the python environment, so no absolute path is baked in.
[ -n "${EVOKE_PYTHON_BIN:-}" ] && export PATH="$EVOKE_PYTHON_BIN:$PATH"

export WANDB_MODE=${WANDB_MODE:-offline}
export ACCELERATE_DEEPSPEED_ZERO3_INIT=false
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TORCH_NCCL_ENABLE_MONITORING=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800000}
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"

# NCCL over IB; harmless on a single node.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_TIMEOUT=3600000
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TC=${NCCL_IB_TC:-160}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-30}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}

MACHINE_RANK=${RANK:-0}
MASTER_ADDR_ARG=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT_ARG=${MASTER_PORT:-29500}

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/accelerate/zero2_1x8.yaml}
TRAINING_CONFIG=${TRAINING_CONFIG:-configs/training/stage1_camera_control.yaml}

mkdir -p logs
echo "[stage1_camera_control] rank=$MACHINE_RANK master=$MASTER_ADDR_ARG:$MASTER_PORT_ARG"
echo "[stage1_camera_control] accelerate=$ACCELERATE_CONFIG"
echo "[stage1_camera_control] training=$TRAINING_CONFIG"

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --machine_rank "$MACHINE_RANK" \
  --main_process_ip "$MASTER_ADDR_ARG" \
  --main_process_port "$MASTER_PORT_ARG" \
  train_evoke.py \
  --config "$TRAINING_CONFIG" \
  2>&1 | tee "logs/stage1_camera_control_$(date +%Y%m%d_%H%M%S)_rank${MACHINE_RANK}.log"
