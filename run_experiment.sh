#!/usr/bin/env bash
# titan-setup experiment launcher (current hardware)
set -euo pipefail

# Change to the torchtitan source directory (cloned as sibling or in this workspace)
cd /workspace/titan-setup/torchtitan

# Ensure the torchtitan package can be imported when using -m torchtitan.train
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

NGPU="${NGPU:-4}"
LOG_RANK="${LOG_RANK:-0}"
CONFIG_FILE="${CONFIG_FILE:-/workspace/titan-setup/configs/infrastructure_run.toml}"

# Note: On the original A40 RunPod pod, NCCL P2P/IB had to be disabled.
# These settings are generally harmful on modern H100 systems (they disable fast
# peer-to-peer communication). Do not set them unless you have a specific reason.
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== titan-setup run ==="
echo "GPUs: ${NGPU}"
echo "Config: ${CONFIG_FILE}"
echo "====================="

torchrun \
  --nproc_per_node="${NGPU}" \
  --rdzv_backend c10d \
  --rdzv_endpoint="localhost:0" \
  --local-ranks-filter "${LOG_RANK}" \
  --role rank \
  --tee 3 \
  -m torchtitan.train \
  --job.config_file "${CONFIG_FILE}" \
  "$@"