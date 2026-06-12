#!/usr/bin/env bash
# titan-setup experiment launcher (4x H100)
set -euo pipefail

cd /workspace/titan-setup/torchtitan

# Make torchtitan importable when running as -m torchtitan.train
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

NGPU="${NGPU:-4}"
LOG_RANK="${LOG_RANK:-0}"
CONFIG_FILE="${CONFIG_FILE:-/workspace/titan-setup/configs/infrastructure_run.toml}"

# Note: NCCL_P2P_DISABLE / NCCL_IB_DISABLE were required on the old A40 RunPod pod
# but are harmful on modern H100 setups (disable fast P2P). Do not set them here.
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