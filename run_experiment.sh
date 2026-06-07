#!/usr/bin/env bash
# titan-exp experiment launcher (2x A40)
set -euo pipefail

cd /workspace/titan-exp/torchtitan

NGPU="${NGPU:-2}"
LOG_RANK="${LOG_RANK:-0}"
CONFIG_FILE="${CONFIG_FILE:-/workspace/titan-exp/configs/infrastructure_run.toml}"

# Required on this RunPod pod: NCCL P2P hangs without these.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== titan-exp run ==="
echo "GPUs: ${NGPU}"
echo "Config: ${CONFIG_FILE}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
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