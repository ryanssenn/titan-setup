#!/usr/bin/env bash
# Launch TorchTitan training.
#
# Usage:
#   NGPU=2 CONFIG_FILE=configs/debugmodel.toml ./run.sh [torchtitan overrides...]
#
# Outputs land in outputs/<config-basename>/ unless --job.dump_folder is passed.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/debugmodel.toml}"
if [[ "${CONFIG_FILE}" != /* ]]; then
  CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: config not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/torchtitan" ]]; then
  echo "ERROR: torchtitan/ not found. Run ./setup_env.sh first." >&2
  exit 1
fi

cd "${REPO_ROOT}/torchtitan"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

NGPU="${NGPU:-1}"
LOG_RANK="${LOG_RANK:-0}"

has_dump=false
for a in "$@"; do
  case "$a" in
    --job.dump_folder|--job.dump_folder=*) has_dump=true; break ;;
  esac
done
if [[ "${has_dump}" = false ]]; then
  base="$(basename "${CONFIG_FILE%.toml}")"
  set -- --job.dump_folder "${REPO_ROOT}/outputs/${base}" "$@"
fi

# NVLS multicast returns CUDA 401 for 3+ GPU collectives with default NCCL settings.
# See docs/NCCL.md.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== titan-setup run ==="
echo "GPUs:   ${NGPU}"
echo "Config: ${CONFIG_FILE}"
echo "NCCL_NVLS_ENABLE: ${NCCL_NVLS_ENABLE}"
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
