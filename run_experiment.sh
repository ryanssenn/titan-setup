#!/usr/bin/env bash
# titan-setup experiment launcher
set -euo pipefail

# Compute the root of this titan-setup clone from the script's own location.
# This makes the launcher fully portable (works from any cwd, any clone path,
# and even when invoked via full path).
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Default the config (relative to the repo root). This is the portable default.
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/infrastructure_run.toml}"

# If the user supplied a *relative* CONFIG_FILE (as shown in the reproduction
# instructions), resolve it against the repo root so it survives the upcoming cd.
if [[ "${CONFIG_FILE}" != /* ]]; then
  CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"
fi

# Change into the torchtitan source tree (expected layout: <repo-root>/torchtitan).
# All subsequent Python/torchrun work happens with this as cwd.
cd "${REPO_ROOT}/torchtitan"

# Ensure `python -m torchtitan.train` can find the package.
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

NGPU="${NGPU:-4}"
LOG_RANK="${LOG_RANK:-0}"

# If the caller did not already supply a --job.dump_folder (either via extra
# args or by editing the .toml), inject a convenient default under the
# titan-setup root. This guarantees artifacts land next to the repo
# (outputs/<config-basename>) even if the .toml contains a legacy absolute path.
has_dump=false
for a in "$@"; do
  case "$a" in
    --job.dump_folder|--job.dump_folder=*) has_dump=true; break ;;
  esac
done
if [ "${has_dump}" = false ]; then
  base="$(basename "${CONFIG_FILE%.toml}")"
  set -- --job.dump_folder "${REPO_ROOT}/outputs/${base}" "$@"
fi

# Optional env tweaks (NCCL disables are intentionally left off for H100-era hardware).
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