#!/usr/bin/env bash
# Download the gated Llama 3.1 8B tokenizer into torchtitan/assets/tokenizer/.
# Reads HF_TOKEN from the environment or from .env in the repo root.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "${REPO_ROOT}/torchtitan" ]]; then
  echo "ERROR: torchtitan/ not found. Run ./setup_env.sh first." >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" && -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env" 2>/dev/null || true
  set +a
  if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN=$(grep -E '^[[:space:]]*HF_TOKEN=' "${REPO_ROOT}/.env" | head -1 | cut -d= -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//") || true
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: No HF_TOKEN found. Add your token to .env in the titan-setup root." >&2
  exit 1
fi

cd "${REPO_ROOT}/torchtitan"

python scripts/download_tokenizer.py \
  --repo_id meta-llama/Meta-Llama-3.1-8B \
  --hf_token "$HF_TOKEN" \
  --local_dir assets/tokenizer

ls -l assets/tokenizer/original/tokenizer.model
echo "Tokenizer ready. Run with CONFIG_FILE=configs/llama3_8b.toml"
