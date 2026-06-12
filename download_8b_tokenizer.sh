#!/usr/bin/env bash
# download_8b_tokenizer.sh
#
# Convenience wrapper for downloading the gated Llama 3.1 8B tokenizer.
# It prefers HF_TOKEN from the environment, then falls back to common
# gitignored files in the titan-setup root: .env, hf_token.env, .hf_token
#
# Usage (after placing your approved token in one of the supported places):
#   ./download_8b_tokenizer.sh
#
# The token is never committed. See README.md and docs/reproduce.md.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}/torchtitan"

# Try to obtain HF_TOKEN (env var takes precedence)
if [[ -z "${HF_TOKEN:-}" ]]; then
  for candidate in \
    "${REPO_ROOT}/.env" \
    "${REPO_ROOT}/hf_token.env" \
    "${REPO_ROOT}/.hf_token" \
    "${REPO_ROOT}/token.env"
  do
    if [[ -f "$candidate" ]]; then
      # Try to source it (works for simple KEY=val)
      # shellcheck disable=SC1090
      set -a
      source "$candidate" 2>/dev/null || true
      set +a

      # Fallback: parse HF_TOKEN=... line if sourcing didn't populate it
      if [[ -z "${HF_TOKEN:-}" ]]; then
        HF_TOKEN=$(grep -E '^[[:space:]]*HF_TOKEN=' "$candidate" | head -1 | cut -d= -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//" ) || true
      fi
      break
    fi
  done
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: No HF_TOKEN found."
  echo ""
  echo "Create one of these gitignored files in the titan-setup root:"
  echo "  .env                 (or hf_token.env, .hf_token, token.env)"
  echo ""
  echo "Example content:"
  echo "  HF_TOKEN=your_real_approved_meta_llama_token_here"
  echo ""
  echo "Then secure it:"
  echo "  chmod 600 .env"
  echo ""
  echo "Or set the variable directly:"
  echo "  export HF_TOKEN=..."
  echo ""
  echo "See docs/reproduce.md for full 8B instructions and Meta approval requirements."
  exit 1
fi

echo "HF token found (length: ${#HF_TOKEN})"

python scripts/download_tokenizer.py \
  --repo_id meta-llama/Meta-Llama-3.1-8B \
  --hf_token "$HF_TOKEN" \
  --local_dir assets/tokenizer

echo ""
echo "Download complete. Verifying file..."
ls -l assets/tokenizer/original/tokenizer.model

echo ""
echo "Tokenizer ready. You can now run the 8B experiment, e.g.:"
echo "  cd .."
echo "  HF_TOKEN=\$HF_TOKEN CONFIG_FILE=configs/llama3_8b_2gpu.toml ./run_experiment.sh ..."
echo ""
echo "(The training run itself only needs the local tokenizer file, not the HF token.)"
