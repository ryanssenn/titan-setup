#!/usr/bin/env bash
# One-time environment setup: venv, PyTorch, TorchTitan v0.1.0.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

TORCHTITAN_TAG="v0.1.0"
TORCHTITAN_URL="https://github.com/pytorch/torchtitan.git"

echo "=== titan-setup: environment setup ==="

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip

pip install --force-reinstall \
  torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

if [[ ! -d torchtitan/.git ]]; then
  echo "Cloning TorchTitan ${TORCHTITAN_TAG}..."
  git clone --depth 1 -b "${TORCHTITAN_TAG}" "${TORCHTITAN_URL}" torchtitan
else
  echo "TorchTitan checkout already present."
fi

pip install -r torchtitan/.ci/docker/requirements.txt

if [[ ! -f .env ]]; then
  cat > .env <<'EOF'
# Hugging Face token (only needed for llama3_8b tokenizer download).
# Requires Meta approval: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
HF_TOKEN=
EOF
  chmod 600 .env
  echo "Created .env - add your HF_TOKEN there if running llama3_8b."
fi

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("nccl:", torch.cuda.nccl.version())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "=== setup complete ==="
echo "Next: source .venv/bin/activate"
echo "Run:  NGPU=1 CONFIG_FILE=configs/debugmodel.toml ./run.sh --training.steps 10"
