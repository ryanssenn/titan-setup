# TorchTitan Validation

A documented validation of TorchTitan distributed training on a multi-GPU setup.

## Purpose

This project verifies that TorchTitan can successfully launch, train, checkpoint, and log metrics in a distributed environment using FSDP.

The goal is not to train a new model or present research results. The goal is to validate a working TorchTitan setup and document the configuration, environment, and results.

## Environment

| Component | Value |
|------------|---------|
| Framework | TorchTitan v0.1.0 |
| PyTorch | 2.6.0 + CUDA 12.4 |
| Hardware | 4× NVIDIA H100 80GB HBM3 |
| Training Mode | FSDP |

## Results

The primary validated work is the full TorchTitan + FSDP pipeline using the tiny `debugmodel` (infrastructure validation run). See `outputs/infrastructure_run/`.

Artifacts, logs, and generated graphs for completed runs are under `outputs/`.

## Run

From inside a clone of this repo (`cd titan-setup`):

```bash
git clone --depth 1 -b v0.1.0 https://github.com/pytorch/torchtitan.git torchtitan

cd torchtitan
pip install -r .ci/docker/requirements.txt
pip install --force-reinstall \
  torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
cd ..

NGPU=1 CONFIG_FILE=configs/infrastructure_run.toml \
  ./run_experiment.sh --training.steps 10 --metrics.log_freq 2
```

Output lands in `outputs/infrastructure_run/`.

For 4 GPUs, the full 500-step run, Llama 3.1 8B, and troubleshooting, see `docs/reproduce.md`.

### Llama 3.1 8B

Requires Meta Llama approval for the gated repo + tokenizer download (see `docs/reproduce.md`).

```bash
echo 'HF_TOKEN=your_approved_hf_token_here' > .env && chmod 600 .env
```

Then use `CONFIG_FILE=configs/llama3_8b_2gpu.toml` (full commands in `docs/reproduce.md`).

## Sources

- TorchTitan v0.1.0 
- Compute: 4× NVIDIA H100 80GB HBM3
- Graphs generated from logs using matplotlib
- Analysis prepared with Grok Build
