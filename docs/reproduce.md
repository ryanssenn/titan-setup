# Reproduction Instructions

This document provides standalone instructions to reproduce runs from this project. The launcher is portable and works from any clone location or current working directory (you just need the `titan-setup/` directory containing `run_experiment.sh`, `configs/`, and a `torchtitan/` checkout next to them). All examples assume you `cd` into the root of your clone of this repo.

**Important:** The debug model validation can (and should) be run without any Hugging Face token. The full Llama 3.1 8B run requires a token with access to the gated `meta-llama` repositories.

## Prerequisites

- NVIDIA GPU(s) with recent drivers (CUDA 12.x or 13.x recommended for H100).
- Python 3.11 (or compatible).
- Git.
- Sufficient disk space (~10+ GB recommended for torchtitan clone + small runs).

## Step-by-Step: Run the Initial Debug Model Validation

This is the recommended first step before any larger training. It exercises the full pipeline using a tiny `debugmodel` (≈6.3M parameters) and the small public `c4_test` dataset.

It does **not** require a gated HF token.

### 1. Clone torchtitan (v0.1.0)

The project was originally validated against torchtitan v0.1.0. Clone it inside the titan-setup directory:

```bash
cd titan-setup   # (or the directory where you cloned this repo)
git clone --depth 1 -b v0.1.0 https://github.com/pytorch/torchtitan.git torchtitan
```

### 2. Set up Python dependencies

Install the requirements for v0.1.0 and pin a compatible PyTorch version (the original project used 2.6.0+cu124; 2.5.x+cu124 also worked in testing):

```bash
cd titan-setup/torchtitan

# Install torchtitan's requirements (from the v0.1.0 tree)
pip install -r .ci/docker/requirements.txt

# Force a known-good torch build compatible with the original setup and this hardware
pip install --force-reinstall \
  torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

**Note:** You may see dependency conflicts during installation (common when mixing with system packages). The debug run has succeeded with the versions above.

### 3. (Optional) Clean previous outputs

If you want a completely fresh run:

```bash
rm -rf outputs/*
```

### 4. Run the debug model

Use the launcher with the project's infrastructure config (which targets the debug model) and override the number of steps for a fast sanity check.

**Recommended quick validation (single GPU):**

```bash
cd titan-setup

NGPU=1 \
CONFIG_FILE=configs/infrastructure_run.toml \
./run_experiment.sh \
  --training.steps 10 \
  --metrics.log_freq 2
```

**Attempt on all available GPUs (4 in the target environment):**

```bash
cd titan-setup

NGPU=4 \
CONFIG_FILE=configs/infrastructure_run.toml \
./run_experiment.sh \
  --training.steps 10 \
  --metrics.log_freq 2
```

**What to expect on success (example from a working 1-GPU run):**
- The job starts and prints "Starting job".
- The debugmodel is built (6.27M parameters).
- The small c4_test dataset is prepared.
- Training begins and you see lines like:
  ```
  step:  1  loss:  8.2412  memory:  1.39GiB ... mfu: 0.19%
  ...
  step: 10  loss:  7.5848  memory:  1.51GiB ... mfu: 3.62%
  ```
- Checkpoints are saved.
- "Training completed" is printed at the end.
- No NCCL or import errors.

**Notes / common issues**
- The launcher (`run_experiment.sh`) is now fully portable: it locates `torchtitan/` and `configs/` relative to itself using the script location. Relative `CONFIG_FILE=...` values (exactly as written in the examples) work and are automatically resolved before changing directory.
- It defaults to NGPU=4, injects a sensible `--job.dump_folder` under `outputs/<config-name>` (next to your titan-setup clone) if you don't override it, and no longer forces the old A40-specific `NCCL_P2P_DISABLE=1` / `NCCL_IB_DISABLE=1`.
- On 4 GPUs you may encounter NCCL initialization errors ("unhandled cuda error", "operation cannot be performed in the present state"). If this happens, fall back to `NGPU=1` for the quick debug validation.
- If you see `ModuleNotFoundError: No module named 'tyro'` (or import errors around `torch.distributed.tensor` / DTensor), re-check that you installed the requirements + pinned torch==2.6.0+cu124 exactly as shown above. The v0.1.0 tree is sensitive to the PyTorch version.
- The run will create output under the `dump_folder` (the launcher ensures a clean default under your clone's `outputs/` directory).

### 5. Inspect results

After a successful run you can look at:

- TensorBoard: `tensorboard --logdir outputs/infrastructure-run/tb --port 6006`
- Raw logs in the configured output directory.
- The config itself (`configs/infrastructure_run.toml`) controls most behavior (you can override many fields on the command line with `--section.key value`).

A 500-step run can be performed the same way by omitting (or increasing) the `--training.steps` override.

The launcher will automatically place results in `outputs/infrastructure-run/` (or the basename of whatever config you point at) unless you pass an explicit `--job.dump_folder`.

## Historical / Original A40 Instructions (for reference only)

The commands below were used on the original 2× NVIDIA A40 RunPod pod. They are kept here for historical context only. Use the "Initial Debug Model" section above when working on the current 4× H100 hardware.

```bash
# Historical smoke test (10 steps)
NGPU=2 \
  CONFIG_FILE=torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml \
  ./run_experiment.sh \
  --job.dump_folder outputs/smoke-test

# Historical full infrastructure validation
./run_experiment.sh
```

## Llama 3.1 8B Run (Requires Gated HF Token)

The debug model instructions above do **not** require any Hugging Face access.

Once you have a token with access to the gated `meta-llama` repositories:

1. Download the tokenizer (run from inside the torchtitan clone):

```bash
cd titan-setup/torchtitan
python torchtitan/scripts/download_tokenizer.py \
  --repo_id meta-llama/Llama-3.1-8B \
  --hf_token "$HF_TOKEN" \
  --local_dir assets/tokenizer
```

   After this step you should have a file at `assets/tokenizer/original/tokenizer.model` (or adjust the path in `configs/llama3_8b_2gpu.toml` accordingly).

2. Run the 8B experiment (example using the updated launcher):

```bash
cd titan-setup
NGPU=4 CONFIG_FILE=configs/llama3_8b_2gpu.toml ./run_experiment.sh \
  --training.steps 50   # use a small number for a short validation run
```

This uses the real `c4` dataset and sequence length 8192.

See `docs/8b-attempt.md` for background on the previous blocked attempt.

## Viewing Results

Results are written to the `dump_folder` specified in the active config (commonly `outputs/infrastructure-run/` or `outputs/llama3-8b-run/`).

Useful commands:

- TensorBoard: `tensorboard --logdir outputs/infrastructure-run/tb --port 6006`
- List recent logs: `ls -l outputs/`
- Full logs are also emitted to stdout/stderr by the launcher.
