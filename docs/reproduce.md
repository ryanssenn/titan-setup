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

**Recommended quick validation (NGPU=1 or 2):**

Use NGPU=1 for the fastest smoke test. Use NGPU=2 when you want to validate actual FSDP sharding (the debug model + c4_test is sufficient for this).

```bash
cd titan-setup

NGPU=1 \
CONFIG_FILE=configs/infrastructure_run.toml \
./run_experiment.sh \
  --training.steps 10 \
  --metrics.log_freq 2
```

**Multi-GPU validation (NGPU=2 recommended for exercising real FSDP):**

NGPU=2 runs FSDP sharding (`data_parallel_shard_degree=-1`) across the available GPUs and has been confirmed to work on the current pod for the debug model:

```bash
cd titan-setup

NGPU=2 \
CONFIG_FILE=configs/infrastructure_run.toml \
./run_experiment.sh \
  --training.steps 10 \
  --metrics.log_freq 2
```

**Note on 4 GPUs (current pod limitation):**

The following command (with the NCCL peer-to-peer / IB disables that helped on some earlier pods) currently fails on this 4× H100 setup:

```bash
cd titan-setup

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
NGPU=4 \
CONFIG_FILE=configs/infrastructure_run.toml \
./run_experiment.sh \
  --training.steps 10 \
  --metrics.log_freq 2
```

It fails very early (before training steps) during `Trainer` initialization in `set_determinism` → `torch.distributed.broadcast` with:

```
torch.distributed.DistBackendError: NCCL error ... ncclUnhandledCudaError: Call to CUDA function failed.
Last error: Cuda failure 401 'the operation cannot be performed in the present state'
```

Neither running without the NCCL flags nor prefixing them has resolved it on the current pod/container. 1-GPU and 2-GPU paths remain reliable for validating the pipeline. See the "Notes / common issues" section below for more context and workarounds to try (pod restart, driver updates, etc.).

**What to expect on success (NGPU=1 or NGPU=2 on current hardware):**
- Launcher prints "=== titan-setup run ===" + "GPUs: N" + the resolved config path.
- "Starting job: debug-model infrastructure validation (FSDP + selective AC, c4_test)"
- Tokenizer + "Preparing c4_test dataset from tests/assets/c4_test" (2000 examples generated quickly).
- "Building llama3 debugmodel ... dim=256, n_layers=6 ..." → "Model llama3 debugmodel size: 6,270,208 total parameters"
- "Applied selective activation checkpointing"
- On NGPU=2 you will additionally see: "Applied FSDP to the model" and a 1-D device mesh log (`Building 1-D device mesh with ['dp_shard'], [2]`).
- "TensorBoard logging enabled. Logs will be saved at .../outputs/infrastructure_run/tb/..."
- Several warnings are normal: warmup steps adjusted (because 10-step short run), "lspci" not found (harmless), etc.
- Training progress examples (log_freq=2):
  - NGPU=1: loss drops (e.g. ~8.20 → ~7.66), throughput ramps from low tens of k tps to hundreds of k tps, MFU rises to a few percent.
  - NGPU=2: similar loss curve, global batch size doubles (16), real FSDP sharding is active.
- Checkpoints: "Saving the checkpoint..." at step 1 and "Saving a full checkpoint at last step, step 10" → directories under `checkpoint/`.
- "Training completed"
- "Process group destroyed."
- No NCCL errors (for 1-2 GPUs).
- Artifacts under `outputs/infrastructure_run/` (the launcher derives the folder name from the basename of the .toml you pointed at via CONFIG_FILE). Subdirs: `checkpoint/`, `tb/`, `comm_trace/`.

**Notes / common issues**
- **Current pod status (debug model):** NGPU=1 works reliably (basic pipeline). NGPU=2 works reliably and exercises real FSDP2 sharding + global batch scaling. NGPU=4 (even with `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`) currently fails during distributed initialization with the NCCL `Cuda failure 401` error shown above. This appears specific to the current RunPod container/pod state; the same flags resolved similar issues on earlier pods.
- **Clear checkpoints when changing GPU count:** Checkpoints contain per-rank dataloader state (e.g. `dataloader.dp_rank_0`, `dataloader.dp_rank_1`). Switching from NGPU=1 to NGPU=2 (or vice versa) without clearing the checkpoint folder will cause `RuntimeError: Missing key in checkpoint state_dict: dataloader.dp_rank_X`. Run `rm -rf outputs/infrastructure_run/checkpoint` (or the relevant dump folder) between different world sizes.
- The launcher is fully portable (locates torchtitan/ relative to itself, auto-derives dump_folder from the CONFIG_FILE basename → `outputs/infrastructure_run/` in this case).
- Benign noise after pip: root-user warning + "new pip available". Detached HEAD after the torchtitan tag clone is expected.
- Other normal short-run warnings: warmup/decay step count adjustments, missing `lspci`.
- Tyro / DTensor import errors almost always mean the torch==2.6.0+cu124 pin step did not fully take effect.
- For the full 500-step validation simply omit the two `--training.steps 10 --metrics.log_freq 2` overrides. On this hardware class it finishes quickly.
- If you need 4-GPU operation and it fails: try restarting the pod, ensuring a clean environment (no stale NCCL state), or testing the NCCL flags plus `NCCL_DEBUG=INFO` for more diagnostics. Report the exact pod/driver/CUDA/NCCL versions if the issue persists across pods.

### 5. Inspect results

After a successful run you can look at:
- TensorBoard: `tensorboard --logdir outputs/infrastructure_run/tb --port 6006`
- List created artifacts: `ls -l outputs/infrastructure_run/` (you will see `checkpoint/step-1/`, `checkpoint/step-10/`, `tb/`, `comm_trace/`)
- The active config and overrides control everything; use `--section.key value` on the command line for quick experiments.

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

**Prerequisite:** You need a Hugging Face token from an account that has been explicitly approved by Meta for the gated `meta-llama/Llama-3.1-8B` (or `Meta-Llama-3.1-8B`) repository. A regular HF token is not sufficient; approval can take time (see `docs/8b-attempt.md`).

1. Set your HF token and download the tokenizer (run from inside the torchtitan clone):

**Recommended approach: use a `.env` file (or `hf_token.env`)**

This is the simplest, most reproducible way — especially in agent/tool sandboxes (like this one) where plain `export` often doesn't propagate to command executions.

```bash
# One-time setup (in titan-setup root)
cp hf_token.env.example .env
# edit .env and put your real approved token
chmod 600 .env
```

Then download using the provided helper (recommended):

```bash
cd titan-setup
./download_8b_tokenizer.sh
```

The helper automatically loads `HF_TOKEN` from the environment **or** from `.env` / `hf_token.env` etc. in the project root. It then calls the torchtitan downloader with the correct arguments and verifies the output file.

**Alternative (manual, no helper):**

```bash
cd titan-setup/torchtitan

# Load from .env if present (works for both normal shells and agent tools)
if [[ -f ../.env ]]; then
  HF_TOKEN=$(grep '^HF_TOKEN=' ../.env | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")
fi

python scripts/download_tokenizer.py \
  --repo_id meta-llama/Meta-Llama-3.1-8B \
  --hf_token "$HF_TOKEN" \
  --local_dir assets/tokenizer

ls -l assets/tokenizer/original/tokenizer.model   # verify
```

The `.env` file is gitignored (see `.gitignore`) and the helper + instructions above work whether you use `export HF_TOKEN=...` (normal interactive shells) or a file (agent/tool environments).

   If the download fails with 403, your account is not yet approved for the gated model.

2. Run the 8B experiment (example using the updated launcher; NGPU=1 or NGPU=2 recommended for initial validation to avoid potential NCCL issues on 4-GPU setups; use 4 when ready and the limitation is resolved):

Once the tokenizer is successfully downloaded, the training run itself does **not** need the HF token (it only needs the local file).

```bash
cd titan-setup

NGPU=1 \
CONFIG_FILE=configs/llama3_8b_2gpu.toml \
./run_experiment.sh \
  --job.dump_folder outputs/llama3-8b-run \
  --training.steps 50 \
  --metrics.log_freq 5   # use a small number for a short validation run
```

NGPU=2 is also a good choice for the 8B model as it exercises actual FSDP sharding while staying within the configurations that are currently reliable on this pod.

   The explicit `--job.dump_folder` ensures artifacts land in the documented `outputs/llama3-8b-run/` (the launcher would otherwise derive a name from the config filename).

This uses the real `c4` dataset and sequence length 8192.

See `docs/8b-attempt.md` for background on the previous blocked attempt and how to resume once approved.

## Viewing Results

Results are written to the `dump_folder` specified in the active config (commonly `outputs/infrastructure-run/` or `outputs/llama3-8b-run/`).

Useful commands:

- TensorBoard (debug/infra): `tensorboard --logdir outputs/infrastructure-run/tb --port 6006`
- TensorBoard (8B): `tensorboard --logdir outputs/llama3-8b-run/tb --port 6006`
- List recent logs / artifacts: `ls -l outputs/`
- Full logs are also emitted to stdout/stderr by the launcher.
