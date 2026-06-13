# titan-exp - Full Details

This file contains the detailed environment, configuration, and results tables that were removed from the main README for readability.

## Compute Environment

| Component | Specification |
|-----------|---------------|
| Provider | RunPod |
| GPUs | 4× NVIDIA H100 80GB HBM3 |
| Driver | 570.195.03 |
| RAM | 100 GB |
| vCPUs | 18 |
| PyTorch | 2.6.0+cu124 |
| TorchTitan | v0.1.0 (pinned checkout at `torchtitan/`) |

**Note:** Newer TorchTitan (main) requires PyTorch ≥2.8 and CUDA 13, which is incompatible with this pod's driver and CUDA version. v0.1.0 was used because it supports PyTorch 2.6 + cu124.

## NCCL / Multi-GPU Notes (Current Pod)

On some RunPod pods (including earlier instances used for this project), multi-GPU NCCL communication required disabling peer-to-peer and InfiniBand:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

These flags are **not** set automatically by `run_experiment.sh`.

**Observed behavior on the current 4× H100 pod (as of 2026-06-13):**
- NGPU=1: Works (no distributed setup required).
- NGPU=2: Works reliably and exercises real FSDP2 (`data_parallel_shard_degree=-1`).
- NGPU=4: Fails early during process group initialization / `set_determinism` (inside `torch.distributed.broadcast`) with `ncclUnhandledCudaError: Call to CUDA function failed. Last error: Cuda failure 401 'the operation cannot be performed in the present state'`.

Prefixing the NCCL disables (as shown in older reproduction steps) has not resolved the 4-GPU failure on this pod/container. The issue may be transient to the specific pod, driver state, or CUDA/NCCL runtime. If 4-GPU operation is required, a pod restart or different instance is currently the most practical path.

The debug model validation can still be performed end-to-end with NGPU=1 or NGPU=2.

## Experiment Plan

| Phase | Config | Steps | Status |
|-------|--------|-------|--------|
| 1. Setup | TorchTitan v0.1.0 + PyTorch 2.6 cu124 | — | ✅ Done |
| 2. Smoke test | `debug_model.toml` (tiny debug model) | 10 | ✅ Done |
| 3. Infrastructure run | `configs/infrastructure_run.toml` | 500 | ✅ Done |
| 4. Llama 3.1 8B run | `configs/llama3_8b_2gpu.toml` (real C4, seq_len=8192) | 500 | ⏸ Pending (awaiting approval from Meta Llama) |

The debug "llama3 debugmodel" (6.27M parameters: dim=256, 6 layers) together with the small `c4_test` dataset is sufficient to validate the full distributed training pipeline (FSDP2, metrics, checkpointing, stability) without requiring any gated Hugging Face assets.

**Note on GPU count for reproduction:** On the current pod, phases 2 and 3 (debug model smoke + infrastructure validation) can be reproduced end-to-end using NGPU=1 or NGPU=2. The 4-GPU configuration in the plan was achieved on compatible hardware/pods in the past.

## Standard Stack (No Extras)

- FSDP2 (`data_parallel_shard_degree=-1` — actual degree matches `NGPU` at runtime; e.g. 2-way sharding when `NGPU=2`)
- Selective activation checkpointing (`selective_ac_option="2"`)
- bf16 mixed precision (handled inside `fully_shard`)
- Fused AdamW optimizer
- No torch.compile, no Float8, no Tensor Parallel, Pipeline Parallel, or Context Parallel

On the current pod, reliable end-to-end runs (including FSDP) have been confirmed with NGPU=1 and NGPU=2 for the debug model. 4-GPU FSDP currently hits the NCCL initialization error described above.

## Detailed Results Tables

### Smoke Test (10 steps)

Completed cleanly in ~13 seconds on NGPU=2 (or equivalent short runs). Loss decreased monotonically. Real FSDP sharding was active when run with NGPU=2.

| Metric | Step 1 | Step 10 |
|--------|--------|---------|
| Loss | 8.2116 | 7.1138 |
| Memory (reserved) | 1.20 GiB | 1.29 GiB |
| Throughput | 15,244 tps | 248,688 tps |
| MFU | 0.35% | 5.73% |

### Infrastructure Validation Run (500 steps)

**Config:** `configs/infrastructure_run.toml`  
**Log:** `outputs/infrastructure-run.log`  
**Wall time:** ~38 seconds (from a previously successful run)

| Metric | Step 1 | Step 250 | Step 500 |
|--------|--------|----------|----------|
| Loss | 8.2066 | 4.3399 | 3.6827 |
| Memory (reserved) | 1.20 GiB | 1.29 GiB | 1.29 GiB |
| Throughput (global) | 15,408 tps | ~250k tps | 232,910 tps |
| MFU (A100 reference) | 0.36% | ~6% | 5.37% |

**Validation checks passed (from the run that produced the committed artifacts):**
- 500 steps completed with no OOM
- Loss trended downward consistently (8.21 → 3.68)
- FSDP2 sharding, checkpointing (DCP), and TensorBoard logging were functional
- Checkpoints saved at steps 1, 250, and 500

**Notes on results:**
- The numbers above come from a prior successful validation run. On the current pod, full 500-step reproduction is straightforward with NGPU=1 or NGPU=2; 4-GPU reproduction is blocked by the NCCL init error described in the NCCL section.
- MFU numbers in the historical table use A100 peak FLOPS as a fallback (the original hardware for some runs was A40-class). Absolute efficiency numbers are therefore not calibrated for the current H100s.
- The `c4_test` dataset contains only ~2K samples and re-loops approximately every 40 steps. This is expected behavior for this validation dataset.
- All metrics were logged at `log_freq=10` (51 data points total).

## Artifacts Location

```
outputs/
├── smoke-test/                 # 10-step smoke test artifacts + log
├── infrastructure-run/
│   ├── checkpoint/             # step-1/, step-250/, step-500/ (DCP; shard count depends on NGPU used)
│   ├── tb/                     # TensorBoard event files
│   ├── graphs/                 # Generated matplotlib plots + summary
│   └── infrastructure-run.log  # Full structured log
├── llama3-8b-attempt.log       # Sanitized log from the 8B attempt
└── llama3-8b-run/              # Stub directory + note (no training occurred)
```

Large binary checkpoint files (`.distcp`) are intentionally excluded from git via `.gitignore`.

## Sources

- TorchTitan v0.1.0: https://github.com/pytorch/torchtitan
- 4× NVIDIA H100 80GB HBM3 (1-GPU and 2-GPU debug model runs confirmed working on current pod; 4-GPU blocked by NCCL init)
- Training metrics and logs generated by TorchTitan
- Graphs created from logs using Python + matplotlib (assisted by Grok Build)
