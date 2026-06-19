# NCCL: 3+ GPU Broadcast Behavior

Factual notes on the `torch.distributed.broadcast` behavior observed during setup, and the `NCCL_NVLS_ENABLE` configuration used in this project.

## Summary

| Setting | 1–2 GPUs | 3+ GPUs |
|---------|----------|---------|
| Default NCCL | Works | `Cuda failure 401` during `broadcast` |
| `NCCL_NVLS_ENABLE=0` | Works | Works |

`run.sh` sets `NCCL_NVLS_ENABLE=0` by default.

## Symptom

With default NCCL settings and 3+ GPUs:

```
torch.distributed.DistBackendError: NCCL error ... ncclUnhandledCudaError
Cuda failure 401 'the operation cannot be performed in the present state'
```

Occurs on the first collective operation (`torch.distributed.broadcast`), including in TorchTitan's `set_determinism` during trainer initialization.

## Cause

NCCL selects **NVLS** (NVLink Sharp / multicast) for 3+ GPU collectives. NCCL debug logs show:

```
NVLS multicast support is available on dev N
```

With default settings, the NVLS code path returns CUDA error 401 in the test environment documented below.

With 2 GPUs, NCCL uses point-to-point copies and does not hit this path.

## Configuration

```bash
export NCCL_NVLS_ENABLE=0
```

`NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` address different communication paths and do not change this behavior.

## Reproducer

```bash
source .venv/bin/activate

# Default NCCL (3+ GPUs may return CUDA 401):
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 repro_broadcast_4gpu.py

# With NVLS disabled:
NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 repro_broadcast_4gpu.py
```

## Test environment

| Component | Value |
|-----------|-------|
| GPUs | 4× NVIDIA H100 80GB HBM3 |
| Interconnect | NVLink (NV18) |
| Driver | 580.126.09 |
| Host CUDA | 13.0 |
| PyTorch | 2.6.0+cu124 |
| NCCL | 2.21.5 |
