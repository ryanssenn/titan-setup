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

Artifacts, logs, and generated graphs are available under `outputs/`.

## Run

```bash
./run_experiment.sh
```

See `docs/reproduce.md` for full reproduction instructions.

## Sources

- TorchTitan v0.1.0 
- Compute: 4× NVIDIA H100 80GB HBM3
- Graphs generated from logs using matplotlib
- Analysis prepared with Grok Build
