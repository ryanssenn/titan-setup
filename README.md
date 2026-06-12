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

The default command runs the **debug model validation** (no tokens or gated access required):

```bash
./run_experiment.sh
```

See `docs/reproduce.md` for full reproduction instructions.

### Llama 3.1 8B

To run the full Llama 3.1 8B model you need approval from Meta Llama for the gated repository.

Create a `.env` file (gitignored) with your token:

```bash
echo 'HF_TOKEN=your_approved_hf_token_here' > .env
chmod 600 .env
```

See `docs/reproduce.md` for the exact download + launch commands. The `.env` is automatically used when present.

```bash
CONFIG_FILE=configs/llama3_8b_2gpu.toml ./run_experiment.sh
```

## Sources

- TorchTitan v0.1.0 
- Compute: 4× NVIDIA H100 80GB HBM3
- Graphs generated from logs using matplotlib
- Analysis prepared with Grok Build
