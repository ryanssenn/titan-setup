# Llama 8B Checkpoint Save Failure

Documentation for the distributed checkpoint (DCP) error observed when training Llama 3.1 8B with default checkpointing enabled.

## Summary

| Setting | debugmodel (~6M) | llama3_8b (8B) |
|---------|------------------|----------------|
| Checkpoint save at step 1 | Works | `RuntimeError: unexpected pos 448 vs 342` |
| Training steps | Works | Works |
| Workaround | None needed | `--checkpoint.no-enable-checkpoint` |

## Symptom

With default config (`enable_checkpoint = true` in `configs/llama3_8b.toml`), training starts and completes step 1, then fails during the first checkpoint save:

```
torch.distributed.checkpoint.api.CheckpointException: CheckpointException ranks:dict_keys([0, 1])
RuntimeError: [enforce fail at inline_container.cc:626] . unexpected pos 448 vs 342
```

Observed on 2-GPU FSDP run (2026-06-19). Failure occurs in `torch.distributed.checkpoint` during `torch.save` inside the filesystem storage writer.

## What works

- Model build, FSDP sharding, C4 data loading, and training steps all succeed.
- Checkpointing works for `debugmodel.toml` (small model) on 1, 2, and 4 GPUs.
- Llama 8B training completes when checkpointing is disabled:

```bash
NGPU=2 CONFIG_FILE=configs/llama3_8b.toml ./run.sh \
  --training.steps 50 \
  --checkpoint.no-enable-checkpoint
```

## Configuration

Disable checkpointing via CLI override:

```bash
--checkpoint.no-enable-checkpoint
```

TorchTitan uses `--checkpoint.no-enable-checkpoint` (not `--checkpoint.enable_checkpoint false`).

## Environment

| Component | Value |
|-----------|-------|
| Model | Llama 3.1 8B, FSDP2, seq_len 8192 |
| PyTorch | 2.6.0+cu124 |
| TorchTitan | v0.1.0 |
| Checkpoint format | DCP (distributed checkpoint), `export_dtype = float32` |
| GPUs tested | 2× H100 (failure), training validated with checkpoints disabled |

## Notes

- GPU memory during 8B training reaches ~78 GiB per GPU on 2-GPU FSDP (~99% of H100 80GB), with CUDA memory allocation retry warnings. 4-GPU FSDP uses ~65 GiB per GPU (~82%).
- TensorBoard logging is unaffected; metrics are written to `outputs/llama3_8b/tb/` regardless of checkpoint setting.
- For short validation runs, disabling checkpoints is sufficient. Long runs requiring checkpoint resume need further investigation of DCP save behavior at 8B scale.
- 4-GPU FSDP training (50 steps) completes successfully with checkpoints disabled. Per-GPU memory ~65 GiB (82% of H100 80GB), lower than 2-GPU (~78 GiB).
