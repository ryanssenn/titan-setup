# titan-exp

TorchTitan 2-GPU FSDP infrastructure validation on A40.

**Status:** Core stack validated. Llama 3.1 8B awaiting approval from Meta Llama.

Prepared with **Grok Build**.

## Results

**Loss curve**

![Loss curve](outputs/infrastructure-run/graphs/01_loss_curve.png)

**Metrics dashboard**

![Dashboard](outputs/infrastructure-run/graphs/00_metrics_dashboard.png)

500-step run: loss dropped from 8.2 to 3.7 with steady throughput and stable memory. Logs and graphs in `outputs/infrastructure-run/`.

## Reproduce

```bash
./run_experiment.sh
```

See `docs/reproduce.md` for smoke test, 8B (pending Meta Llama approval), and TensorBoard.

## Sources

- TorchTitan v0.1.0 (https://github.com/pytorch/torchtitan)
- Compute: RunPod 2× NVIDIA A40
- Raw metrics and logs produced by TorchTitan
- Graphs generated from logs using matplotlib
- Analysis and report prepared with **Grok Build**

See `docs/details.md` for full environment, metrics tables, and experiment details.