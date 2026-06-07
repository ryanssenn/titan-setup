# Reproduction Instructions

## Validated Runs (No Hugging Face Token Required)

These commands reproduce the successful smoke test and 500-step infrastructure validation. They use the small debug model and `c4_test` dataset.

```bash
cd /workspace/titan-exp

# Smoke test (10 steps, quick sanity check)
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NGPU=2 \
  CONFIG_FILE=torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml \
  ./run_experiment.sh \
  --job.dump_folder outputs/smoke-test

# Full infrastructure validation (500 steps)
./run_experiment.sh
```

The launcher (`run_experiment.sh`) automatically sets the required NCCL environment variables for this pod.

Results will appear under `outputs/`.

## Llama 3.1 8B Run (Pending Meta Llama Approval)

Once a Hugging Face token with access to the gated model is available:

1. Download the tokenizer (one time):

```bash
cd torchtitan
python scripts/download_tokenizer.py \
  --repo_id meta-llama/Meta-Llama-3.1-8B \
  --hf_token "$HF_TOKEN" \
  --local_dir assets/tokenizer
```

2. Run the 8B experiment:

```bash
cd /workspace/titan-exp
CONFIG_FILE=configs/llama3_8b_2gpu.toml ./run_experiment.sh
```

This will use the real `c4` dataset, sequence length 8192, and the full 8B Llama 3.1 model.

See `docs/8b-attempt.md` for details on the previous blocked attempt.

## Viewing Results

- TensorBoard: `tensorboard --logdir outputs/infrastructure-run/tb --port 6006`
- Graphs: `outputs/infrastructure-run/graphs/`
- Full logs: `outputs/infrastructure-run.log` and `outputs/smoke-test.log`
