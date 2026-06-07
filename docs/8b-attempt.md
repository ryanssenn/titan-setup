# Llama 3.1 8B Run - Pending Work

## Status

The Llama 3.1 8B experiment was attempted but could not complete because the required tokenizer is gated under the `meta-llama` organization on Hugging Face.

**We are awaiting approval from Meta Llama** for the account associated with the provided HF token.

## What Happened During the Attempt

- The tokenizer download was attempted using the supplied token for `meta-llama/Meta-Llama-3.1-8B` (and the `Llama-3.1-8B` variant).
- Hugging Face returned a 403 GatedRepoError: the account is not (yet) in the authorized list for the repository.
- The training launcher was still invoked with `configs/llama3_8b_2gpu.toml` to document the exact failure point.
- The run failed immediately when TorchTitan tried to load the tokenizer (file `assets/tokenizer/original/tokenizer.model` did not exist).

No training steps were executed, no checkpoints were written, and no metrics were collected for the 8B configuration.

## Artifacts from the Attempt

- `outputs/llama3-8b-attempt.log` — Sanitized full transcript of the torchrun failure.
- `outputs/llama3-8b-run/` — Stub directory created by the launcher (contains only comm_trace and a note file).

## How to Resume

1. Obtain a Hugging Face token from an account that has been granted access to `meta-llama/Meta-Llama-3.1-8B`.
2. Follow the reproduction steps in `docs/reproduce.md` (Llama 3.1 8B section).
3. Re-run the tokenizer download and then the training command.

The C4 dataset itself (`allenai/c4`) is public and does not require authentication. The only blocker is the gated Llama 3.1 tokenizer.

## Related Files

- `outputs/llama3-8b-attempt.log`
- `outputs/llama3-8b-run/NOTE-ATTEMPT.txt`
- Main README.md (high-level status only)
- `docs/details.md` (full environment and plan context)
