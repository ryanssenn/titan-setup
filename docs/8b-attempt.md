# Llama 3.1 8B Run - Pending Work

## Current Status (latest attempt)

**Auth / token handling issues are now resolved.**

- Created `.env` (or `hf_token.env`) + `hf_token.env.example` (gitignored).
- Added `download_8b_tokenizer.sh` helper in repo root. It automatically loads `HF_TOKEN` from the environment **or** from the `.env` file in the titan-setup root. This makes the flow reliable even in agent/tool sandboxes (e.g. Grok Build) where plain `export` often does not propagate to command executions.
- Updated `README.md` (minimal) and `docs/reproduce.md` with clear instructions and sandbox notes.
- Improved error messages in the (local) vendored download script and added early length checks + verification steps in docs.
- `.env` approach + helper now allows successful tokenizer download for the gated `meta-llama/Meta-Llama-3.1-8B`.

**We are currently stuck on NCCL / distributed init for the 4-GPU training run.**

Latest attempt (after successful download):
- Tokenizer download via `./download_8b_tokenizer.sh` (using `.env`) succeeded and produced `torchtitan/assets/tokenizer/original/tokenizer.model`.
- Training launch with `NGPU=4 CONFIG_FILE=configs/llama3_8b_2gpu.toml ./run_experiment.sh --job.dump_folder outputs/llama3-8b-run --training.steps 50 ...` failed immediately in `Trainer` init during `set_determinism` → `torch.distributed.broadcast`.
- Error: `ncclUnhandledCudaError: Call to CUDA function failed. Last error: Cuda failure 401 'the operation cannot be performed in the present state'`.
- This matches the pre-existing caveat in `docs/reproduce.md` ("On 4 GPUs you may encounter NCCL initialization errors...").

The download path is now reproducible. The 4-GPU NCCL issue on this H100 setup remains (see historical notes in reproduce.md; common workaround is `NGPU=1` for initial validation or the old `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` flags).

See the updated `NOTE-ATTEMPT.txt` in `outputs/llama3-8b-run/` and the reproduction logs for full details.

## How to Resume

1. Use the `.env` + helper for the tokenizer (see `docs/reproduce.md`).
2. For a first successful end-to-end validation now that the tokenizer works, try `NGPU=1` (as recommended in docs for quick 8B validation).
3. Once basic 8B training succeeds on 1 GPU, investigate NCCL flags or other env tweaks for 4-GPU.
4. Full 500-step run can follow.

## Status (historical summary)

The Llama 3.1 8B experiment was attempted (multiple times, including on 4× H100) but could not complete because the required tokenizer is gated under the `meta-llama` organization on Hugging Face.

**Approval has now been obtained by the user.** The gated tokenizer can (and should) be downloaded and the 8B training run executed (auth handling is fixed; see Current Status above).

The top-level `README.md` now has a dedicated "Llama 3.1 8B Run (Real Model)" section that explicitly states:
- The requirement for **explicit Meta approval** (not just any HF token).
- How to provide the token **securely as an environment variable** (`export HF_TOKEN=...`, one-liner `HF_TOKEN=...`, and `! export` usage) **or via `.env` file** (recommended for agent/tool environments).

Detailed commands, the verification `ls` step after download, and NGPU recommendations are in `docs/reproduce.md`.

Reproduction instructions (and this file) were significantly improved to eliminate common pitfalls (wrong script path, confusing token errors, output directory mismatches, etc.).

## Previous Blocked Attempts + Latest Status

Previous attempts (before approval) failed at the tokenizer download / load step with 403 or path-related errors (detailed below for history).

**As of now (user has obtained Meta approval):** The 8B run can be executed. The main README now clearly documents the approval requirement and `export HF_TOKEN=...` method.

## Reproduction Attempt History (pre-approval, 2026-06-12 on current workspace)

Following the (then-current) `docs/reproduce.md` exactly surfaced several documentation issues (now fixed):
- Wrong script path in the download command (`torchtitan/scripts/...` instead of `scripts/...` when cwd is the torchtitan root) → "can't open file" error.
- Passing empty `--hf_token` (from unset $HF_TOKEN) produced a low-level httpx "Illegal header value b'Bearer '" instead of a helpful message.
- Repo ID inconsistency (Llama-3.1-8B vs Meta-Llama-3.1-8B).
- Dump folder naming: launcher derives `llama3_8b_2gpu` from config filename unless `--job.dump_folder` is passed explicitly. Instructions and examples now use explicit folder + NGPU=1 recommendation for quick validation.
- Added verification step (`ls ...tokenizer.model`) and clearer gated 403 guidance.

A fresh attempt with the (pre-fix) instructions + NGPU=1 + 5 steps produced:
- Clear early warning: "Tokenizer path ./assets/tokenizer/original/tokenizer.model does not exist!"
- Clean failure: `AssertionError` in `tiktoken.py` (as designed).
- Launcher created `outputs/llama3_8b_2gpu/` stub (with comm_trace).
- No training steps, no model build, no dataset work.

The fixes in `docs/reproduce.md`, `configs/llama3_8b_2gpu.toml`, and a small robustness improvement in the vendored download script should allow a smooth run the moment a properly approved HF token is available.

## What Happened During the Attempt(s)

- Tokenizer download attempted (with and without token) for `meta-llama/Meta-Llama-3.1-8B`.
- Without approved access: 403 (or bad header errors before script improvements).
- Training launcher invoked with `configs/llama3_8b_2gpu.toml` (and explicit dump folder in later tries).
- Failure always immediate at tokenizer load: `AssertionError: The tokenizer path does not exist: ./assets/tokenizer/original/tokenizer.model` (from `tiktoken.py` after a helpful warning in the job startup).

No training steps were executed, no checkpoints were written, and no metrics were collected for the 8B configuration in any attempt. The real C4 dataset was never reached.

## Artifacts from the Attempt(s)

- `outputs/llama3-8b-run/` (or `llama3_8b_2gpu/` in runs without explicit --job.dump_folder) — Stub directory with `comm_trace/`.
- A `NOTE-ATTEMPT.txt` (or similar) can be added with the failure transcript.
- Full failure output is captured in the reproduction session logs / terminal (the exact AssertionError + traceback is stable and documented in the updated reproduction steps).
- Historical attempt log was removed during repo cleanup for the H100 port.

## How to Resume (Updated Instructions)

1. Obtain a Hugging Face token from an account that has been **granted access** to `meta-llama/Meta-Llama-3.1-8B` (approval from Meta is required; not all HF accounts have it).
2. Follow the (now improved) reproduction steps in `docs/reproduce.md` (Llama 3.1 8B section). Key improvements:
   - Correct `python scripts/...` path.
   - `export HF_TOKEN=...` + verification `ls`.
   - Explicit `--job.dump_folder outputs/llama3-8b-run`.
   - NGPU=1 recommended for initial quick validation runs.
   - Better error messages for 403 gated case.
3. Re-run the tokenizer download (must succeed with the .model file) and then the training command.
4. For a short validation: use `--training.steps 50` (or smaller). Full 500-step run can follow once basic stability is confirmed.

The C4 dataset itself (`allenai/c4`) is public and does not require authentication. The only blocker is the gated Llama 3.1 tokenizer + Meta approval.

## Related Files

- `docs/reproduce.md` (primary instructions; updated with fixes from this reproduction attempt)
- `configs/llama3_8b_2gpu.toml` (updated comments)
- `outputs/llama3-8b-run/` (target location when using the explicit dump_folder)
- Main README.md and `docs/details.md` (high-level + plan status)
- This file (`docs/8b-attempt.md`) for history of blockers.
