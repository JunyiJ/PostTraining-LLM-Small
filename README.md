# PostTraining-LLM-Small

Lightweight GRPO + LoRA post-training experiments on a local Gemma 2B Instruct checkpoint.

## Repo structure
- `run_train.py` — main GRPO+LoRA training loop (multi-epoch, sampling, ratio loss, periodic eval, checkpoints in `checkpoints/`).
- `eval_math.py` — evaluate math accuracy; can load base model or base+LoRA checkpoint.
- `eval_math_base.py` — baseline math eval without LoRA.
- `grpo/` — helpers:
  - `sampler.py` — sampling with optional logprobs.
  - `advantage.py` — mean-centered advantages.
  - `reward.py` — numeric reward extractor.
  - `lora.py` — LoRA wrappers and utils.
  - `*_test.py` — small unit tests for helpers.
- `data/` — e.g., `math_grpo_200.jsonl`, `test_math.jsonl`.
- `models/` — local model download (e.g., Gemma 2B Instruct).
- `logs/` — training logs.
- `checkpoints/` — saved LoRA checkpoints.

## Setup
1) Create/activate env (example):
   ```bash
   conda create -n grpo-lora python=3.10 -y
   conda activate grpo-lora
   pip install torch transformers tqdm
   ```
   (or `conda env update -f environment.yml` if you maintain it).
2) Download model locally, e.g.:
   ```bash
   huggingface-cli download google/gemma-2-2b --local-dir ./models/gemma-2-2b --include "*"
   ```
3) (Optional) Install pytest for helper tests:
   ```bash
   pip install pytest
   ```

## GRPO + LoRA flow
- LoRA: wrap target linear layers (`q_proj`/`v_proj`) via `apply_lora_to_model`, freeze base weights, optimize only LoRA params.
- Sampling: `sample_k` generates K answers per prompt with the current LoRA policy, storing tokens and old logprobs.
- Rewards: `compute_reward` extracts the last numeric answer and compares to gold (binary reward).
- Advantages: mean-center rewards per prompt via `compute_advantage`.
- Loss: recompute new logprobs under the current model, form ratio `exp(new - old)`, and optimize `-(adv * ratio).mean()`. Periodic eval and checkpointing are built in.
- Checkpoints: saved under `checkpoints/` as `lora_epoch{N}_step{S}.pt` (contains LoRA weights + optimizer state).
- Eval: `eval_math.py` can load base or LoRA checkpoint (set `USE_LORA` and `LORA_CKPT`).

To train:
```bash
python run_train.py | tee logs/train.log
```
Adjust `NUM_EPOCHS`, `NUM_TRAINING_DATA`, `NUM_SAMPLES_PER_PROMPT`, etc. in `run_train.py`.

## TODO
- Refine reward function (format checks, partial credit, reasoning-based reward, numeric robustness).
- Add PPO-style clipping/advantage normalization for stability.
- Add proper batching and gradient accumulation.
- Expand eval beyond math and add more unit tests.

## Baseline Performance
Model: Gemma 2B Instruct Total: 200 Correct: 74 Accuracy: 37.00%
