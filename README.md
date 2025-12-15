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
   `conda env update -f environment.yml`.
2) Download model locally, e.g.:
   ```bash
   huggingface-cli download google/gemma-2-2b --local-dir ./models/gemma-2-2b --include "*"
   ```
3) I used local machine (mac mini, device=`mps`) to do the post-training, but the code should
 be able to be adapted to other devices.

## GRPO + LoRA flow
### Code structure
- LoRA: wrap target linear layers (`q_proj`/`v_proj`) via `apply_lora_to_model`, freeze base weights, optimize only LoRA params.
- Sampling: `sample_k` generates K answers per prompt with the current LoRA policy, storing tokens and old logprobs.
- Rewards: `compute_reward` extracts the last numeric answer and compares to gold (binary reward).
- Advantages: mean-center rewards per prompt via `compute_advantage`.
- Loss: recompute new logprobs under the current model, form ratio `exp(new - old)`, and optimize `-(adv * ratio).mean()`. Periodic eval and checkpointing are built in.
- Checkpoints: saved under `checkpoints/` as `lora_epoch{N}_step{S}.pt` (contains LoRA weights + optimizer state).
- Eval: `eval_math.py` can load base or LoRA checkpoint (set `USE_LORA` and `LORA_CKPT`).

To train:
```bash
python run_train.py | tee logs/train_test.log
```
Adjust `NUM_EPOCHS`, `NUM_TRAINING_DATA`, `NUM_SAMPLES_PER_PROMPT`, etc. in `run_train.py`.

### Overview of LoRA
For each target layer(usually q, k, v, o), add a learnable weight with low rank while base weights are frozen:
output = original_output + alpha/low_rank * B(A(x))
where A with dimension input_dim * low_rank, B with dimension low_rank * output_dim.
A is usually initialized with normal distribution and B is initialized as 0 so that initially the delta is 0 
and model learns from update.

### Overview of GRPO
loss = - advantage * (prob_new / prob_old) + KL_weight * KL_divergency
advantage = (reward - mean(reward)) / (std(reward) + 0.00001) for a group of answers (e.g. sample k answers)
KL_divergency ~= sum(log_prob_new / log_prob_old) per token and then take the mean of all samples

## Performance Comparison
### Gemma 2B Instruct as base model
* Baseline Model: Gemma 2B Instruct Total: 200 Correct: 74 Accuracy: 37.00%
* GRPO + LORA Model checkpoint (base): Gemma 2B Instruct + LoRA with GRPO loss Total: 200 Correct: 126 Accuracy: 63.00% (before running optimization)
* GRPO + LORA Model checkpoint(efficient): Gemma 2B Instruct + LoRA with GRPO loss with improved efficiency. Total: 199 Correct: 118 Accuracy: 59.3%.

### Qwen2.5-Math-1.5B-Instruct as base model
* Baseline Model: Total: 200 Correct: 16 Accuracy: 8.00%

## Interesting Learnings
### Reward definition is key to the quality
Reward is probably the most critical part for the RL reasoning training for LLM. Soley relying 
on correctness of the final answer is not enough mainly because
1) It doesn't encourage reasoning behavior and
2) The reward is relatively sparse and not distinguishable among different answers.

Instead, we need to take other things into consideration such as
1) Format checking (e.g. having an "answer" token in reward.)
2) Encourage reasoning
3) Numeric robustness.
...

In this project, the quality breakthrough is through better reward definition.

### Reward hacking and volatility of training loss/accuracy
Both the base model, the dataset and the post-training parameter are small for this project.
I do sometimes the training loss/accuracy jump back and forth. Interestingly, I also observed 
reward hacking sometimes during the RL training process (e.g. got `Answer:\nAnswer:\nAnswer\n...` 
for example question).

To deal with these issues,
1) I tried to update the reward function to punish reward hacking (negative score when there are repetitive patterns).
2) In GRPO advantage calculation, instead of using `reward - mean(reward)`, use 
`(reward - mean(reward))/std(reward)` to reduce the gradient variance.
3) Add KL divergency term into the original GRPO formula: `-advantage * P(new)/P(old) + alpha * logP(new) / logP(old)`

### Post training efficiency
I tried a few things to speed up the training on MPS
1) Update sample_k logic to avoid looping over k samples, but instead batching the k samples. 
Ideally we could use the `model.generate` call to avoid iterating with tokens as well, however, it 
causes NaN issue probably due to a known unstability of Gemma model on MPS.
2) Update the second pass to a batch mode instead of looping over the k samples.
3) Using shorter max_token.
4) Early stop if all sampling reached EOF.

After the optimization step above (combining, 1/2/3/4), the overall training time is able to reduce 
by 3-4x.
3) TODO: update to larger batches.

