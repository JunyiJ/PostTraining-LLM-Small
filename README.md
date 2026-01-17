# PostTraining-LLM-Small

Lightweight GRPO + LoRA post-training experiments on a local Gemma 2B Instruct checkpoint.

## Repo structure
- `run_train_grpo.py` — main GRPO+LoRA training loop (multi-epoch, sampling, ratio loss, periodic eval, checkpoints in `checkpoints/`).
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
- `ppo/` — PPO utilities: sampler, advantage, reward helpers, LoRA critic wrapper.

## Setup
1) Create/activate env (example):
   `conda env update -f environment.yml`.
2) Download model locally, e.g.:
   ```bash
   huggingface-cli download google/gemma-2-2b --local-dir ./models/gemma-2-2b --include "*"
   ```
3) I used a local Mac mini for post-training, but device selection now auto-detects
   `cuda`/`mps`/`cpu` based on what's available.

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
python run_train_grpo.py | tee logs/train_test.log
```
Adjust `NUM_EPOCHS`, `NUM_TRAINING_DATA`, `NUM_SAMPLES_PER_PROMPT`, etc. in `run_train_grpo.py`.

### Overview of LoRA
For each target layer(usually q, k, v, o), add a learnable weight with low rank while base weights are frozen:
output = original_output + alpha/low_rank * B(A(x))
where A with dimension input_dim * low_rank, B with dimension low_rank * output_dim.
A is usually initialized with normal distribution and B is initialized as 0 so that initially the delta is 0 
and model learns from update.

### Overview of KL divergency
The KL (Kullback-Leibler) divergence term in RLHF (Reinforcement Learning from Human Feedback) is primarily designed to prevent reward hacking (by forcing the model to remain similar to the original model), maintaining diversity(avoid model collapse into a few high-score answers) and stability.

In model post-training, KL divergence is a penalty added to the reward.
`R' = R - beta * KL($$\pi_{policy} \| \pi_{ref})$$)`
where the reference model is almost always the frozen copy of the base model of SFT model.

There are different ways to estimate KL divergence $D_{KL}(q \| p)$ and the most commonly used method is the Shulman estimation($k_{3}$) in RLHF.

* Let $r = \frac{p(x)}{q(x)}$

* $k_{1}$: The Native Estimator (unbiased, high variance, can <0): $$k_1 = -\log r = \log q(x) - \log p(x)$$
* $k_{2}$: The Squared Log-Ratio (biased, always >=0, low variance. It's stable when 2 distributions are close): $$k_2 = \frac{1}{2}(\log r)^2$$
* $k_{3}$: Schulman Estimator(unbiased, always >=0, moderate variance): $$k_3 = r - 1 - \log r$$


### Overview of GRPO

$$L_{GRPO}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \left( \sum_{t=1}^{T} \min \left( r_{i,t}(\theta) \hat{A}_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right)$$

And a simplified version is `loss = - advantage * (prob_new / prob_old) + KL_weight * KL_divergency`

where
* `advantage = (reward - mean(reward)) / (std(reward) + 0.00001)` for a group of answers (e.g. sample k answers)
* KL_divergency is estimated by average over all effective tokens/sample and then average overall samples.


#### Implementation details
* Only take answer tokens (excluding prompt) into account - Basically one need to implement the answer_mask to mask out non-answer tokens for followup calculations.
* Pay attention to the logit shift for logprobs and target tokens: input_token_0 -> produces logits_0 -> predicts target_token_1. In the shift, one usually need to discard the last logits because it's predict a target token that is meaningless and shift target_token by 1 like target = target[, 1:] to align the target token at index 1 with logits at index 0.

### Overview of PPO
Unlike GRPO, PPO is an actor–critic method: it trains both a policy (actor) and a value head (critic).

- **Total loss** (multi-part):  
  ```
  L_total = L_CLIP + c1 * L_VF + c2 * L_ENT
  ```
  where `L_CLIP` is the clipped policy loss, `L_VF` the value loss, and `L_ENT` an optional entropy bonus.

- **Actor / policy loss** (clipped surrogate):

  The Actor maximizes the Clipped Surrogate Objective to ensure stable training:
  ```
  r_t = exp(log P_new - log P_old)
  PolicyLoss = -min(r_t * Â_t, clip(r_t, 1-ε, 1+ε) * Â_t)
  ```
  `Â_t` is the advantage for token/time step `t` and it represents how much better an action is compared to state baseline.

- **Critic / value loss**:  
  The Critic is typically a Value Head—a linear layer added to the final hidden state of the base model that outputs a single scalar value predicts `V_s`. A common target is `R_target = V_old + Â_t`, and value loss is MSE (often with clipping) between `V_new` and `R_target`.

- **Generalized Advantage Estimation (GAE)**:  
  ```
  δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
  Â_t = δ_t + (γλ) δ_{t+1} + ... + (γλ)^{T-t+1} δ_{T-1}
  ```
  Use `V_old` from data collection to reflect surprise under the generating policy. GAE is basically the TD lambda implementation based on token level rewards (see below) and value estimation.

- **Token-level rewards & KL penalty** (LLM setting):  
  ```
  R_t = HeuristicReward - β * D_KL
  D_KL ≈ log P_new - log P_ref
  ```
  Typically every token gets the KL penalty; only the final token gets the task reward (e.g., correctness/helpfulness) to handle sparse rewards and keep the policy near a reference model. A reference model is used for KL divergency: This prevents the model from drifting too far from a frozen reference model (the "base" or "SFT" model).

#### Implementation details
* It's common to save the rollout into a buffer and replay the buffer > 1 times to increase sample-efficiency, stable update by shuffling/mini-batching the same batch of experience and for memory control (store on CPU move to GPU per mini-batch to avoid VRAM spikes)
* Global batch normalization: use the whole buffer to compute a single mean/std and scale to every token advantage to reduce variance, prevents a few high-reward episodes from dominating the gradient and keep a stable scale across epoches. Pay attention to only take answer tokens into account.
* KL divergence is not explicit in the final loss, instead it's built into the reward (see above "Token-level rewards & KL penalty").

### Overview of DPO
Unlike GRPO or PPO, DPO don't need a reward definition (LLM as the reward model), instead it directly train the base model with a pair of answers and maximize the probability of the choosen answer and minimize the probablity of the rejected answer.

- ** Loss ** -
  ```
  loss = -log_sigmoid(beta * log_prob(chosen_policy / chosen_ref) - beta * logprob(rejected_policy / rejected_ref))
  ```
  To optimize how well the policy model likes the chosen answer over the old model, over how much the policy model likes the rejected answer over the old model.

#### implementation details
* Need to make sure only the answer tokens is taken into account. It's easier to separate answer tokens from prompt if the model is right padded.

## Performance Comparison
### Gemma 2B Instruct as base model
* Baseline Model: Gemma 2B Instruct Total: 61% (on GSM8K_200), 60.25% (on GSM8K_800)
* Best GRPO performance: 68.5% (on GSM8K_200), 66% (on GSM8K_800)
* Best PPO performance: 64.5%
* Best DPO performance: 71% (on GSM8K_200), 65.62% (on GSM8K_800)

Below are from local mac mini run
* GRPO + LORA Model checkpoint (base): Gemma 2B Instruct + LoRA with GRPO loss Total: 200 Correct: 126 Accuracy: 63.00% (before running optimization)
* GRPO + LORA Model checkpoint(efficient): Gemma 2B Instruct + LoRA with GRPO loss with improved efficiency. Total: 199 Correct: 118 Accuracy: 59.3%.
* GRPO + LORA Model checkpoint(Train on 200 harder hand curated + AI generated examples, with efficiency improvement, MAX_NEW_TOKENS=400, TEMP=0.9, NUM_SAMPLES=5, lr=2*1e-4, KL_COEFF=0.1): Gemma 2B Instruct + LoRA Total: 200 Correct: 132 Accuracy: 66.00%
* PPO + Critic (single layer) + LORA(Actor) (Initial trial with 160 training examples, batch_size=8, VF_COEFF=0.01, EPS=0.1): Model: Gemma 2B Instruct + LoRA Total: 200 Correct: 126 Accuracy: 63.00%
* DPO + LORA: Total: 200 Correct: 127 Accuracy: 63.5%


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
`(reward - mean(reward))/std(reward)` to reduce the gradient variance. Later changed to a rank based advantage function that is more robust to outliers, providing consistent gradient signals for stable training and effective tie-breaking. Rank-based method is more suitable for sparse, discrete rewards such as Math/Coding problems.
3) Add KL divergency term into the original GRPO formula: `-advantage * P(new)/P(old) + alpha * logP(new) / logP(old)`

### Post training efficiency
I tried a few things to speed up the training on MPS
1) Update sample_k logic to avoid looping over k samples, but instead batching the k samples. 
Ideally we could use the `model.generate` call to avoid iterating with tokens as well, however, it 
causes NaN issue probably due to a known unstability of Gemma model on MPS.
2) Update the second pass to a batch mode instead of looping over the k samples.
3) Enable cache for sampling logic
4) Early stop if all sampling reached EOF.
5) Use Top-P sampling to avoid unexpected tokens that confuse the model.

After the optimization step above (combining, 1/2/3/4), the overall training time is able to reduce 
by 3-4x.
3) TODO: update to larger batches.

## Device comparison:
While it's not surprising that GPU(cuda) is much more efficient, it's still good to record some numbers comparing mac mini and cuda:
For eval of 200 math examples, mac mini took around 1 hr and cuda (4090 with 24vram) only took 2-3 min!
For running 1 epoch(100 examples) of GRPO training, mac mini can do at most 5 as group size and it took around 2-3 hrs, cuda(4090 with 24 vram) can support much larger group size (e.g. 32) and took around 20-30 min.

## Model Soup
For GRPO training on GPU with more training data (1000 total, 10 epoches with 100 per epoch), the model accuracy (200 math test set) is 63.5%, 68.5%, 65%, 67.5%, 62.5%, 61%, 63.5%, 63.5%, 67%, 60.5%. Epoch 2, 4 and 9 showed best performance. As a result, I decided to use the model soup method to combine these 3 epochs and try to re-do the evaluation on a larger test dataset (800 examples). In summary, soup indeed achieved better performance to individual snapshot!

For math eval with 800 examples
Epoch 2 accuracy: 65.88%
Epoch 4 accuracy: 66%
Soup accuracy: 66.88%

For math eval with 200 examples
Epoch 2 accuracy: 68.5%
Epoch 4 accuracy: 67.5%
Soup accuracy: 71%
