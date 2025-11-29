import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from grpo.utils import load_model
from grpo.sampler import sample_k
from grpo.advantage import compute_advantage
from grpo.reward import compute_reward
from grpo.lora import apply_lora_to_model, freeze_non_lora_params, get_lora_parameters

MODEL_PATH = Path(__file__).resolve().parent / "models" / "gemma-2-2b"
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "math_grpo_200.jsonl"
NUM_SAMPLES_PER_PROMPT = 3
NUM_TRAINING_DATA = 10
EVAL_EVERY = 50
SAMPLING_TEMPERATURE = 0.7
DEVICE = torch.device("mps")

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))
# Frozen reference model for old logprobs (no LoRA, no grads)
_, ref_model = load_model(str(MODEL_PATH))
ref_model.to(DEVICE)
for p in ref_model.parameters():
    p.requires_grad = False
ref_model.eval()

# Wrap target linear layers with LoRA adapters
model = apply_lora_to_model(
    model,
    r=8,
    alpha=16,
    target_modules=("q_proj", "v_proj"),
    dropout=0.0,
)
freeze_non_lora_params(model)
model.to(DEVICE)
lora_params = get_lora_parameters(model)
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

global_step = 0
running_loss = 0.0
running_correct = 0
running_total = 0

def extract_answer(text):
    if text is None:
        return None
    # Find all numeric spans and pick the last one (closest to the end of the output)
    # Matches numbers like: 3, -2, 3.1415, 0.00001, .5, -0.25
    matches = list(re.finditer(r"[-+]?\d*\.?\d+", text))
    if not matches:
        return None

    last_match = matches[-1].group(0)
    try:
        cleaned = last_match.replace(",", "")
        return float(cleaned)
    except Exception:
        return None

"""
load data
load model (with LoRA enabled)
for each batch:
    generate k initial answers
    generate k refined answers
    compute rewards
    compute advantages
    recompute logprobs with grad
    compute GRPO loss
    backprop
    step optimizer
    periodically evaluate
"""
# Load training data
with open(TRAIN_FILE) as f:
    test_data = [json.loads(line) for line in f]

for line in tqdm(test_data[:NUM_TRAINING_DATA]):
    question = line['question']
    print(f"question is {question}")
    gold_answer = str(line["gold_answer"]).strip()
    print(f"gold_answer is {gold_answer}")
    print("start sampling passes")
    # Generate K initial answers and get each answer token's logprob and old_logprob
    # for the answer.
    model.eval()    # disable dropout
    with torch.no_grad():
        samples = sample_k(
            model,
            tokenizer,
            question,
            k=NUM_SAMPLES_PER_PROMPT,
            temperature=SAMPLING_TEMPERATURE,
            max_new_tokens=500,
        )
    # Second pass (enable gradient) to get each answer token's logprob and new_logprob
    # and collect rewards
    model.train()
    with torch.enable_grad():
        rewards = []
        new_logprobs = []
        old_logprobs = []
        for i, r in enumerate(samples, start=1):
            print(f"\nSample {i}: {r['text']}")
            print(f"prompt token id length: {r['prompt_id_length']}")
            print(f"tokens size: {r['tokens'].shape}")
            print(f"sum answer token logprobs: {r['sum_token_logprobs']}")
            print(f"token logprobs shape: {r['token_logprobs'].shape}")
            
            tokens = r["tokens"].to(DEVICE)
            attention_mask = torch.ones_like(tokens)
            # New logprobs from trainable model (with LoRA)
            out_new = model(input_ids=tokens, attention_mask=attention_mask)
            logits_new = out_new.logits  # [batch, seq_length, vocab]
            shifted_log_probs_new = F.log_softmax(logits_new / SAMPLING_TEMPERATURE, dim=-1)[:, :-1, :]
            targets = tokens[:, 1:].unsqueeze(-1)
            log_probs_new = shifted_log_probs_new.gather(-1, targets).squeeze(-1)  # [batch, seq_length-1]
            answer_log_probs_new = log_probs_new[:, r['prompt_id_length'] - 1:]
            sum_token_logprobs_new = answer_log_probs_new.sum(dim=1)

            # Old logprobs from frozen reference model
            with torch.no_grad():
                out_old = ref_model(input_ids=tokens, attention_mask=attention_mask)
                logits_old = out_old.logits
                shifted_log_probs_old = F.log_softmax(logits_old / SAMPLING_TEMPERATURE, dim=-1)[:, :-1, :]
                log_probs_old = shifted_log_probs_old.gather(-1, targets).squeeze(-1)
                answer_log_probs_old = log_probs_old[:, r['prompt_id_length'] - 1:]
                sum_token_logprobs_old = answer_log_probs_old.sum(dim=1)

            # Start to prepare for the GRPO loss
            reward = compute_reward(question, r['text'], gold_answer)
            rewards.append(reward)
            new_logprobs.append(sum_token_logprobs_new.squeeze(0))
            old_logprobs.append(sum_token_logprobs_old.squeeze(0))
        # Compute advantages
        advantages = torch.tensor(
            compute_advantage(rewards),
            device=DEVICE,
            dtype=new_logprobs[0].dtype if new_logprobs else torch.float32,
        )
        print(f"rewards: {rewards}")
        print(f"advantages: {advantages.tolist()}")
        # Compute GRPO loss
        old_logprobs_t = torch.stack(old_logprobs)
        new_logprobs_t = torch.stack(new_logprobs)
        print(f"new_logprobs_t shape: {new_logprobs_t.shape}")
        # GRPO/PPO-style ratio
        log_prob_ratio = new_logprobs_t - old_logprobs_t
        ratio = log_prob_ratio.exp()
        loss = -(advantages * ratio).mean()
        print(f"loss: {loss.item()}")
        running_loss += loss.item()
        running_correct += sum(1 for r in rewards if r > 0)
        running_total += len(rewards)
        global_step += 1
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # periodically evaluate
        if global_step % EVAL_EVERY == 0:
            avg_loss = running_loss / max(global_step, 1)
            acc = running_correct / max(running_total, 1)
            print(f"[step {global_step}] avg_loss={avg_loss:.4f} acc={acc:.4f}")
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            eval_prompt = "11+123=?"
            eval_inputs = tokenizer(eval_prompt, return_tensors="pt").to(DEVICE)
            model.eval()
            with torch.no_grad():
                out = model.generate(
                    **eval_inputs,
                    max_new_tokens=8,
                    do_sample=False,
                )
                print(f"[eval] {eval_prompt} -> {tokenizer.decode(out[0], skip_special_tokens=True)}")
            model.train()
    print("==end-of-training-sample==")
