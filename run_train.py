import json
import re
import random
import time
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
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
NUM_SAMPLES_PER_PROMPT = 3
NUM_TRAINING_DATA = 50
NUM_EPOCHS = 3
EVAL_EVERY = 50
SAMPLING_TEMPERATURE = 0.7
DEVICE = torch.device("mps")

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))
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
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

global_step = 0
running_loss = 0.0
running_correct = 0
running_total = 0

def save_lora_checkpoint(model, optimizer, epoch, global_step):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "lora_state_dict": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer_state_dict": optimizer.state_dict(),
    }
    ckpt_path = CHECKPOINT_DIR / f"lora_epoch{epoch}_step{global_step}.pt"
    torch.save(state, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

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

train_samples = test_data[:NUM_TRAINING_DATA]

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
    random.shuffle(train_samples)
    for line in tqdm(train_samples, desc=f"epoch {epoch}", leave=False):
        t_start = time.perf_counter()
        question = line['question']
        gold_answer = str(line["gold_answer"]).strip()
        # Generate K initial answers and get each answer token's logprob and old_logprob
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
        model.train()
        with torch.enable_grad():
            rewards = []
            new_logprobs = []
            old_logprobs = []
            for r in samples:
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

                # Start to prepare for the GRPO loss
                reward = compute_reward(question, r['text'], gold_answer)
                rewards.append(reward)
                new_logprobs.append(sum_token_logprobs_new.squeeze(0))
                old_logprobs.append(r['sum_token_logprobs'].to(DEVICE).squeeze(0))
            # Calculate advantages
            advantages = torch.tensor(
                compute_advantage(rewards),
                device=DEVICE,
                dtype=new_logprobs[0].dtype if new_logprobs else torch.float32,
            )
            # Compute GRPO loss
            old_logprobs_t = torch.stack(old_logprobs)
            new_logprobs_t = torch.stack(new_logprobs)
            log_prob_ratio = new_logprobs_t - old_logprobs_t
            ratio = log_prob_ratio.exp()
            loss = -(advantages * ratio).mean()
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
        t_end = time.perf_counter()
        print(f"[timing] sample processed in {(t_end - t_start):.2f}s")
    save_lora_checkpoint(model, optimizer, epoch, global_step)
    print(f"==end-of-epoch {epoch}==")
