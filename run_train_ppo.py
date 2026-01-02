"""
Script to load a pretrained model and do GRPO with math data to fine-tune the model with LoRA
"""
import gc
import json
import os
import re
import random
import time
from pathlib import Path
import psutil

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from grpo.utils import load_model
from ppo.ppo_advantage import advantage_gae
from ppo.ppo_sampler import sample_batch
from ppo.lora_critic import Critic, apply_lora_to_model, freeze_non_lora_critic_params, get_optimizer_params
from ppo.ppo_reward import refined_advanced_cot_reward

# To avoid the known issue of gemma2 x MPS memory allocator bug.
# This hapens because hugging face automatically runs FP16 warmup allocations
# even request fp32 or bfloat16
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TRANSFORMERS_NO_MPS_CACHE_ALLOCATOR"] = "1"

MODEL_PATH = Path(__file__).resolve().parent / "models" / "gemma-2-2b"
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "gsm8k_grpo_train.jsonl"
LORA_CKPT = None
# LORA_CKPT = Path("./gemma-2-2b-checkpoints/sft_lora_epoch0_step200.pt")  # Set to None if training from base

CHECKPOINT_DIR = Path(__file__).resolve().parent / "gemma-2-2b-checkpoints"
# CHECKPOINT_DIR = Path(__file__).resolve().parent / "Qwen2.5-Math-1.5B-Instruct-checkpoints"
NUM_TRAINING_DATA = 64
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4    # Accumulate 4 mini-batches (Total effective batch = 32)
TOTAL_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS
PPO_EPOCHS = 4          # Number of optimization passes per batch
NUM_EPOCHS = 10
EVAL_EVERY = 11
MAX_INPUT_TOKENS = 150
MAX_NEW_TOKENS = 400
TARGET_KL = 6.0
BETA = 0.1
VF_COEF = 0.01
ENT_COEF = 0.01
DEVICE = torch.device("mps")
EPS = 0.2
PROMPT = " Please reason step-by-step,  then give: Final answer."

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))
model = apply_lora_to_model(
    model,
    r=16,
    alpha=32,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    dropout=0.05,
)
model = Critic(model)
freeze_non_lora_critic_params(model)
trainable_params = [p for p in model.parameters() if p.requires_grad]

"""
load data
load model (with LoRA enabled)
for each batch:
    Get old policy log-probs (tokens, log-probs, values from critic, reward)
    get reference policy log-probs
    advantage estimation (GAE)
    The surrogate loss
    The value loss
    Back prop
"""
# Load training data
test_data = []
with open(TRAIN_FILE) as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        test_data.append(json.loads(ln))
# random.shuffle(test_data)
print(f"Print found {len(test_data)} lines of training data")

if LORA_CKPT and LORA_CKPT.exists():
    ckpt = torch.load(LORA_CKPT, map_location="cpu")
    missing = model.load_state_dict(ckpt.get("lora_state_dict", {}), strict=False)
    print(f"Loaded LoRA checkpoint {LORA_CKPT} (missing/unexpected: {missing})")
else:
    print(f"LoRA checkpoint {LORA_CKPT} not found; training from base model.")

model.to(DEVICE)
# Setup optimizer and scheduler
params = get_optimizer_params(model, lora_lr=2e-4, critic_lr=1e-4, weight_decay=0.01)
optimizer = torch.optim.AdamW(params, eps=1e-6)

total_steps = min(len(test_data), NUM_TRAINING_DATA * NUM_EPOCHS) // BATCH_SIZE * NUM_EPOCHS
warmup_steps = int(0.05 * total_steps) # 5% warmup is a safe default
# 3. Setup Scheduler
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

global_step = 0
running_loss = 0.0
running_correct = 0
running_total = 0

def check_memory_health():
    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    # Color coding the output for visibility
    color = "\033[93m" if vmem.percent > 85 else "\033[92m"
    reset = "\033[0m"
    print(f"{color}📊 [System Health] RAM: {vmem.percent}% | Swap Used: {swap.used / 1e9:.2f} GB{reset}")

def save_lora_checkpoint(model, optimizer, epoch, global_step):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "lora_state_dict": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer_state_dict": optimizer.state_dict(),
    }
    ckpt_path = CHECKPOINT_DIR / f"ppo_lora_epoch{epoch}_step{global_step}.pt"
    torch.save(state, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")



for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
    start = (epoch - 1) * NUM_TRAINING_DATA
    end = start + NUM_TRAINING_DATA
    train_samples = test_data[start:end]
    for i in range(0, len(train_samples), TOTAL_BATCH_SIZE):
        chunk = test_data[i: i + TOTAL_BATCH_SIZE]
        if len(chunk) < TOTAL_BATCH_SIZE: continue
        check_memory_health()
        t_start = time.perf_counter()
        buffer_memory = [] # Experience Buffer (Stored on CPU)
        
        # ====================================================
        # PHASE 1: EXPERIENCE COLLECTION (No Gradients)
        # ====================================================
        model.eval()
        t0 = time.perf_counter()
        print(f"Collecting experience for step {global_step}...")
        for idx in tqdm(range(0, len(chunk), BATCH_SIZE), desc=f"epoch {epoch} - mini-batch {i}", leave=False):
            batch = train_samples[idx : idx + BATCH_SIZE]
            questions = [sample["question"] + PROMPT for sample in batch]
            golds = [str(sample["gold_answer"]).strip() for sample in batch]
            model.eval()
            with torch.no_grad():
                res = sample_batch(
                    model,
                    tokenizer,
                    questions,
                    device=DEVICE,
                    dtype=torch.bfloat16,
                    max_input_tokens=MAX_INPUT_TOKENS,
                    max_new_tokens=MAX_NEW_TOKENS
                )
                t1 = time.perf_counter()
                B, T = res["tokens"].size()
                prompt_len = res["prompt_id_length"]
                padded_batch_tokens = res["tokens"].to(DEVICE)
                attention_mask = res["attention_mask"].to(DEVICE)
                eos_mask = (padded_batch_tokens[:, prompt_len:] == tokenizer.eos_token_id)
                has_eos = eos_mask.any(dim=1)
                first_eos_offset = torch.where(
                    has_eos,
                    eos_mask.float().argmax(dim=1),
                    (attention_mask.sum(dim=1) - prompt_len - 1)   # fallback to last token
                )
                eos_pos = first_eos_offset + prompt_len
                # build mask for answer region: shape [B, T-1]
                arange = torch.arange(T - 1, device=DEVICE).unsqueeze(0)  # [1, T-1]
                answer_mask = ((arange >= (prompt_len - 1)) & (arange < (eos_pos.unsqueeze(1)))).float()
                
                targets = padded_batch_tokens[:, 1:].unsqueeze(-1)
                ## Get old policy logprobs.
                old_out, old_values = model(input_ids=padded_batch_tokens, attention_mask=attention_mask, return_values=True)
                # [B, T_max, vocab]
                logits_old = old_out.logits
                values_old = old_values[:, :-1]
                # [B, T_max-1, vocab]
                shifted_log_probs_old = F.log_softmax(logits_old, dim=-1)[:, :-1, :]
                # [B, T_max-1]
                log_probs_old = shifted_log_probs_old.gather(-1, targets).squeeze(-1)

                with model.disable_adapter():
                ## Get reference policy logprobs.
                    ref_out = model(input_ids=padded_batch_tokens, attention_mask=attention_mask, return_values=False)
                    # [B, T_max, vocab]
                    logits_ref = ref_out.logits
                    # [B, T_max-1, vocab]
                    shifted_log_probs_ref = F.log_softmax(logits_ref, dim=-1)[:, :-1, :]
                    # [B, T_max-1]
                    log_probs_ref = shifted_log_probs_ref.gather(-1, targets).squeeze(-1)
            masked_log_probs_ref = log_probs_ref * answer_mask
            masked_log_probs_old = log_probs_old * answer_mask
            kl_divergence_est = masked_log_probs_old - masked_log_probs_ref
            kl_penalty_est = - BETA * kl_divergence_est
            rewards = kl_penalty_est.detach().clone()
            final_rewards = [
                refined_advanced_cot_reward(
                    txt,
                    gold_answer,
                    truncated=tr,
                )
                for txt, tr, gold_answer in zip(res["text"], res["truncated"], golds)
            ]
            batch_indices = torch.arange(B, device=DEVICE)
            eos_reward_idx = torch.clamp(eos_pos - 1, min=0, max=rewards.size(1) - 1)
            final_rewards_t = torch.tensor(final_rewards, device=rewards.device, dtype=rewards.dtype)
            rewards[batch_indices, eos_reward_idx] += final_rewards_t
            # Get advantage
            advantages = advantage_gae(rewards, old_values, res["truncated"]).detach()
            returns = (advantages + values_old) * answer_mask
            buffer_memory.append({
                    "tokens": padded_batch_tokens.cpu(),
                    "attention_mask": attention_mask.cpu(),
                    "old_log_probs": log_probs_old.cpu(),
                    "ref_log_probs": log_probs_ref.cpu(), # Storing reference probs
                    "old_values": values_old.cpu(),
                    "advantages": advantages.cpu(),
                    "returns": returns.cpu(),
                    "answer_mask": answer_mask.cpu(),
                    "rewards": rewards.cpu() 
                })
            # Cleanup GPU
            del res, padded_batch_tokens, attention_mask, old_out, old_values, ref_out
            del logits_old, log_probs_old, log_probs_ref, rewards, advantages, returns
            torch.mps.empty_cache()
        # ====================================================
        # PHASE 2: GLOBAL ADVANTAGE NORMALIZATION
        # ====================================================
        t2 = time.perf_counter()
        total_sum = torch.tensor(0.0)
        total_sq = torch.tensor(0.0)
        total_count = 0
        for b in buffer_memory:
            active_advs = b["advantages"][b["answer_mask"] > 0].float()
            if active_advs.numel() == 0:
                continue
            total_sum += active_advs.sum()
            total_sq += (active_advs ** 2).sum()
            total_count += active_advs.numel()
        adv_mean = total_sum / max(total_count, 1)
        adv_var = total_sq / max(total_count, 1) - adv_mean ** 2
        adv_std = torch.sqrt(torch.clamp(adv_var, min=1e-8))

        # Standardize only the active tokens
        for b in buffer_memory:
            b["advantages"] = (b["advantages"] - adv_mean) / (adv_std + 1e-8)
            b["advantages"] = b["advantages"] * b["answer_mask"] # Ensure padding stays 0        
        # ====================================================
        # PHASE 3: PPO OPTIMIZATION (Multiple Epochs)
        # ====================================================
        print(f"PPO optimization step {global_step}...")
        model.train()
        for ppo_epoch in range(PPO_EPOCHS):
            random.shuffle(buffer_memory)
            for mini_batch in buffer_memory:
                # Move Batch to GPU
                b_tokens = mini_batch["tokens"].to(DEVICE)
                b_mask = mini_batch["attention_mask"].to(DEVICE)
                b_old_log_probs = mini_batch["old_log_probs"].to(DEVICE)
                b_ref_log_probs = mini_batch["ref_log_probs"].to(DEVICE) # Load Ref
                b_old_values = mini_batch["old_values"].to(DEVICE)
                b_advantages = mini_batch["advantages"].to(DEVICE)
                b_returns = mini_batch["returns"].to(DEVICE)
                b_answer_mask = mini_batch["answer_mask"].to(DEVICE)
                with torch.enable_grad():
                    # Get logprobs of policy model
                    new_out, new_values = model(input_ids=b_tokens, attention_mask=b_mask, return_values=True)
                    targets = b_tokens[:, 1:].unsqueeze(-1)
                    logits_new = new_out.logits[:, :-1, :]   # [B, T-1, V]
                    values_new = new_values[:, :-1] # [B, T-1]
                    shifted_log_probs_new = F.log_softmax(logits_new, dim=-1)
                    # Gather logprobs of the actually generated tokens
                    log_probs_new = shifted_log_probs_new.gather(-1, targets).squeeze(-1)  # [B, T-1]
                    # Mask out padded tokens
                    masked_values_new = values_new * b_answer_mask
                    masked_values_old = b_old_values * b_answer_mask
                    ratio = torch.exp(log_probs_new - b_old_log_probs) * b_answer_mask

                # Policy loss for actor based on ratio and advantage
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * b_advantages
                policy_loss = -(torch.min(surr1, surr2) * b_answer_mask).sum() / (b_answer_mask.sum() + 1e-8)
                
                # Clipped MSE loss for critic
                v_clipped = masked_values_old + torch.clamp(masked_values_new - masked_values_old, -EPS, EPS)
                v_loss_1 = (masked_values_new - b_returns).pow(2)
                v_loss_2 = (v_clipped - b_returns).pow(2)
                value_loss = 0.5 * (torch.max(v_loss_1, v_loss_2) * b_answer_mask).sum() / (b_answer_mask.sum() + 1e-8)

                # entropy loss to encourage exploration
                entropy = -(torch.exp(log_probs_new) * log_probs_new * b_answer_mask).sum() / (b_answer_mask.sum() + 1e-8)
                entropy_loss = -ENT_COEF * entropy

                # KL Penalty Loss (CORRECTED CALCULATION)
                # Ensure the new policy doesn't drift too far from reference
                kl_divergence = (log_probs_new - b_ref_log_probs) * b_answer_mask
                kl_loss = BETA * kl_divergence.sum() / (b_answer_mask.sum() + 1e-8)

                total_loss = policy_loss + VF_COEF * value_loss + entropy_loss + kl_loss
                t3 = time.perf_counter()
                if ppo_epoch == PPO_EPOCHS - 1:
                    print(f"Step {global_step} | Loss: {total_loss.item():.3f} | Pol: {policy_loss.item():.3f} | KL: {kl_loss.item():.3f}")
                    print(
                        {
                            "tokens_shape": tuple(b_tokens.shape),
                            "attention_mask_shape": tuple(b_mask.shape),
                            "ratio_mean": (ratio.sum() / (b_answer_mask.sum() + 1e-8)).item(),
                            "adv_mean": b_advantages.mean().item(),
                            "policy_loss": policy_loss.item(),
                            "value_loss": value_loss.item(),
                        }
                    )
                optimizer.zero_grad(set_to_none=True)
                t4 = time.perf_counter()
                total_loss.backward()
                # Check if gradients are reaching the LoRA weights
                lora_grads = []
                critic_grads = []
                for n, p in model.named_parameters():
                    if p.requires_grad and ("lora_" in n): # Match your custom LoRA names
                        if p.grad is not None:
                            lora_grads.append(p.grad.abs().mean().item())
                    if p.requires_grad and ("value_layer" in n):
                        if p.grad is not None:
                            critic_grads.append(p.grad.abs().mean().item())

                avg_lora_grad = sum(lora_grads) / len(lora_grads) if lora_grads else 0.0
                avg_critic_grad = sum(critic_grads) / len(critic_grads) if critic_grads else 0.0
                print(f"--- Gradient Check ---")
                print(f"Average LoRA Gradient Magnitude: {avg_lora_grad:.10f}")
                print(f"Average Critic Gradient Magnitude: {avg_critic_grad:.10f}")
                # For MPS gradient stability
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                for param in trainable_params:
                    if param.grad is not None:
                        param.grad.data = param.grad.data.contiguous()
                optimizer.step()
                scheduler.step()
                t5 = time.perf_counter()
                with torch.no_grad():
                    valid_kl = (kl_divergence * b_answer_mask).sum() / (b_answer_mask.sum() + 1e-8)
                    print(f"Step: {global_step} | Loss: {total_loss.item():.4f} | KL: {valid_kl.item():.4f}")
                    # SAFETY: Stop if KL explodes
                    if valid_kl.item() > 15.0:
                        print("!!! ALERT: KL Exploded. Saving and exiting.")
                        save_lora_checkpoint(model, optimizer, epoch, global_step)
                        break
                # Cleanup GPU for next mini-batch
                del b_tokens, b_mask, b_old_log_probs, b_ref_log_probs, b_old_values, b_advantages, b_returns, b_answer_mask
                del new_out, new_values, logits_new, shifted_log_probs_new, log_probs_new
                del total_loss, kl_divergence
                torch.mps.empty_cache()

            global_step += 1
            if global_step % EVAL_EVERY == 0:
                avg_loss = running_loss / max(global_step, 1)
                acc = running_correct / max(running_total, 1)
                print(f"[step {global_step}] avg_loss={avg_loss:.4f} acc={acc:.4f}")
                running_loss = 0.0
                running_correct = 0
                running_total = 0

                eval_prompt = "A car travels at 62 km/h for 2 hours, then twice that speed for 3 hours. Compute total distance in km."
                eval_inputs = tokenizer(eval_prompt + PROMPT, return_tensors="pt").to(DEVICE)
                model.eval()
                with torch.no_grad():
                    out = model.generate(
                        **eval_inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )
                    print(f"[eval] {eval_prompt} -> {tokenizer.decode(out[0], skip_special_tokens=True)}")
                model.train()
                del eval_inputs, out
                torch.mps.empty_cache()
        t_end = time.perf_counter()
        print(f"[timing] sample processed in {(t_end - t_start):.2f}s")
        print("\n=== PROFILER ===")
        print(f"Sampling:          {(t1-t0):.2f}s")
        print(f"Logprob forward:   {(t3-t2):.2f}s")
        print(f"Backward:          {(t5-t4):.2f}s")
    save_lora_checkpoint(model, optimizer, epoch, global_step)
    print(f"==end-of-epoch {epoch}==")
