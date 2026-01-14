"""
Script to load a pretrained model and do GRPO with math data to fine-tune the model with LoRA
"""
import bitsandbytes as bnb
import gc
import json
import os
import re
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from grpo.device import get_default_device, empty_cache
from grpo.utils import load_model, append_jsonl, check_memory_health, save_lora_checkpoint
from grpo.sampler import sample_k_parallel
from grpo.advantage import compute_advantage, compute_rank_advantage
from grpo.reward import compute_reward, refined_advanced_cot_reward
from grpo.lora import ModelAdapterWrapper, apply_lora_to_model, freeze_non_lora_params, get_lora_parameters

DEVICE = get_default_device()
IS_CUDA = (DEVICE.type == "cuda")
IS_MPS = (DEVICE.type == "mps")
if IS_MPS:
    # To avoid the known issue of gemma2 x MPS memory allocator bug.
    # This hapens because hugging face automatically runs FP16 warmup allocations
    # even request fp32 or bfloat16
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["TRANSFORMERS_NO_MPS_CACHE_ALLOCATOR"] = "1"
if IS_CUDA:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = Path(__file__).resolve().parent / "models" / "gemma-2-2b"
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "gsm8k_grpo_train.jsonl"
LORA_CKPT = None
# LORA_CKPT = Path("./gemma-2-2b-checkpoints/2026_lora_epoch3_step300.pt")  # Set to None if training from base

CHECKPOINT_DIR = Path(__file__).resolve().parent / "gemma-2-2b-checkpoints"
# CHECKPOINT_DIR = Path(__file__).resolve().parent / "Qwen2.5-Math-1.5B-Instruct-checkpoints"
HARD_QUESTION_FILE = Path(__file__).resolve().parent / "data" / "gsm8k_grpo_hard.jsonl"
NUM_SAMPLES_PER_PROMPT = 16 if IS_CUDA else 5
NUM_TRAINING_DATA = 100
NUM_EPOCHS = 10
EVAL_EVERY = 25
LOG_EVERY = 10
SAMPLING_TEMPERATURE = 0.9
MAX_NEW_TOKENS = 350
KL_COEF = 0.1

PROMPT = " Reason step-by-step,  then give: Final answer."

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH), device=DEVICE)
# Wrap target linear layers with LoRA adapters
model = apply_lora_to_model(
    model,
    r=16,
    alpha=32,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    dropout=0.05,
)
model = ModelAdapterWrapper(model)
if LORA_CKPT and LORA_CKPT.exists():
    ckpt = torch.load(LORA_CKPT, map_location="cpu")
    state_dict = ckpt.get("lora_state_dict", {})
    
    # --- FIX START: Sanitize Keys ---
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'base_model.' prefix if it exists
        if k.startswith("base_model.model."):
            new_key = k.replace("base_model.model.", "model.")
        elif k.startswith("base_model."):
            new_key = k.replace("base_model.", "")
        else:
            new_key = k
        new_state_dict[new_key] = v
    # --- FIX END ---
    
    missing = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded LoRA checkpoint {LORA_CKPT}")
    # Verify we actually loaded something relevant
    if len(missing.missing_keys) > 0 and 'model.layers.0.self_attn.q_proj.A.weight' in missing.missing_keys:
        print("⚠️ CRITICAL WARNING: LoRA weights were NOT loaded correctly!")
freeze_non_lora_params(model)
if IS_CUDA:
    print("🚀 Enabling CUDA-specific optimizations (RTX 4090)...")
    model.gradient_checkpointing_enable()
    model.base_model.enable_input_require_grads()
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(get_lora_parameters(model), lr=1e-5)
    except ImportError:
        print("⚠️ bitsandbytes not found. Using standard AdamW.")
        optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=1e-5)
else:
    print("🍏 Using standard optimization (MPS/CPU)...")
    optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=1e-5)


CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

global_step = 0
running_loss = 0.0
running_correct = 0
running_total = 0

def summarize_rewards(values):
    count = len(values)
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / count
    var = sum((v - mean) ** 2 for v in values) / count
    std = var ** 0.5
    return mean, std, min(values), max(values)

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
test_data = []
with open(TRAIN_FILE) as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        test_data.append(json.loads(ln))
# random.shuffle(test_data)
print(f"Print found {len(test_data)} lines of training data")
print(
    f"[config] device={DEVICE} samples_per_prompt={NUM_SAMPLES_PER_PROMPT} "
    f"max_new_tokens={MAX_NEW_TOKENS} lr={optimizer.param_groups[0]['lr']} "
    f"temperature={SAMPLING_TEMPERATURE} KL_COEF={KL_COEF}"
)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
    start = (epoch - 1) * NUM_TRAINING_DATA
    end = start + NUM_TRAINING_DATA
    train_samples = test_data[start:end]
    for line in tqdm(train_samples, desc=f"epoch {epoch}", leave=False):
        if global_step % 10 == 0:
            check_memory_health()
        t_start = time.perf_counter()
        question = line['question']
        prompt = PROMPT
        gold_answer = str(line["gold_answer"]).strip()
        t0 = time.perf_counter()
        # Sample K initial answers and get each answer token's sum_logprob_old.
        model.eval()    # disable dropout
        with torch.no_grad():
            res = sample_k_parallel(
                model,
                tokenizer,
                question + prompt,
                k=NUM_SAMPLES_PER_PROMPT,
                device=DEVICE,
                dtype=torch.bfloat16,
                temperature=SAMPLING_TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        # print("get results from sampling")
        t1 = time.perf_counter()
        t2 = time.perf_counter()
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
        with torch.no_grad():
            old_out = model(input_ids=padded_batch_tokens, attention_mask=attention_mask)
            logits_old = old_out.logits[:, :-1, :]   # [B, T-1, V]
            shifted_log_probs_old = F.log_softmax(logits_old / SAMPLING_TEMPERATURE, dim=-1)
            # Gather logprobs of the actually generated tokens
            log_probs_old = shifted_log_probs_old.gather(-1, targets).squeeze(-1)  # [B, T-1]
            with model.disable_adapter():
                ref_out = model(input_ids=padded_batch_tokens, attention_mask=attention_mask)
                logits_ref = ref_out.logits[:, :-1, :]   # [B, T-1, V]
                shifted_log_probs_ref = F.log_softmax(logits_ref / SAMPLING_TEMPERATURE, dim=-1)
                log_probs_ref = shifted_log_probs_ref.gather(-1, targets).squeeze(-1).detach()  # [B, T-1]

        # Second pass (enable gradient) to get each answer token's sum_logprob_new.
        # Reward & Advantage
        with torch.no_grad():
            rewards = [
                refined_advanced_cot_reward(
                    txt,
                    gold_answer,
                    truncated=tr,
                )
                for txt, tr in zip(res["text"], res["truncated"])
            ]
        if global_step % 10 == 0:
            print(question)
            print(f"answer is {gold_answer}")
            for txt, r, tr in zip(res['text'], rewards, res["truncated"]):
                print(txt)
                print(f"reward is {r}")
                print(f"is result truncated? {tr}")
        # Calculate advantages
        advantages = compute_rank_advantage(rewards, device=DEVICE, dtype=torch.float32).detach()
        advantages = advantages.to(shifted_log_probs_old.dtype)
        print(f"advantages is {advantages}")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        if IS_CUDA:
            model.base_model.config.use_cache = False # Required for Gradient Checkpointing
        # We process each sample in the group individually to save VRAM
        # but normalize the loss by K so the total gradient is an average.
        accumulated_kl = 0.0
        accumulated_grpo_loss = 0.0
        ratio_sum = 0.0
        ratio_count = 0
        with torch.enable_grad():
            accumulated_kl = 0.0
            accumulated_grpo_loss = 0.0
            if max(rewards) > 0:
                for i in range(NUM_SAMPLES_PER_PROMPT):
                    m_tokens = padded_batch_tokens[i:i+1]
                    m_mask = attention_mask[i:i+1]
                    m_ans_mask = answer_mask[i:i+1]
                    m_targets = targets[i:i+1]
                    m_adv = advantages[i:i+1]
                    m_log_old = log_probs_old[i:i+1].sum(dim=1)
                    m_log_ref = log_probs_ref[i:i+1]

                    # Single Forward Pass
                    out_new = model(input_ids=m_tokens, attention_mask=m_mask)
                    logits_new = out_new.logits[:, :-1, :] / SAMPLING_TEMPERATURE
                    log_probs_new = F.log_softmax(logits_new, dim=-1).gather(-1, m_targets).squeeze(-1)
                    
                    # Masked log-probs for this specific sample
                    m_masked_new = log_probs_new * m_ans_mask
                    m_masked_ref = m_log_ref * m_ans_mask
                    m_sum_new = m_masked_new.sum(dim=1)
                    
                    # 1. KL Loss (Schulman approximation)
                    log_ratio = m_masked_ref.detach() - m_masked_new
                    kl_per_token = torch.exp(log_ratio) - log_ratio - 1
                    m_sum_kl = (kl_per_token * m_ans_mask).sum(dim=1)
                    m_actual_len = m_ans_mask.sum(dim=1).clamp(min=1.0)
                    m_kl_loss = (m_sum_kl / m_actual_len).mean()

                    # 2. GRPO Loss
                    ratio = torch.exp(m_sum_new - m_log_old)
                    m_grpo_loss = -(m_adv * ratio).mean()
                    ratio_sum += ratio.detach().item()
                    ratio_count += 1

                    # 3. Combined Loss scaled by group size
                    total_sample_loss = (m_grpo_loss + KL_COEF * m_kl_loss) / NUM_SAMPLES_PER_PROMPT
                    
                    # Backprop this specific sample's gradient (gradient accumulation)
                    total_sample_loss.backward()
                    
                    # Track metrics
                    accumulated_kl += m_kl_loss.item()
                    accumulated_grpo_loss += m_grpo_loss.item()

                    # CRITICAL: Clean up heavy tensors immediately
                    del out_new, logits_new, log_probs_new, m_masked_new, log_ratio, kl_per_token
                    
                # Step the optimizer after processing the whole group
                torch.nn.utils.clip_grad_norm_(get_lora_parameters(model), max_norm=1.0)
                optimizer.step()
            
                # Final logging values for the step
                final_kl = accumulated_kl / NUM_SAMPLES_PER_PROMPT
                final_grpo = accumulated_grpo_loss / NUM_SAMPLES_PER_PROMPT
                final_loss = final_grpo + KL_COEF * final_kl
                print(f"grpo_loss is {final_grpo} and kl is {final_kl}")
            else:
                print("All rewards are negative; skipping gradient update.")
                final_kl, final_grpo, final_loss = 0.0, 0.0, 0.0
                append_jsonl(HARD_QUESTION_FILE, line)

        running_loss += final_loss
        running_correct += sum(1 for r in rewards if r > 0)
        running_total += len(rewards)
        global_step += 1
        if global_step % LOG_EVERY == 0:
            lr = optimizer.param_groups[0]["lr"]
            reward_mean, reward_std, reward_min, reward_max = summarize_rewards(rewards)
            truncated_rate = sum(res["truncated"]) / max(len(res["truncated"]), 1)
            avg_answer_len = answer_mask.sum(dim=1).mean().item()
            ratio_mean = ratio_sum / max(ratio_count, 1)
            gen_tokens = answer_mask.sum().item()
            sampling_time = max(t1 - t0, 1e-6)
            gen_toks_per_s = gen_tokens / sampling_time 
            print(
                f"[step {global_step}] loss={final_loss:.4f} grpo={final_grpo:.4f} kl={final_kl:.4f} "
                f"reward_mean={reward_mean:.3f} reward_std={reward_std:.3f} "
                f"reward_min={reward_min:.3f} reward_max={reward_max:.3f} "
                f"ratio_mean={ratio_mean:.3f} truncated={truncated_rate:.2f} "
                f"avg_len={avg_answer_len:.1f} gen_tok/s={gen_toks_per_s:.1f} lr={lr}"
            )
        if IS_CUDA:
            model.base_model.config.use_cache = True
        # --- THE DEEP CLEAN BLOCK ---
        # Tensors from the Forward Passes (The biggest memory hogs)
        del old_out, logits_old, shifted_log_probs_old, log_probs_old
        del ref_out, logits_ref, shifted_log_probs_ref, log_probs_ref

        # Intermediate Tensors and Sampler output
        del res, padded_batch_tokens, attention_mask, answer_mask, targets

        # Final scalars
        del advantages, rewards
        del final_grpo, final_kl, final_loss
        gc.collect()
        empty_cache(DEVICE)
        
        # periodically evaluate
        if global_step % EVAL_EVERY == 0:
            avg_loss = running_loss / EVAL_EVERY
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
            empty_cache(DEVICE)
        t_end = time.perf_counter()
        print(f"[timing] sample processed in {(t_end - t_start):.2f}s")
    save_lora_checkpoint(model, optimizer, epoch, global_step, CHECKPOINT_DIR, prefix="grpo")
    print(f"==end-of-epoch {epoch}==")
