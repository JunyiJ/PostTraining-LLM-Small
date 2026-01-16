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

import torch
import torch.nn.functional as F
from tqdm import tqdm

from grpo.device import get_default_device, empty_cache
from grpo.utils import load_model, check_memory_health, save_lora_checkpoint
from dpo.dpo_loss import dpo_loss
from dpo.helper import get_tokens_and_masks
from dpo.lora import ModelAdapterWrapper, apply_lora_to_model, freeze_non_lora_params, get_lora_parameters

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
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "gsm8k_dpo_pairs.jsonl"
LORA_CKPT = None
# LORA_CKPT = Path("./gemma-2-2b-checkpoints/sft_lora_epoch0_step200.pt")  # Set to None if training from base

CHECKPOINT_DIR = Path(__file__).resolve().parent / "gemma-2-2b-checkpoints"
# CHECKPOINT_DIR = Path(__file__).resolve().parent / "Qwen2.5-Math-1.5B-Instruct-checkpoints"
NUM_TRAINING_DATA = 128
MICRO_BATCH_SIZE = 8 
# ACCUMULATION_STEPS: How many micro-batches to accumulate before updating weights
# Effective Batch Size = MICRO_BATCH_SIZE * ACCUMULATION_STEPS
ACCUMULATION_STEPS = 4
NUM_EPOCHS = 50
EVAL_EVERY = 10
LOG_EVERY = 5
MAX_INPUT_TOKENS = 512
LR = 5e-6
BETA = 0.1
if IS_CUDA:
    MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ATTN_IMPLEMENTATION = "sdpa"
elif IS_MPS:
    MODEL_DTYPE = torch.bfloat16
    ATTN_IMPLEMENTATION = None
else:
    MODEL_DTYPE = torch.float32
    ATTN_IMPLEMENTATION = None
PROMPT = "  Reason step-by-step,  then give: Final answer. "

# CUDA performance knobs.
if IS_CUDA and hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load model/tokenizer using helper
tokenizer, model = load_model(
    str(MODEL_PATH),
    device=DEVICE,
    attn_implementation=ATTN_IMPLEMENTATION,
    torch_dtype=MODEL_DTYPE,
)
# Critical for correct identify the answer tokens from the prompt tokens
tokenizer.padding_side = "right"
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
    missing = model.load_state_dict(ckpt.get("lora_state_dict", {}), strict=False)
    print(f"Loaded LoRA checkpoint {LORA_CKPT} (missing/unexpected: {missing})")
else:
    print(f"LoRA checkpoint {LORA_CKPT} not found; training from base model.")
if IS_CUDA and hasattr(model, "base_model"):
    model.base_model.gradient_checkpointing_enable()
    model.base_model.enable_input_require_grads()
    model.base_model.config.use_cache = False
freeze_non_lora_params(model)
model.to(DEVICE)
lora_params = get_lora_parameters(model)
if IS_CUDA:
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(lora_params, lr=LR)
    except ImportError:
        print("WARN: bitsandbytes not found. Using standard AdamW.")
        optimizer = torch.optim.AdamW(lora_params, lr=LR)
else:
    optimizer = torch.optim.AdamW(lora_params, lr=LR)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

global_step = 0
running_loss = 0.0
running_kl_sum = 0.0
running_kl_tokens = 0.0
running_policy_chosen_sum = 0.0
running_policy_chosen_tokens = 0.0
running_policy_rejected_sum = 0.0
running_policy_rejected_tokens = 0.0
running_ref_chosen_sum = 0.0
running_ref_chosen_tokens = 0.0
running_ref_rejected_sum = 0.0
running_ref_rejected_tokens = 0.0
running_eos = 0.0
running_trunc = 0.0
running_sequences = 0.0
running_tokens = 0.0
running_time = 0.0
running_mem_alloc = 0.0
running_mem_reserved = 0.0
running_eval_loss_sum = 0.0
running_eval_steps = 0.0

"""
load data
load model (with LoRA enabled)
for each batch:
    get a pair of <chosen, rejected> prompt and get the log-prob sum of the answer
    calculate the DPO loss and backpropogate.
"""
# Load training data
test_data = []
with open(TRAIN_FILE) as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        test_data.append(json.loads(ln))
print(f"Print found {len(test_data)} lines of training data")
print(
    f"[config] device={DEVICE} micro_batch={MICRO_BATCH_SIZE} accum_steps={ACCUMULATION_STEPS} "
    f"effective_batch={MICRO_BATCH_SIZE * ACCUMULATION_STEPS} lr={optimizer.param_groups[0]['lr']} "
    f"beta={BETA}"
)
optimizer.zero_grad(set_to_none=True)
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
    start = (epoch - 1) * NUM_TRAINING_DATA
    end = start + NUM_TRAINING_DATA
    train_samples = test_data[start:end]
    # Note: total iterations = len(train_samples) / MICRO_BATCH_SIZE
    pbar = tqdm(range(0, len(train_samples), MICRO_BATCH_SIZE), desc=f"epoch {epoch}", leave=False)
    for step_idx, idx in enumerate(pbar):
        if global_step % 10 == 0:
            check_memory_health()
        if IS_CUDA:
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        batch = train_samples[idx : idx + MICRO_BATCH_SIZE]
        prompts = [sample["prompt"] + PROMPT for sample in batch]
        chosens = [sample["prompt"] + PROMPT + sample["chosen"] + tokenizer.eos_token for sample in batch]
        rejects = [sample["prompt"] + PROMPT + sample["rejected"] + tokenizer.eos_token for sample in batch]
        responses = chosens + rejects
        response_ids, response_attn, response_answer_mask = get_tokens_and_masks(
            prompts,
            responses,
            tokenizer,
            DEVICE,
            max_length = MAX_INPUT_TOKENS
        )
        response_targets = response_ids[:, 1:].unsqueeze(-1)
        model.eval()    # disable dropout
        with torch.no_grad():
            with model.disable_adapter():
                ref_response = model(input_ids=response_ids, attention_mask=response_attn)
                response_logits_ref = ref_response.logits[:, :-1, :]   # [2B, T-1, V]
                response_shifted_log_probs_ref = F.log_softmax(response_logits_ref, dim=-1)
                response_log_probs_ref = response_shifted_log_probs_ref.gather(-1, response_targets).squeeze(-1)  # [2B, T-1]

        model.train()
        with torch.enable_grad():
            policy_response = model(input_ids=response_ids, attention_mask=response_attn)
            response_logits_policy = policy_response.logits[:, :-1, :]   # [2B, T-1, V]
            response_shifted_log_probs_policy = F.log_softmax(response_logits_policy, dim=-1)
            response_log_probs_policy = response_shifted_log_probs_policy.gather(-1, response_targets).squeeze(-1)  # [2B, T-1]
            split_idx = len(prompts)
            chosen_log_probs_ref = response_log_probs_ref[:split_idx]
            rejected_log_probs_ref = response_log_probs_ref[split_idx:]
            chosen_log_probs_policy = response_log_probs_policy[:split_idx]
            rejected_log_probs_policy = response_log_probs_policy[split_idx:]
            chosen_answer_mask = response_answer_mask[:split_idx]
            rejected_answer_mask = response_answer_mask[split_idx:]
            loss, chosen_rewards, rejected_rewards = dpo_loss(
                (chosen_log_probs_policy * chosen_answer_mask).sum(dim=1),
                (rejected_log_probs_policy * rejected_answer_mask).sum(dim=1),
                (chosen_log_probs_ref * chosen_answer_mask).sum(dim=1),
                (rejected_log_probs_ref * rejected_answer_mask).sum(dim=1),
                beta=BETA
            )
            if not torch.isfinite(loss).all().item():
                raise ValueError("Non-finite loss detected.")
            with torch.no_grad():
                chosen_policy_sum = (chosen_log_probs_policy.detach() * chosen_answer_mask).sum().item()
                chosen_policy_tokens = chosen_answer_mask.sum().item()
                rejected_policy_sum = (rejected_log_probs_policy.detach() * rejected_answer_mask).sum().item()
                rejected_policy_tokens = rejected_answer_mask.sum().item()
                chosen_ref_sum = (chosen_log_probs_ref.detach() * chosen_answer_mask).sum().item()
                chosen_ref_tokens = chosen_answer_mask.sum().item()
                rejected_ref_sum = (rejected_log_probs_ref.detach() * rejected_answer_mask).sum().item()
                rejected_ref_tokens = rejected_answer_mask.sum().item()
                kl_sum = (
                    (response_log_probs_policy.detach() - response_log_probs_ref.detach())
                    * response_answer_mask
                ).sum().item()
                kl_tokens = response_answer_mask.sum().item()
                eos_count = (response_ids == tokenizer.eos_token_id).any(dim=1).float().sum().item()
                trunc_count = (response_attn.sum(dim=1) >= MAX_INPUT_TOKENS).float().sum().item()
                seq_count = float(response_ids.size(0))
                token_count = response_attn.sum().item()
            running_policy_chosen_sum += chosen_policy_sum
            running_policy_chosen_tokens += chosen_policy_tokens
            running_policy_rejected_sum += rejected_policy_sum
            running_policy_rejected_tokens += rejected_policy_tokens
            running_ref_chosen_sum += chosen_ref_sum
            running_ref_chosen_tokens += chosen_ref_tokens
            running_ref_rejected_sum += rejected_ref_sum
            running_ref_rejected_tokens += rejected_ref_tokens
            running_kl_sum += kl_sum
            running_kl_tokens += kl_tokens
            running_eos += eos_count
            running_trunc += trunc_count
            running_sequences += seq_count
            running_tokens += token_count
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

        running_loss += loss.item()
        is_update_step = ((step_idx + 1) % ACCUMULATION_STEPS == 0) or ((step_idx + 1) == len(pbar))
        log_payload = None
        if is_update_step:
            # Backprop
            # pre-backprop cleanup. adding set_to_none is more memory efficiency
            global_step += 1
            # For MPS gradient stability
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            for param in lora_params:
                if param.grad is not None:
                    param.grad.data = param.grad.data.contiguous()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            chosen_mean = chosen_rewards.mean().item()
            rejected_mean = rejected_rewards.mean().item()
            reward_gap = (chosen_rewards - rejected_rewards).mean().item()
            pref_acc = (chosen_rewards > rejected_rewards).float().mean().item()
            avg_len = (
                (chosen_answer_mask.sum() + rejected_answer_mask.sum())
                / max(len(batch) * 2, 1)
            ).item()
            step_loss = running_loss / ACCUMULATION_STEPS
            log_payload = {
                "step_loss": step_loss,
                "chosen_mean": chosen_mean,
                "rejected_mean": rejected_mean,
                "reward_gap": reward_gap,
                "pref_acc": pref_acc,
                "avg_len": avg_len,
                "grad_norm": grad_norm,
            }
            running_eval_loss_sum += step_loss
            running_eval_steps += 1.0
            running_loss = 0.0
        # --- THE DEEP CLEAN BLOCK ---
        # Tensors from the Forward Passes (The biggest memory hogs)
        del response_ids, response_attn, response_answer_mask, response_targets
        del ref_response, response_logits_ref, response_shifted_log_probs_ref, response_log_probs_ref
        del policy_response, response_logits_policy, response_shifted_log_probs_policy, response_log_probs_policy
        del chosen_log_probs_ref, rejected_log_probs_ref, chosen_log_probs_policy, rejected_log_probs_policy
        del chosen_answer_mask, rejected_answer_mask
        if IS_MPS:
            empty_cache(DEVICE)
        # periodically evaluate
        if is_update_step and global_step % EVAL_EVERY == 0:
            gc.collect()
            avg_loss = running_eval_loss_sum / max(running_eval_steps, 1.0)
            print(f"[step {global_step}] avg_loss={avg_loss:.4f}")
            running_eval_loss_sum = 0.0
            running_eval_steps = 0.0

            eval_prompt = "A car travels at 62 km/h for 2 hours, then twice that speed for 3 hours. Compute total distance in km."
            eval_inputs = tokenizer(eval_prompt + PROMPT, return_tensors="pt").to(DEVICE)
            model.eval()
            with torch.no_grad():
                out = model.generate(
                    **eval_inputs,
                    max_new_tokens=MAX_INPUT_TOKENS,
                    do_sample=False,
                )
                print(f"[eval] {eval_prompt} -> {tokenizer.decode(out[0], skip_special_tokens=True)}")
            model.train()
            del eval_inputs, out
            if IS_MPS:
                empty_cache(DEVICE)
        t_end = time.perf_counter()
        step_time = t_end - t_start
        running_time += step_time
        if IS_CUDA:
            mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            running_mem_alloc = max(running_mem_alloc, mem_alloc)
            running_mem_reserved = max(running_mem_reserved, mem_reserved)
        if log_payload is not None:
            kl_mean = running_kl_sum / max(running_kl_tokens, 1.0)
            policy_chosen_logp = running_policy_chosen_sum / max(running_policy_chosen_tokens, 1.0)
            policy_rejected_logp = running_policy_rejected_sum / max(running_policy_rejected_tokens, 1.0)
            ref_chosen_logp = running_ref_chosen_sum / max(running_ref_chosen_tokens, 1.0)
            ref_rejected_logp = running_ref_rejected_sum / max(running_ref_rejected_tokens, 1.0)
            eos_rate = running_eos / max(running_sequences, 1.0)
            trunc_rate = running_trunc / max(running_sequences, 1.0)
            tok_per_s = running_tokens / max(running_time, 1e-6)
            if IS_CUDA:
                mem_str = f"{running_mem_alloc:.2f}/{running_mem_reserved:.2f}"
            else:
                mem_str = "n/a"
            if global_step % LOG_EVERY == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[step {global_step}] loss={log_payload['step_loss']:.4f} "
                    f"rewards(c/r)={log_payload['chosen_mean']:.2f}/{log_payload['rejected_mean']:.2f} "
                    f"pref_acc={log_payload['pref_acc']:.2f} gap={log_payload['reward_gap']:.2f} "
                    f"avg_len={log_payload['avg_len']:.1f} grad_norm={log_payload['grad_norm']:.3f} "
                    f"lr={lr} beta={BETA}"
                )
                print(
                    f"[metrics] kl={kl_mean:.4f} logp_p(c/r)={policy_chosen_logp:.3f}/{policy_rejected_logp:.3f} "
                    f"logp_r(c/r)={ref_chosen_logp:.3f}/{ref_rejected_logp:.3f} "
                    f"eos={eos_rate:.2f} trunc={trunc_rate:.2f} tok/s={tok_per_s:.1f} "
                    f"mem_gb={mem_str}"
                )
            else:
                print(
                    f"[step {global_step}] loss={log_payload['step_loss']:.4f} "
                    f"rewards(c/r)={log_payload['chosen_mean']:.2f}/{log_payload['rejected_mean']:.2f}"
                )
            running_kl_sum = 0.0
            running_kl_tokens = 0.0
            running_policy_chosen_sum = 0.0
            running_policy_chosen_tokens = 0.0
            running_policy_rejected_sum = 0.0
            running_policy_rejected_tokens = 0.0
            running_ref_chosen_sum = 0.0
            running_ref_chosen_tokens = 0.0
            running_ref_rejected_sum = 0.0
            running_ref_rejected_tokens = 0.0
            running_eos = 0.0
            running_trunc = 0.0
            running_sequences = 0.0
            running_tokens = 0.0
            running_time = 0.0
            running_mem_alloc = 0.0
            running_mem_reserved = 0.0
        print(f"[timing] sample processed in {step_time:.2f}s")
    save_lora_checkpoint(model, optimizer, epoch, global_step, CHECKPOINT_DIR, prefix="dpo")
    print(f"==end-of-epoch {epoch}==")
