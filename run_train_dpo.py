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

from grpo.utils import load_model
from dpo.dpo_loss import dpo_loss
from dpo.helper import check_memory_health, save_lora_checkpoint, get_tokens_and_masks
from dpo.lora import ModelAdapterWrapper, apply_lora_to_model, freeze_non_lora_params, get_lora_parameters

# To avoid the known issue of gemma2 x MPS memory allocator bug.
# This hapens because hugging face automatically runs FP16 warmup allocations
# even request fp32 or bfloat16
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TRANSFORMERS_NO_MPS_CACHE_ALLOCATOR"] = "1"

MODEL_PATH = Path(__file__).resolve().parent / "models" / "gemma-2-2b"
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "gsm8k_dpo_pairs.jsonl"
LORA_CKPT = None
# LORA_CKPT = Path("./gemma-2-2b-checkpoints/sft_lora_epoch0_step200.pt")  # Set to None if training from base

CHECKPOINT_DIR = Path(__file__).resolve().parent / "gemma-2-2b-checkpoints"
# CHECKPOINT_DIR = Path(__file__).resolve().parent / "Qwen2.5-Math-1.5B-Instruct-checkpoints"
NUM_TRAINING_DATA = 100
MICRO_BATCH_SIZE = 2 
# ACCUMULATION_STEPS: How many micro-batches to accumulate before updating weights
# Effective Batch Size = MICRO_BATCH_SIZE * ACCUMULATION_STEPS
ACCUMULATION_STEPS = 5
NUM_EPOCHS = 50
EVAL_EVERY = 10
MAX_INPUT_TOKENS = 412
KL_COEF = 0.1
DEVICE = torch.device("mps")
PROMPT = " Instruction: Solve the math problem. You MUST output the full reasoning process followed by the final answer. Do not ask for confirmation. Do not stop until the answer is reached. "

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))
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
freeze_non_lora_params(model)
if LORA_CKPT and LORA_CKPT.exists():
    ckpt = torch.load(LORA_CKPT, map_location="cpu")
    missing = model.load_state_dict(ckpt.get("lora_state_dict", {}), strict=False)
    print(f"Loaded LoRA checkpoint {LORA_CKPT} (missing/unexpected: {missing})")
else:
    print(f"LoRA checkpoint {LORA_CKPT} not found; training from base model.")
model.to(DEVICE)
lora_params = get_lora_parameters(model)
optimizer = torch.optim.AdamW(lora_params, lr=2e-5)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

global_step = 0
running_loss = 0.0

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
                beta=0.1
            )
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

        running_loss += loss.item()
        is_update_step = ((step_idx + 1) % ACCUMULATION_STEPS == 0) or ((step_idx + 1) == len(pbar))
        if is_update_step:
            # Backprop
            # pre-backprop cleanup. adding set_to_none is more memory efficiency
            global_step += 1
            # For MPS gradient stability
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            for param in lora_params:
                if param.grad is not None:
                    param.grad.data = param.grad.data.contiguous()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print(f"[step {global_step}] loss={running_loss/ACCUMULATION_STEPS:.4f} rewards(c/r)={chosen_rewards.mean().item():.2f}/{rejected_rewards.mean().item():.2f}")
            running_loss = 0.0
        # --- THE DEEP CLEAN BLOCK ---
        # Tensors from the Forward Passes (The biggest memory hogs)
        del response_ids, response_attn, response_answer_mask, response_targets
        del ref_response, response_logits_ref, response_shifted_log_probs_ref, response_log_probs_ref
        del policy_response, response_logits_policy, response_shifted_log_probs_policy, response_log_probs_policy
        del chosen_log_probs_ref, rejected_log_probs_ref, chosen_log_probs_policy, rejected_log_probs_policy
        del chosen_answer_mask, rejected_answer_mask
        torch.mps.empty_cache()
        # periodically evaluate
        if is_update_step and global_step % EVAL_EVERY == 0:
            gc.collect()
            avg_loss = running_loss / EVAL_EVERY
            print(f"[step {global_step}] avg_loss={avg_loss:.4f}")
            running_loss = 0.0

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
            torch.mps.empty_cache()
        t_end = time.perf_counter()
        print(f"[timing] sample processed in {(t_end - t_start):.2f}s")
    save_lora_checkpoint(model, optimizer, epoch, global_step, CHECKPOINT_DIR)
    print(f"==end-of-epoch {epoch}==")
