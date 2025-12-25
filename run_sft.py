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
from grpo.sampler import sample_k_parallel
from grpo.advantage import compute_advantage, compute_rank_advantage
from grpo.reward import compute_reward, advanced_cot_reward,refined_advanced_cot_reward
from grpo.lora import apply_lora_to_model, freeze_non_lora_params, get_lora_parameters

# To avoid the known issue of gemma2 x MPS memory allocator bug.
# This hapens because hugging face automatically runs FP16 warmup allocations
# even request fp32 or bfloat16
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TRANSFORMERS_NO_MPS_CACHE_ALLOCATOR"] = "1"

MODEL_PATH = Path(__file__).resolve().parent / "models" / "gemma-2-2b"
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "gsm8k_sft_warmup.jsonl"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "gemma-2-2b-checkpoints"
DEVICE = torch.device("mps")

def save_lora_checkpoint(model, optimizer, epoch, global_step):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "lora_state_dict": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer_state_dict": optimizer.state_dict(),
    }
    ckpt_path = CHECKPOINT_DIR / f"sft_lora_epoch{epoch}_step{global_step}.pt"
    torch.save(state, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

def run_memory_optimized_sft(model, tokenizer, sft_data, optimizer, device="mps", epochs=1, grad_accum_steps=4):
    """
    SFT Warm-up optimized for 16GB RAM.
    - grad_accum_steps=4 simulates a Batch Size of 4 while using memory of 1.
    - labels are masked to focus on the reasoning/answer only.
    """
    model.train()
    print(f"🚀 Starting Memory-Optimized SFT Warm-up...")
    
    # Use bfloat16 for training if supported, else float32
    # Note: float16 can sometimes be unstable on early M1/M2 chips
    train_dtype = torch.bfloat16 

    global_step = 0

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        
        for i, item in enumerate(tqdm(sft_data, desc=f"SFT Epoch {epoch+1}")):
            # 1. Construct parts
            prompt_text = f"{item['question']} Please reason step-by-step, then give: Final answer."
            # Ensure reasoning exists in your JSONL
            completion_text = f" {item.get('reasoning', '')} Final answer: {item['gold_answer']}"
            full_text = prompt_text + completion_text

            # 2. Tokenize and Mask
            enc_full = tokenizer(full_text, truncation=True, max_length=450, return_tensors="pt")
            enc_prompt = tokenizer(prompt_text, truncation=True, max_length=450, return_tensors="pt")
            
            input_ids = enc_full["input_ids"].to(device)
            attention_mask = enc_full["attention_mask"].to(device)
            
            # Create labels and mask the prompt part with -100
            labels = input_ids.clone()
            prompt_len = enc_prompt["input_ids"].size(1)
            labels[:, :prompt_len] = -100 

            # 3. Forward Pass (Wrapped in autocast if using mixed precision)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # Scale loss for gradient accumulation
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            # 4. Step Optimizer every N examples
            if (i + 1) % grad_accum_steps == 0:
                # Clip gradients to prevent 'spikes' that cause logic collapse
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            # 5. AGGRESSIVE CLEANUP
            # This is the secret to avoiding the 3000s stall
            del outputs, loss, input_ids, labels, attention_mask
            if i % 10 == 0:
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        # handle leftover gradients if dataset size not divisible by grad_accum_steps
        if (len(sft_data) % grad_accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        save_lora_checkpoint(model, optimizer, epoch, global_step)

    print("✅ SFT Warm-up Complete.")

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))
# Wrap target linear layers with LoRA adapters
model = apply_lora_to_model(
    model,
    r=16,
    alpha=32,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    dropout=0.05,
)
freeze_non_lora_params(model)
model.to(DEVICE)
lora_params = get_lora_parameters(model)
optimizer = torch.optim.AdamW(lora_params, lr=5e-5)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
sft_data = []
with open(TRAIN_FILE) as f:
    for ln in f:
        ln = ln.strip()
        if not ln:
            continue
        sft_data.append(json.loads(ln))
run_memory_optimized_sft(model, tokenizer, sft_data, optimizer, device=DEVICE, epochs=1, grad_accum_steps=4)
