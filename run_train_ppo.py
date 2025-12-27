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
from ppo.ppo_sampler import sample_batch
from ppo.lora_critic import Critic, apply_lora_to_model, freeze_non_lora_critic_params, get_lora_critic_parameters

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
NUM_SAMPLES_PER_PROMPT = 1
NUM_TRAINING_DATA = 2
BATCH_SIZE = 2
NUM_EPOCHS = 1
EVAL_EVERY = 1
SAMPLING_TEMPERATURE = 0.9
MAX_INPUT_TOKENS = 150
MAX_NEW_TOKENS = 400
KL_COEF = 0.1
DEVICE = torch.device("mps")
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
if LORA_CKPT and LORA_CKPT.exists():
    ckpt = torch.load(LORA_CKPT, map_location="cpu")
    missing = model.load_state_dict(ckpt.get("lora_state_dict", {}), strict=False)
    print(f"Loaded LoRA checkpoint {LORA_CKPT} (missing/unexpected: {missing})")
else:
    print(f"LoRA checkpoint {LORA_CKPT} not found; training from base model.")
model.to(DEVICE)

lora_params = get_lora_critic_parameters(model)
optimizer = torch.optim.AdamW(lora_params, lr=2e-5)
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

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
    start = (epoch - 1) * NUM_TRAINING_DATA
    end = start + NUM_TRAINING_DATA
    train_samples = test_data[start:end]
    for idx in tqdm(range(0, min(len(test_data), NUM_TRAINING_DATA), BATCH_SIZE),  desc=f"epoch {epoch}", leave=False):
        batch = test_data[idx : idx + BATCH_SIZE]
        questions = [sample["question"] + PROMPT for sample in batch]
        golds = [str(sample["gold_answer"]).strip() for sample in batch]
        print(questions)
        print(golds)
        model.eval()
        with torch.no_grad():
            res = sample_batch(
                model,
                tokenizer,
                questions,
                device=DEVICE,
                dtype=torch.bfloat16,
                temperature=SAMPLING_TEMPERATURE,
                max_input_tokens=MAX_INPUT_TOKENS,
                max_new_tokens=MAX_NEW_TOKENS
            )
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
            with model.disable_adapter():
            ## Second pass (enable gradient) to get reference policy logprobs.
                ref_out, ref_values = model(input_ids=padded_batch_tokens, attention_mask=attention_mask, return_values=True)
                # [B, T_max, vocab]
                logits_ref = ref_out.logits
                values_ref = ref_values[:, :-1]
                # [B, T_max-1, vocab]
                shifted_log_probs_ref = F.log_softmax(logits_ref / SAMPLING_TEMPERATURE, dim=-1)[:, :-1, :]
                # [B, T_max-1]
                log_probs_ref = shifted_log_probs_ref.gather(-1, targets).squeeze(-1)
        model.train()
        with torch.enable_grad():
            # Get logprobs of policy model
            new_out, new_values = model(input_ids=padded_batch_tokens, attention_mask=attention_mask, return_values=True)
            logits_new = new_out.logits[:, :-1, :]   # [B, T-1, V]
            values_new = new_values[:, :-1] # [B, T-1]
            shifted_log_probs_new = F.log_softmax(logits_new / SAMPLING_TEMPERATURE, dim=-1)
            # Gather logprobs of the actually generated tokens
            log_probs_new = shifted_log_probs_new.gather(-1, targets).squeeze(-1)  # [B, T-1]
            # Mask out padded tokens
            masked_log_probs_ref = log_probs_ref * answer_mask
            masked_values_ref = values_ref * answer_mask
            masked_log_probs_new = log_probs_new * answer_mask
            masked_values_new = values_new * answer_mask

            r = torch.exp(masked_log_probs_new - masked_log_probs_ref)
            print(
                {
                    "text": res.get("text"),
                    "prompt_id_length": res.get("prompt_id_length"),
                    "tokens_shape": tuple(res.get("tokens", torch.empty(0)).shape),
                    "attention_mask_shape": tuple(res.get("attention_mask", torch.empty(0)).shape),
                    "truncated": res.get("truncated"),
                    "steps_taken": res.get("steps_taken"),
                }
            )

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
        global_step += 1
        # print(f"[timing] sample processed in {(t_end - t_start):.2f}s")
        # print("\n=== PROFILER ===")
        # print(f"Sampling:          {(t1-t0):.2f}s")
        # print(f"Logprob forward:   {(t3-t2):.2f}s")
        # print(f"Reward + Adv:      {(t5-t4):.2f}s")
        # print(f"Backward:          {(t7-t6):.2f}s")
    save_lora_checkpoint(model, optimizer, epoch, global_step)
    print(f"==end-of-epoch {epoch}==")
