"""
Script to load a pretrained model and do GRPO with math data to fine-tune the model with LoRA
"""
import json
import os
import re
import random
import time
from pathlib import Path

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
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "math_grpo_200.jsonl"
# LORA_CKPT = Path("./gemma-2-2b-checkpoints/lora_epoch4_step160.pt")  # Set to None if training from base
LORA_CKPT = None
CHECKPOINT_DIR = Path(__file__).resolve().parent / "gemma-2-2b-checkpoints"
# CHECKPOINT_DIR = Path(__file__).resolve().parent / "Qwen2.5-Math-1.5B-Instruct-checkpoints"
NUM_SAMPLES_PER_PROMPT = 6
NUM_TRAINING_DATA = 50
NUM_EPOCHS = 1
EVAL_EVERY = 25
SAMPLING_TEMPERATURE = 0.6
MAX_NEW_TOKENS = 256
KL_COEF = 0.2
DEVICE = torch.device("mps")

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))
# Wrap target linear layers with LoRA adapters
model = apply_lora_to_model(
    model,
    r=8,
    alpha=16,
    target_modules=("q_proj", "v_proj"),
    dropout=0.05,
)
freeze_non_lora_params(model)
if LORA_CKPT and LORA_CKPT.exists():
    ckpt = torch.load(LORA_CKPT, map_location="cpu")
    missing = model.load_state_dict(ckpt.get("lora_state_dict", {}), strict=False)
    print(f"Loaded LoRA checkpoint {LORA_CKPT} (missing/unexpected: {missing})")
else:
    print(f"LoRA checkpoint {LORA_CKPT} not found; training from base model.")
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
# random.shuffle(test_data)

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
    start = (epoch - 1) * NUM_TRAINING_DATA
    end = start + NUM_TRAINING_DATA
    train_samples = test_data[start:end]
    for line in tqdm(train_samples, desc=f"epoch {epoch}", leave=False):
        t_start = time.perf_counter()
        question = line['question']
        prompt = " Please reason step-by-step,  then give: Final answer."
        # print(question)
        gold_answer = str(line["gold_answer"]).strip()
        print(question)
        print(f"answer is {gold_answer}")
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
            # Second pass (enable gradient) to get each answer token's sum_logprob_new.
        model.train()
        with torch.enable_grad():
            out_new = model(input_ids=padded_batch_tokens, attention_mask=attention_mask)
            # [B, T_max, vocab]
            logits_new = out_new.logits
            # [B, T_max-1, vocab]
            shifted_log_probs_new = F.log_softmax(logits_new / SAMPLING_TEMPERATURE, dim=-1)[:, :-1, :]
        # [B, T_max-1]
        log_probs_new = shifted_log_probs_new.gather(-1, targets).squeeze(-1)
        # Mask out padded tokens
        masked_log_probs_old = log_probs_old * answer_mask
        masked_log_probs_new = log_probs_new * answer_mask
        sum_token_logprobs_old = masked_log_probs_old.sum(dim=1)
        sum_token_logprobs_new = masked_log_probs_new.sum(dim=1)

        # print("\n--- ALIGNMENT DEBUG START ---")

        # print(f"prompt_len = {prompt_len}")
        # print(f"steps_taken = {res['steps_taken'] if 'steps_taken' in res else 'UNKNOWN'}")
        # print(f"eos_pos = {eos_pos.tolist()}")
        # print(f"answer_mask.sum(dim=1) = {answer_mask.sum(dim=1).tolist()}")

        # # 1. Check mask starts at prompt_len-1
        # mask_starts = answer_mask.argmax(dim=1)
        # print(f"mask_starts = {mask_starts.tolist()} (should be prompt_len-1)")

        # # 2. Check mask ends at eos_pos-1
        # mask_lengths = answer_mask.sum(dim=1)
        # mask_ends = mask_starts + mask_lengths - 1
        # print(f"mask_ends = {mask_ends.tolist()} (should be eos_pos-1)")

        # # 3. Compare OLD logprob sum with masked sum of token_logprobs_old
        # masked_old_manual = masked_log_probs_old.sum(dim=1)
        # print("\nOLD LOGPROB CHECK:")
        # for i in range(min(4, len(sum_token_logprobs_old))):
        #     print(f" sample {i}: sum_old={sum_token_logprobs_old[i].item():.6f}, "
        #         f"manual={masked_old_manual[i].item():.6f}")

        # # 4. Compare NEW logprob sum with masked sum of log_probs_new
        # masked_new_manual = masked_log_probs_new.sum(dim=1)
        # print("\nNEW LOGPROB CHECK:")
        # for i in range(min(4, len(sum_token_logprobs_new))):
        #     print(f" sample {i}: sum_new={sum_token_logprobs_new[i].item():.6f}, "
        #         f"manual={masked_new_manual[i].item():.6f}")

        # # 5. Ensure mask does not overlap prompt tokens
        # print("\nFIRST MASKED TOKEN (SHOULD BE FIRST GENERATED TOKEN):")
        # for i in range(min(4, padded_batch_tokens.size(0))):
        #     idx = mask_starts[i].item()
        #     tok = padded_batch_tokens[i, idx+1].item()
        #     print(f" sample {i}: pos={idx}, token='{tokenizer.decode([tok])}'")

        # print("--- ALIGNMENT DEBUG END ---\n")
        # Calculate rewards
        rewards = [
            refined_advanced_cot_reward(
                txt,
                gold_answer,
                truncated=tr,
            )
            for txt, tr in zip(res["text"], res["truncated"])
        ]
        if global_step % 5 == 0:
            for txt, r, tr in zip(res['text'], rewards, res["truncated"]):
                print(txt)
                print(f"reward is {r}")
                print(f"is result truncated? {tr}")
        # Calculate advantages
        advantages = compute_rank_advantage(rewards, device=DEVICE, dtype=torch.float32)
        advantages = advantages.to(sum_token_logprobs_new.dtype)
        print(f"advantages is {advantages}")
        # Compute GRPO loss
        log_prob_ratio = sum_token_logprobs_new - sum_token_logprobs_old
        ratio = log_prob_ratio.exp()
        kl_loss = (masked_log_probs_old.detach() - masked_log_probs_new).sum(dim=1).mean()
        kl_loss = torch.clamp(kl_loss, 0.0, 5.0)
        grpo_loss = -(advantages * ratio).mean()
        loss = grpo_loss + KL_COEF * kl_loss
        print(f"grpo_loss is {grpo_loss} and kl is {kl_loss}")
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

            eval_prompt = "A car travels at 62 km/h for 2 hours, then twice that speed for 3 hours. Compute total distance in km."
            eval_inputs = tokenizer(eval_prompt, return_tensors="pt").to(DEVICE)
            model.eval()
            with torch.no_grad():
                out = model.generate(
                    **eval_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
                print(f"[eval] {eval_prompt} -> {tokenizer.decode(out[0], skip_special_tokens=True)}")
            model.train()
        t_end = time.perf_counter()
        print(f"[timing] sample processed in {(t_end - t_start):.2f}s")
    save_lora_checkpoint(model, optimizer, epoch, global_step)
    print(f"==end-of-epoch {epoch}==")
