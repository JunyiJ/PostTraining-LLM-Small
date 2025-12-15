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
from grpo.advantage import compute_advantage
from grpo.reward import compute_reward
from grpo.lora import apply_lora_to_model, freeze_non_lora_params, get_lora_parameters

# To avoid the known issue of gemma2 x MPS memory allocator bug.
# This hapens because hugging face automatically runs FP16 warmup allocations
# even request fp32 or bfloat16
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TRANSFORMERS_NO_MPS_CACHE_ALLOCATOR"] = "1"

MODEL_PATH = Path(__file__).resolve().parent / "models" / "gemma-2-2b"
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "math_grpo_200.jsonl"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "gemma-2-2b-checkpoints"
# CHECKPOINT_DIR = Path(__file__).resolve().parent / "Qwen2.5-Math-1.5B-Instruct-checkpoints"
NUM_SAMPLES_PER_PROMPT = 4
NUM_TRAINING_DATA = 50
NUM_EPOCHS = 3
EVAL_EVERY = 25
SAMPLING_TEMPERATURE = 0.8
MAX_NEW_TOKENS = 205
KL_COEF = 0.01
DEVICE = torch.device("mps")

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))
# Wrap target linear layers with LoRA adapters
model = apply_lora_to_model(
    model,
    r=8,
    alpha=16,
    target_modules=("q_proj", "v_proj", "k_proj", "o_proj"),
    dropout=0.05,
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
        question = line['question'] + "\nGive numeric answer with concise reasoning, please avoid long reasoning. "
        # print(question)
        gold_answer = str(line["gold_answer"]).strip()
        # Sample K initial answers and get each answer token's sum_logprob_old.
        model.eval()    # disable dropout
        with torch.no_grad():
            res = sample_k_parallel(
                model,
                tokenizer,
                question,
                k=NUM_SAMPLES_PER_PROMPT,
                device=DEVICE,
                dtype=torch.bfloat16,
                temperature=SAMPLING_TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        # Second pass (enable gradient) to get each answer token's sum_logprob_new.
        model.train()
        with torch.enable_grad():
            B, T = res["tokens"].size()
            prompt_len = res["prompt_id_length"]
            padded_batch_tokens = res["tokens"].to(DEVICE)
            attention_mask = res["attention_mask"].to(DEVICE)
            # First eos after prompt_len
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
            # print("TOKENS:", padded_batch_tokens[0].tolist())
            # print("PROMPT LEN:", prompt_len)
            # print("EOS POS:", eos_pos[0].item())
            # print("ANSWER TOKENS:", padded_batch_tokens[0, prompt_len:eos_pos[0]].tolist())
            # print("ANSWER MASK ROW:", answer_mask[0].nonzero().squeeze().tolist())

            out_new = model(input_ids=padded_batch_tokens, attention_mask=attention_mask)
            # [B, T_max, vocab]
            logits_new = out_new.logits
            # [B, T_max-1, vocab]
            shifted_log_probs_new = F.log_softmax(logits_new / SAMPLING_TEMPERATURE, dim=-1)[:, :-1, :]
            targets = padded_batch_tokens[:, 1:].unsqueeze(-1)
            # [B, T_max-1]
            log_probs_new = shifted_log_probs_new.gather(-1, targets).squeeze(-1)
            # Mask out padded tokens
            masked_log_probs_new = log_probs_new * answer_mask
            # [B]
            sum_token_logprobs_new = masked_log_probs_new.sum(dim=1)
            sum_token_logprobs_old = res["sum_token_logprobs"].to(DEVICE).detach()
            # Calculate rewards
            rewards = [
                compute_reward(
                    question,
                    txt,
                    gold_answer,
                    truncated=tr,
                )
                for txt, tr in zip(res["text"], res["truncated"])
            ]
            # for txt, r in zip(res['text'], rewards):
            #     print(txt)
            #     print(f"reward is {r}")
            # Calculate advantages
            advantages = compute_advantage(rewards, device=DEVICE, dtype=sum_token_logprobs_new.dtype).to(DEVICE)
            # Compute GRPO loss
            log_prob_ratio = sum_token_logprobs_new - sum_token_logprobs_old
            ratio = log_prob_ratio.exp()
            answer_logprobs_old = res["token_logprobs"].to(DEVICE) * answer_mask
            answer_logprobs_new = masked_log_probs_new
            kl_loss = (answer_logprobs_old - answer_logprobs_new).sum(dim=1).mean()
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
