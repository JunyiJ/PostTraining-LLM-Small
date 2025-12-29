import torch
import torch.nn.functional as F
import re
import time


FAKE_PAD_ID = -100   # any ID not used by model

import time

def profile_sampling(func):
    """Decorator to measure sampling performance and detect stalls."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        t1 = time.perf_counter()

        duration = t1 - t0
        steps = out.get("steps_taken", None)
        truncated = out.get("truncated", None)

        # Immediately warn if sampling too slow
        if duration > 200:
            print("\n🚨 SAMPLING WARNING")
            print(f"  Time: {duration:.2f}s (too slow)")
            print(f"  Steps taken: {steps}")
            print(f"  Truncated flags: {truncated}")
            print(f"  Prompt: {args[2][:120]}...")
            print("  → Early-stop likely failed OR logits collapsed.")
            print("  → Inspect sampler immediately.\n")

        # Soft warning
        elif duration > 20:
            print(f"⚠️ Sampling slow: {duration:.1f}s, steps={steps}")

        return out
    return wrapper

# Get  log-probs (tokens, log-probs, values from critic)
def sample_batch(
    model,
    tokenizer,
    prompts,
    device="mps",
    dtype=torch.float32,
    max_input_tokens=150,
    max_new_tokens=256,
    enable_grad=False,
    top_p=0.95,
    repetition_penalty=1.05,
    timeout_seconds=500,
):
    t_start_global = time.perf_counter()
    batch_size = len(prompts)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    input_ids = enc["input_ids"].to(device)  # [B, seq_len]
    attn = enc["attention_mask"]
    prompt_lens = attn.sum(dim=1)
    # padded prompt length
    prompt_id_length = input_ids.size(1)
    
    # Pre-allocate tensor for input_ids_k
    total_max_len = prompt_id_length + max_new_tokens
    input_ids_k = torch.full((batch_size, total_max_len), FAKE_PAD_ID, device=device)
    input_ids_k[:, :prompt_id_length] = input_ids
    
    attention_mask_k = torch.zeros((batch_size, total_max_len), dtype=torch.long, device=device)
    attention_mask_k[:, :prompt_id_length] = 1

    values_k = torch.zeros((batch_size, total_max_len), dtype=dtype, device=device)

    sampling_active = torch.ones((batch_size,), dtype=torch.bool, device=device)
    finished_with_eos = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    
    steps_taken = 0
    past_key_values = None
    pad_id_for_model = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    grad_context = torch.enable_grad() if enable_grad else torch.no_grad()

    with torch.no_grad():
        for i in range(max_new_tokens):
            curr_pos = prompt_id_length + i
            # --- WATCHDOG: Skip if batch is taking too long ---
            if (time.perf_counter() - t_start_global) > timeout_seconds:
                print(f"🚨 WATCHDOG TRIGGERED: Sampling took >{timeout_seconds}s. Aborting.")
                sampling_active[:] = False
                break
            # STOP IF ALL SAMPLES DONE
            if not sampling_active.any():
                break

            if i == 0:
                model_inputs = {
                    "input_ids": input_ids_k[:, :prompt_id_length],
                    "attention_mask": attention_mask_k[:, :prompt_id_length],
                    "return_values": False,
                    "use_cache": True,
                }
            else:
                model_inputs = {
                    "input_ids": input_ids_k[:, curr_pos - 1 : curr_pos],
                    "past_key_values": past_key_values,
                    "attention_mask": attention_mask_k[:, :curr_pos],
                    "return_values": False,
                    "use_cache": True,
                }
            out = model(**model_inputs)

            del past_key_values
            past_key_values = out.past_key_values
            # [K, vocab]
            step_logits = out.logits[:, -1, :].to(torch.float32)
            del out
            step_logits[:, tokenizer.eos_token_id] += 1.0
            # # Repetition penalty: penalize tokens that already appeared
            # if repetition_penalty > 1.0:
            #     for batch_idx in range(k):
            #         if sampling_active[batch_idx]:
            #             lookback = input_ids_k[batch_idx, max(0, curr_pos - 64) : curr_pos]
            #             valid_mask = (lookback >= 0) & (lookback < model.config.vocab_size)
            #             unique_tokens = torch.unique(lookback[valid_mask])
            #             step_logits[batch_idx, unique_tokens] /= repetition_penalty

            # TOP-P sampling
            sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            indices_to_remove = mask.scatter(1, sorted_indices, mask)
            step_logits.masked_fill_(indices_to_remove, -1e10)

            
            # --- Check for flat logits (stall detector) ---
            # if torch.allclose(step_logits.max(dim=-1).values,
            #                    step_logits.min(dim=-1).values,
            #                    atol=1e-7):
            #     sampling_active[:] = False
            #     finished_with_eos[:] = True
            #     print("🚨 Flat logits detected — model confused.")
            #     break

            log_probs = F.log_softmax(step_logits, dim=-1)
            probs = torch.exp(log_probs - log_probs.max(dim=-1, keepdim=True).values)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            # if probs.min() < 1e-12:
            #     print(f"⚠️ Extremely small probs detected: min={probs.min().item():.3e}")
            # [K, 1]
            next_token_raw = torch.multinomial(probs, num_samples=1)
            # For finished rows, append pad instead of sampled token to keep shapes aligned
            next_token = torch.where(
                sampling_active.view(-1, 1),
                next_token_raw,
                torch.full_like(next_token_raw, FAKE_PAD_ID),
            )
            # Build attention mask for next step: active rows attend, finished rows are masked
            attention_mask_k[:, curr_pos] = sampling_active.long()
            # Append token for model (replace fake pads with a real id)
            append_for_model = next_token.clone()
            append_for_model[append_for_model == FAKE_PAD_ID] = pad_id_for_model
            input_ids_k[:, curr_pos] = append_for_model.squeeze(-1)

            # # Early stopping
            # if i > 50 and i % 10 == 0:
            #     for j in range(k):
            #         if i == max_new_tokens - 1 and sampling_active[j]:
            #             print(f"⚠️ No early-stop for sample {j}, generated full length.")
            #         if sampling_active[j]:
            #             start_ind = max(prompt_id_length, curr_pos - 30)
            #             tail_text = tokenizer.decode(input_ids_k[j, start_ind:curr_pos+1], skip_special_tokens=True).lower().strip()
            #             # find any numeric candidate in text
            #             marker = "final answer:"
            #             if marker in tail_text.lower():
            #                 parts = re.split(marker, tail_text, flags=re.IGNORECASE)
            #                 after_answer = parts[-1].strip()
            #                 if re.search(r"[-+]?\d[\d,./]*\s*(\n|\.|$)", after_answer):
            #                     # We found a number followed by a terminator or end of string
            #                     # To be safe, let's ensure it's not just a single digit mid-sentence
            #                     if len(after_answer) > 0 and (after_answer[-1] in ['.', '\n'] or i == max_new_tokens - 1):
            #                         print(f"🟢 Controlled stop (sample {j}): {after_answer}")
            #                         sampling_active[j] = False
            #                         finished_with_eos[j] = True

            # if sampling_active.sum() == 0:
            #     print("🟢 All samples terminated — stopping sampler loop")
            #     break

            hit_eos = (next_token_raw.squeeze(-1) == tokenizer.eos_token_id)
            finished_with_eos |= (hit_eos & sampling_active)
            sampling_active &= (~hit_eos)
            steps_taken += 1
    
    # Cleanup
    del past_key_values
    torch.mps.empty_cache()
    
    # Replace fake pads for model consumption; keep track of real padding locations
    tokens_for_model = input_ids_k.clone()[:, :prompt_id_length+steps_taken]
    fake_pad_positions = (tokens_for_model == FAKE_PAD_ID)
    tokens_for_model[fake_pad_positions] = pad_id_for_model
    answer_tokens = tokens_for_model[:, prompt_id_length:prompt_id_length + steps_taken]
    texts = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
    return {
        "text": texts,  # list length K
        "prompt_id_length": prompt_id_length,  # scaler
        "tokens": tokens_for_model,  # [K, seq_len]
        "attention_mask": attention_mask_k[:, : prompt_id_length + steps_taken],
        "truncated": (~finished_with_eos).tolist(),  # list of bools
        "steps_taken": steps_taken,
    }

# sample_batch = profile_sampling(sample_batch)