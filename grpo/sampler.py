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


def sample_k(
    model,
    tokenizer,
    prompt,
    k,
    temperature=1.0,
    max_new_tokens=256,
    enable_grad=False,
):
    samples = []
    enc = tokenizer(prompt, return_tensors="pt").to("mps")
    for _ in range(k):
        input_ids = enc.input_ids.clone()
        prompt_id_length = len(input_ids[0])
        collected_token_logprobs = []
        truncated = True
        grad_context = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_context:
            for _ in range(max_new_tokens):
                out = model(input_ids=input_ids)
                step_logits = out.logits[:, -1, :]
                log_probs = F.log_softmax(step_logits / temperature, dim=-1)
                probs = log_probs.exp()
                next_token = torch.multinomial(probs, num_samples=1)
                token_log_prob = log_probs.gather(-1, next_token)  # [batch, 1]
                target_token_log_prob = token_log_prob if enable_grad else token_log_prob.detach().cpu()
                collected_token_logprobs.append(target_token_log_prob)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    truncated = False
                    break
        token_logprob_tensor = torch.cat(collected_token_logprobs, dim=1)  # [batch, steps]
        sum_token_logprobs = token_logprob_tensor.sum(dim=1)              # [batch]
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        samples.append({
            "text": text,
            "prompt_id_length": prompt_id_length,
            "tokens": input_ids,
            "token_logprobs": token_logprob_tensor,      # [batch, steps] or None
            "sum_token_logprobs": sum_token_logprobs,    # [batch] or None
            "truncated": truncated,
        })
    return samples

def sample_k_parallel(
    model,
    tokenizer,
    prompt,
    k,
    device="mps",
    dtype=torch.float32,
    temperature=1.0,
    max_new_tokens=256,
    enable_grad=False,
    top_p=0.95,
    repetition_penalty=1.05,
):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # [1, seq_len]
    prompt_id_length = input_ids.size(1)
    
    # Pre-allocate tensor for input_ids_k
    total_max_len = prompt_id_length + max_new_tokens
    input_ids_k = torch.full((k, total_max_len), FAKE_PAD_ID, device=device)
    input_ids_k[:, :prompt_id_length] = input_ids.repeat(k, 1)
    
    attention_mask_k = torch.zeros((k, total_max_len), dtype=torch.long, device=device)
    attention_mask_k[:, :prompt_id_length] = 1

    sampling_active = torch.ones((k,), dtype=torch.bool, device=device)
    finished_with_eos = torch.zeros((k,), dtype=torch.bool, device=device)
    
    steps_taken = 0
    past_key_values = None
    pad_id_for_model = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    grad_context = torch.enable_grad() if enable_grad else torch.no_grad()

    with torch.no_grad():
        for i in range(max_new_tokens):
            curr_pos = prompt_id_length + i
            # STOP IF ALL SAMPLES DONE
            if i % 16 == 0:
                if not sampling_active.any():
                    break

            if curr_pos > 200:
                print(f"⚠️ KV cache getting large: {curr_pos}")
            if i == 0:
                model_inputs = {
                    "input_ids": input_ids_k[:, :prompt_id_length],
                    "attention_mask": attention_mask_k[:, :prompt_id_length],
                    "use_cache": True,
                }
            else:
                model_inputs = {
                    "input_ids": input_ids_k[:, curr_pos - 1 : curr_pos],
                    "past_key_values": past_key_values,
                    "attention_mask": attention_mask_k[:, :curr_pos],
                    "use_cache": True,
                }
            out = model(**model_inputs)
            del past_key_values
            past_key_values = out.past_key_values
            # [K, vocab]
            step_logits = out.logits[:, -1, :] / max(temperature, 1e-5)
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

            # # TOP-P sampling
            # sorted_logits, sorted_indices = torch.sort(step_logits, descending=True)
            # cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # mask = cumulative_probs > top_p
            # mask[..., 1:] = mask[..., :-1].clone()
            # mask[..., 0] = False
            # indices_to_remove = mask.scatter(1, sorted_indices, mask)
            # step_logits = step_logits.masked_fill(indices_to_remove, -1e10)

            
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

sample_k_parallel = profile_sampling(sample_k_parallel)


# model.generate(...) Had issues of NaN tensor on MPS with gemma-2-2b
def sample_k_generate(
    model,
    tokenizer,
    prompt,
    k,
    temperature=1.0,
    max_new_tokens=256,
    enable_grad=False,
):
    enc = tokenizer(prompt, return_tensors="pt").to("mps")
    # ==debug==
    with torch.no_grad():
        out_logits = model(**enc).logits
        print("logits stats:", out_logits.min(), out_logits.max(), out_logits.mean())
        print("any nan:", torch.isnan(out_logits).any())
    print(prompt)
    # ==
    prompt_id_length = enc.input_ids.size(1)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    # ensure generation config has pad/eos set to avoid NaNs in sampling
    model.generation_config.pad_token_id = pad_id
    model.generation_config.eos_token_id = eos_id
    grad_context = torch.enable_grad() if enable_grad else torch.no_grad()
    with grad_context:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature),
            num_return_sequences=k,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=False
        )

    # sequences: [batch*num_return_sequences, seq_len]
    sequences = out.sequences
    attention_mask = (sequences != pad_id).long()
    # scores: list length num_steps, each [batch*num_return_sequences, vocab]
    num_steps = len(out.scores)
    logits = torch.stack(out.scores, dim=1)  # [B*k, num_steps, vocab]
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log-probs of generated tokens only (after prompt)
    gen_tokens = sequences[:, prompt_id_length : prompt_id_length + num_steps]
    token_log_probs = log_probs.gather(-1, gen_tokens.unsqueeze(-1)).squeeze(-1)  # [B*k, num_steps]
    # Sum over generated tokens
    sum_token_logprobs = token_log_probs.sum(dim=1)
    if not enable_grad:
        sequences = sequences.cpu()
        attention_mask = attention_mask.cpu()
        token_log_probs = token_log_probs.detach().cpu()
        sum_token_logprobs = sum_token_logprobs.detach().cpu()

    # Truncation: no EOS and generated max_new_tokens
    truncated_flags = []
    for seq in sequences:
        seq_gen = seq[prompt_id_length:]
        truncated_flags.append((eos_id is not None and eos_id not in seq_gen) and (seq_gen.numel() >= max_new_tokens))

    return {
        "tokens": sequences,  # [k, seq_length]
        "attention_mask": attention_mask,  # [k, seq_length]
        "prompt_id_length": prompt_id_length,  #scaler
        "sum_token_logprobs": sum_token_logprobs, #[k]
        "truncated": truncated_flags,  # list of length k
        "text": tokenizer.batch_decode(sequences, skip_special_tokens=True),
    }
