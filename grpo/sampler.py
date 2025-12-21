import torch
import torch.nn.functional as F


FAKE_PAD_ID = -100   # any ID not used by model

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
):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # [1, seq_len]
    prompt_id_length = input_ids.size(1)
    # [K, seq_len]
    input_ids_k = input_ids.repeat(k, 1)
    attention_mask_k = torch.ones_like(input_ids_k, dtype=torch.long, device=device)
    sampling_active = torch.ones((k,), dtype=torch.bool, device=device)
    finished_with_eos = torch.zeros((k,), dtype=torch.bool, device=device)
    steps_taken = 0
    past_key_values = None
    pad_id_for_model = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    grad_context = torch.enable_grad() if enable_grad else torch.no_grad()
    with grad_context:
        for _ in range(max_new_tokens):
            if past_key_values is None:
                model_inputs = {
                    "input_ids": input_ids_k,
                    "attention_mask": attention_mask_k,
                    "use_cache": True,
                }
            else:
                last_token = input_ids_k[:, -1:].clone()
                last_token[last_token == FAKE_PAD_ID] = pad_id_for_model
                model_inputs = {
                    "input_ids": last_token,
                    "attention_mask": attention_mask_k,
                    "use_cache": True,
                    "past_key_values": past_key_values,
                }
            out = model(**model_inputs)
            past_key_values = out.past_key_values
            # [K, vocab]
            step_logits = out.logits[:, -1, :]
            log_probs = F.log_softmax(step_logits / temperature, dim=-1)
            probs = log_probs.exp()
            # [K, 1]
            next_token_raw = torch.multinomial(probs, num_samples=1)
            # For finished rows, append pad instead of sampled token to keep shapes aligned
            if tokenizer.pad_token_id is not None:
                next_token = torch.where(
                    sampling_active.view(-1, 1),
                    next_token_raw,
                    torch.full_like(next_token_raw, FAKE_PAD_ID),
                )
            else:
                next_token = next_token_raw
            # Build attention mask for next step: active rows attend, finished rows are masked
            attention_mask_k = torch.cat(
                [attention_mask_k, torch.ones((k,1), dtype=torch.long, device=device)], dim=1
            )
            # Append token for model (replace fake pads with a real id)
            append_for_model = next_token.clone()
            append_for_model[append_for_model == FAKE_PAD_ID] = pad_id_for_model
            input_ids_k = torch.cat([input_ids_k, append_for_model], dim=1)  # [K, seq_len+step]
            hit_eos = (next_token_raw.squeeze(-1) == tokenizer.eos_token_id)
            finished_with_eos = finished_with_eos | (hit_eos & sampling_active)
            sampling_active = sampling_active & (~hit_eos)
            steps_taken += 1
            if steps_taken >= max_new_tokens or sampling_active.sum() == 0:
                break
    
    # Replace fake pads for model consumption; keep track of real padding locations
    tokens_for_model = input_ids_k.clone()
    fake_pad_positions = (tokens_for_model == FAKE_PAD_ID)
    tokens_for_model[fake_pad_positions] = pad_id_for_model
    answer_tokens = tokens_for_model[:, prompt_id_length:]
    texts = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
    # Attention mask should ignore only the fake pads (not EOS, even if pad==eos)
    attention_mask = torch.ones_like(tokens_for_model, dtype=torch.long)
    attention_mask[fake_pad_positions] = 0
    return {
        "text": texts,  # list length K
        "prompt_id_length": prompt_id_length,  # scaler
        "tokens": tokens_for_model,  # [K, seq_len]
        "attention_mask": attention_mask,  # [K, seq_len]
        "truncated": (~finished_with_eos).tolist(),  # list of bools
        "steps_taken": steps_taken,
    }

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
