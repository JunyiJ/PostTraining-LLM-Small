import torch
import torch.nn.functional as F

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

# model.generate(...) Had issues of NaN tensor on MPS
def sample_k_parallel(
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
