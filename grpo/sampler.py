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
        })
    return samples
