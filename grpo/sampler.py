import torch
import torch.nn.functional as F

def sample_k(
    model,
    tokenizer,
    prompt,
    k,
    temperature=1.0,
    max_new_tokens=256,
    store_logits=True,
    enable_grad=False,
):
    samples = []
    enc = tokenizer(prompt, return_tensors="pt").to("mps")
    for _ in range(k):
        input_ids = enc.input_ids.clone()
        collected_logits = []
        collected_probs = []
        grad_context = torch.enable_grad() if enable_grad else torch.no_grad()
        with grad_context:
            for _ in range(max_new_tokens):
                out = model(input_ids=input_ids)
                step_logits = out.logits[:, -1, :]
                probs = F.softmax(step_logits / temperature, dim=-1)
                if store_logits:
                    if enable_grad:
                        collected_logits.append(step_logits)
                        collected_probs.append(probs)
                    else:
                        # Detach and move off MPS to avoid holding GPU memory
                        collected_logits.append(step_logits.detach().cpu())
                        collected_probs.append(probs.detach().cpu())
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        samples.append({
            "text": text,
            "logits": collected_logits,
            "probs": collected_probs
        })
    return samples
