import torch

def get_tokens_and_masks(prompt_lens, responses, tokenizer, device, max_length=None):
    def _encode(texts):
        kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        if max_length is not None:
            kwargs["max_length"] = max_length
        return tokenizer(texts, **kwargs)

    response_enc = _encode(responses)
    chosen_ids = response_enc["input_ids"].to(device)  # [B, T]
    chosen_attn = response_enc["attention_mask"].to(device)
    chosen_lens = chosen_attn.sum(dim=1)

    seq_len = chosen_ids.size(1)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    eos_candidates = (
        (chosen_ids == tokenizer.eos_token_id)
        & (positions >= prompt_lens.unsqueeze(1))
        & (positions < chosen_lens.unsqueeze(1))
    )
    has_eos = eos_candidates.any(dim=1)
    eos_pos = torch.where(
        has_eos,
        eos_candidates.float().argmax(dim=1),
        chosen_lens - 1,
    )

    arange = torch.arange(seq_len - 1, device=device).unsqueeze(0)
    start = (prompt_lens - 1).clamp(min=0).unsqueeze(1)
    end = eos_pos.unsqueeze(1)
    answer_mask = ((arange >= start) & (arange < end)).float()
    return chosen_ids, chosen_attn, answer_mask
