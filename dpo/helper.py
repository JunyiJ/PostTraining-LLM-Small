import torch
import psutil

def get_tokens_and_masks(prompts, responses, tokenizer, device, max_length=None):
    """
    Please make sure the batch is right padded
    """
    def _encode(texts):
        kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        if max_length is not None:
            kwargs["max_length"] = max_length
        return tokenizer(texts, **kwargs)

    prompt_enc = _encode(prompts)
    prompt_attn = prompt_enc["attention_mask"].to(device)
    prompt_lens = prompt_attn.sum(dim=1)
    response_prompt_lens = prompt_lens.repeat(2)

    response_enc = _encode(responses)
    response_ids = response_enc["input_ids"].to(device)  # [B, T]
    response_attn = response_enc["attention_mask"].to(device)
    response_lens = response_attn.sum(dim=1)

    seq_len = response_ids.size(1)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    eos_candidates = (
        (response_ids == tokenizer.eos_token_id)
        & (positions >= response_prompt_lens.unsqueeze(1))
        & (positions < response_lens.unsqueeze(1))
    )
    has_eos = eos_candidates.any(dim=1)
    eos_pos = torch.where(
        has_eos,
        eos_candidates.float().argmax(dim=1),
        response_lens - 1,
    )

    arange = torch.arange(seq_len - 1, device=device).unsqueeze(0)
    start = (response_prompt_lens - 1).clamp(min=0).unsqueeze(1)
    end = eos_pos.unsqueeze(1)
    answer_mask = ((arange >= start) & (arange < end)).float()
    return response_ids, response_attn, answer_mask

def check_memory_health():
    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    # Color coding the output for visibility
    color = "\033[93m" if vmem.percent > 85 else "\033[92m"
    reset = "\033[0m"
    print(f"{color}📊 [System Health] RAM: {vmem.percent}% | Swap Used: {swap.used / 1e9:.2f} GB{reset}")

def save_lora_checkpoint(model, optimizer, epoch, global_step, checkpoint_dir):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "lora_state_dict": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer_state_dict": optimizer.state_dict(),
    }
    ckpt_path = checkpoint_dir / f"dpo_lora_epoch{epoch}_step{global_step}.pt"
    torch.save(state, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
