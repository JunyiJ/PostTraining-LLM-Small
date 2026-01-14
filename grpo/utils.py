import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from grpo.device import resolve_device, configure_torch_for_device
except ImportError:
    from device import resolve_device, configure_torch_for_device


def load_model(model_path="./models/gemma-2-2b", device=None):
    device = resolve_device(device)
    configure_torch_for_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",  # Critical for Qwen model
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,  # Using float32/bf16 instead of FP16 e MPS FP16 matmul has limited exponent range and MPS has buggy FP16 softmax
        device_map={"": "cpu"},  # Load model to cpu first and later moved to desired device to avoid the hugging face buggy warmup with FP16
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return tokenizer, model


def append_jsonl(path: Path, record: dict) -> None:
    """Append a JSON record to a jsonl file, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def check_memory_health() -> None:
    import psutil

    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    color = "\033[93m" if vmem.percent > 85 else "\033[92m"
    reset = "\033[0m"
    print(f"{color}📊 [System Health] RAM: {vmem.percent}% | Swap Used: {swap.used / 1e9:.2f} GB{reset}")


def _collect_lora_state_dict(model):
    # Prefer the inner model when it avoids a wrapper prefix and no critic head exists.
    target = model
    if hasattr(model, "base_model") and not hasattr(model, "value_layer"):
        target = model.base_model
    return {n: p.detach().cpu() for n, p in target.named_parameters() if p.requires_grad}


def save_lora_checkpoint(model, optimizer, epoch, global_step, checkpoint_dir: Path, prefix: str) -> Path:
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "lora_state_dict": _collect_lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    date_str = time.strftime("%Y%m%d")
    ckpt_path = checkpoint_dir / f"{prefix}_{date_str}_epoch{epoch}_step{global_step}.pt"
    torch.save(state, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path
