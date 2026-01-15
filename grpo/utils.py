import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from grpo.device import resolve_device, configure_torch_for_device
except ImportError:
    from device import resolve_device, configure_torch_for_device


def load_model(
    model_path="./models/gemma-2-2b",
    device=None,
    use_flash_attn_2=False,
    quantization=None,
    torch_dtype=None,
    device_map=None,
):
    device = resolve_device(device)
    configure_torch_for_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",  # Critical for Qwen model
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if torch_dtype is None:
        if device.type == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device.type == "mps":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

    quantization_mode = str(quantization).lower() if quantization else None
    if quantization_mode == "none":
        quantization_mode = None
    use_bnb = quantization_mode in ("4bit", "8bit")

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if use_flash_attn_2 and device.type == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if use_bnb:
        if device.type != "cuda":
            raise ValueError("8-bit/4-bit quantization requires CUDA.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes/transformers BitsAndBytesConfig is required for quantization."
            ) from exc
        if quantization_mode == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = "auto" if device_map is None else device_map
    else:
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        elif device.type == "mps":
            # Load model to CPU first to avoid MPS FP16 warmup allocations.
            model_kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if not use_bnb:
        model = model.to(device)
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
