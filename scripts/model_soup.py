from pathlib import Path
import torch

from grpo.device import get_default_device
from grpo.lora import ModelAdapterWrapper, apply_lora_to_model, freeze_non_lora_params, get_lora_parameters
from grpo.utils import load_model, save_lora_checkpoint

DEVICE = get_default_device()
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gemma-2-2b"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "gemma-2-2b-checkpoints"
LORA_CKPT_2 = Path("../gemma-2-2b-checkpoints/grpo_20260115_epoch2_step200.pt")
LORA_CKPT_4 = Path("../gemma-2-2b-checkpoints/grpo_20260115_epoch4_step400.pt")
LORA_CKPT_9 = Path("../gemma-2-2b-checkpoints/grpo_20260115_epoch9_step900.pt")

def _load_lora_state(ckpt_path: Path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        print(f"Error loading checkpoint {ckpt_path}: {exc}")
        raise
    if "lora_state_dict" not in ckpt:
        print(f"Error loading checkpoint {ckpt_path}: missing lora_state_dict")
        raise KeyError(f"Missing lora_state_dict in {ckpt_path}")
    return ckpt["lora_state_dict"]

def blend_checkpoints(ckpt_paths, weights):
    # Load the first checkpoint as the base
    blended_state = _load_lora_state(ckpt_paths[0])
    
    # Scale base by its weight
    for key in blended_state:
        blended_state[key] *= weights[0]
        
    # Add subsequent checkpoints
    for i in range(1, len(ckpt_paths)):
        next_ckpt = _load_lora_state(ckpt_paths[i])
        for key in blended_state:
            blended_state[key] += next_ckpt[key] * weights[i]
            
    return blended_state

# recommended blend: 50% Epoch 2, 25% Epoch 4, 25% Epoch 9
best_lora = blend_checkpoints(
    [LORA_CKPT_2, LORA_CKPT_4, LORA_CKPT_9], 
    [0.50, 0.25, 0.25]
)

# Load base model with LoRA adapters so we can save a standard LoRA checkpoint.
_, model = load_model(str(MODEL_PATH), device=DEVICE)
model = apply_lora_to_model(
    model,
    r=16,
    alpha=32,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    dropout=0.05,
)
model = ModelAdapterWrapper(model)
freeze_non_lora_params(model)
try:
    missing = model.load_state_dict(best_lora, strict=False)
except Exception as exc:
    print(f"Error loading blended LoRA state: {exc}")
    raise
if missing.missing_keys or missing.unexpected_keys:
    print(f"Warning: missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
if DEVICE.type == "cuda":
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(get_lora_parameters(model), lr=1e-5)
    except ImportError:
        optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=1e-5)
else:
    optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=1e-5)
save_lora_checkpoint(model, optimizer, epoch=0, global_step=0, checkpoint_dir=CHECKPOINT_DIR, prefix="soup")
