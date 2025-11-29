from pathlib import Path

import torch
import torch.nn.functional as F

from lora import (
    LoRALinear,
    apply_lora_to_model,
    freeze_non_lora_params,
    get_lora_parameters,
)
from utils import load_model

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gemma-2-2b"
DEVICE = torch.device("mps")

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))

# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         print(name)

# Wrap target linear layers with LoRA adapters
model = apply_lora_to_model(
    model,
    r=8,
    alpha=16,
    target_modules=("q_proj", "v_proj"),
    dropout=0.0,
)
freeze_non_lora_params(model)
model.to(DEVICE)
lora_params = get_lora_parameters(model)

# print("\n=== LoRALinear layers detected ===")
# for name, module in model.named_modules():
#     if isinstance(module, LoRALinear):
#         print(name)
# print("\n")
# print(model.model.layers[0].self_attn)

# Test1: only LoRA params are trainable
print("\n=== Trainable parameters (should ONLY be LoRA A/B) ===")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(name, p.shape)

# Test2: LoRA output should change model output
print("\n=== Testing LoRA influence on output ===")
eval_prompt = "11+123=?"
eval_inputs = tokenizer(eval_prompt, return_tensors="pt").to(DEVICE)
model.eval()
with torch.no_grad():
    out1 = model(**eval_inputs).logits.clone()

# Find the first LoRA-wrapped linear
lora_modules = [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]
if not lora_modules:
    raise RuntimeError("No LoRA modules found; check target_modules names.")

print("Modifying first LoRA A & B matrix by +0.01 ...")
with torch.no_grad():
    lora_modules[0].A.weight += 0.01
    lora_modules[0].B.weight += 0.01
    out2 = model(**eval_inputs).logits.clone()

difference = (out2 - out1).abs().max().item()
print(f"\nMax absolute difference: {difference}")

if difference == 0:
    print("LoRA update NOT affecting output — check forward() logic.")
else:
    print("LoRA update is affecting output.")

# Test3: LoRA can actually learn
print("\n=== Testing LoRA can actually learn ===")
freeze_non_lora_params(model)
model.to(DEVICE)
lora_params = get_lora_parameters(model)
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
train_prompt = "a"
target_text = "bcd"
prompt_ids = tokenizer(train_prompt, add_special_tokens=False).input_ids
target_ids_list = tokenizer(target_text, add_special_tokens=False).input_ids
input_ids = torch.tensor([prompt_ids + target_ids_list], device=DEVICE)
labels = torch.full_like(input_ids, -100)
labels[0, len(prompt_ids) :] = torch.tensor(target_ids_list, device=DEVICE)
train_inputs = {"input_ids": input_ids}

model.eval()
with torch.no_grad():
    out = model(**train_inputs)
    log_probs = out.logits.log_softmax(dim=-1)
    gathered = []
    for idx, tok in enumerate(target_ids_list):
        pos = len(prompt_ids) + idx
        gathered.append(log_probs[0, pos - 1, tok])  # logits at pos-1 predict token at pos
    prob_seq = torch.stack(gathered).sum().exp().item()
    print(f"Pr('{target_text}' | '{train_prompt}') before LoRA fine-tune: {prob_seq:.6f}")

model.train()
with torch.enable_grad():
    for _ in range(20):
        out = model(**train_inputs)
        logits = out.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    out = model(**train_inputs)
    log_probs = out.logits.log_softmax(dim=-1)
    gathered = []
    for idx, tok in enumerate(target_ids_list):
        pos = len(prompt_ids) + idx
        gathered.append(log_probs[0, pos - 1, tok])
    prob_seq = torch.stack(gathered).sum().exp().item()
    print(f"Pr('{target_text}' | '{train_prompt}') after LoRA fine-tune: {prob_seq:.6f}")
