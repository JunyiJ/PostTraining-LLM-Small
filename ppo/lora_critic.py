import torch.nn as nn
import torch.nn.init as init
from typing import Iterable, Tuple

import torch

class Critic(nn.Module):
    """
    LM based critic head for PPO.
    Using a sidecar method for the implementation so only the hidden layer is shared
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        print(f"base_model.config.hidden_size is {base_model.config.hidden_size}")
        self.value_layer = nn.Linear(base_model.config.hidden_size, 1)
        # Align dtype/device with base model to avoid mixed-type matmul on MPS
        ref_param = next(base_model.parameters())
        self.value_layer.to(device=ref_param.device, dtype=ref_param.dtype)
        init.normal_(self.value_layer.weight, mean=0.0, std=0.02)
        init.constant_(self.value_layer.bias, 0)

    def forward(self, input_ids, attention_mask=None, return_values=True):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        if not return_values:
            return outputs.logits
        x = outputs.hidden_states[-1]
        values = self.value_layer(x).squeeze(-1)
        return outputs.logits, values

    def generate(self, *args, **kwargs):
        """Delegate generation to the underlying base_model."""
        return self.base_model.generate(*args, **kwargs)

class LoRALinear(nn.Module):
    """
    LoRA wrapper around an nn.Linear layer.
    Base weights are frozen; low-rank A/B matrices are trainable.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
        self.base_layer.weight.requires_grad = False
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha * 1.0 / r
        self.dropout = nn.Dropout(p=dropout)

        self.A = nn.Linear(self.in_features, self.r, bias=False)
        self.B = nn.Linear(self.r, self.out_features, bias=False)
        init.kaiming_normal_(self.A.weight, a=0.0, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.B.weight)
        
    def forward(self, x):
        old = self.base_layer(x)
        delta = self.B(self.dropout(self.A(x)))
        return old + self.scale * delta

def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    target_modules=("q_proj", "v_proj", "k_proj", "o_proj"),
    critic_module='lm_head',
    dropout: float = 0.0,
) -> nn.Module:
    """
    Recursively replace target Linear layers with LoRALinear.
    Works properly with Gemma, Llama, Mistral, Qwen, etc.
    """

    # Convert to tuple of strings
    target_modules = tuple(target_modules)

    # ---- recursive function ----
    def replace_recursive(module: nn.Module):

        for name, child in list(module.named_children()):

            # ---- CASE 1: LoRA-wrap target linears ----
            if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                continue

            # ---- CASE 2: recursively search deeper ----
            replace_recursive(child)

    # start recursion
    replace_recursive(model)
    return model


def freeze_non_lora_critic_params(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapter weights."""
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.weight.requires_grad = True
            module.B.weight.requires_grad = True
        if isinstance(module, Critic):
            for p in module.value_layer.parameters():
                p.requires_grad = True

def get_lora_critic_parameters(model):
    """
    Gather only LoRA A and B matrix params.
    Base model params remain frozen.
    """
    return [p for _, p in model.named_parameters() if p.requires_grad]
