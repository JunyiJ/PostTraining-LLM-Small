
import torch
import torch.nn as nn
import torch.nn.init as init
from contextlib import contextmanager
from typing import Tuple, Iterable

class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        ref = next(base_model.parameters())
        self.value_layer = nn.Linear(base_model.config.hidden_size, 1, device=ref.device, dtype=ref.dtype)
        init.normal_(self.value_layer.weight, mean=0.0, std=0.02)
        init.zeros_(self.value_layer.bias)

    def forward(self, *args, return_values=True, **kwargs):
        # Preserve user args/kwargs while ensuring hidden states for value head
        if "return_values" in kwargs:
            return_values = kwargs.pop("return_values")
        kwargs.setdefault("output_hidden_states", True)
        outs = self.base_model(*args, **kwargs)
        if not return_values:
            return outs
        hidden = outs.hidden_states[-1]
        values = self.value_layer(hidden).squeeze(-1)
        return outs, values

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        # Temporarily disable LoRA adapters (used for reference/pass-through runs)
        lora_modules = []
        for module in self.modules():
            if isinstance(module, LoRALinear):
                lora_modules.append((module, module._lora_enabled))
                module._lora_enabled = False
        try:
            yield
        finally:
            for module, prev in lora_modules:
                module._lora_enabled = prev

class LoRALinear(nn.Module):
    """
    LoRA wrapper around an nn.Linear layer.
    Base weights are frozen; low-rank A/B matrices are trainable.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()

        self.base = base_layer
        # Freeze the base layer weights immediately
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout)
        self._lora_enabled = True

        # Low-rank adapters
        self.lora_A = nn.Linear(self.base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.base.out_features, bias=False)
        init.kaiming_normal_(self.lora_A.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        base_out = self.base(x)
        if not self._lora_enabled:
            return base_out
        lora_out = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    target_modules=("q_proj", "v_proj", "k_proj", "o_proj"),
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

            # ---- CASE 1: child is a Linear that should be LoRA-wrapped ----
            if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
                setattr(module, name,
                        LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
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
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True
        if isinstance(module, Critic):
            module.value_layer.weight.requires_grad = True
            module.value_layer.bias.requires_grad = True

def get_optimizer_params(model, lora_lr, critic_lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    
    # We create four groups: (Lora + Decay), (Lora + No Decay), (Critic + Decay), (Critic + No Decay)
    optimizer_grouped_parameters = [
        # LORA Group
        {
            "params": [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lora_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lora_lr,
        },
        # CRITIC Group
        {
            "params": [p for n, p in model.named_parameters() if "value_layer" in n and p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": critic_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "value_layer" in n and p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": critic_lr,
        },
    ]
    return optimizer_grouped_parameters
