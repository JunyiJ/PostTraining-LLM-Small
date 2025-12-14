
import torch.nn as nn
import torch.nn.init as init
from typing import Iterable, Tuple

import torch

class LoRALinear(nn.Module):
    """
    LoRA wrapper around an nn.Linear layer.
    Base weights are frozen; low-rank A/B matrices are trainable.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()

        self.base = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout)

        self.weight = nn.Parameter(
            base_layer.weight.detach().clone(),
            requires_grad=False
        )
        self.bias = None
        if base_layer.bias is not None:
            self.bias = nn.Parameter(
                base_layer.bias.detach().clone(),
                requires_grad=False
            )

        # Low-rank adapters
        self.A = nn.Linear(self.base.in_features, r, bias=False)
        self.B = nn.Linear(r, self.base.out_features, bias=False)
        init.kaiming_normal_(self.A.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        init.zeros_(self.B.weight)
        
    def forward(self, x):
        base_out = x @ self.weight.T
        lora_out = self.B(self.dropout(self.A(x))) * self.scaling
        if self.bias is not None:
            return self.bias + base_out +lora_out
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


def freeze_non_lora_params(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapter weights."""
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.weight.requires_grad = True
            module.B.weight.requires_grad = True

def get_lora_parameters(model):
    """
    Gather only LoRA A and B matrix params.
    Base model params remain frozen.
    """
    return [p for _, p in model.named_parameters() if p.requires_grad]
