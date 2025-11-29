
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
        for p in self.base.parameters():
            p.requires_grad = False  # freeze base weights/bias

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout)

        # Low-rank adapters
        self.A = nn.Linear(self.base.in_features, r, bias=False)
        self.B = nn.Linear(r, self.base.out_features, bias=False)
        init.kaiming_normal_(self.A.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        init.zeros_(self.B.weight)
        
    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.B(self.dropout(self.A(x))) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
    dropout: float = 0.0,
) -> nn.Module:
    """Recursively wrap target linear submodules with LoRALinear."""

    def _replace(module: nn.Module, parents: Tuple[str, ...] = ()) -> None:
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and child_name in target_modules:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            else:
                _replace(child, (*parents, child_name))

    _replace(model, ())
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
