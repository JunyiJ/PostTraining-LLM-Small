
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Skeleton for LoRA-wrapped Linear layer.
    You will fill in:
      - how to freeze the base weight
      - how to create LoRA A and B matrices
      - how to combine base layer output with LoRA output
    """
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        
        # === 1. Save base frozen weight ===
        self.base = base_layer          # nn.Linear
        # freeze base weights
        
        # === 2. Store LoRA hyperparams ===
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        # === 3. Create A and B low-rank matrices ===
        # A: (in_dim -> r)
        # B: (r -> out_dim)

        # initialize A and B properly

        # === 4. Mark LoRA weights as trainable ===
        # base weights remain frozen
        
    def forward(self, x):
        # === 5. Output from frozen base layer ===
        base_out = self.base(x)
        
        # === 6. Output from LoRA low rank update ===
        # lora_out = B(A(x)) * scaling
        
        # === 7. Combine outputs ===
        return base_out + lora_out


def apply_lora_to_model(model, r=8, alpha=16, target_modules=("q_proj", "v_proj")):
    """
    Walk model module tree.
    If a module name matches target_modules, replace its nn.Linear with LoRALinear.
    """
    for name, module in model.named_modules():
        # if module is a linear layer and name matches
        # wrap it with LoRALinear
        pass

    return model

def get_lora_parameters(model):
    """
    Gather only LoRA A and B matrix params.
    Base model params remain frozen.
    """
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
    return params