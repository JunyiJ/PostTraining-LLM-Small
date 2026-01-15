import torch
# Check if Torch's internal Flash Attention is available
print(f"SDPA Flash Attention enabled: {torch.backends.cuda.flash_sdp_enabled()}")
