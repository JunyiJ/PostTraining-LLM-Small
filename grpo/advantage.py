import torch

def compute_advantage(rewards, device, dtype=torch.float32, std_eps: float = 1e-4):
    """
    Standardize rewards to compute advantages: (r - mean) / std.
    Returns a 1D torch.Tensor on CPU.
    """
    if len(rewards) == 0:
        return torch.tensor([])
    rs = torch.tensor(rewards, device=device, dtype=dtype)
    mean_reward = rs.mean()
    std_reward = rs.std(unbiased=False)
    if std_reward.item() < std_eps:
        return torch.zeros_like(rs)
    return (rs - mean_reward) / (std_reward + 1e-6)
