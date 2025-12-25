import torch

def compute_advantage(rewards, device, dtype=torch.float32):
    """
    Standardize rewards to compute advantages: (r - mean) / std.
    Returns a 1D torch.Tensor on CPU.
    """
    if len(rewards) == 0:
        return torch.tensor([])
    rs = torch.tensor(rewards, device=device, dtype=dtype)
    mean_reward = rs.mean()
    std_reward = rs.std(unbiased=False)
    if std_reward > 1e-6:
        advantages = (rs - mean_reward) / std_reward
    else:
        advantages = rs - mean_reward
    return advantages

def compute_rank_advantage(rewards, device, dtype=torch.float32):
    if len(rewards) == 0:
        return torch.tensor([])
    rs = torch.tensor(rewards, device=device, dtype=dtype).detach()
    if torch.allclose(rs, rs[0]):
        return torch.zeros_like(rs)
    ranks = torch.argsort(torch.argsort(rs))
    advantages = ranks.float()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


