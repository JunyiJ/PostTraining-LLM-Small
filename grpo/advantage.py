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
    rs = torch.tensor(rewards, device=device, dtype=dtype)
    if torch.all(rs == rs[0]):
        return torch.zeros_like(rs)
    unique_vals, inverse_indices = torch.unique(rs, return_inverse=True)
    counts = torch.bincount(inverse_indices)
    cumulative_counts = torch.cumsum(counts, dim=0)
    mean_ranks = (cumulative_counts - counts.float() / 2.0)
    advantages = mean_ranks[inverse_indices]
    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean())/ advantages.std()
    else:
        advantages = advantages - advantages.mean()
    return advantages
