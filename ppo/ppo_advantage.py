import torch

def advantage_gae(rewards, values, truncated, gamma=0.99, gae_lambda=0.95):
    """
    rewards: [B, T-1] - The token-level rewards (KL + Sparse Reward)
    values:  [B, T]   - The values for all tokens (un-shifted)
    masks:   [B, T-1] - The answer_mask
    """
    B, T = values.shape
    advantages = torch.zeros_like(rewards)

    # Use bootstrap value if truncated, otherwise zero for terminal
    trunc_mask = torch.as_tensor(truncated, device=values.device, dtype=torch.bool)
    last_values = torch.where(trunc_mask, values[:, -1], torch.zeros_like(values[:, -1]))
    next_values = torch.cat([values[:, 1:-1], last_values.unsqueeze(1)], dim=1)  # [B, T-1]

    # TD errors
    deltas = rewards + gamma * next_values - values[:, :-1]

    running_gae = torch.zeros(B, device=values.device, dtype=values.dtype)
    for t in range(T - 2, -1, -1):
        running_gae = deltas[:, t] + gamma * gae_lambda * running_gae
        advantages[:, t] = running_gae
    return advantages


def advantage_gae_vectorized(rewards, values, truncated, gamma=0.99, gae_lambda=0.95):
    """
    rewards: [B, T-1] - The token-level rewards (KL + Sparse Reward)
    values:  [B, T]   - The values for all tokens (un-shifted)
    masks:   [B, T-1] - The answer_mask
    """
    B, T = values.shape
     # Use bootstrap value if truncated, otherwise zero for terminal
    trunc_mask = torch.as_tensor(truncated, device=values.device, dtype=torch.bool)
    last_values = torch.where(trunc_mask, values[:, -1], torch.zeros_like(values[:, -1]))
    next_values = torch.cat([values[:, 1:-1], last_values.unsqueeze(1)], dim=1)  # [B, T-1]

    deltas = rewards + gamma * next_values - values[:, :-1]  # [B, T]


    idx = torch.arange(T-1, device=values.device)
    dist = idx.view(1, -1) - idx.view(-1, 1)  # [T, T], dist[i,j] = j - i
    weights = torch.where(
        dist >= 0,
        (gamma * gae_lambda) ** dist.float(),
        torch.zeros_like(dist, dtype=values.dtype),
    ).to(values.dtype)

    # For each timestep i, weights[i, j] is the factor for delta_j when j >= i
    # advantages[:, i] = sum_j deltas[:, j] * weights[i, j]
    advantages = deltas @ weights.T  # [B, T]
    return advantages
