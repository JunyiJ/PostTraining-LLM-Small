import torch

def advantage_gae(rewards, values, gamma=0.99, gae_lambda=0.95):
    """
    Generalized Advantage Estimation (scalar loop version).
    rewards, values: [B, T]
    """
    B, T = values.shape
    advantages = torch.zeros_like(rewards)

    # TD errors
    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=-1)
    deltas = rewards + gamma * next_values - values

    running_gae = torch.zeros(B, device=values.device, dtype=values.dtype)
    for t in range(T - 1, -1, -1):
        running_gae = deltas[:, t] + gamma * gae_lambda * running_gae
        advantages[:, t] = running_gae
    return advantages


def advantage_gae_vectorized(rewards, values, gamma=0.99, gae_lambda=0.95):
    """
    Vectorized GAE computation using matrix weights.
    rewards, values: [B, T]
    """
    B, T = values.shape
    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=-1)
    deltas = rewards + gamma * next_values - values  # [B, T]

    idx = torch.arange(T, device=values.device)
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
