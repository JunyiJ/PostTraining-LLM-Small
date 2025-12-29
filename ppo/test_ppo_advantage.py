import torch

from ppo.ppo_advantage import advantage_gae, advantage_gae_vectorized


def test_advantage_gae_matches_manual():
    rewards = torch.tensor([[1.0, 0.0, 2.0]])
    values = torch.tensor([[0.5, 0.25, 0.1]])
    gamma = 0.9
    gae_lambda = 0.8

    # Manual backward accumulation
    next_values = torch.tensor([[0.25, 0.1, 0.0]])
    deltas = rewards + gamma * next_values - values

    adv_t2 = deltas[:, 2]
    adv_t1 = deltas[:, 1] + gamma * gae_lambda * adv_t2
    adv_t0 = deltas[:, 0] + gamma * gae_lambda * adv_t1
    expected = torch.stack([adv_t0, adv_t1, adv_t2], dim=1)

    out = advantage_gae(rewards, values, gamma=gamma, gae_lambda=gae_lambda)
    assert torch.allclose(out, expected)


def test_vectorized_matches_loop():
    rewards = torch.tensor([[0.0, 1.0, 0.5, 0.0]])
    values = torch.tensor([[0.2, 0.1, 0.05, 0.01]])
    gamma = 0.95
    gae_lambda = 0.9

    loop = advantage_gae(rewards, values, gamma=gamma, gae_lambda=gae_lambda)
    vec = advantage_gae_vectorized(rewards, values, gamma=gamma, gae_lambda=gae_lambda)

    assert torch.allclose(loop, vec)
