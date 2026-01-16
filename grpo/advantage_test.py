import pytest
import torch

from grpo.advantage import compute_advantage, compute_rank_advantage

@pytest.mark.parametrize(
    "rewards, expected",
    [
        ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),             # all equal -> std 0
        ([1.0, 0.0], [1.0, -1.0]),                      # standardized (population std)
        ([1.0, 0.5, 0.0], [1.2247449, 0.0, -1.2247449]),# standardized
        ([-1.0, 2.0], [-1.0, 1.0]),                     # standardized (population std)
    ],
)
def test_compute_advantage(rewards, expected):
    out = compute_advantage(rewards, device="cpu")
    assert pytest.approx(out.tolist(), rel=1e-5, abs=1e-5) == expected

def test_rank_advantage():
    rewards = [1.0, -1.0, 1.0, 0.5]
    out = compute_rank_advantage(rewards, "cpu")
    advantages = torch.tensor([2.0, 0.0, 3.0, 1.0], device="cpu")
    advantages = (advantages - advantages.mean())/ advantages.std()
    assert pytest.approx(out.tolist(), rel=1e-5, abs=1e-5) == advantages.tolist()
