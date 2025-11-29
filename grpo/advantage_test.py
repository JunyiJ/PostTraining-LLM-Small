import pytest

from advantage import compute_advantage

@pytest.mark.parametrize(
    "rewards, expected",
    [
        ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),             # all equal
        ([1.0, 0.0], [0.5, -0.5]),                      # two values
        ([1.0, 0.5, 0.0], [0.5, 0.0, -0.5]),            # mean-centering
        ([-1.0, 2.0], [-1.5, 1.5]),                     # negatives/positives
    ],
)
def test_compute_advantage(rewards, expected):
    assert compute_advantage(rewards) == expected