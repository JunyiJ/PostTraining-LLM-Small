import pytest
from reward import compute_reward

@pytest.mark.parametrize(
    "answer, gold, expected",
    [
        ("Mathematically, 1 + 1 = 2.", "2", 1.0),                       # clean correct
        ("답がわかっていますか?\n\n... 답: 2", "2", 1.0),                  # multilingual, ends with 2
        ("1+1=? ... Answer: 2", "2", 1.0),                              # explicit answer at end
        ("1+1=? 3", "2", 0.0),                                          # incorrect numeric
        ("No number here", "2", -1.0),                                  # no numeric value → punish
        ("1+1=?\n\nHere's how: 2", "2", 1.0),                           # last number correct
    ],
)
def test_compute_reward(answer, gold, expected):
    assert compute_reward("1+1=?", answer, gold) == expected