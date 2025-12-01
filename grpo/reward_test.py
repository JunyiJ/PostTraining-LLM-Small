import pytest
from reward import compute_reward, MIN_REASON_TOKENS

@pytest.mark.parametrize(
    "answer, gold, expected",
    [
        ("Mathematically, 1 + 1 = 2.", "2", 1.0),                       # correct, no keyword, short reasoning
        ("답がわかっていますか?\n\n... 답: 2", "2", 1.0),                  # multilingual, ends with 2, no keyword
        ("1+1=? ... Answer: 2", "2", 1.2),                              # explicit answer with keyword bonus
        ("1+1=? 3", "2", -0.5),                                         # incorrect numeric, short reasoning penalty
        ("No number here", "2", -1.0),                                  # no numeric value → punish
        ("1+1=?\n\nHere's how: 2", "2", 1.0),                           # last number correct, no keyword
    ],
)
def test_compute_reward(answer, gold, expected):
    assert abs(compute_reward("1+1=?", answer, gold) - expected) < 1e-6


def test_truncated_reward():
    assert compute_reward("q", "partial", "1", truncated=True) == -1.0


def test_reasoning_bonus_and_sanity():
    reasoning = " ".join(["step"] * MIN_REASON_TOKENS)
    answer = f"{reasoning} verify: result is 2"
    reward = compute_reward("1+1=?", answer, "2")
    # base 1.0 + reasoning 0.5 + sanity 0.5 = 2.0 (clipped)
    assert abs(reward - 2.0) < 1e-6


def test_keyword_bonus():
    answer = "Here is my answer: 42"
    reward = compute_reward("Q", answer, "42")
    # base 1.0 + answer bonus 0.2 (reasoning too short, no sanity)
    assert reward == 1.2


def test_keyword_picks_answer_over_later_number():
    answer = "Numbers: 1 2. My answer is 4 and then some text with 5 later."
    reward = compute_reward("Q", answer, "4")
    # base 1.0 + keyword bonus 0.2 (reasoning too short for bonus)
    assert abs(reward - 1.2) < 1e-6


def test_fallback_to_last_numeric_when_no_keyword():
    answer = "Earlier number 3 but final number 7"
    reward = compute_reward("Q", answer, "7")
    # base 1.0, short reasoning, no bonuses
    assert abs(reward - 1.0) < 1e-6
