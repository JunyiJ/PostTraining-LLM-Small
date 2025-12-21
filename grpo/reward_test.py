import pytest
from reward import (
    compute_reward,
    MIN_REASON_TOKENS,
    reward_fn,
    advanced_cot_reward,
)

@pytest.mark.parametrize(
    "answer, gold, expected",
    [
        ("Mathematically, 1 + 1 = 2.", "2", 1.0),                       # correct, keyword-style "=", short reasoning + keyword bonus
        ("답がわかっていますか?\n\n... 답: 2", "2", 1.0),                  # multilingual, ends with 2, no keyword
        ("1+1=? ... Answer: 2", "2", 1.2),                              # explicit answer with keyword bonus
        ("1+1=? 3", "2", -0.25),                                         # incorrect numeric, short reasoning penalty
        ("No number here", "2", -1.0),                                  # no numeric value → punish
        ("1+1=?\n\nHere's how: 2", "2", 1.0),                           # last number correct, no keyword
    ],
)
def test_compute_reward(answer, gold, expected):
    assert abs(compute_reward("1+1=?", answer, gold) - expected) < 1e-4


def test_truncated_reward():
    assert compute_reward("q", "partial", "1", truncated=True) == -1.0


def test_reasoning_bonus_and_sanity():
    reasoning = " ".join(["step"] * MIN_REASON_TOKENS)
    answer = f"{reasoning} verify: result is 2"
    reward = compute_reward("1+1=?", answer, "2")
    # base 1.0 + reasoning 0.2 + sanity 0.5 + CoT 0.05 - 0.5 repetition = 1.25
    assert abs(reward - 1.25) < 1e-6


def test_keyword_bonus():
    answer = "Here is my answer: 42"
    reward = compute_reward("Q", answer, "42")
    # base 1.0 + answer bonus 0.2 (reasoning too short, no sanity)
    assert reward == 1.2

def test_keyword_hacking():
    answer = "Answer: Answer: Answer: Answer: Answer: "
    reward = compute_reward("Q", answer, "42")
    assert reward == -1.0

def test_keyword_hacking_with_answer():
    answer = "Answer: Answer: Answer: Answer: Answer: 42"
    reward = compute_reward("Q", answer, "42")
    # base 1.0 - 0.5 (spam)
    assert reward == 0.5


def test_keyword_picks_answer_over_later_number():
    answer = "Numbers: 1 2. My answer is 4 and then some text with 5 later."
    reward = compute_reward("Q", answer, "4")
    # With lower MIN_REASON_TOKENS, we now get reasoning bonus + keyword bonus
    assert abs(reward - 1.4) < 1e-6


def test_fallback_to_last_numeric_when_no_keyword():
    answer = "Earlier number 3 but final number 7"
    reward = compute_reward("Q", answer, "7")
    # base 1.0, short reasoning, no bonuses
    assert abs(reward - 1.0) < 1e-6


def test_repetition_penalty():
    answer = "word word word word word"
    reward = compute_reward("Q", answer, "0")
    # Should incur repetition penalty; incorrect + short + repetition => negative
    assert reward < 0

def test_train_example():
    answer = """
    **Here's how to solve this problem:**

    **1.  Find the combined speed:**

    *   Since the trains are moving towards each other, we need to add their speeds.

    *   Combined Speed = 47 km/h + 39 km/h = 86 km/h

    **2.  Use the formula:**

    *   Distance = Speed x Time

    *   We know the distance (96 km) and the combined speed.

    *   Time = Distance / Speed

    **3. Calculate the time:**

    *   Time = 96 km / 86 km/h = 1.10 hours (approximately)

    **Answer:** They will meet in approximately **1.10 hours**.
    """
    reward = compute_reward("Q", answer, "1.116279")
    # Near-correct numeric (approximate), reasonable reasoning length; should get partial credit
    assert abs(reward - 1.3) < 0.05


def test_fraction_answer():
    answer = "Compute 1/9 + 2/9.\nAnswer: 1/3"
    reward = compute_reward("Q", answer, "0.3333")
    assert 1.0 < reward < 1.3  # fraction should be parsed and rewarded


def test_equals_in_last_sentence_only():
    answer = "Work: 90-55 = 35. But actually it's 45"
    # Should look at last sentence; no '=' there, so fallback last numeric 45 (incorrect vs gold 35)
    reward = compute_reward("Q", answer, "35")
    assert reward < 0  # incorrect after fallback


def test_cot_verification_bonus_improves_score():
    correct = "Step 1: 2 + 3 = 5\nFinal answer: 5"
    incorrect = "Step 1: 2 + 3 = 6\nFinal answer: 5"
    base = reward_fn(correct, 5.0, verify_cot=False)
    with_verify = reward_fn(correct, 5.0, verify_cot=True)
    with_verify_bad = reward_fn(incorrect, 5.0, verify_cot=True)
    assert with_verify > base
    assert with_verify > with_verify_bad


def test_advanced_cot_reward_bounds():
    text = "Calc: 2 * 3 = 6\nFinal answer: 6"
    score = advanced_cot_reward(text, 6.0)
    assert -1.5 <= score <= 1.5


def test_truncated_penalty_applied():
    text = "Calc: 2 * 3 = 6\nFinal answer: 6"
    base = advanced_cot_reward(text, 6.0, truncated=False)
    penalized = advanced_cot_reward(text, 6.0, truncated=True, trunc_penalty=-0.2)
    assert penalized < base
