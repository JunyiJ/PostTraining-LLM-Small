import pytest
from grpo.reward import (
    compute_reward,
    MIN_REASON_TOKENS,
    advanced_cot_reward,
    refined_advanced_cot_reward,
    extract_final_answer,
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


def test_advanced_cot_reward_bounds():
    text = "Calc: 2 * 3 = 6\nFinal answer: 6"
    score = advanced_cot_reward(text, 6.0)
    assert -1.5 <= score <= 1.5


def test_truncated_penalty_applied():
    text = "Calc: 2 * 3 = 6\nFinal answer: 6"
    base = advanced_cot_reward(text, 6.0, truncated=False)
    penalized = advanced_cot_reward(text, 6.0, truncated=True)
    assert penalized < base


# === refined_advanced_cot_reward coverage ===
def test_refined_exact_and_near():
    exact = "Final answer: 42"
    near = "Final answer: 42.5"
    exact_score = refined_advanced_cot_reward(exact, 42.0, truncated=False)
    near_score = refined_advanced_cot_reward(near, 42.0, truncated=False)
    assert exact_score > 0.9
    assert 0.1 <= near_score <= 0.4


def test_refined_no_answer_penalty():
    text = "I refuse to answer."
    score = refined_advanced_cot_reward(text, 10.0, truncated=False)
    assert score <= -1.0


def test_refined_equation_bonus_and_length_penalty():
    base_text = "Final answer: 5"
    eq_text = "Step 1: 2+3=5\nFinal answer: 5"
    base_score = refined_advanced_cot_reward(base_text, 5.0, truncated=False)
    long_text = ("Final answer: 5 " + "filler " * 200).strip()
    long_score = refined_advanced_cot_reward(long_text, 5.0, truncated=False)
    assert long_score < base_score  # length penalty applied


def test_refined_extraction_1():
    text = "Final answer: 10,000"
    base = refined_advanced_cot_reward(text, 10000, truncated=False)
    assert base == 0.98

def test_refined_extraction_2():
    text = """
    **Reasoning:**

    1. **Year 1:** The car loses 30% of its value, meaning it's worth 70% of its original price. We can express this as: 
    Original Price * 0.70 = $5600
    2. **Year 2:**  The car loses 20% of its value, meaning it's worth 80% of what it was last year. We can express this as:
    $5600 * 0.80 = $4480
    3. **Year 1 Value:** The value after Year 1 is $5600.
    4. **Year 2 Value:** The value after Year 2 is $4480.

    Now we can use these values to find the original price:


    **Final Answer:** $10,000 
    """
    base = refined_advanced_cot_reward(text, 10000, truncated=False)
    assert base >= 0.9

def test_refined_extraction_3():
    text = """
    **Reasoning:**

    1. **Year 1:** The car loses 30% of its value, meaning it's worth 70% of its original price. We can express this as: 
    Original Price * 0.70 = $5600
    2. **Year 2:**  The car loses 20% of its value, meaning it's worth 80% of what it was last year. We can express this as:
    $5600 * 0.80 = $4480
    3. **Year 1 Value:** The value after Year 1 is $5600.
    4. **Year 2 Value:** The value after Year 2 is $4480.

    Now we can use these values to find the original price:


    **Final Answer:** $10,000 
    """
    base = refined_advanced_cot_reward(text, "10,000", truncated=False)
    assert base >= 0.9


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Final Answer: 100 hours for 4 workers", 100.0),
        ("Final Answer: 100", 100.0),
        ("Final Answer: 2 + 2 = 4", 4.0),
        ("Final Answer: The 4 workers take 100 hours", 4.0),
        ("Final Answer: 4 (Hallucinated variable)", 4.0),
        ("**Final answer:** 1000 people had not voted by 16:00.", 1000)
    ],
)
def test_extract_final_answer_examples(text, expected):
    assert extract_final_answer(text) == expected
