import re
from typing import Optional

MIN_REASON_TOKENS = 10
MAX_REASON_TOKENS = 120
REWARD_CLIP = (-1.0, 2.5)
ANSWER_BONUS = 0.2


def extract_answer(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    matches = list(re.finditer(r"[-+]?\d*\.?\d+", text))
    if not matches:
        return None
    try:
        cleaned = matches[-1].group(0).replace(",", "")
        return float(cleaned)
    except Exception:
        return None


def extract_answer_after_keyword(text: Optional[str]) -> Optional[float]:
    """
    Try to extract a numeric answer that appears after the keyword 'answer'.
    """
    if text is None:
        return None
    matches = list(re.finditer(r"answer[^0-9\-+]*([-+]?\d*\.?\d+)", text, re.IGNORECASE))
    if not matches:
        return None
    try:
        cleaned = matches[-1].group(1).replace(",", "")
        return float(cleaned)
    except Exception:
        return None


def extract_answer_with_keyword(text: Optional[str]) -> (Optional[float], bool):
    """Return (value, used_keyword)."""
    val = extract_answer_after_keyword(text)
    if val is not None:
        return val, True
    return extract_answer(text), False


def _reasoning_token_count(text: str) -> int:
    matches = list(re.finditer(r"[-+]?\d*\.?\d+", text))
    if not matches:
        return len(text.split())
    last_span = matches[-1].span()
    reasoning_text = text[: last_span[0]]
    return len(reasoning_text.split())


def _has_sanity_check(text: str) -> bool:
    return bool(re.search(r"\b(check|verify|re-check|sanity)\b", text, re.IGNORECASE))


def _has_repetition(
    text: str,
    n_values=(1, 2, 3),
    min_freq: int = 4,
    unique_ratio_thresh: float = 0.3,
) -> bool:
    """
    Detect repetitive patterns:
    - Unique token ratio very low
    - Any n-gram (1,2,3) repeated at least min_freq times
    """
    from collections import Counter

    toks = re.findall(r"\b\w+\b", text.lower())
    if not toks:
        return False
    if len(set(toks)) / len(toks) < unique_ratio_thresh:
        return True
    for n in n_values:
        ngrams = [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        if not ngrams:
            continue
        if Counter(ngrams).most_common(1)[0][1] >= min_freq:
            return True
    return False


def compute_reward(question, answer, gold, tol=1e-6, truncated: bool = False):
    """
    Heuristic reward:
    - Truncated outputs: -1.0
    - No numeric answer: -1.0
    - Correct numeric: +1.0
    - Incorrect numeric: -0.25
    - Reasoning bonus: +0.5 if reasoning length within [MIN_REASON_TOKENS, MAX_REASON_TOKENS]
      Short reasoning: 0 bonus if correct, -0.25 if incorrect
    - Sanity check mentions: +0.5 (only if correct)
    - Keyword "answer" bonus: +0.2 (only if correct)
    Clipped to REWARD_CLIP.
    """
    if truncated:
        return REWARD_CLIP[0]

    pred_val, used_keyword = extract_answer_with_keyword(answer)
    if pred_val is None:
        return REWARD_CLIP[0]

    try:
        gold_val = float(str(gold).replace(",", ""))
    except Exception:
        return REWARD_CLIP[0]

    correct = abs(pred_val - gold_val) <= tol
    reward = 1.0 if correct else -0.25

    reasoning_len = _reasoning_token_count(answer)
    if MIN_REASON_TOKENS <= reasoning_len <= MAX_REASON_TOKENS:
        reward += 0.5
    elif reasoning_len < MIN_REASON_TOKENS and not correct:
        reward -= 0.25

    if _has_sanity_check(answer) and correct:
        reward += 0.5
    if used_keyword and correct:
        reward += ANSWER_BONUS
    if _has_repetition(answer):
        reward -= 0.5

    reward = max(REWARD_CLIP[0], min(REWARD_CLIP[1], reward))
    return reward
