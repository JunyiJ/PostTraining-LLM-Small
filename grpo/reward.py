import re
import math
from typing import Optional

MIN_REASON_TOKENS = 10
MAX_REASON_TOKENS = 120
REWARD_CLIP = (-1.0, 2.5)
ANSWER_BONUS = 0.2
NEAR_ABS_TOL = 0.02
NEAR_REL_TOL = 0.02
NEAR_REWARD = 0.5
COT_BONUS = 0.05
COT_MIN_REASON_TOKENS = 5
COT_MAX_REL_ERR = 0.05
COT_TRIGGER = ("step", "steps", "calculat", "here is how", "here's how", "1.")


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


def _parse_fraction(text: str) -> Optional[float]:
    m = re.search(r"(-?\d+)\s*/\s*(-?\d+)", text)
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        return None
    return num / den


def extract_answer_with_keyword(text: Optional[str]) -> (Optional[float], bool):
    """Return (value, used_keyword)."""
    if text is None:
        return None, False

    number_pat = r"[-+]?\d*\.?\d+"
    frac_pat = r"-?\d+\s*/\s*-?\d+"

    # Prefer structured “answer” sections: take the last number in the answer clause
    answer_section_patterns = [
        r"answer[:\s]*(.*)",              # Answer: 101 - 44 = 57
        r"answer\s+is[:\s]*(.*)",         # Answer is 57
        r"final\s+answer[:\s]*(.*)",      # Final answer: 57
    ]
    for pat in answer_section_patterns:
        matches = list(re.finditer(pat, text, flags=re.IGNORECASE))
        for match in reversed(matches):  # prefer the last occurrence of the keyword
            section = match.group(1)
            # If section contains an explicit "=", prefer the value after the last "="
            eq_in_section = list(re.finditer(r"=+\s*((" + frac_pat + r")|(" + number_pat + r"))", section))
            if eq_in_section:
                token = eq_in_section[-1].group(1)
                frac_val = _parse_fraction(token)
                if frac_val is not None:
                    return float(frac_val), True
                try:
                    cleaned = token.replace(",", "")
                    return float(cleaned), True
                except Exception:
                    continue
            # Otherwise take the first numeric after the keyword
            frac_val = _parse_fraction(section)
            if frac_val is not None:
                return float(frac_val), True
            nums = re.findall(number_pat, section)
            if nums:
                try:
                    cleaned = nums[0].replace(",", "")
                    return float(cleaned), True
                except Exception:
                    continue

    # Fallback: look in the last sentence for an “= number” or “= fraction”
    sentences = re.split(r"[.!?\n]+", text)
    last_sent = sentences[-1] if sentences else text
    eq_matches = list(re.finditer(r"=+\s*((" + frac_pat + r")|(" + number_pat + r"))", last_sent))
    if eq_matches:
        token = eq_matches[-1].group(1)
        frac_val = _parse_fraction(token)
        if frac_val is not None:
            return float(frac_val), True
        try:
            cleaned = token.replace(",", "")
            return float(cleaned), True
        except Exception:
            pass

    # Fallback to any numeric, preferring last occurrence
    frac_val = _parse_fraction(text)
    if frac_val is not None:
        return float(frac_val), False
    val = extract_answer(text)
    return val, False


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
    unique_ratio_thresh: float = 0.25,
    length_relax: int = 80,
) -> bool:
    """
    Detect repetitive patterns:
    - Unique token ratio very low
    - Any n-gram (1,2,3) repeated at least min_freq times
    Relax repetition penalty for longer, more diverse texts.
    """
    from collections import Counter

    toks = re.findall(r"\b\w+\b", text.lower())
    if not toks:
        return False
    unique_ratio = len(set(toks)) / len(toks)
    # If text is long and reasonably diverse, skip repetition penalty
    if len(toks) >= length_relax and unique_ratio >= (unique_ratio_thresh + 0.1):
        return False
    for n in n_values:
        ngrams = [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        if not ngrams:
            continue
        if Counter(ngrams).most_common(1)[0][1] >= min_freq:
            return True
    return unique_ratio < unique_ratio_thresh


ANSWER_SPAM_THRESH = 4


def _has_answer_spam(text: str, threshold: int = ANSWER_SPAM_THRESH) -> bool:
    """Detects repeated 'answer' tokens (case-insensitive)."""
    return len(re.findall(r"\banswer\b", text, flags=re.IGNORECASE)) >= threshold


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

    err = abs(pred_val - gold_val)
    rel_err = err / max(abs(gold_val), 1.0)
    near = err <= max(NEAR_ABS_TOL, NEAR_REL_TOL * max(abs(gold_val), 1.0))
    correct = err <= tol
    # Smooth base reward: high when correct/near, decays with relative error
    if pred_val is None:
        reward = -1.0
    else:
        reward = 1.0 - math.tanh(rel_err / 0.1)
        if not near and not correct:
            reward = max(reward, -0.25)

    reasoning_len = _reasoning_token_count(answer)
    if MIN_REASON_TOKENS <= reasoning_len <= MAX_REASON_TOKENS:
        reward += 0.2
    elif reasoning_len < MIN_REASON_TOKENS and (not correct and not near):
        reward -= 0.25

    # Small CoT-style bonus if near/correct and has simple reasoning cues
    if (correct or near) and (COT_MIN_REASON_TOKENS <= reasoning_len <= MAX_REASON_TOKENS):
        lower_answer = answer.lower()
        if any(tok in lower_answer for tok in COT_TRIGGER) and rel_err <= COT_MAX_REL_ERR:
            reward += COT_BONUS

    if _has_sanity_check(answer) and (correct or near):
        reward += 0.5
    if used_keyword and (correct or near) and (not _has_answer_spam(answer)):
        reward += ANSWER_BONUS
    # Penalize answer spam explicitly; otherwise apply repetition guard
    if _has_answer_spam(answer):
        reward -= 0.5
    elif _has_repetition(answer):
        reward -= 0.5

    reward = max(REWARD_CLIP[0], min(REWARD_CLIP[1], reward))
    return reward


# === Optional smoother CoT-aware reward ===
def extract_final_answer(text: str) -> Optional[float]:
    if text is None:
        return None
    
    # Strict final answer pattern
    m = re.search(
        r"final\s*answer\s*:\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)(?!.*final\s*answer)",
        text,
        flags=re.IGNORECASE
    )
    if m:
        token = m.group(1).strip().rstrip(".")
        frac_val = _parse_fraction(token)
        if frac_val is not None:
            return float(frac_val)
        try:
            return float(token)
        except:
            pass
    
    # Backup: extract first number AFTER "Final answer"
    m = re.search(
        r"final\s*answer[^0-9\-+]*([-+]?\d*\.?\d+)",
        text, flags=re.IGNORECASE
    )
    if m:
        return float(m.group(1))
    
    # Absolute fallback: DO NOT scan entire text
    return None


def cot_quality_score(text: str) -> float:
    """
    Soft reward for structured reasoning patterns.
    Range: 0.0 → +0.15
    Encourages: step-by-step, intermediate calculations, logical flow.
    """
    if text is None:
        return 0.0
    score = 0.0
    t = text.lower()
    if "step" in t:
        score += 0.03
    if "first" in t:
        score += 0.03
    if "next" in t or "then" in t:
        score += 0.03
    if "therefore" in t or "so" in t:
        score += 0.03
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+", text):
        score += 0.03
    return min(score, 0.15)


def extract_equations(text: str):
    """
    Extract simple arithmetic equations of the form: <expr> = <expr>
    Returns a list of (lhs_str, rhs_str).
    """
    equations = []
    for line in text.splitlines():
        if "=" not in line:
            continue
        if not re.search(r"\d", line):
            continue
        parts = line.split("=")
        if len(parts) != 2:
            continue
        lhs, rhs = parts[0].strip(), parts[1].strip()
        equations.append((lhs, rhs))
    return equations


def safe_eval_arith(expr: str):
    """
    Safely evaluate a simple arithmetic expression with + - * / and parentheses.
    Strips non-allowed chars to avoid weird code execution.
    """
    cleaned = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", expr)
    if not cleaned.strip():
        raise ValueError("Empty or invalid expression")
    return eval(cleaned, {"__builtins__": None}, {})


def cot_verification_bonus(text) -> float:
    """
    Analyze intermediate equations in CoT and reward correct math.
    Penalize clearly incorrect equations.
    Returns a small bonus/penalty ~ [-0.2, +0.2].
    """
    equations = extract_equations(text)
    if not equations:
        return 0.0

    correct = 0
    incorrect = 0
    for lhs, rhs in equations:
        try:
            lhs_val = safe_eval_arith(lhs)
            rhs_val = safe_eval_arith(rhs)
            if isinstance(lhs_val, (int, float)) and isinstance(rhs_val, (int, float)):
                if abs(lhs_val - rhs_val) < 1e-6:
                    correct += 1
                else:
                    incorrect += 1
        except Exception:
            continue

    total = correct + incorrect
    if total == 0:
        return 0.0
    ratio = (correct - incorrect) / total  # [-1, 1]
    return 0.2 * ratio


def numeric_reward(pred: float, gold_val: float, tol: float = 1e-6) -> float:
    err = abs(pred - gold_val)
    reward = 1 - math.tanh(err)
    if err <= tol:
        reward += 0.2
    return reward


def reward_fn(text: str, gold_answer: float, tol: float = 1e-6, verify_cot: bool = False) -> float:
    """
    Continuous reward combining:
      1) Numeric correctness (smooth)
      2) CoT structural quality
      3) Optional CoT verification bonus
    Clips to [-1.2, 1.4].
    """
    pred = extract_final_answer(text)
    if pred is None:
        return -1.0
    try:
        gold_val = float(gold_answer)
    except Exception:
        return -1.0

    err = abs(pred - gold_val)
    rel_err = err / max(abs(gold_val), 1.0)
    base = numeric_reward(pred, gold_val, tol=tol)
    cot_bonus = cot_quality_score(text)
    verify_bonus = cot_verification_bonus(text) if verify_cot else 0.0
    total_reward = base + cot_bonus + verify_bonus
    return max(-1.2, min(1.4, total_reward))


def advanced_cot_reward(
    text: str,
    gold_answer: float,
    tol: float = 1e-4,
    truncated: bool = False,
    trunc_penalty: float = -0.6,
) -> float:
    """
    Advanced CoT-aware reward:
      - Final numeric correctness (smooth)
      - CoT structure bonus (weak)
      - CoT equation verification bonus
      - Optional truncation penalty
    Output clipped to [-1.5, 1.5].
    """
    pred = extract_final_answer(text)
    if pred is None:
        return -1.2
    try:
        gold_val = float(gold_answer)
    except Exception:
        return -1.2
    num_r = numeric_reward(pred, gold_val, tol=tol)
    struct_r = cot_quality_score(text)
    verify_r = cot_verification_bonus(text)
    total = num_r + struct_r + verify_r
    if truncated:
        total += trunc_penalty
    return max(-1.5, min(1.5, total))
