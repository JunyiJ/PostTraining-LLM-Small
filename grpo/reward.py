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



import re
from typing import Optional


# ============================================================
# Helper: parse numbers, including simple fractions like 3/4
# ============================================================

def _parse_number(token: str) -> Optional[float]:
    token = token.strip().rstrip(".")
    # Support fractions like "3/4"
    if "/" in token:
        try:
            num, den = token.split("/")
            return float(num) / float(den)
        except:
            pass
    try:
        return float(token)
    except:
        return None


# ============================================================
# Extract final numeric answer from the model output
# ============================================================

NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:/\d+)?")

def extract_final_answer(text: str) -> Optional[float]:
    """
    Robust answer extraction:
    1) Prefer LAST explicit "Final Answer: <num>"
    2) Otherwise use the LAST number in the text (e.g., "... = 72").
    """

    if text is None:
        return None
    # Remove currency, commas, and percentage signs before float conversion
    # Pattern to find the number after the tag
    pattern = r"final\s*answer[:\s]*[\$]?\s*([-+]?\d[0-9,]*\.?\d*(?:/\d+)?)\s*[\%]*"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        raw_num = match.group(1).strip()
        # Remove commas: "1,200" -> "1200"
        clean_num = raw_num.replace(",", "")
        
        # Handle Fractions: "3/4" -> 0.75
        if "/" in clean_num:
            try:
                num, den = clean_num.split("/")
                return float(num) / float(den)
            except: return None
        try:
            return float(clean_num)
        except:
            return None

    # 2) Fallback: use LAST number in the answer text
    nums = NUMBER_PATTERN.findall(text)
    if nums:
        return _parse_number(nums[-1])

    return None


# ============================================================
# Numeric reward: stable, binary math correctness
# ============================================================

def numeric_reward(pred: float, gold: float, tol: float = 1e-4) -> float:
    """
    Math-centric reward:
    +1    if exactly correct
    +0.2  if within small relative error (<5%)
    -1    otherwise
    """
    err = abs(pred - gold)
    if err <= tol:
        return 1.0
    rel_err = err / max(1.0, abs(gold))
    if rel_err < 0.05:
        return 0.2
    return -1.0


# ============================================================
# Light, safe equation correctness bonus
# ============================================================

def extract_equations(text: str):
    equations = []
    for line in text.splitlines():
        if "=" not in line:
            continue
        # extract "LHS = RHS"
        parts = line.split("=")
        if len(parts) != 2:
            continue
        lhs, rhs = parts[0].strip(), parts[1].strip()
        # must contain digits
        if not re.search(r"\d", lhs) or not re.search(r"\d", rhs):
            continue
        equations.append((lhs, rhs))
    return equations


def _safe_eval_arith(expr: str) -> float:
    """
    Safely evaluate a simple arithmetic expression; raises on empty/invalid.
    Only allows digits, ., +, -, *, /, parentheses, spaces.
    """
    cleaned = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", expr)
    if not cleaned.strip():
        raise ValueError("Empty or invalid expression")
    # allow bare numbers
    if not re.search(r"\d\s*[\+\-\*\/]\s*\d", cleaned):
        return float(cleaned)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        return eval(cleaned, {"__builtins__": None}, {})


def equation_bonus(text: str) -> float:
    """
    Reward correct intermediate math gently:
    +0.02 per correct equation (max +0.1)
    No penalty for missing/bad equations: avoids training collapse.
    """
    eqs = extract_equations(text)
    if not eqs:
        return 0.0

    bonus = 0.0
    for lhs, rhs in eqs:
        try:
            lhs_val = _safe_eval_arith(lhs)
            rhs_val = _safe_eval_arith(rhs)
            if abs(lhs_val - rhs_val) < 1e-6:
                bonus += 0.02
        except:
            continue

    return min(bonus, 0.10)   # cap at +0.1


# ============================================================
# FINAL COMBINED REWARD
# ============================================================

def advanced_cot_reward(text: str, gold_answer: float, truncated: bool = False) -> float:
    """
    Robust, stable math reward for GRPO/PPO.

    Components:
        + numeric correctness     (dominant signal)
        + equation bonus          (optional small add-on)
        + truncation penalty      (mild)
    
    Output range: [-1.5, +1.5]
    """

    pred = extract_final_answer(text)
    if pred is None:
        return -1.0
    
    try:
        gold = float(gold_answer)
    except:
        return -1.0

    # 1. Dominant term
    num_r = numeric_reward(pred, gold)   # [-1, 1]

    # 2. Small positive bonus only
    eq_r = equation_bonus(text)          # [0, 0.1]

    # 3. Truncation penalty
    trunc_r = -0.05 if truncated else 0.0

    total = num_r + eq_r + trunc_r

    return max(-1.5, min(1.5, total))

def refined_advanced_cot_reward(text: str, gold_answer: float, truncated: bool = False) -> float:
    """
    Optimized for Rank-based GRPO on Gemma-2-2B.
    """
    # 1. Improved Extraction Logic
    pred = extract_final_answer(text)
    
    try:
        gold = float(gold_answer)
    except:
        return -1.0

    # 2. Strict Numeric Reward with a "Near Miss" buffer
    # This helps the model if it's off by a tiny rounding error
    if pred is None:
        # Penalty for failing to follow the format 'Final answer: <num>'
        num_r = -1.2 
    elif abs(pred - gold) < 1e-4:
        num_r = 1.0
    elif abs(pred - gold) < 1.0: # Near miss (off by less than 1)
        num_r = 0.2
    else:
        num_r = -1.0

    # 3. Intermediate Logic Bonus
    # Only award equation bonus if the CoT isn't total gibberish
    eq_r = 0.0
    if len(text) > 20: 
        eq_r = equation_bonus(text)

    # 4. Format/Structure Bonus
    # Reward the model slightly for actually using the step-by-step format
    format_r = 0.0
    # if "Step 1" in text or "1." in text:
    #     format_r += 0.05

    # 5. Soft Truncation Penalty
    # We lowered this to -0.05 per your latest update, which is good.
    # It stops the 'Builder Problem' from being a total loss.
    trunc_r = -0.05 if truncated else 0.0

    # 6. Efficiency Bonus (The Tie-Breaker)
    # Rewards the model for getting the answer in fewer tokens.
    # If two samples are correct, the shorter one ranks higher.
    length_penalty = min(len(text) / 1000, 0.1)

    total = num_r + eq_r + format_r + trunc_r - length_penalty

    # Clip to ensure advantages don't explode
    return max(-1.5, min(1.5, total))