import re
import math
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
    2) Otherwise prefer LAST "therefore ... <num>"
    3) Otherwise use the LAST number in the text (e.g., "... = 72").
    """

    if not text:
        return None

    # Split by the LAST "final answer" to avoid getting trapped by intermediate thoughts
    parts = re.split(r"final\s*answer[:\s]*", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        after_tag = parts[-1].strip()
        num_match = re.search(r"([-+]?\d[0-9,./]*)", after_tag)
        if num_match:
            raw_num = num_match.group(1).replace(",", "").rstrip(".")
            try:
                if "/" in raw_num:
                    n, d = raw_num.split("/")
                    return float(n) / float(d)
                return float(raw_num)
            except:
                pass

    # Try the last occurrence of "therefore ... <num>"
    parts = re.split(r"therefore[:\s]*", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        after_tag = parts[-1].strip()
        num_match = re.search(r"([-+]?\d[0-9,./]*)", after_tag)
        if num_match:
            raw_num = num_match.group(1).replace(",", "").rstrip(".")
            try:
                if "/" in raw_num:
                    n, d = raw_num.split("/")
                    return float(n) / float(d)
                return float(raw_num)
            except:
                pass

    # Fallback to the very last number found anywhere
    nums = re.findall(r"[-+]?\d[0-9,]*\.?\d*", text)
    if nums:
        try:
            return float(nums[-1].replace(",", "").rstrip("."))
        except:
            return None
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

def refined_advanced_cot_reward(text: str, gold_answer: float, truncated: bool = False) -> float:
    """
    Optimized for Rank-based GRPO on Gemma-2-2B.
    """
    # 1. Improved Extraction Logic
    pred = extract_final_answer(text)
    
    try:
        gold = float(str(gold_answer).replace(",", ""))
    except:
        return -1.0

    # 2. Strict Numeric Reward with a "Near Miss" buffer
    # This helps the model if it's off by a tiny rounding error
    if pred is None:
        # Penalty for failing to follow the format 'Final answer: <num>'
        num_r = -1.2
    else:
        rel_error = abs(pred - gold) / max(1.0, abs(gold))
        if rel_error < 0.01:
            num_r = 1.0
        elif rel_error < 0.05: # Near miss
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
    trunc_r = -0.2 if truncated else 0.0

    # 6. Efficiency Bonus (The Tie-Breaker)
    # Rewards the model for getting the answer in fewer tokens.
    # If two samples are correct, the shorter one ranks higher.
    length_penalty = min(len(text) / 1000, 0.1)

    total = num_r + eq_r + format_r + trunc_r - length_penalty

    # Clip to ensure advantages don't explode
    return max(-1.5, min(1.5, total))