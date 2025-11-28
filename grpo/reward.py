import re

def extract_answer(text):
    if text is None:
        return None
    # find last integer in the output
    # match numbers like: 3, -2, 3.1415, 0.00001, .5, -0.25
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    
    try:
        cleaned = match.group(0).replace(",", "")
        return float(cleaned)
    except:
        return None

def compute_reward(question, answer, gold, tol=1e-6):
    # 1) correctness. TODO: add format checking, reasoning length reward, refinement reward.
    pred_val = extract_answer(answer)
    # 2) format - punish if results have no numeric value.
    if pred_val is None:
        return -1.0
    try:
        gold_val = float(str(gold).replace(",", ""))
    except Exception:
        return -1.0
    correct = (pred_val == gold_val) or (abs(pred_val - gold_val) <= tol)
    return 1.0 if correct else 0.0
