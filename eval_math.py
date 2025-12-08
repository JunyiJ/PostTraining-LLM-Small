"""
Script to evaluate the model's performance on the test math dataset
"""
import json, re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from grpo.utils import load_model
from grpo.lora import apply_lora_to_model, freeze_non_lora_params

MODEL_PATH = "./models/gemma-2-2b"
TEST_FILE = "./data/test_math.jsonl"
LORA_CKPT = Path("./checkpoints/lora_epoch3_step75.pt")
USE_LORA = True  # set False to eval base model only
BATCH_SIZE = 40
MAX_NEW_TOKENS = 205
TOL = 1e-6

def extract_answer(text):
    if text is None:
        return None
    # Prefer a numeric after the keyword "answer", otherwise fall back to last numeric span.
    keyword_matches = list(re.finditer(r"answer[^0-9\-+]*([-+]?\d*\.?\d+)", text, re.IGNORECASE))
    if keyword_matches:
        try:
            cleaned = keyword_matches[-1].group(1).replace(",", "")
            return float(cleaned)
        except Exception:
            pass

    matches = list(re.finditer(r"[-+]?\d*\.?\d+", text))
    if not matches:
        return None
    try:
        cleaned = matches[-1].group(0).replace(",", "")
        return float(cleaned)
    except Exception:
        return None

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))

if USE_LORA:
    model = apply_lora_to_model(
        model,
        r=8,
        alpha=16,
        target_modules=("q_proj", "v_proj"),
        dropout=0.0,
    )
    freeze_non_lora_params(model)
    if LORA_CKPT.exists():
        ckpt = torch.load(LORA_CKPT, map_location="cpu")
        missing = model.load_state_dict(ckpt.get("lora_state_dict", {}), strict=False)
        print(f"Loaded LoRA checkpoint {LORA_CKPT} (missing/unexpected: {missing})")
    else:
        print(f"LoRA checkpoint {LORA_CKPT} not found; evaluating base model.")

model.to("mps")
model.eval()

correct, total = 0, 0

with open(TEST_FILE) as f:
    test_data = [json.loads(line) for line in f]

# Compute the maximum question length in tokens so we can set a no-truncation limit
question_lengths = [
    len(tokenizer(q, add_special_tokens=True)["input_ids"])
    for q in (sample["question"] for sample in test_data)
]
MAX_INPUT_TOKENS = max(question_lengths) if question_lengths else 0
print(f"Max question tokens: {MAX_INPUT_TOKENS}")

for idx in tqdm(range(0, len(test_data), BATCH_SIZE)):
    batch = test_data[idx : idx + BATCH_SIZE]
    questions = [sample["question"] for sample in batch]
    golds = [str(sample["gold_answer"]).strip() for sample in batch]

    inputs = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy decode; temperature/top-k/p ignored
        )

    for out, gold in zip(outputs, golds):
        text = tokenizer.decode(out, skip_special_tokens=True)
        try:
            gold_val = float(gold.replace(",", ""))
        except Exception:
            continue
        pred = extract_answer(text)
        if pred is None:
            continue
        print("pred is {} and gold is {}".format(pred, gold))
        if abs(pred - gold_val) <= TOL:
            correct += 1
        total += 1
    print("total questions processed is {} and correct answer is {}".format(total, correct))

accuracy = correct / total * 100

print(f"\n--- Baseline Evaluation ---")
print(f"Model: Gemma 2B Instruct{' + LoRA' if USE_LORA and LORA_CKPT.exists() else ''}")
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
