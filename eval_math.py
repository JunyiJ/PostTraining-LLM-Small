"""
Script to evaluate the model's performance on the test math dataset
"""
import json, re, gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from grpo.reward import extract_final_answer
from grpo.utils import load_model
from grpo.lora import apply_lora_to_model, freeze_non_lora_params

MODEL_PATH = "./models/gemma-2-2b"
# MODEL_PATH = "./models/Qwen2.5-Math-1.5B-Instruct"
TEST_FILE = "./data/test_math.jsonl"
LORA_CKPT = Path("./gemma-2-2b-checkpoints/lora_epoch2_step200.pt")
USE_LORA = True  # set False to eval base model only
BATCH_SIZE = 20
MAX_NEW_TOKENS = 400
TOL = 1e-1

prompt = " Please reason step-by-step,  then give: Final answer."

def extract_answer(text):
    if text is None:
        return None
    patterns = [
        r"answer[^0-9\-+]*([-+]?\d*\.?\d+)",          # Answer: 42
        r"final\s+answer[^0-9\-+]*([-+]?\d*\.?\d+)",  # Final answer: 42
        r"=+\s*([-+]?\d*\.?\d+)",                     # x = 42 or = 42
    ]
    for pat in patterns:
        keyword_matches = list(re.finditer(pat, text, re.IGNORECASE))
        if keyword_matches:
            try:
                cleaned = keyword_matches[-1].group(1).replace(",", "")
                return float(cleaned)
            except Exception:
                continue

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
        r=16,
        alpha=32,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        dropout=0.05,
    )
    freeze_non_lora_params(model)
    if LORA_CKPT is not None and LORA_CKPT.exists():
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
    questions = [sample["question"] + prompt for sample in batch]
    golds = [str(sample["gold_answer"]).strip() for sample in batch]

    inputs = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to("mps")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy decode; temperature/top-k/p ignored
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True # Critical for speed
        )
        texts = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

    for q, text, gold in zip(questions, texts, golds):
        print(f"Question is: {q}\n\n")
        print(f"model output is: {text}\n")
        try:
            gold_val = float(gold.replace(",", ""))
        except Exception:
            continue
        pred = extract_final_answer(text)
        if pred is None:
            continue
        print("pred is {} and gold is {}".format(pred, gold))
        if abs(pred - gold_val) <= TOL:
            correct += 1
        total += 1
        print(">>>>>>>>>>>>.")
    del inputs, outputs, texts
    gc.collect()
    torch.mps.empty_cache()
    print("total questions processed is {} and correct answer is {}".format(total, correct))

accuracy = correct / total * 100

print(f"\n--- Baseline Evaluation ---")
print(f"Model: Gemma 2B Instruct{' + LoRA' if USE_LORA and LORA_CKPT.exists() else ''}")
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
