# Script to evaluate the model's performance on the test math dataset
import json, re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

MODEL_PATH = "./models/gemma-2-2b"
TEST_FILE = "./data/test_math.jsonl"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 128
TOL = 1e-6

def extract_answer(text):
    if text is None:
        return None
    # Find all numeric spans and pick the last one (closest to the end of the output)
    # Matches numbers like: 3, -2, 3.1415, 0.00001, .5, -0.25
    matches = list(re.finditer(r"[-+]?\d*\.?\d+", text))
    if not matches:
        return None

    last_match = matches[-1].group(0)
    try:
        cleaned = last_match.replace(",", "")
        return float(cleaned)
    except Exception:
        return None

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="mps",
    dtype=torch.float16,
)
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
print(f"Model: Gemma 2B Instruct")
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
