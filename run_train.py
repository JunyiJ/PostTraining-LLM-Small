import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from grpo.utils import load_model
from grpo.sampler import sample_k
from grpo.advantage import compute_advantage

MODEL_PATH = Path(__file__).resolve().parent / "models" / "gemma-2-2b"
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "math_grpo_200.jsonl"
NUM_SAMPLES_PER_PROMPT = 2
NUM_TRAINING_DATA = 1

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))

prompts = ["Hello world", "1+1=?"]

# Sample multiple completions per prompt for demonstration
for prompt in prompts:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("mps")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=1.0
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Example advantage computation over dummy rewards
dummy_rewards = [1.0, 0.5, 0.0]
print(f"Advantages for {dummy_rewards}: {compute_advantage(dummy_rewards)}")



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

"""
load data
load model (with LoRA enabled)
for each batch:
    generate k initial answers
    generate k refined answers
    compute rewards
    compute advantages
    recompute logprobs with grad
    compute GRPO loss
    backprop
    step optimizer
    periodically evaluate
"""
# Load training data
with open(TRAIN_FILE) as f:
    test_data = [json.loads(line) for line in f]

for line in tqdm(test_data[:NUM_TRAINING_DATA]):
    question = line['question']
    print(f"question is {question}")
    gold_answer = str(line["gold_answer"]).strip()
    print(f"gold_answer is {gold_answer}")
    with torch.no_grad():
        samples = sample_k(
            model,
            tokenizer,
            question,
            k=NUM_SAMPLES_PER_PROMPT,
            temperature=0.2,
            max_new_tokens=50,
        )
        for i, r in enumerate(samples, start=1):
            print(f"\nSample {i}: {r['text']}")
            print(f"logits size : {len(r['logits'])}")
            print(f"probs size : {len(r['probs'])}")
