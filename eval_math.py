"""
Script to evaluate the model's performance on the test math dataset
"""
import json, re, gc, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from grpo.reward import extract_final_answer
from grpo.device import get_default_device, empty_cache
from grpo.utils import load_model

MODEL_PATH = "./models/gemma-2-2b"
# MODEL_PATH = "./models/Qwen2.5-Math-1.5B-Instruct"
TEST_FILE = "./data/test_math.jsonl"
LORA_CKPT = Path("./gemma-2-2b-checkpoints/grpo_20260115_epoch1_step100.pt")
USE_LORA = True  # set False to eval base model only
LORA_BACKEND = "grpo"  # "auto", "grpo", "dpo", "ppo"
BATCH_SIZE = 20
MAX_NEW_TOKENS = 300
TOL = 1e-1

prompt = " Reason step-by-step,  then give: Final answer."
DEVICE = get_default_device()

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
tokenizer, model = load_model(str(MODEL_PATH), device=DEVICE)

if USE_LORA:
    if LORA_CKPT is not None and LORA_CKPT.exists():
        ckpt = torch.load(LORA_CKPT, map_location="cpu")
        state_dict = ckpt.get("lora_state_dict", {})

        def sanitize_state_dict(state):
            new_state_dict = {}
            for k, v in state.items():
                if k.startswith("base_model.model."):
                    new_key = k.replace("base_model.model.", "model.")
                elif k.startswith("base_model."):
                    new_key = k.replace("base_model.", "")
                else:
                    new_key = k
                new_state_dict[new_key] = v
            return new_state_dict

        def select_state_dict(state, target_model):
            candidates = [
                ("raw", state),
                ("sanitized", sanitize_state_dict(state)),
            ]
            model_keys = set(target_model.state_dict().keys())
            best_name = None
            best_state = None
            best_score = -1
            for name, cand in candidates:
                score = sum(1 for key in cand.keys() if key in model_keys)
                if score > best_score:
                    best_name = name
                    best_state = cand
                    best_score = score
            if best_name != "raw":
                print(f"Using {best_name} checkpoint keys for loading.")
            return best_state

        def detect_backend(state, ckpt_path):
            if LORA_BACKEND != "auto":
                return LORA_BACKEND
            for key in state.keys():
                if "value_layer" in key:
                    return "ppo"
            ckpt_name = ckpt_path.name.lower()
            if "dpo" in ckpt_name:
                return "dpo"
            if "grpo" in ckpt_name:
                return "grpo"
            return "grpo"

        backend = detect_backend(state_dict, LORA_CKPT)
        print(f"Using LoRA backend: {backend}")

        if backend == "ppo":
            from ppo.lora_critic import Critic, apply_lora_to_model, freeze_non_lora_critic_params

            model = apply_lora_to_model(
                model,
                r=16,
                alpha=32,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                dropout=0.05,
            )
            model = Critic(model)
            freeze_non_lora_critic_params(model)
            model.to(DEVICE)
            selected_state_dict = select_state_dict(state_dict, model)
            missing = model.load_state_dict(selected_state_dict, strict=False)
        else:
            if backend == "dpo":
                from dpo.lora import apply_lora_to_model, freeze_non_lora_params
            else:
                from grpo.lora import apply_lora_to_model, freeze_non_lora_params

            model = apply_lora_to_model(
                model,
                r=16,
                alpha=32,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                dropout=0.05,
            )
            freeze_non_lora_params(model)
            model.to(DEVICE)
            selected_state_dict = select_state_dict(state_dict, model)
            missing = model.load_state_dict(selected_state_dict, strict=False)

        print(f"Loaded LoRA checkpoint {LORA_CKPT}")
        if missing.missing_keys:
            lora_missing = [
                key for key in missing.missing_keys
                if key.endswith(".A.weight") or "lora_A.weight" in key
            ]
            if lora_missing:
                print("⚠️ CRITICAL WARNING: LoRA weights were NOT loaded correctly!")

model.eval()

correct, total = 0, 0
start_time = time.perf_counter()

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
    batch_start = time.perf_counter()
    batch = test_data[idx : idx + BATCH_SIZE]
    questions = [sample["question"] + prompt for sample in batch]
    golds = [str(sample["gold_answer"]).strip() for sample in batch]

    inputs = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy decode; temperature/top-k/p ignored
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True, # Critical for speed
            repetition_penalty=1.0
        )
        texts = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

    for q, text, gold in zip(questions, texts, golds):
        print(f"Question is: {q}\n\n")
        print(f"model output is: {text}\n")
        total += 1
        try:
            clean_gold = str(gold).replace(",", "").replace("$", "").replace("%", "").strip()
            gold_val = float(clean_gold)
        except Exception:
            print(f"⚠️ SKIPPING Question {idx}: Could not parse gold answer '{gold}'. Error: {e}")
            continue
        pred = extract_final_answer(text)
        if pred is None:
            continue
        print("pred is {} and gold is {}".format(pred, gold))
        if abs(pred - gold_val) <= TOL:
            correct += 1
        print(">>>>>>>>>>>>.")
    del inputs, outputs, texts
    gc.collect()
    empty_cache(DEVICE)
    batch_elapsed = time.perf_counter() - batch_start
    batch_count = len(batch)
    avg_per_sample = batch_elapsed / batch_count if batch_count else 0.0
    print(f"Batch time: {batch_elapsed:.2f}s ({batch_count} samples, {avg_per_sample:.2f}s/sample)")
    print("total questions processed is {} and correct answer is {}".format(total, correct))

accuracy = correct / total * 100

total_elapsed = time.perf_counter() - start_time
avg_elapsed = total_elapsed / total if total else 0.0
print(f"\n--- Baseline Evaluation ---")
print(f"Model: Gemma 2B Instruct{' + LoRA' if USE_LORA and LORA_CKPT.exists() else ''}")
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Elapsed: {total_elapsed:.2f}s ({avg_elapsed:.2f}s/sample)")
