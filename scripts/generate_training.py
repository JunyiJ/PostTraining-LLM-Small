import json
import re
from datasets import load_dataset
from tqdm import tqdm

def prepare_gsm8k_files(sft_count=200, rl_count=2000):
    # 1. Load official GSM8K dataset
    print("📥 Downloading GSM8K...")
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")

    def clean_gsm8k(text):
        # Remove internal calculator tags: <<1+1=2>> -> 2
        text = re.sub(r"<<.*?>>", "", text)
        return text.strip()

    # 2. Generate SFT Warm-up Data (Reasoning + Answer)
    print(f"✍️ Creating SFT file ({sft_count} examples)...")
    with open("./data/gsm8k_sft_warmup.jsonl", "w") as f:
        for i in range(sft_count):
            item = ds_train[i]
            reasoning, gold = item["answer"].split("####")
            f.write(json.dumps({
                "question": item["question"],
                "reasoning": clean_gsm8k(reasoning),
                "gold_answer": gold.strip()
            }) + "\n")

    # 3. Generate GRPO Training Data (Question + Answer)
    # We start from where SFT ended to avoid overlap
    print(f"✍️ Creating GRPO file ({rl_count} examples)...")
    with open("gsm8k_grpo_train.jsonl", "w") as f:
        end_idx = min(sft_count + rl_count, len(ds_train))
        for i in range(sft_count, end_idx):
            item = ds_train[i]
            _, gold = item["answer"].split("####")
            f.write(json.dumps({
                "question": item["question"],
                "gold_answer": gold.strip()
            }) + "\n")

    # 4. Generate Clean Test File
    print(f"✍️ Creating Evaluation file ({len(ds_test)} examples)...")
    with open("gsm8k_test_eval.jsonl", "w") as f:
        for item in ds_test:
            _, gold = item["answer"].split("####")
            f.write(json.dumps({
                "question": item["question"],
                "gold_answer": gold.strip()
            }) + "\n")

    print("✅ All files ready: gsm8k_sft_warmup.jsonl, gsm8k_grpo_train.jsonl, gsm8k_test_eval.jsonl")

if __name__ == "__main__":
    prepare_gsm8k_files()