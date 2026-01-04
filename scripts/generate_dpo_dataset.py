import json
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from pathlib import Path

PROMPT = " Please reason step-by-step,  then give: Final answer. "

# --- CONFIGURATION ---
# OPTION 1: specialized Step-DPO dataset (Recommended)
DATASET_CONFIG = {
    "name": "xinlai/Math-Step-DPO-10K",
    "split": "train",
    "cols": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}
}

# OPTION 2: Orca (Only GSM8K subset)
# DATASET_CONFIG = {
#     "name": "argilla/distilabel-intel-orca-dpo-pairs",
#     "split": "train",
#     "cols": {"prompt": "input", "chosen": "chosen", "rejected": "rejected"},
#     "filter_fn": lambda x: x.get('in_gsm8k_train', False) # Only use confirmed GSM8K data
# }

OUTPUT_FILE = "./data/gsm8k_dpo_pairs.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "gemma-2-2b"
MODEL_ID = str(MODEL_PATH) if MODEL_PATH.exists() else "google/gemma-2-2b"
MAX_LENGTH = 400

def process_dpo_data():
    print(f"Loading tokenizer: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print("Loading GSM8K Test Set (Blocklist)...")
    gsm8k_test = load_dataset("gsm8k", "main", split="test")
    # Normalized blocklist (strip whitespace and lower case for strict safety)
    forbidden_prompts = set(q.strip().lower() for q in gsm8k_test['question'])
    
    print(f"Loading Source Dataset: {DATASET_CONFIG['name']}...")
    ds = load_dataset(DATASET_CONFIG['name'], split=DATASET_CONFIG['split'])
    
    # Apply special filters (like checking for 'in_gsm8k_train' column)
    if "filter_fn" in DATASET_CONFIG:
        print("Applying dataset-specific internal filters...")
        ds = ds.filter(DATASET_CONFIG["filter_fn"])

    kept = 0
    filtered_leakage = 0
    filtered_length = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in tqdm(ds, desc="Processing"):
            
            # 1. Map columns based on config
            prompt = entry.get(DATASET_CONFIG['cols']['prompt'])
            chosen = entry.get(DATASET_CONFIG['cols']['chosen'])
            rejected = entry.get(DATASET_CONFIG['cols']['rejected'])
            
            if not prompt or not chosen or not rejected:
                continue
                
            # 2. Safety Check (Leakage)
            # We check if the prompt exists in the GSM8K test set
            if prompt.strip().lower() in forbidden_prompts:
                filtered_leakage += 1
                continue
                
            # 3. Length Check
            # Check chosen/rejected response length
            len_c = len(tokenizer.encode(prompt + PROMPT + chosen, add_special_tokens=False))
            len_r = len(tokenizer.encode(prompt + PROMPT + rejected, add_special_tokens=False))
            
            if len_c > MAX_LENGTH or len_r > MAX_LENGTH:
                filtered_length += 1
                continue
                
            # 4. Write
            out_entry = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }
            f.write(json.dumps(out_entry) + "\n")
            kept += 1

    print("\n--- RESULTS ---")
    print(f"Source size: {len(ds)}")
    print(f"🚫 Leakage Removed: {filtered_leakage}")
    print(f"✂️  Too Long (> {MAX_LENGTH}): {filtered_length}")
    print(f"✅ Final Dataset: {kept}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dpo_data()
