import json
import os
import torch
from tqdm import tqdm

from pathlib import Path
from grpo.sampler import sample_k_parallel
from grpo.reward import extract_final_answer
from grpo.utils import load_model

# To avoid the known issue of gemma2 x MPS memory allocator bug.
# This hapens because hugging face automatically runs FP16 warmup allocations
# even request fp32 or bfloat16
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TRANSFORMERS_NO_MPS_CACHE_ALLOCATOR"] = "1"

MODEL_PATH = Path(__file__).resolve().parent / "models" / "gemma-2-2b"
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "math_grpo_200.jsonl"
OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "math_grpo_hard_mining.jsonl"

NUM_SAMPLES_PER_PROMPT = 4
NUM_TRAINING_DATA = 50
NUM_EPOCHS = 8
EVAL_EVERY = 10
SAMPLING_TEMPERATURE = 0.8
MAX_NEW_TOKENS = 300
DEVICE = torch.device("mps")
PROMPT = " Please reason step-by-step,  then give: Final answer."

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))

def mine_hard_examples(model, tokenizer, input_file, output_file, k=NUM_SAMPLES_PER_PROMPT):
    hard_examples = []
    
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    print(f"🔍 Mining {len(data)} examples for training signal...")
    model.eval()

    for item in tqdm(data[10:100]):
        question = item['question']
        gold = str(item["gold_answer"]).strip()
        gold = float(str(gold).replace(",", ""))
        
        with torch.no_grad():
            # Run a quick sample pass
            res = sample_k_parallel(
                model, tokenizer, question + PROMPT,
                k=k, temperature=SAMPLING_TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS
            )
            
            # Count correct answers using your robust extraction
            correct_count = 0
            for text in res['text']:
                pred = extract_final_answer(text)
                if pred is not None and abs(pred - gold) < 1e-2:
                    correct_count += 1
        
        # --- THE SELECTION LOGIC ---
        # 0/5 correct: Model is totally lost (Keep for 'Discovery' learning)
        # 1/5 to 4/5 correct: PERFECT training signal (High Advantage variance)
        # 5/5 correct: Model already knows this (SKIP - causes 0.0 loss)
        print(item)
        print(correct_count)
        print(correct_count / k)
        if correct_count < k:
            hard_examples.append(item)
    
    mode = "a" if output_file.exists() else "w"
    with open(output_file, mode) as f:
        for item in hard_examples:
            f.write(json.dumps(item) + "\n")
            
    print(f"✅ Done! Found {len(hard_examples)} usable examples out of {len(data)}.")
    print(f"🚀 Training on this set will prevent the 0.0 GRPO loss issue.")

# Run it
mine_hard_examples(model, tokenizer, TRAIN_FILE, OUTPUT_FILE)
