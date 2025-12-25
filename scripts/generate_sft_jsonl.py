import json
import re
from datasets import load_dataset

OUTPUT_FILE = "./data/gsm8k_math_sft.jsonl"
NUM_SAMPLES = 300

def generate_sft_jsonl(output_path="sft_logic_data.jsonl", count=200):
    # Load official GSM8K training set
    ds = load_dataset("openai/gsm8k", "main", split="train")
    
    with open(output_path, "w") as f:
        for i in range(min(count, len(ds))):
            item = ds[i]
            # Clean OpenAI's internal calculation tags <<1+1=2>>
            reasoning_clean = re.sub(r"<<.*?>>", "", item["answer"])
            # Split reasoning from the final answer (OpenAI uses #### as separator)
            steps, final_val = reasoning_clean.split("####")
            
            sft_entry = {
                "question": item["question"],
                "reasoning": steps.strip(),
                "gold_answer": final_val.strip()
            }
            f.write(json.dumps(sft_entry) + "\n")
    
    print(f"✅ Successfully generated {count} SFT examples in {output_path}")

generate_sft_jsonl(OUTPUT_FILE, count=NUM_SAMPLES)