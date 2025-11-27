import json
import random
from datasets import load_dataset

OUTPUT_FILE = "./data/test_math.jsonl"
NUM_SAMPLES = 200

def extract_gold_answer(answer_text):
    """
    GSM8K answers often end with '#### <answer>'
    We extract the integer/string after ####.
    """
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()

def main():
    print("Loading GSM8K dataset...")
    ds = load_dataset("gsm8k", "main")["test"]     # test split preferred for evaluation

    print(f"Total test samples: {len(ds)}")

    # Randomly select 200 samples
    samples = random.sample(list(ds), NUM_SAMPLES)

    print(f"Saving {NUM_SAMPLES} samples to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w") as f:
        for i, item in enumerate(samples):
            entry = {
                "id": f"gsm8k_{i}",
                "question": item["question"],
                "gold_answer": extract_gold_answer(item["answer"])
            }
            f.write(json.dumps(entry) + "\n")

    print("Done! File written to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()