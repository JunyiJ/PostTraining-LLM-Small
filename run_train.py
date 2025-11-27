from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use the locally downloaded Gemma 2B Instruct weights
model_name = str(Path(__file__).resolve().parent / "models" / "gemma-2-2b")

t = AutoTokenizer.from_pretrained(model_name)
m = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",
    dtype=torch.float16,
)

input_ids = t(["Hello world", "how many r in strawberry"],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).input_ids.to("mps")
out = m.generate(input_ids, max_new_tokens=50)
print(len(out))
print(t.decode(out[0]))
print(t.decode(out[1]))
