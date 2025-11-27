from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use the locally downloaded Gemma 2B Instruct weights
model_name = str(Path(__file__).resolve().parent / "models" / "gemma-2-2b")
print(model_name)

t = AutoTokenizer.from_pretrained(model_name)
m = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",
    torch_dtype=torch.float16,
)

input_ids = t("Hello world", return_tensors="pt").input_ids.to("mps")
out = m.generate(input_ids, max_new_tokens=50)
print(len(out))
print(t.decode(out[0]))
