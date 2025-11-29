from pathlib import Path

from utils import load_model

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gemma-2-2b"
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