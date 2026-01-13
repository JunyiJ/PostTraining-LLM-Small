from pathlib import Path

try:
    from grpo.device import get_default_device
    from grpo.utils import load_model
except ImportError:
    from device import get_default_device
    from utils import load_model

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gemma-2-2b"
DEVICE = get_default_device()
# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH), device=DEVICE)

prompts = ["Hello world", "1+1=?"]

# Sample multiple completions per prompt for demonstration
for prompt in prompts:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=1.0
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
