import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="./models/gemma-2-2b"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,  # Using float32/bf16 instead of FP16 e MPS FP16 matmul has limited exponent range and MPS has buggy FP16 softmax
        device_map={"": "cpu"},  # Load model to cpu first and later moved to desired device to avoid the hugging face buggy warmup with FP16
        low_cpu_mem_usage=True,
    )
    model = model.to("mps")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return tokenizer, model