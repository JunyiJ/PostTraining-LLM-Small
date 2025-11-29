from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="./models/gemma-2-2b"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="mps",
        dtype="float16"
    )
    return tokenizer, model