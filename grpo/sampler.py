def sample_k(model, tokenizer, prompt, k, max_new_tokens=256):
    samples = []
    for _ in range(k):
        out = model.generate(
            tokenizer.encode(prompt, return_tensors="pt").to("mps"),
            do_sample=True,
            temperature=1.0,
            top_k=40,
            max_new_tokens=max_new_tokens
        )
        samples.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return samples