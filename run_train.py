import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from grpo.utils import load_model
from grpo.sampler import sample_k
from grpo.advantage import compute_advantage
from grpo.reward import compute_reward

MODEL_PATH = Path(__file__).resolve().parent / "models" / "gemma-2-2b"
TRAIN_FILE = Path(__file__).resolve().parent / "data" / "math_grpo_200.jsonl"
NUM_SAMPLES_PER_PROMPT = 2
NUM_TRAINING_DATA = 1
SAMPLING_TEMPERATURE = 0.7
DEVICE = torch.device("mps")

# Load model/tokenizer using helper
tokenizer, model = load_model(str(MODEL_PATH))
model.to(DEVICE)

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

# Example advantage computation over dummy rewards
dummy_rewards = [1.0, 0.5, 0.0]
print(f"Advantages for {dummy_rewards}: {compute_advantage(dummy_rewards)}")



def extract_answer(text):
    if text is None:
        return None
    # Find all numeric spans and pick the last one (closest to the end of the output)
    # Matches numbers like: 3, -2, 3.1415, 0.00001, .5, -0.25
    matches = list(re.finditer(r"[-+]?\d*\.?\d+", text))
    if not matches:
        return None

    last_match = matches[-1].group(0)
    try:
        cleaned = last_match.replace(",", "")
        return float(cleaned)
    except Exception:
        return None

"""
load data
load model (with LoRA enabled)
for each batch:
    generate k initial answers
    generate k refined answers
    compute rewards
    compute advantages
    recompute logprobs with grad
    compute GRPO loss
    backprop
    step optimizer
    periodically evaluate
"""
# Load training data
with open(TRAIN_FILE) as f:
    test_data = [json.loads(line) for line in f]

# TODO load model with Lora and only enable Lora params for optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for line in tqdm(test_data[:NUM_TRAINING_DATA]):
    question = line['question']
    print(f"question is {question}")
    gold_answer = str(line["gold_answer"]).strip()
    print(f"gold_answer is {gold_answer}")
    print("start sampling passes")
    # Generate K initial answers and get each answer token's logprob and old_logprob
    # for the answer.
    model.eval()    # disable dropout
    with torch.no_grad():
        samples = sample_k(
            model,
            tokenizer,
            question,
            k=NUM_SAMPLES_PER_PROMPT,
            temperature=SAMPLING_TEMPERATURE,
            max_new_tokens=100,
        )
    # Second pass to get each answer token's logprob and new_logprob
    model.train()
    with torch.enable_grad():
        rewards = []
        new_logprobs = []
        old_logprobs = []
        for i, r in enumerate(samples, start=1):
            print(f"\nSample {i}: {r['text']}")
            print(f"prompt token id length: {r['prompt_id_length']}")
            print(f"tokens size: {r['tokens'].shape}")
            print(f"sum answer token logprobs: {r['sum_token_logprobs']}")
            print(f"token logprobs shape: {r['token_logprobs'].shape}")
            print('second pass')
            
            tokens = r["tokens"].to(DEVICE)
            attention_mask = torch.ones_like(tokens)
            out = model(input_ids=tokens, attention_mask=attention_mask)
            logits = out.logits  # [batch, seq_length, vocab]
            # Shift logits vs targets: logits at t predict token at t+1
            shifted_log_probs = F.log_softmax(logits / SAMPLING_TEMPERATURE, dim=-1)[:, :-1, :]
            targets = tokens[:, 1:].unsqueeze(-1)
            log_probs = shifted_log_probs.gather(-1, targets).squeeze(-1)  # [batch, seq_length-1]
            print(f'shape of log_probs: {log_probs.shape}')
            answer_log_probs = log_probs[:, r['prompt_id_length'] - 1:]
            print(f'shape of answer_log_probs: {answer_log_probs.shape}')
            sum_token_logprobs_new = answer_log_probs.sum(dim=1)
            print(f"sum answer token logprobs new: {sum_token_logprobs_new}")

            # Start to prepare for the GRPO loss
            reward = compute_reward(question, r['text'], gold_answer)
            rewards.append(reward)
            new_logprobs.append(sum_token_logprobs_new.squeeze(0))
            old_logprobs.append(r['sum_token_logprobs'].to(DEVICE).squeeze(0))

        advantages = torch.tensor(
            compute_advantage(rewards),
            device=DEVICE,
            dtype=new_logprobs[0].dtype if new_logprobs else torch.float32,
        )
        old_logprobs_t = torch.stack(old_logprobs)
        new_logprobs_t = torch.stack(new_logprobs)
        print(f"new_logprobs_t shape: {new_logprobs_t.shape}")
        log_prob_ratio = new_logprobs_t - old_logprobs_t
        loss = -(advantages * log_prob_ratio.exp()).mean()
        print(f"loss: {loss.item()}")
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    print("==end-of-training-sample==")
