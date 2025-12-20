import torch
import pytest

from sampler import sample_k_parallel


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=10, pad_id=0, eos_id=9):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = type("cfg", (), {"pad_token_id": pad_id, "eos_token_id": eos_id})()
        # simple embedding to produce logits
        self.embed = torch.nn.Linear(1, vocab_size)

    @property
    def generation_config(self):
        return self.config

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None, **kwargs):
        # produce deterministic logits ignoring input content
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self.vocab_size, device=input_ids.device)
        logits[:, :, 1] = 5.0  # favor token id 1
        logits[:, :, self.config.eos_token_id] = -5.0
        return type("Out", (), {"logits": logits, "past_key_values": None})


class DummyTokenizer:
    def __init__(self, pad_token_id=0, eos_token_id=9):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, text, return_tensors=None):
        # encode each char as 1 for simplicity
        ids = torch.tensor([[1] * len(text)], dtype=torch.long)
        return {"input_ids": ids}

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["decoded"] * sequences.size(0)


def test_sample_k_parallel_shapes():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    res = sample_k_parallel(
        model=model,
        tokenizer=tokenizer,
        prompt="hi",
        k=3,
        device="cpu",
        dtype=torch.float32,
        temperature=1.0,
        max_new_tokens=2,
        enable_grad=False,
    )
    assert res["tokens"].shape[0] == 3
    assert res["sum_token_logprobs"].shape[0] == 3
    assert len(res["text"]) == 3
    assert len(res["truncated"]) == 3


def test_truncated_flags():
    model = DummyModel()
    tokenizer = DummyTokenizer()
    res = sample_k_parallel(
        model=model,
        tokenizer=tokenizer,
        prompt="hi",
        k=2,
        device="cpu",
        dtype=torch.float32,
        temperature=1.0,
        max_new_tokens=1,
        enable_grad=False,
    )
    # with no eos generated, both should be marked truncated
    assert all(res["truncated"])


def test_padding_after_eos():
    # Make the model emit eos_token_id for the first row immediately
    class EosFirstModel(DummyModel):
        def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None, **kwargs):
            batch, seq = input_ids.shape
            logits = torch.zeros(batch, seq, self.vocab_size, device=input_ids.device)
            logits[:, :, self.config.eos_token_id] = 10.0  # force eos
            return type("Out", (), {"logits": logits, "past_key_values": None})

    model = EosFirstModel(pad_id=0, eos_id=9)
    tokenizer = DummyTokenizer(pad_token_id=0, eos_token_id=9)
    res = sample_k_parallel(
        model=model,
        tokenizer=tokenizer,
        prompt="hi",
        k=2,
        device="cpu",
        dtype=torch.float32,
        temperature=1.0,
        max_new_tokens=2,
        enable_grad=False,
    )
    tokens = res["tokens"]
    mask = res["attention_mask"]
    # Row 0 should contain EOS; mask should be 0 after EOS token
    eos_positions = (tokens[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    assert len(eos_positions) > 0
    eos_pos = eos_positions[0].item()
    # Positions after EOS should be padding (or ignored) and mask 0
    if eos_pos + 1 < tokens.size(1):
        assert mask[0, eos_pos + 1:].sum().item() == 0
