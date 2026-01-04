import torch

from dpo.helper import get_tokens_and_masks


class FakeTokenizer:
    def __init__(self, eos_token_id=2, pad_token_id=0):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self._vocab = {"<pad>": pad_token_id, "<eos>": eos_token_id}
        self._next_id = max(self._vocab.values()) + 1

    def _encode_text(self, text):
        ids = []
        for tok in text.split():
            if tok == "<eos>":
                ids.append(self.eos_token_id)
            else:
                if tok not in self._vocab:
                    self._vocab[tok] = self._next_id
                    self._next_id += 1
                ids.append(self._vocab[tok])
        return ids

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=None):
        encoded = [self._encode_text(t) for t in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        max_len = max(len(ids) for ids in encoded) if padding else None
        padded = []
        attention = []
        for ids in encoded:
            if padding:
                pad_len = max_len - len(ids)
                padded.append(ids + [self.pad_token_id] * pad_len)
                attention.append([1] * len(ids) + [0] * pad_len)
            else:
                padded.append(ids)
                attention.append([1] * len(ids))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def test_answer_mask_includes_response_and_eos():
    tokenizer = FakeTokenizer()
    device = torch.device("cpu")
    prompts = ["p1 p2 p3", "q1 q2"]
    responses = [
        "p1 p2 p3 a1 a2 <eos>",
        "q1 q2 b1 <eos>",
        "p1 p2 p3 c1 <eos>",
        "q1 q2 d1 d2 <eos>",
    ]

    _, _, answer_mask = get_tokens_and_masks(
        prompts,
        responses,
        tokenizer,
        device,
    )

    expected = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(answer_mask, expected)


def test_answer_mask_fallback_without_eos():
    tokenizer = FakeTokenizer()
    device = torch.device("cpu")
    prompts = ["p1 p2", "q1"]
    responses = [
        "p1 p2 a1 a2",
        "q1 b1",
        "p1 p2 c1",
        "q1 d1 d2",
    ]

    _, _, answer_mask = get_tokens_and_masks(
        prompts,
        responses,
        tokenizer,
        device,
    )

    expected = torch.tensor(
        [
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(answer_mask, expected)
