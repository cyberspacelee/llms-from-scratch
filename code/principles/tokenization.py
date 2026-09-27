"""Byte BPE, embedding lookup, and repeated-token gradient checks on CPU."""

from collections import Counter

import torch
from torch import nn


def merge_pair(ids, pair, merged_id):
    result = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            result.append(merged_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


class ByteBPE:
    """Teaching tokenizer: raw UTF-8, no normalization or pre-tokenization."""

    def __init__(self):
        self.pieces = {i: bytes([i]) for i in range(256)}
        self.merges = []

    def fit(self, documents, num_merges):
        if num_merges < 0 or self.merges:
            raise ValueError("use a fresh tokenizer and a nonnegative merge count")
        sequences = [list(text.encode("utf-8")) for text in documents]
        if not sequences or not any(sequences):
            raise ValueError("training documents must contain text")
        for _ in range(num_merges):
            counts = Counter(pair for ids in sequences for pair in zip(ids, ids[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            merged_id = len(self.pieces)
            self.pieces[merged_id] = self.pieces[pair[0]] + self.pieces[pair[1]]
            self.merges.append((pair, merged_id))
            sequences = [merge_pair(ids, pair, merged_id) for ids in sequences]
        return self

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for pair, merged_id in self.merges:
            ids = merge_pair(ids, pair, merged_id)
        return ids

    def decode(self, ids):
        if any(i not in self.pieces for i in ids):
            raise ValueError("unknown token ID")
        return b"".join(self.pieces[i] for i in ids).decode("utf-8")


def verify():
    torch.manual_seed(7)
    tokenizer = ByteBPE().fit(["aba"] * 4 + ["abb"] * 2 + ["bab"], 2)
    assert tokenizer.merges == [((97, 98), 256), ((256, 97), 257)]
    assert tokenizer.encode("aba") == [257]
    assert tokenizer.encode("abb") == [256, 98]
    assert tokenizer.encode("bab") == [98, 256]
    for text in ("", "aba", "abb", "bab", "Hello, world!", "\u4e2d\u6587\u548c AI"):
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert merge_pair([97, 97, 97], (97, 97), 256) == [256, 97]
    ids = torch.tensor([[1, 2, 1], [3, 1, 0]])
    embedding = nn.Embedding(8, 4).double()
    selected = embedding(ids)
    one_hot = torch.nn.functional.one_hot(ids, 8).to(torch.float64)
    torch.testing.assert_close(selected, one_hot @ embedding.weight)
    selected.sum().backward()
    counts = torch.bincount(ids.flatten(), minlength=8).double()
    torch.testing.assert_close(embedding.weight.grad, counts[:, None].expand(8, 4))
    assert ids[0, 0] == ids[0, 2]
    torch.testing.assert_close(selected[0, 0], selected[0, 2])
    print("PASS: BPE weighted pairs, merge order, overlap, UTF-8 round trips")
    print("PASS: embedding lookup equals one-hot selection; repeated IDs sum gradients")


if __name__ == "__main__":
    verify()
