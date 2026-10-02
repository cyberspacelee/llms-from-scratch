"""Byte BPE, embedding lookup, and repeated-token gradient checks on CPU."""

import json
import re
from collections import Counter
from pathlib import Path

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

    def __init__(self, special_tokens=()):
        if any(not isinstance(s, str) or not s for s in special_tokens) or len(
            set(special_tokens)
        ) != len(special_tokens):
            raise ValueError("special token spellings must be nonempty and unique")
        self.pieces = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.special_tokens = tuple(special_tokens)

    @property
    def special_ids(self):
        return {s: len(self.pieces) + i for i, s in enumerate(self.special_tokens)}

    @property
    def vocab_size(self):
        return len(self.pieces) + len(self.special_tokens)

    def _parts(self, text):
        if not self.special_tokens:
            return [text]
        pattern = (
            "("
            + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
            + ")"
        )
        return re.split(pattern, text)

    def fit(self, documents, num_merges):
        if num_merges < 0 or self.merges:
            raise ValueError("use a fresh tokenizer and a nonnegative merge count")
        sequences = [
            list(part.encode("utf-8"))
            for text in documents
            for part in self._parts(text)
            if part not in self.special_ids
        ]
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

    def _encode_plain(self, text):
        ids = list(text.encode("utf-8"))
        for pair, merged_id in self.merges:
            ids = merge_pair(ids, pair, merged_id)
        return ids

    def encode(self, text, allowed_special=False):
        if not allowed_special:
            return self._encode_plain(text)
        return [
            i
            for part in self._parts(text)
            for i in (
                [self.special_ids[part]] if part in self.special_ids else self._encode_plain(part)
            )
        ]

    def decode(self, ids, errors="strict"):
        pieces = {**self.pieces, **{i: s.encode("utf-8") for s, i in self.special_ids.items()}}
        if any(i not in pieces for i in ids):
            raise ValueError("unknown token ID")
        return b"".join(pieces[i] for i in ids).decode("utf-8", errors=errors)

    def state_dict(self):
        return dict(
            version=1,
            kind=type(self).__name__,
            special_tokens=list(self.special_tokens),
            merges=[[a, b] for (a, b), _ in self.merges],
        )

    @staticmethod
    def from_state_dict(state):
        if (
            not isinstance(state, dict)
            or state.get("version") != 1
            or state.get("kind") not in ("ByteBPE", "PreSplitBPE")
        ):
            raise ValueError("unsupported tokenizer state")
        if not isinstance(state.get("special_tokens"), list) or not isinstance(
            state.get("merges"), list
        ):
            raise ValueError("tokenizer state needs special_tokens and merges lists")
        tokenizer = (PreSplitBPE if state["kind"] == "PreSplitBPE" else ByteBPE)(
            state["special_tokens"]
        )
        for pair in state["merges"]:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(i) is not int or i not in tokenizer.pieces for i in pair)
            ):
                raise ValueError("merge references must precede their output ID")
            if any(old_pair == tuple(pair) for old_pair, _ in tokenizer.merges):
                raise ValueError("duplicate merge rule")
            merged_id = len(tokenizer.pieces)
            tokenizer.pieces[merged_id] = tokenizer.pieces[pair[0]] + tokenizer.pieces[pair[1]]
            tokenizer.merges.append((tuple(pair), merged_id))
        return tokenizer

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.state_dict(), ensure_ascii=True, indent=2), encoding="utf-8"
        )

    @staticmethod
    def load(path):
        return ByteBPE.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# A simplified, ASCII-only version of GPT-2 style pre-tokenization: a word keeps its leading space.
PRE_SPLIT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


class PreSplitBPE(ByteBPE):
    """Byte BPE whose merges never cross pre-tokenized chunk boundaries."""

    def fit(self, documents, num_merges):
        return super().fit(
            [
                chunk
                for text in documents
                for part in self._parts(text)
                if part not in self.special_ids
                for chunk in PRE_SPLIT.findall(part)
            ],
            num_merges,
        )

    def _encode_plain(self, text):
        return [i for chunk in PRE_SPLIT.findall(text) for i in super()._encode_plain(chunk)]


def verify():
    torch.manual_seed(7)
    main = ByteBPE(["<bos>", "<eos>"]).fit(["猫 sat"] * 4, 2)
    assert main.merges == [((32, 115), 256), ((97, 116), 257)]
    main_ids = [231, 140, 171, 256, 257]
    assert main.encode("猫 sat") == main_ids
    assert main.encode("猫 sat!") == main_ids + [33]
    assert main.decode(main_ids) == "猫 sat"
    assert main.special_ids == {"<bos>": 258, "<eos>": 259}
    assert main.vocab_size == 260
    assert main.encode("<eos>") != [259]
    assert main.encode("<eos>", allowed_special=True) == [259]
    main_batch = torch.tensor([main_ids, main_ids])
    main_embedding = nn.Embedding(main.vocab_size, 2).double()
    main_selected = main_embedding(main_batch)
    assert main_selected.shape == (2, 5, 2)
    torch.testing.assert_close(
        main_selected,
        torch.nn.functional.one_hot(main_batch, main.vocab_size).double() @ main_embedding.weight,
    )
    upstream = torch.zeros_like(main_selected)
    upstream[0, 3] = torch.tensor([1.0, 2.0])
    upstream[1, 3] = torch.tensor([3.0, -1.0])
    main_selected.backward(upstream)
    torch.testing.assert_close(
        main_embedding.weight.grad[256], torch.tensor([4.0, 1.0], dtype=torch.float64)
    )
    assert torch.count_nonzero(main_embedding.weight.grad).item() == 2
    tokenizer = ByteBPE().fit(["aba"] * 4 + ["abb"] * 2 + ["bab"], 2)
    assert tokenizer.merges == [((97, 98), 256), ((256, 97), 257)]
    assert tokenizer.encode("aba") == [257]
    assert tokenizer.encode("abb") == [256, 98]
    assert tokenizer.encode("bab") == [98, 256]
    for text in ("", "aba", "abb", "bab", "Hello, world!", "\u4e2d\u6587\u548c AI"):
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert merge_pair([97, 97, 97], (97, 97), 256) == [256, 97]
    # Without pre-splitting, a frequent phrase merges across the space into one token.
    raw = ByteBPE().fit(["a b"] * 5, 2)
    assert raw.pieces[257] == b"a b" and raw.encode("a b") == [257]
    split = PreSplitBPE().fit(["a b"] * 5, 2)
    assert PRE_SPLIT.findall("a b") == ["a", " b"]
    assert [split.pieces[i] for i in split.encode("a b")] == [b"a", b" b"]
    assert all(b" " not in piece[1:] for piece in split.pieces.values())
    assert split.decode(split.encode("Hi, a b 42!")) == "Hi, a b 42!"
    print(
        "PASS: 猫 sat byte BPE, enexamples/decode, special IDs, embedding lookup and repeated-ID gradients"
    )
    print("PASS: BPE weighted pairs, merge order, overlap, UTF-8 round trips, pre-split boundaries")
    import tempfile

    for cls in (ByteBPE, PreSplitBPE):
        special = cls(["<eos>", "<bos>"]).fit(["abc<eos>abc", "cab"], 3)
        text = "abc<eos>cab"
        encoded = special.encode(text, allowed_special=True)
        assert special.special_ids["<eos>"] in encoded
        assert special.special_ids["<eos>"] not in special.encode(text)
        assert special.decode(encoded) == text
        assert all(b"<eos>" not in p for p in special.pieces.values())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            special.save(path)
            loaded = ByteBPE.load(path)
            assert loaded.state_dict() == special.state_dict()
            assert loaded.encode(text, allowed_special=True) == encoded
        bad = special.state_dict()
        bad["merges"] = [[9999, 1]]
        try:
            ByteBPE.from_state_dict(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid tokenizer merge accepted")
    print("PASS: special-token opt-in/boundaries and JSON tokenizer round trips")


if __name__ == "__main__":
    verify()
