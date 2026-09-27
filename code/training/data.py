"""Document splitting and next-token packing, with CPU self-checks."""

import hashlib
import random

import torch


def unique_documents(documents):
    seen = set()
    result = []
    for text in documents:
        identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if identity not in seen:
            seen.add(identity)
            result.append((identity, text))
    return result


def split_documents(documents, validation_count, seed=7):
    unique = unique_documents(documents)
    if not 0 < validation_count < len(unique):
        raise ValueError("Both splits must contain at least one unique document")
    random.Random(seed).shuffle(unique)
    return unique[validation_count:], unique[:validation_count]


def fit_character_vocabulary(training_documents):
    return {character: index for index, character in enumerate(sorted(
        set("".join(text for _, text in training_documents))
    ))}


def pack_documents(documents, eos_id, isolate=True):
    if eos_id < 0 or any(not doc or eos_id in doc for doc in documents):
        raise ValueError("Nonempty documents must not contain the reserved EOS ID")
    tokens, owners = [], []
    for owner, document in enumerate(documents):
        tokens.extend([*document, eos_id])
        owners.extend([owner] * (len(document) + 1))
    if len(tokens) < 2:
        raise ValueError("Need at least one prediction")
    x = torch.tensor(tokens[:-1], dtype=torch.long)
    y = torch.tensor(tokens[1:], dtype=torch.long)
    owner_x = torch.tensor(owners[:-1])
    owner_y = torch.tensor(owners[1:])
    causal = torch.ones(len(x), len(x), dtype=torch.bool).tril()
    valid = torch.ones_like(x, dtype=torch.bool)
    positions = torch.arange(len(x))
    if isolate:
        causal &= owner_x[:, None] == owner_x[None, :]
        valid = owner_x == owner_y
        start = 0
        for i in range(len(x)):
            if i and owner_x[i] != owner_x[i - 1]:
                start = i
            positions[i] -= start
    return x, y, causal, valid, positions


def verify():
    documents = ["red cat", "blue dog", "green bird", "red cat", "white fish",
                 "black horse", "orange fox", "pink rabbit", "blue dog", "gray wolf"]
    train, validation = split_documents(documents, 2)
    assert len(train) == 6 and len(validation) == 2
    assert set(i for i, _ in train).isdisjoint(i for i, _ in validation)
    # A character vocabulary is fitted using the training split only.
    vocabulary = fit_character_vocabulary(train)
    assert set(vocabulary) == set("".join(text for _, text in train))
    assert "Z" not in fit_character_vocabulary([("training", "abc")])
    assert "Z" in fit_character_vocabulary([("training", "abc"), ("validation", "Z")])
    windows = ["abcdef"[:4], "abcdef"[2:]]
    assert set(windows[0]) & set(windows[1]) == {"c", "d"}
    x, y, mask, valid, positions = pack_documents([[1, 2], [3, 4]], eos_id=5)
    assert x.tolist() == [1, 2, 5, 3, 4]
    assert y.tolist() == [2, 5, 3, 4, 5]
    assert valid.tolist() == [True, True, False, True, True]
    assert positions.tolist() == [0, 1, 2, 0, 1]
    assert not mask[3, :3].any() and mask.diag().all()
    _, _, continuous_mask, continuous_valid, _ = pack_documents(
        [[1, 2], [3, 4]], eos_id=5, isolate=False
    )
    assert continuous_valid.all() and continuous_mask[3, 0]
    assert valid.sum().item() == 4
    print("data: 10 records -> 8 unique documents -> 6 train / 2 validation")
    print("isolated packing: inputs", x.tolist(), "targets", y.tolist())


if __name__ == "__main__":
    verify()
