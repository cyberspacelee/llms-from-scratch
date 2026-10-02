"""Document splitting and next-token packing, with CPU self-checks."""

import hashlib

import torch


def unique_documents(records):
    seen = {}
    result = []
    for source, text, assignment in records:
        identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if identity in seen:
            if seen[identity] != assignment:
                raise ValueError("Identical text appears in different splits")
            continue
        seen[identity] = assignment
        result.append((identity, source, text, assignment))
    return result


def fit_character_vocabulary(training_documents):
    return {
        character: index
        for index, character in enumerate(
            sorted(set("".join(text for _, text in training_documents))), start=1
        )
    }


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
    records = [
        ("s1", "ab", "train"),
        ("s1", "ab", "train"),
        ("s2", "cd", "train"),
        ("s3", "az", "validation"),
    ]
    unique = unique_documents(records)
    train = [(identity, text) for identity, _, text, split in unique if split == "train"]
    validation = [(identity, text) for identity, _, text, split in unique if split == "validation"]
    assert [text for _, text in train] == ["ab", "cd"]
    assert [text for _, text in validation] == ["az"]
    assert set(i for i, _ in train).isdisjoint(i for i, _ in validation)
    assert {source for _, source, _, split in unique if split == "train"}.isdisjoint(
        {source for _, source, _, split in unique if split == "validation"}
    )
    try:
        unique_documents([("s1", "ab", "train"), ("s4", "ab", "validation")])
    except ValueError:
        pass
    else:
        raise AssertionError("Cross-split duplicates must be rejected")
    vocabulary = fit_character_vocabulary(train)
    assert vocabulary == {"a": 1, "b": 2, "c": 3, "d": 4}
    unk_id, eos_id = 0, len(vocabulary) + 1

    def encode(text):
        return [vocabulary.get(char, unk_id) for char in text]

    assert encode(validation[0][1]) + [eos_id] == [1, 0, 5]
    x, y, mask, valid, positions = pack_documents(
        [encode(text) for _, text in train], eos_id=eos_id
    )
    assert x.tolist() == [1, 2, 5, 3, 4]
    assert y.tolist() == [2, 5, 3, 4, 5]
    assert valid.tolist() == [True, True, False, True, True]
    assert positions.tolist() == [0, 1, 2, 0, 1]
    assert not mask[3, :3].any() and mask.diag().all()
    _, _, continuous_mask, continuous_valid, continuous_positions = pack_documents(
        [encode(text) for _, text in train], eos_id=eos_id, isolate=False
    )
    assert continuous_valid.all() and continuous_mask[3, 0]
    assert continuous_positions.tolist() == [0, 1, 2, 3, 4]
    assert valid.sum().item() == 4
    losses = torch.tensor([1.0, 2.0, 9.0, 3.0, 4.0], dtype=torch.float64)
    assert losses[valid].mean().item() == 2.5
    assert (losses * valid).mean().item() == 2.0
    print("data: 4 records -> 3 unique documents -> 2 train / 1 validation")
    print("isolated packing: inputs", x.tolist(), "targets", y.tolist())


if __name__ == "__main__":
    verify()
