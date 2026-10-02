"""A tiny LM training run and exact CPU checkpoint recovery."""

import copy
import io

import torch
from torch.utils.flop_counter import FlopCounterMode

from llms_from_scratch import Transformer, basic_decoder_config
from llms_from_scratch import token_loss as sequence_loss

CONFIG = dict(vocab_size=4, dim=16, heads=2, ff_dim=32, layers=2, max_length=12)
TRAIN = torch.tensor(
    [
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 3],
        [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3],
        [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3],
    ]
)
VALIDATION = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 3], [2, 0, 1, 2, 0, 1, 2, 0, 1, 3]])


def new_run():
    model = Transformer(basic_decoder_config(**CONFIG)).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01, foreach=False)
    return model, optimizer


def train_steps(model, optimizer, count, cursor=0, order=None):
    model.train()
    order = list(range(len(TRAIN))) if order is None else list(order)
    losses = []
    for _ in range(count):
        if cursor == len(order):
            order = torch.randperm(len(TRAIN)).tolist()
            cursor = 0
        index = order[cursor]
        cursor += 1
        document = TRAIN[index : index + 1]
        optimizer.zero_grad(set_to_none=True)
        loss = sequence_loss(model(document[:, :-1]).logits, document[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return cursor, order, losses


@torch.no_grad()
def evaluate(model):
    model.eval()
    return sequence_loss(model(VALIDATION[:, :-1]).logits, VALIDATION[:, 1:]).item()


def verify():
    torch.set_num_threads(1)
    torch.manual_seed(29)
    model, optimizer = new_run()
    initial = evaluate(model)
    cursor, order, _ = train_steps(model, optimizer, 18)
    checkpoint = dict(
        config=CONFIG,
        model=copy.deepcopy(model.state_dict()),
        optimizer=copy.deepcopy(optimizer.state_dict()),
        step=18,
        cursor=cursor,
        order=order,
        torch_rng=torch.get_rng_state(),
        tokenizer={"a": 0, "b": 1, "c": 2, "EOS": 3},
    )
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    cursor, order, continuous_losses = train_steps(model, optimizer, 12, cursor, order)
    expected_parameters = copy.deepcopy(model.state_dict())
    final = evaluate(model)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=True)
    restored, restored_optimizer = new_run()
    assert loaded["config"] == CONFIG
    assert loaded["step"] == 18 and loaded["tokenizer"] == {"a": 0, "b": 1, "c": 2, "EOS": 3}
    restored.load_state_dict(loaded["model"])
    restored_optimizer.load_state_dict(loaded["optimizer"])
    torch.set_rng_state(loaded["torch_rng"])
    restored_cursor, restored_order, restored_losses = train_steps(
        restored, restored_optimizer, 12, loaded["cursor"], loaded["order"]
    )
    assert restored_cursor == cursor and restored_order == order
    assert restored_losses == continuous_losses
    for key, value in expected_parameters.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert final < initial
    counts = torch.ones(4, 4, dtype=torch.float64)
    for document in TRAIN:
        for current, following in zip(document[:-1], document[1:]):
            counts[current, following] += 1
    probabilities = counts / counts.sum(dim=1, keepdim=True)
    baseline = -probabilities[VALIDATION[:, :-1], VALIDATION[:, 1:]].log().mean().item()
    restored.eval()
    with torch.no_grad():
        inputs = VALIDATION[:, :-1]
        reference = restored(inputs).logits
        for chunks in ([9], [1] * 9, [3, 2, 4]):
            caches, outputs, offset = None, [], 0
            for length in chunks:
                result = restored(inputs[:, offset : offset + length], cache=caches, use_cache=True)
                output, caches = result.logits, result.cache
                outputs.append(output)
                offset += length
                assert all(layer.self_attention.length == offset for layer in caches.layers)
            torch.testing.assert_close(torch.cat(outputs, dim=1), reference, atol=1e-10, rtol=1e-10)
    generated = torch.tensor([[0, 1]])
    with torch.no_grad():
        for _ in range(6):
            following = restored(generated).logits[:, -1].argmax(-1, keepdim=True)
            generated = torch.cat([generated, following], dim=1)
            if following.item() == 3:
                break
    assert TRAIN.shape == (3, 12) and VALIDATION.shape == (2, 10)
    print("pretraining: 30 updates x 11 targets = 330 training targets; validation targets=18")
    print(
        f"pretraining: initial validation NLL={initial:.6f}, final={final:.6f}, bigram={baseline:.6f}"
    )
    print("pretraining: resumed 12 steps exactly equal uninterrupted parameters and losses")
    print("pretraining: trained two-layer model full/cached logits agree for three chunkings")
    print("pretraining: generated IDs", generated.tolist())

    # Separate correctness probe: deliberately overfit one fixed training document.
    torch.manual_seed(31)
    probe, probe_optimizer = new_run()
    fixed = TRAIN[:1]
    with torch.no_grad():
        start = sequence_loss(probe(fixed[:, :-1]).logits, fixed[:, 1:]).item()
    for _ in range(60):
        probe_optimizer.zero_grad(set_to_none=True)
        objective = sequence_loss(probe(fixed[:, :-1]).logits, fixed[:, 1:])
        objective.backward()
        probe_optimizer.step()
    with torch.no_grad():
        end = sequence_loss(probe(fixed[:, :-1]).logits, fixed[:, 1:]).item()
    assert end < 0.03 and end < start
    print(
        f"pretraining: fixed-batch overfit NLL {start:.6f} -> {end:.6f} (not a generalization result)"
    )


def training_flops(matmul_weights, layers, width, length, tokens):
    """Forward + backward FLOP: 6 per matmul weight per token, plus full-square attention matmuls."""
    return 6 * matmul_weights * tokens + 12 * layers * length * width * tokens


def verify_compute():
    torch.manual_seed(37)
    model = Transformer(basic_decoder_config(**CONFIG)).double()
    batch, length = 3, 12
    ids = torch.randint(0, CONFIG["vocab_size"], (batch, length))
    # Embedding lookups are not matrix multiplications; every Linear weight is.
    weights = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Linear))
    with FlopCounterMode(display=False) as counter:
        sequence_loss(model(ids[:, :-1]).logits, ids[:, 1:]).backward()
    predicted = training_flops(
        weights, CONFIG["layers"], CONFIG["dim"], length - 1, batch * (length - 1)
    )
    assert weights == 4160 and counter.get_total_flops() == predicted
    print(f"pretraining: counted {counter.get_total_flops()} FLOP for one three-document batch")


if __name__ == "__main__":
    verify()
    verify_compute()
