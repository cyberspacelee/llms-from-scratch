"""Independent Pre-LN/learned-position oracle for checking the canonical Transformer.

ReferenceDecoder is used only for numerical verification, never for training runs.
"""

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.heads, self.head_width = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, x, cache=None, return_cache=False):
        batch, length, width = x.shape
        q, k, v = [
            part.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
            for part in self.qkv(x).chunk(3, -1)
        ]
        past = 0 if cache is None else cache[0].shape[-2]
        if cache is not None:
            k, v = torch.cat((cache[0], k), -2), torch.cat((cache[1], v), -2)
        query_positions = past + torch.arange(length, device=x.device)
        mask = torch.arange(past + length, device=x.device)[None, :] <= query_positions[:, None]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_width)
        weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        result = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        result = self.output(result)
        return (result, (k, v)) if return_cache else result


class DecoderBlock(nn.Module):
    def __init__(self, width, heads, ff_width):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = MultiHeadAttention(width, heads)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(), nn.Linear(ff_width, width))

    def forward(self, x, cache=None, return_cache=False):
        result = self.attention(self.norm_attention(x), cache, return_cache)
        if return_cache:
            result, cache = result
        x = x + result
        x = x + self.ff(self.norm_ff(x))
        return (x, cache) if return_cache else x


class ReferenceDecoder(nn.Module):
    def __init__(self, vocab_size, width=32, heads=4, ff_width=64, layers=2, max_length=32):
        super().__init__()
        if min(vocab_size, width, heads, ff_width, layers, max_length) <= 0:
            raise ValueError("model dimensions must be positive")
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList([DecoderBlock(width, heads, ff_width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, ids):
        return self.forward_cached(ids)[0]

    def forward_cached(self, ids, caches=None):
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError("one cache is required per decoder layer")
        past = 0 if caches is None else caches[0][0].shape[-2]
        if caches is not None and any(k.shape[-2] != past or v.shape != k.shape for k, v in caches):
            raise ValueError("all layer caches must cover the same prefix")
        if ids.ndim != 2 or ids.dtype != torch.long or not 0 < ids.shape[1] <= self.max_length:
            raise ValueError("expected nonempty integer IDs [B,T] within the position table")
        if past + ids.shape[1] > self.max_length:
            raise ValueError("cached positions exceed the learned position table")
        if ids.numel() == 0 or ids.min() < 0 or ids.max() >= self.embedding.num_embeddings:
            raise ValueError("token IDs must belong to the vocabulary")
        if caches is not None:
            for block, (k, v) in zip(self.blocks, caches):
                expected = (ids.shape[0], block.attention.heads, past, block.attention.head_width)
                if (
                    k.shape != expected
                    or k.device != ids.device
                    or k.dtype != self.embedding.weight.dtype
                ):
                    raise ValueError(
                        "cache batch, heads, width, device and dtype must match this model"
                    )
        positions = past + torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding(ids) + self.position(positions)
        next_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if caches is None else caches[i], return_cache=True)
            next_caches.append(cache)
        return self.head(self.norm(x)), next_caches


def verify():
    from examples.principles.language_model import sequence_loss

    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = ReferenceDecoder(8, width=4, heads=2, ff_width=8, layers=1, max_length=4).double()
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    logits = model(ids)
    assert logits.shape == (2, 4, 8)
    assert sum(p.numel() for p in model.parameters()) == 268
    changed = ids.clone()
    changed[:, 2:] = 7 - changed[:, 2:]
    torch.testing.assert_close(model(changed)[:, :2], logits[:, :2])
    torch.testing.assert_close(model(ids[:1]), logits[:1])
    sequence_loss(logits[:, :-1], ids[:, 1:]).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    attention = model.blocks[0].attention
    x = torch.randn(2, 4, 4, dtype=torch.float64)
    q, k, v = attention.qkv(x).chunk(3, -1)
    reference = torch.zeros_like(x)
    for b in range(2):
        for i in range(4):
            for h in range(2):
                span = slice(2 * h, 2 * h + 2)
                scores = k[b, : i + 1, span] @ q[b, i, span] / math.sqrt(2)
                reference[b, i, span] = scores.softmax(0) @ v[b, : i + 1, span]
    torch.testing.assert_close(attention(x), attention.output(reference))

    # Residual stream: the final hidden state is the embedding plus every branch write.
    with torch.no_grad():
        deep = ReferenceDecoder(8, width=4, heads=2, ff_width=8, layers=3, max_length=4).double()
        x0 = deep.embedding(ids) + deep.position(torch.arange(4))
        stream, writes = x0, []
        for block in deep.blocks:
            writes.append(block.attention(block.norm_attention(stream)))
            stream = stream + writes[-1]
            writes.append(block.ff(block.norm_ff(stream)))
            stream = stream + writes[-1]
        assert len(writes) == 6
        torch.testing.assert_close(x0 + sum(writes), stream)
        torch.testing.assert_close(deep.head(deep.norm(stream)), deep(ids))

    with torch.no_grad():
        model.position.weight.zero_()
        for block in model.blocks:
            for parameter in block.attention.parameters():
                parameter.zero_()
            for parameter in block.ff.parameters():
                parameter.zero_()
        model.embedding.weight[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        model.head.weight.zero_()
        model.head.weight[:4] = torch.eye(4)
        model.head.bias.zero_()
    expected = (torch.arange(1.0, 5.0, dtype=torch.float64) - 2.5) / math.sqrt(1.25 + 1e-5)
    torch.testing.assert_close(model(torch.tensor([[0]]))[0, 0, :4], expected)
    verify_hand_path(sequence_loss)
    print("PASS: complete forward/backward, 268 parameters, causal and batch isolation")
    print(
        "PASS: loop vs multihead matrix attention; residual stream sum; hand LayerNorm-to-logits example"
    )


def verify_hand_path(sequence_loss):
    model = ReferenceDecoder(4, width=2, heads=1, ff_width=2, layers=1, max_length=3).double()
    assert sum(p.numel() for p in model.parameters()) == 74
    block = model.blocks[0]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for norm in (block.norm_attention, block.norm_ff, model.norm):
            norm.weight.fill_(1)
            norm.eps = 1
        model.embedding.weight.copy_(
            torch.tensor([[2, 0], [1, 2], [1, 3], [0, 0]], dtype=torch.float64)
        )
        model.position.weight.copy_(torch.tensor([[0, 0], [-1, 0], [1, 1]], dtype=torch.float64))
        block.attention.qkv.weight[4:6] = torch.eye(2, dtype=torch.float64)
        block.attention.output.weight.copy_(torch.eye(2, dtype=torch.float64))
        block.ff[0].bias[0] = 1
        block.ff[2].weight[0, 0] = 1
        model.head.weight[1, 0] = 1
        model.head.weight[2, 1] = 1

    ids = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])
    logits = model(ids)
    s = 1 / math.sqrt(2)
    g = (1 + math.erf(1 / math.sqrt(2))) / 2
    differences = [2 + 2 * s + g, g - 2, g - 2 - 2 * s / 3]
    r = torch.tensor(
        [delta / math.sqrt(delta * delta + 4) for delta in differences], dtype=torch.float64
    )
    expected_logits = torch.stack((torch.zeros_like(r), r, -r, torch.zeros_like(r)), -1)[None]
    torch.testing.assert_close(logits, expected_logits)
    loss = sequence_loss(logits, targets)
    torch.testing.assert_close(loss, torch.tensor(1.0370171040287348, dtype=torch.float64))
    logit_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(logit_grad[0, 0, 1], (logits[0, 0].softmax(-1)[1] - 1) / 3)
    loss.backward()
    assert model.embedding.weight.grad[0].abs().sum() > 0
    assert block.attention.qkv.weight.grad[4:6].abs().sum() > 0
    assert block.ff[2].weight.grad.abs().sum() > 0
    print("PASS: one BOS,A,B path through embedding, attention, FFN, logits, loss and backward")


if __name__ == "__main__":
    verify()
