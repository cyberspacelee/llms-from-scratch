"""RMSNorm, SwiGLU, grouped-query attention, and shared embedding checks."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from position_encoding import apply_rope


def rms_norm(x, weight, epsilon=1e-6):
    statistics = x if x.dtype == torch.float64 else x.float()
    return (statistics * torch.rsqrt(statistics.square().mean(-1, keepdim=True) + epsilon)).to(x.dtype) * weight


class RMSNorm(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x):
        return rms_norm(x, self.weight, self.epsilon)


class SwiGLU(nn.Module):
    def __init__(self, width, ff_width):
        super().__init__()
        self.gate = nn.Linear(width, ff_width, bias=False)
        self.up = nn.Linear(width, ff_width, bias=False)
        self.down = nn.Linear(ff_width, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GroupedAttention(nn.Module):
    def __init__(self, width, heads, kv_heads, head_width, rope_base):
        super().__init__()
        self.heads, self.kv_heads, self.head_width = heads, kv_heads, head_width
        self.rope_base = rope_base
        self.q = nn.Linear(width, heads * head_width, bias=False)
        self.k = nn.Linear(width, kv_heads * head_width, bias=False)
        self.v = nn.Linear(width, kv_heads * head_width, bias=False)
        self.output = nn.Linear(heads * head_width, width, bias=False)

    def forward(self, x, past, cache=None, backend="manual"):
        batch, length, _ = x.shape
        q = self.q(x).reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = self.k(x).reshape(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
        v = self.v(x).reshape(batch, length, self.kv_heads, self.head_width).transpose(1, 2)
        positions = past + torch.arange(length, device=x.device)
        q, k = apply_rope(q, positions, self.rope_base), apply_rope(k, positions, self.rope_base)
        if cache is not None:
            k, v = torch.cat((cache[0], k), -2), torch.cat((cache[1], v), -2)
        next_cache = (k, v)
        # ponytail: reference GQA expands KV for compute; a grouped GPU kernel avoids this temporary.
        k, v = [part.repeat_interleave(self.heads // self.kv_heads, 1) for part in (k, v)]
        mask = torch.arange(past + length, device=x.device)[None, :] <= positions[:, None]
        if backend == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        elif backend == "manual":
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_width)
            y = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v
        else:
            raise ValueError("backend must be manual or sdpa")
        return self.output(y.transpose(1, 2).reshape(batch, length, -1)), next_cache


class ModernBlock(nn.Module):
    def __init__(self, width, ff_width, heads, kv_heads, head_width, rope_base, epsilon):
        super().__init__()
        self.norm_attention = RMSNorm(width, epsilon)
        self.attention = GroupedAttention(width, heads, kv_heads, head_width, rope_base)
        self.norm_ff = RMSNorm(width, epsilon)
        self.ff = SwiGLU(width, ff_width)

    def forward(self, x, past, cache=None, backend="manual"):
        branch, cache = self.attention(self.norm_attention(x), past, cache, backend)
        x = x + branch
        return x + self.ff(self.norm_ff(x)), cache


class ModernDecoder(nn.Module):
    """Dense decoder-only LM, adjacent-pair RoPE, equal-length unpadded inputs."""
    def __init__(self, vocab_size, width=32, ff_width=64, layers=2, heads=4,
                 kv_heads=2, head_width=8, max_length=64, rope_base=10000.0,
                 epsilon=1e-6, tie_embeddings=True):
        super().__init__()
        dimensions = (vocab_size, width, ff_width, layers, heads, kv_heads, head_width, max_length)
        if any(type(n) is not int or n <= 0 for n in dimensions):
            raise ValueError("model dimensions must be positive integers")
        if heads % kv_heads or head_width % 2:
            raise ValueError("heads must divide into KV groups; RoPE head width must be even")
        if not math.isfinite(rope_base) or rope_base <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("RoPE base and epsilon must be finite and positive")
        if type(tie_embeddings) is not bool:
            raise ValueError("tie_embeddings must be boolean")
        self.config = dict(vocab_size=vocab_size, width=width, ff_width=ff_width, layers=layers,
                           heads=heads, kv_heads=kv_heads, head_width=head_width,
                           max_length=max_length, rope_base=rope_base, epsilon=epsilon,
                           tie_embeddings=tie_embeddings)
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList([ModernBlock(width, ff_width, heads, kv_heads, head_width,
                                                rope_base, epsilon) for _ in range(layers)])
        self.norm = RMSNorm(width, epsilon)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._initialize)
        if tie_embeddings:
            self.head.weight = self.embedding.weight
        for block in self.blocks:
            for weight in (block.attention.output.weight, block.ff.down.weight):
                nn.init.normal_(weight, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids, backend="manual"):
        return self.forward_cached(ids, backend=backend)[0]

    def forward_cached(self, ids, caches=None, backend="manual"):
        if ids.ndim != 2 or ids.dtype != torch.long or ids.numel() == 0:
            raise ValueError("expected nonempty long token IDs [B,T]")
        if ids.device != self.embedding.weight.device or ids.min() < 0 or ids.max() >= self.config["vocab_size"]:
            raise ValueError("IDs must be on the model device and inside its vocabulary")
        past = 0
        if caches is not None:
            if not isinstance(caches, (list, tuple)) or len(caches) != len(self.blocks):
                raise ValueError("one (K,V) pair is required per layer")
            first = caches[0]
            if not isinstance(first, (list, tuple)) or len(first) != 2 or not isinstance(first[0], torch.Tensor) or first[0].ndim != 4:
                raise ValueError("cache must contain four-dimensional K,V tensors")
            past = first[0].shape[-2]
            expected = (ids.shape[0], self.config["kv_heads"], past, self.config["head_width"])
            for pair in caches:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError("each cache must be a K,V pair")
                if any(not isinstance(t, torch.Tensor) or t.shape != expected or
                       t.device != ids.device or t.dtype != self.embedding.weight.dtype for t in pair):
                    raise ValueError("cache shape, prefix length, device and dtype must match the model")
        if past + ids.shape[1] > self.max_length:
            raise ValueError("input and cached prefix exceed the declared context limit")
        x, next_caches = self.embedding(ids), []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, past, None if caches is None else caches[i], backend)
            next_caches.append(cache)
        return self.head(self.norm(x)), next_caches

    def parameter_count(self):
        c = self.config
        vocabulary = c["vocab_size"] * c["width"] * (1 if c["tie_embeddings"] else 2)
        block = (2 * c["width"] * (c["heads"] + c["kv_heads"]) * c["head_width"]
                 + 3 * c["width"] * c["ff_width"] + 2 * c["width"])
        return vocabulary + c["layers"] * block + c["width"]


@torch.no_grad()
def generate_cached(model, prompt, max_new_tokens, eos_id=None, generator=None,
                    temperature=1., top_k=None, top_p=1.):
    from generation import distribution

    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
        raise ValueError("generation accepts one nonempty prompt")
    if type(max_new_tokens) is not int or max_new_tokens < 0 or prompt.shape[1] + max_new_tokens > model.max_length:
        raise ValueError("requested generation must fit the declared context")
    if eos_id is not None and not 0 <= eos_id < model.config["vocab_size"]:
        raise ValueError("EOS must belong to the vocabulary")
    result, caches, current = prompt.clone(), None, prompt
    for _ in range(max_new_tokens):
        logits, caches = model.forward_cached(current, caches)
        probabilities = distribution(logits[0, -1], temperature, top_k, top_p)
        current = torch.multinomial(probabilities, 1, generator=generator).reshape(1, 1)
        result = torch.cat((result, current), 1)
        if current.item() == eos_id:
            break
    return result


def verify_model():
    from language_model import sequence_loss

    torch.set_num_threads(1)
    torch.manual_seed(13)
    ids = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    for kv_heads in (1, 2, 4):
        for tied in (False, True):
            model = ModernDecoder(8, width=12, ff_width=20, layers=2, heads=4,
                                  kv_heads=kv_heads, head_width=4, max_length=12,
                                  tie_embeddings=tied).double()
            logits = model(ids)
            assert model.parameter_count() == sum(p.numel() for p in model.parameters())
            assert (model.head.weight is model.embedding.weight) == tied
            altered = ids.clone()
            altered[:, 3:] = 7 - altered[:, 3:]
            torch.testing.assert_close(logits[:, :3], model(altered)[:, :3])
            torch.testing.assert_close(logits[:1], model(ids[:1]))
            torch.testing.assert_close(logits, model(ids, backend="sdpa"))
            for chunks in ([5], [1] * 5, [2, 1, 2]):
                caches, outputs, offset = None, [], 0
                for length in chunks:
                    output, caches = model.forward_cached(ids[:, offset:offset + length], caches)
                    outputs.append(output)
                    offset += length
                    assert all(k.shape == (2, kv_heads, offset, 4) and v.shape == k.shape for k, v in caches)
                torch.testing.assert_close(torch.cat(outputs, 1), logits)
            sequence_loss(logits[:, :-1], ids[:, 1:]).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
    for _ in range(70):
        optimizer.zero_grad(set_to_none=True)
        loss = sequence_loss(model(ids[:, :-1]), ids[:, 1:])
        loss.backward()
        optimizer.step()
    assert loss.item() < .05
    sample_a = generate_cached(model, ids[:1, :2], 3, generator=torch.Generator().manual_seed(9))
    sample_b = generate_cached(model, ids[:1, :2], 3, generator=torch.Generator().manual_seed(9))
    assert torch.equal(sample_a, sample_b)
    assert torch.equal(generate_cached(model, ids[:1, :2], 0), ids[:1, :2])
    for call in (lambda: ModernDecoder(8, heads=3, kv_heads=2),
                 lambda: model(torch.tensor([[8]])),
                 lambda: model(torch.zeros(1, 13, dtype=torch.long)),
                 lambda: model.forward_cached(ids, [(torch.zeros(2, 4, 1, 4), torch.zeros(1))] * 2),
                 lambda: model(ids, backend="unknown")):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid model input accepted")
    print("PASS: modern LM forward/backward, MHA/GQA/MQA, independent head width, tied parameter count")
    print(f"PASS: causal/batch isolation, three cache chunkings, SDPA reference, overfit NLL={loss.item():.6f}")


def verify():
    torch.manual_seed(7)
    x = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64)
    scale = torch.ones(4, dtype=torch.float64)
    torch.testing.assert_close(rms_norm(x, scale), x / math.sqrt(7.5 + 1e-6))
    assert not torch.allclose(rms_norm(x + 1, scale), rms_norm(x, scale))
    gate, up, down = [torch.randn(*shape, dtype=torch.float64) for shape in [(6, 4), (6, 4), (4, 6)]]
    y = (F.silu(x @ gate.T) * (x @ up.T)) @ down.T
    assert y.shape == x.shape
    q = torch.randn(2, 4, 5, 2, dtype=torch.float64)
    k, v = [torch.randn(2, 2, 5, 2, dtype=torch.float64) for _ in range(2)]
    expanded_k, expanded_v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    result = (q @ expanded_k.transpose(-1, -2) / math.sqrt(2)).masked_fill(~mask, -torch.inf).softmax(-1) @ expanded_v
    for head in range(4):
        shared = head // 2
        ref = (q[:, head] @ k[:, shared].transpose(-1, -2) / math.sqrt(2)).masked_fill(~mask, -torch.inf).softmax(-1) @ v[:, shared]
        torch.testing.assert_close(result[:, head], ref)
    embedding = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([1, 2, 1])
    hidden = embedding[ids]
    logits = hidden @ embedding.T
    logits.sum().backward()
    assert embedding.grad is not None and embedding.grad[0].abs().sum() > 0
    print("PASS: RMSNorm hand case, non-shift-invariance, SwiGLU shapes")
    print("PASS: GQA group mapping, tied embedding output gradients")


if __name__ == "__main__":
    verify()
    verify_model()
