"""Modern decoder mechanisms: independent hand calculation and canonical model checks.

No alternate multi-layer model lives here; Transformer is the implementation used
by all course training, generation and architecture experiments.
"""

import math
from dataclasses import replace

import torch
from torch.nn import functional as F

from examples.principles.position_encoding import apply_rope
from llms_from_scratch import Transformer, decoder_config, token_loss
from llms_from_scratch.analysis import parameter_count
from llms_from_scratch.inference.sampling import sample_generate


def rms_norm(x, weight, epsilon=1e-6):
    """Independent hand oracle, intentionally not an import of the tested layer."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * weight


def verify_chapter_walkthrough():
    dtype = torch.float64
    embedding = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=dtype)
    x = embedding[:2]
    a = 2 / math.sqrt(3)
    c, s = math.cos(1), math.sin(1)
    wq = torch.tensor([[0.0, 0.0, 0.0, 0.0], [c / a, -s / a, s / a, c / a]], dtype=dtype)
    wk = torch.tensor([[1 / a, 0.0], [s / a, c / a]], dtype=dtype)
    wv = torch.eye(2, dtype=dtype) / a
    wo = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 0.0]], dtype=dtype)
    normalized = rms_norm(x, torch.ones(2, dtype=dtype), epsilon=1)
    torch.testing.assert_close(normalized, torch.tensor([[a, 0.0], [0.0, a]], dtype=dtype))
    positions = torch.arange(2)
    pre_q = (normalized @ wq).reshape(1, 2, 2, 2).transpose(1, 2)
    pre_k = (normalized @ wk).reshape(1, 2, 1, 2).transpose(1, 2)

    def attend(query, key):
        q = apply_rope(query, positions)
        k = apply_rope(key, positions).repeat_interleave(2, dim=1)
        weights = ((q[:, :, 1:2] @ k.transpose(-1, -2)) / math.sqrt(2)).softmax(-1)
        v = (normalized @ wv).reshape(1, 2, 1, 2).transpose(1, 2).repeat_interleave(2, dim=1)
        output = weights @ v
        return weights, output.transpose(1, 2).reshape(1, 4) @ wo

    weights, branch = attend(pre_q, pre_k)
    p = 1 / (1 + math.exp(-1 / math.sqrt(2)))
    torch.testing.assert_close(
        weights[0, :, 0], torch.tensor([[p, 1 - p], [1 - p, p]], dtype=dtype)
    )
    h = x[1] + branch[0]
    torch.testing.assert_close(h, torch.tensor([2 * p, 2.0], dtype=dtype))
    r = rms_norm(h, torch.ones(2, dtype=dtype), epsilon=1)
    y = h + F.silu(r) * r
    f = rms_norm(y, torch.ones(2, dtype=dtype), epsilon=1)
    logits = f @ embedding.T
    loss = F.cross_entropy(logits[None], torch.tensor([2]))
    torch.testing.assert_close(
        y, torch.tensor([1.6449702029359567, 2.7529870797605573], dtype=dtype)
    )
    torch.testing.assert_close(
        logits, torch.tensor([1.3274489792026454, 2.2215903256264036, 0.0], dtype=dtype)
    )
    torch.testing.assert_close(loss, torch.tensor(2.6385854561101554, dtype=dtype))

    norm_q = rms_norm(pre_q, torch.ones(2, dtype=dtype), epsilon=1)
    norm_k = rms_norm(pre_k, torch.ones(2, dtype=dtype), epsilon=1)
    qk_weights, _ = attend(norm_q, norm_k)
    p_qk = 1 / (1 + math.exp(-2 / (3 * math.sqrt(2))))
    torch.testing.assert_close(qk_weights[0, 0, 0, 0], torch.tensor(p_qk, dtype=dtype))
    print(f"PASS: chapter walkthrough p={p:.6f}, QK Norm p={p_qk:.6f}, loss={loss.item():.6f}")


def verify_model():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    ids = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    for kv_heads in (1, 2, 4):
        for tied in (False, True):
            config = decoder_config(
                8,
                dim=12,
                ff_dim=20,
                layers=2,
                heads=4,
                kv_heads=kv_heads,
                head_dim=4,
                max_length=12,
                tie_embeddings=tied,
            )
            model = Transformer(config).double()
            logits = model(ids).logits
            assert (model.head.weight is model.embedding.weight) == tied
            expected = (
                8 * 12 * (1 if tied else 2)
                + 2 * (2 * 12 * (4 + kv_heads) * 4 + 3 * 12 * 20 + 2 * 12)
                + 12
            )
            assert parameter_count(model) == expected
            altered = ids.clone()
            altered[:, 3:] = 7 - altered[:, 3:]
            torch.testing.assert_close(logits[:, :3], model(altered).logits[:, :3])
            torch.testing.assert_close(logits[:1], model(ids[:1]).logits)
            sdpa = Transformer(
                replace(
                    config,
                    block=replace(
                        config.block, attention=replace(config.block.attention, backend="sdpa")
                    ),
                )
            ).double()
            sdpa.load_state_dict(model.state_dict())
            torch.testing.assert_close(logits, sdpa(ids).logits)
            for chunks in ([5], [1] * 5, [2, 1, 2]):
                cache, outputs, offset = None, [], 0
                for length in chunks:
                    output = model(ids[:, offset : offset + length], cache=cache, use_cache=True)
                    cache = output.cache
                    outputs.append(output.logits)
                    offset += length
                    assert all(
                        layer.self_attention.key.shape == (2, kv_heads, offset, 4)
                        for layer in cache.layers
                    )
                torch.testing.assert_close(torch.cat(outputs, 1), logits)
            token_loss(logits[:, :-1], ids[:, 1:]).backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(70):
        optimizer.zero_grad(set_to_none=True)
        loss = token_loss(model(ids[:, :-1]).logits, ids[:, 1:])
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05
    a = sample_generate(model, ids[:1, :2], 3, generator=torch.Generator().manual_seed(9))
    b = sample_generate(model, ids[:1, :2], 3, generator=torch.Generator().manual_seed(9))
    assert torch.equal(a, b)
    assert torch.equal(sample_generate(model, ids[:1, :2], 0), ids[:1, :2])
    print("PASS: canonical model MHA/GQA/MQA, tied parameter count, causal/batch isolation")
    print(f"PASS: three cache chunkings, SDPA reference, overfit NLL={loss.item():.6f}")


def verify():
    torch.manual_seed(7)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    scale = torch.ones(4, dtype=torch.float64)
    torch.testing.assert_close(rms_norm(x, scale), x / math.sqrt(7.5 + 1e-6))
    assert not torch.allclose(rms_norm(x + 1, scale), rms_norm(x, scale))
    gate, up, down = [
        torch.randn(*shape, dtype=torch.float64) for shape in [(6, 4), (6, 4), (4, 6)]
    ]
    y = (F.silu(x @ gate.T) * (x @ up.T)) @ down.T
    assert y.shape == x.shape
    q = torch.randn(2, 4, 5, 2, dtype=torch.float64)
    k, v = [torch.randn(2, 2, 5, 2, dtype=torch.float64) for _ in range(2)]
    expanded_k, expanded_v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    result = (q @ expanded_k.transpose(-1, -2) / math.sqrt(2)).masked_fill(
        ~mask, -torch.inf
    ).softmax(-1) @ expanded_v
    for head in range(4):
        shared = head // 2
        ref = (q[:, head] @ k[:, shared].transpose(-1, -2) / math.sqrt(2)).masked_fill(
            ~mask, -torch.inf
        ).softmax(-1) @ v[:, shared]
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
    verify_chapter_walkthrough()
    verify()
    verify_model()
