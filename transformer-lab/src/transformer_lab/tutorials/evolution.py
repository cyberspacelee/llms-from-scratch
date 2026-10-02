"""第 01–13 步：以配置差异表达演进，以小型不变量核对原理。

各分支共享核心实现；MLA、稀疏、MoE、递归结构并不是必须全部叠加的升级。
run_lesson 每次只运行一个步骤，不下载数据、不做性能测试。
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..analysis import attention_cost, ffn_cost, parameter_count
from ..attention import gathered_attention, scaled_dot_product_attention
from ..attention.patterns import attention_mask
from ..cache import PagedKVCache, QuantizedTensor
from ..config import AttentionConfig, BlockConfig, ModelConfig, PositionConfig
from ..experiments.runners import consistency
from ..inference import compressed_causal_attention, greedy_speculative_generate
from ..layers import MixtureOfExperts
from ..models import Transformer, make_attention
from ..training import language_model_loss
from .mvp import MinimalTransformer, demo

LESSONS = (
    "最小单头 Encoder–Decoder MVP",
    "保留数学：多头与多层",
    "三类模型：Encoder / Decoder / Encoder–Decoder",
    "从绝对位置到 RoPE",
    "Prefill / Decode 与 Self / Cross KV Cache",
    "Pre-Norm、RMSNorm、QK-Norm 与 SwiGLU",
    "MHA → MQA → GQA",
    "MLA：Latent KV 与矩阵吸收",
    "长上下文：Scaling、窗口、稀疏与序列压缩",
    "Dense FFN → Top-K MoE 与共享专家",
    "NTP → 顺序 MTP 训练目标",
    "Linear / Delta / Gated Delta 与混合层",
    "因果 Encoder–Decoder 与非对称计算",
    "SDPA 对照、分页、前缀共享、量化与推测解码",
)


def classic_config(heads: int = 1, layers: int = 1) -> ModelConfig:
    """与 MVP 完全同构的配置；heads 增加时总投影宽度仍保持 D=32。"""
    if type(heads) is not int or heads < 1 or 32 % heads:
        raise ValueError("heads must be a positive divisor of 32 for this lesson")
    return ModelConfig(
        architecture="encoder_decoder",
        vocab_size=32,
        dim=32,
        layers=layers,
        encoder_layers=layers,
        tie_embeddings=False,
        block=BlockConfig(
            attention=AttentionConfig(
                kind="mha",
                heads=heads,
                head_dim=32 // heads,
                position=PositionConfig(kind="sinusoidal"),
            ),
            norm="layer",
            norm_order="post",
            activation="relu",
            ff_dim=64,
        ),
    )


def lift_mvp(model: MinimalTransformer) -> Transformer:
    """将固定 MVP 的权重放入统一模型，验证重组接口并未改动计算。

    教学桥梁仅适用于默认尺寸、单头、单层 MVP；后续升级创建各自模型，
    不声称结构不同的模型能加载相同 checkpoint。
    """
    if (
        model.dim != 32
        or model.embedding.num_embeddings != 32
        or model.encoder_ff[0].out_features != 64
    ):
        raise ValueError("teaching bridge expects the default MVP dimensions")
    result = Transformer(classic_config()).to(
        device=model.embedding.weight.device, dtype=model.embedding.weight.dtype
    )
    result.embedding.load_state_dict(model.embedding.state_dict())
    result.head.load_state_dict(model.head.state_dict())
    encoder, decoder = result.encoder_blocks[0], result.blocks[0]
    encoder.attention.load_state_dict(model.encoder_attention.state_dict())
    decoder.attention.load_state_dict(model.decoder_attention.state_dict())
    decoder.cross.load_state_dict(model.cross_attention.state_dict())
    for block, ff, norms in (
        (encoder, model.encoder_ff, model.encoder_norms),
        (decoder, model.decoder_ff, model.decoder_norms),
    ):
        block.ff.up.load_state_dict(ff[0].state_dict())
        block.ff.down.load_state_dict(ff[2].state_dict())
        block.norms.load_state_dict(norms.state_dict())
    result.train(model.training)
    return result


def lesson_config(step: int) -> ModelConfig:
    """每一步写出配置变化；08/11/12/13 是不同研究分支，见演进文档。"""
    if not 0 <= step < len(LESSONS):
        raise ValueError("lesson step must be between 0 and 13")
    c = classic_config() if step == 0 else classic_config(heads=4, layers=2)
    if step <= 1:
        return c
    c = replace(c, architecture="decoder")
    if step == 2:
        return c
    a = replace(c.block.attention, position=PositionConfig(kind="rope"))
    c = replace(c, block=replace(c.block, attention=a))
    if step <= 4:
        return c
    c = replace(
        c,
        block=replace(
            c.block,
            norm_order="pre",
            norm="rms",
            activation="swiglu",
            attention=replace(a, qk_norm=True),
        ),
    )
    if step == 5:
        return c
    c = replace(
        c, block=replace(c.block, attention=replace(c.block.attention, kind="gqa", kv_heads=2))
    )
    if step in {6, 11, 12, 13}:
        return c
    # MLA 的 query/KV 压缩与普通 QK-Norm 不等价，不能直接沿用其开关。
    mla = AttentionConfig(
        kind="mla", heads=4, head_dim=8, value_dim=8, kv_rank=8, q_rank=8, rope_dim=4
    )
    c = replace(c, block=replace(c.block, attention=mla))
    if step <= 8:
        return c
    c = replace(
        c,
        block=replace(
            c.block, experts=4, top_k=2, shared_experts=1, router_score="sigmoid", balance="bias"
        ),
    )
    return replace(c, mtp_depth=2) if step == 10 else c


def multihead_lesson(device: torch.device) -> dict[str, object]:
    """先比较完全同构模型；再增加 head/layer，不比较随机模型的 logits。"""
    mvp = MinimalTransformer().to(device).double().eval()
    core = lift_mvp(mvp)
    source = torch.tensor([[1, 2, 3]], device=device)
    decoder = torch.tensor([[0, 4, 5, 6]], device=device)
    with torch.no_grad():
        left, right = mvp(source, decoder), core(decoder, source_ids=source).logits
        torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-7)
    report = consistency(lesson_config(1), device)
    report["mvp_to_core_max_error"] = (left - right).abs().max().item()
    report["change"] = "1 head / 1 layer -> 4 heads / 2 layers; H*d remains 32"
    return report


def position_lesson(device: torch.device) -> dict[str, object]:
    """同一 RoPE 原语作用于 Encoder、Decoder、Cross 的 Q/K，各自使用本侧位置。"""
    c = lesson_config(3)
    seq2seq = replace(c, architecture="encoder_decoder")
    return {
        "decoder": consistency(c, device),
        "encoder_decoder_with_cross_rope": consistency(seq2seq, device),
    }


def head_lesson(device: torch.device) -> dict[str, object]:
    """比较相同 H/d、不同 KV head 数；不宣称不同架构输出相同。"""
    c = lesson_config(6)
    report = {}
    for kind in ("mha", "mqa", "gqa"):
        a = replace(c.block.attention, kind=kind)
        config = replace(c, block=replace(c.block, attention=a))
        report[kind] = {
            "query_heads": a.heads,
            "kv_heads": a.effective_kv_heads,
            "cost_fp32_S1024": attention_cost(
                a, c.dim, key_tokens=1024, bytes_per_element=4
            ).to_dict(),
            "check": consistency(config, device),
        }
    return report


def mla_lesson(device: torch.device) -> dict[str, object]:
    """两个路径必须共享权重，decode 必须共享同一段 latent 前缀。"""
    c = lesson_config(7)
    a = c.block.attention
    naive = make_attention(c.dim, replace(a, mla_impl="naive")).to(device).double().eval()
    absorbed = make_attention(c.dim, replace(a, mla_impl="absorbed")).to(device).double().eval()
    absorbed.load_state_dict(naive.state_dict())
    x = torch.randn(1, 6, c.dim, device=device, dtype=torch.float64)
    with torch.no_grad():
        _, prefix = naive(x[:, :5], causal=True, use_cache=True)
        left, _ = naive(x[:, 5:], cache=prefix, causal=True, query_offset=5)
        right, state = absorbed(x[:, 5:], cache=prefix, causal=True, query_offset=5, use_cache=True)
        torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-7)
    return {
        "latent_shape": list(state.key.shape),
        "rotary_key_shape": list(state.value.shape),
        "naive_absorbed_decode_error": (left - right).abs().max().item(),
        "cost_fp32_S1024": attention_cost(a, c.dim, key_tokens=1024, bytes_per_element=4).to_dict(),
        "model_check": consistency(c, device),
    }


def context_lesson(device: torch.device) -> dict[str, object]:
    """区分可见关系、真实稀疏 gather、序列近似压缩与位置频率扩展。"""
    a = lesson_config(6).block.attention
    positions = torch.arange(6, device=device)
    patterns = {
        name: attention_mask(
            positions, positions, replace(a, pattern=name, window=2, block_size=2), True
        )[0, 0]
        .int()
        .tolist()
        for name in ("global", "sliding", "local", "block_sparse", "token_sparse")
    }
    q, k, v = (torch.randn(1, 2, 6, 8, device=device, dtype=torch.float64) for _ in range(3))
    # 每个 query 读取自己和前一 token；第一行用 masked filler 维持固定 M=2。
    indices = torch.stack(((positions - 1).clamp_min(0), positions), -1)
    valid = torch.ones_like(indices, dtype=torch.bool)
    valid[0, 0] = False
    dense_mask = torch.zeros(6, 6, device=device, dtype=torch.bool)
    dense_mask[positions[:, None].expand_as(indices)[valid], indices[valid]] = True
    sparse = gathered_attention(q, k, v, indices, valid)
    dense = scaled_dot_product_attention(q, k, v, dense_mask)
    torch.testing.assert_close(sparse, dense)
    compressed = compressed_causal_attention(q, k, v, block_size=2, window=2)
    changed = v.clone()
    changed[:, :, 4:] += 100
    torch.testing.assert_close(
        compressed[:, :, :4], compressed_causal_attention(q, k, changed, 2, 2)[:, :, :4]
    )
    hybrid = replace(
        lesson_config(6),
        layer_blocks=(
            replace(lesson_config(6).block, attention=replace(a, pattern="sliding", window=2)),
            lesson_config(6).block,
        ),
    )
    return {
        "causal_visible_matrices": patterns,
        "gather_dense_error": (sparse - dense).abs().max().item(),
        "compression_shape": list(compressed.shape),
        "local_global_check": consistency(hybrid, device),
        "scaling_checks": {
            scaling: consistency(
                replace(
                    lesson_config(6),
                    block=replace(
                        lesson_config(6).block,
                        attention=replace(
                            a, position=replace(a.position, scaling=scaling, factor=4)
                        ),
                    ),
                ),
                device,
            )
            for scaling in ("linear", "ntk", "yarn")
        },
        "note": "mask alone still allocates dense scores; pooling changes the model; scaling does not prove long-context quality",
    }


def moe_lesson(device: torch.device) -> dict[str, object]:
    """路由只激活 K 个 routed expert；shared expert 对每个有效 token 激活。"""
    c = lesson_config(9)
    moe = MixtureOfExperts(c.dim, c.block).to(device)
    x = torch.randn(2, 3, c.dim, device=device, requires_grad=True)
    y, auxiliary, counts = moe(x)
    assert counts.sum().item() == x.shape[0] * x.shape[1] * c.block.top_k
    (y.square().mean() + auxiliary).backward()
    assert moe.router.weight.grad is not None and torch.isfinite(moe.router.weight.grad).all()
    # 训练循环应在 optimizer.step 之后汇总 counts，再更新 selection bias。
    moe.update_balance(counts)
    return {
        "dispatch_counts": counts.tolist(),
        "cost": ffn_cost(c.dim, c.block),
        "total_parameters": parameter_count(moe),
        "model_check": consistency(c, device),
    }


def mtp_lesson(device: torch.device) -> dict[str, object]:
    """NTP 标签为 x(t+1)；第一个额外 MTP 深度标签为 x(t+2)。"""
    c = lesson_config(10)
    model = Transformer(c).to(device)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    output = model(ids)
    total, terms, routing = language_model_loss(model, output, ids)
    total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is not None for p in model.mtp.parameters())
    predictions, _, _ = model.mtp(output.hidden, ids, model.embedding, model.head)
    assert [p.shape[1] for p in predictions] == [4, 3]
    return {
        "terms": {name: term.item() for name, term in terms.items()},
        "total": total.item(),
        "ntp_labels": ids[:, 1:].tolist(),
        "mtp_labels": [ids[:, 2:].tolist(), ids[:, 3:].tolist()],
        "mtp_shapes": [list(p.shape) for p in predictions],
        "routing_records": len(routing),
    }


def recurrent_lesson(device: torch.device) -> dict[str, object]:
    """递归层与 softmax 层共享 Block 接口；递归状态大小不随 S 增长。"""
    c = lesson_config(11)
    report = {}
    for kind in ("linear", "delta", "gated_delta"):
        a = replace(
            c.block.attention, kind=kind, qk_norm=False, position=PositionConfig(kind="none")
        )
        config = replace(c, block=replace(c.block, attention=a))
        short = attention_cost(a, c.dim, key_tokens=16, bytes_per_element=4).cache_bytes
        long = attention_cost(a, c.dim, key_tokens=4096, bytes_per_element=4).cache_bytes
        assert short == long
        report[kind] = {"state_bytes_fp32": short, "check": consistency(config, device)}
    hybrid = replace(c, layer_blocks=(config.block, c.block))
    report["gated_delta_then_gqa"] = consistency(hybrid, device)
    # iRoPE 入门分支：在不同层之间交替使用 RoPE/NoPE，不是对单个 Q/K 做半旋转。
    nope = replace(
        c.block, attention=replace(c.block.attention, position=PositionConfig(kind="none"))
    )
    report["rope_then_nope"] = consistency(replace(c, layer_blocks=(c.block, nope)), device)
    return report


def encoder_lesson(device: torch.device) -> dict[str, object]:
    """因果 Encoder 可追加；输入层数和输出层数不同；编码 memory 可以复用。"""
    c = replace(
        lesson_config(12),
        architecture="encoder_decoder",
        encoder_causal=True,
        cross_causal=True,
        encoder_layers=1,
    )
    model = Transformer(c).to(device).double().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    with torch.no_grad():
        full = model.encode(ids, use_cache=True)
        first = model.encode(ids[:, :3], use_cache=True)
        second = model.encode(ids[:, 3:], past=first, use_cache=True)
        torch.testing.assert_close(full.hidden, second.hidden, atol=1e-9, rtol=1e-7)
        direct = model(ids, source_ids=ids).logits
        reused = model(ids, memory=second).logits
        torch.testing.assert_close(direct, reused, atol=1e-9, rtol=1e-7)
    return {
        "encoder_layers": len(model.encoder_blocks),
        "decoder_layers": len(model.blocks),
        "encoder_append_error": (full.hidden - second.hidden).abs().max().item(),
        "memory_reuse_error": (direct - reused).abs().max().item(),
        "check": consistency(c, device),
    }


def inference_lesson(device: torch.device) -> dict[str, object]:
    """原理核对不计时：SDPA 同权重、分页保值、量化误差、greedy 接受对齐。"""
    c = lesson_config(13)
    manual = make_attention(c.dim, c.block.attention).to(device).double().eval()
    sdpa = (
        make_attention(c.dim, replace(c.block.attention, backend="sdpa")).to(device).double().eval()
    )
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(1, 5, c.dim, device=device, dtype=torch.float64)
    with torch.no_grad():
        left, state = manual(x, causal=True, use_cache=True)
        right, _ = sdpa(x, causal=True)
    torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-7)
    pages = PagedKVCache(page_size=3)
    pages.append(state.key[:, :, :4], state.value[:, :, :4])
    fork = pages.fork()
    pages.append(state.key[:, :, 4:], state.value[:, :, 4:])
    assert fork.materialize().length == 4 and pages.materialize().length == 5
    assert pages.block_table[0] == fork.block_table[0]
    torch.testing.assert_close(pages.materialize().key, state.key)
    quantized = QuantizedTensor.encode(state.key)
    assert ((quantized.decode() - state.key).abs() <= quantized.scales / 2 + 1e-12).all()
    target = Transformer(c).to(device).eval()
    draft = Transformer(c).to(device).eval()
    draft.load_state_dict(target.state_dict())
    prompt = torch.tensor([[1, 2]], device=device)
    speculative, stats = greedy_speculative_generate(target, draft, prompt, 5, draft_length=2)
    assert torch.equal(speculative, target.generate(prompt, 5))
    return {
        "manual_sdpa_error": (left - right).abs().max().item(),
        "prefix_lengths": [fork.materialize().length, pages.materialize().length],
        "quantization_max_error": (quantized.decode() - state.key).abs().max().item(),
        "original_bytes": state.key.numel() * state.key.element_size(),
        "int8_with_scale_bytes": quantized.nbytes,
        "speculative": stats,
        "note": "storage primitives and verification only; no performance claim",
    }


def run_lesson(step: int, device: torch.device) -> dict[str, object]:
    """统一入口；步号映射可直接阅读，不需要插件注册或额外教程框架。"""
    config = lesson_config(step)  # 同时校验步号
    if step == 0:
        result = demo(device)
    elif step == 1:
        result = multihead_lesson(device)
    elif step == 2:
        result = {
            architecture: consistency(replace(config, architecture=architecture), device)
            for architecture in ("encoder", "decoder", "encoder_decoder")
        }
    elif step == 3:
        result = position_lesson(device)
    elif step == 4:
        # 绝对位置、Cross 静态缓存与 Decoder 动态缓存同时演示。
        result = consistency(replace(config, architecture="encoder_decoder"), device)
    elif step == 5:
        result = consistency(config, device)
    elif step == 6:
        result = head_lesson(device)
    elif step == 7:
        result = mla_lesson(device)
    elif step == 8:
        result = context_lesson(device)
    elif step == 9:
        result = moe_lesson(device)
    elif step == 10:
        result = mtp_lesson(device)
    elif step == 11:
        result = recurrent_lesson(device)
    elif step == 12:
        result = encoder_lesson(device)
    else:
        result = inference_lesson(device)
    return {"step": f"{step:02}", "title": LESSONS[step], "result": result}
