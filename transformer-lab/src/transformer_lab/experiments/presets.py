"""Tiny idea presets, never checkpoint-compatible replicas of named models."""

from __future__ import annotations

from dataclasses import replace

from ..config import AttentionConfig, BlockConfig, ModelConfig, PositionConfig


def preset(name: str, architecture: str | None = None) -> ModelConfig:
    """根据名称创建小规模教学配方，不加载官方权重。

    Args:
        name: 教学配方名称，必须出现在 PRESETS。
        architecture: 可选架构覆盖：encoder/decoder/encoder_decoder。

    Returns:
        ModelConfig 教学配方。
    """
    rope = PositionConfig()
    gqa = AttentionConfig(position=rope)
    modern = BlockConfig(attention=gqa)
    if name == "classic":
        block = BlockConfig(
            attention=AttentionConfig(kind="mha", position=PositionConfig(kind="sinusoidal")),
            activation="relu",
            norm="layer",
            norm_order="post",
        )
        result = ModelConfig(architecture="encoder_decoder", block=block)
    elif name in {"llama", "qwen3_dense"}:
        result = ModelConfig(
            block=replace(modern, attention=replace(gqa, qk_norm=name == "qwen3_dense"))
        )
    elif name in {"gemma", "gpt_oss"}:
        local = replace(
            modern,
            attention=replace(gqa, pattern="sliding", qk_norm=name == "gemma"),
            activation="geglu" if name == "gemma" else "swiglu",
            experts=4 if name == "gpt_oss" else 0,
        )
        result = ModelConfig(layers=2, block=modern, layer_blocks=(local, modern))
    elif name in {"deepseek_v3", "kimi_k2"}:
        mla = AttentionConfig(kind="mla", position=rope)
        block = replace(
            modern,
            attention=mla,
            experts=4,
            top_k=2,
            shared_experts=1,
            router_score="sigmoid",
            balance="bias",
        )
        result = ModelConfig(block=block, mtp_depth=1 if name == "deepseek_v3" else 0)
    elif name in {"qwen3_hybrid", "kimi_linear"}:
        recurrent = replace(
            modern,
            attention=AttentionConfig(kind="gated_delta", position=PositionConfig(kind="none")),
        )
        result = ModelConfig(
            layers=4, block=modern, layer_blocks=(recurrent, recurrent, recurrent, modern)
        )
    elif name == "irope":
        nope = replace(modern, attention=replace(gqa, position=PositionConfig(kind="none")))
        result = ModelConfig(layers=4, block=modern, layer_blocks=(modern, modern, modern, nope))
    elif name == "causal_seq2seq":
        result = ModelConfig(
            architecture="encoder_decoder", block=modern, encoder_causal=True, cross_causal=True
        )
    else:
        raise ValueError(f"unknown preset: {name}")
    return replace(result, architecture=architecture) if architecture else result


PRESETS = (
    "classic",
    "llama",
    "qwen3_dense",
    "gemma",
    "gpt_oss",
    "deepseek_v3",
    "kimi_k2",
    "qwen3_hybrid",
    "kimi_linear",
    "irope",
    "causal_seq2seq",
)
