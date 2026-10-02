"""章节目录和执行入口；机制实现分别位于每个 chNN 模块。"""

from __future__ import annotations

from importlib import import_module

import torch

CHAPTERS = (
    ("ch01_naive_transformer", "2017 · Naive Transformer / Encoder–Decoder"),
    ("ch02_decoder_only", "2018 · Decoder-only / GPT / NTP"),
    ("ch03_encoder_only", "2018 · Encoder-only / BERT / MLM"),
    ("ch04_mqa", "2019 · Multi-Query Attention (MQA)"),
    ("ch05_normalization", "2019–2020 · Pre-Norm / RMSNorm"),
    ("ch06_gated_ffn", "2020 · Gated FFN / SwiGLU"),
    ("ch07_efficient_attention", "2020 · Sparse / Linear Attention"),
    ("ch08_rope", "2021 · Rotary Position Embedding (RoPE)"),
    ("ch09_moe", "2021–2024 · Mixture of Experts (MoE)"),
    ("ch10_flash_attention", "2022 · FlashAttention / Online Softmax / SDPA"),
    ("ch11_gqa", "2023 · Grouped-Query Attention (GQA)"),
    ("ch12_rope_scaling", "2023 · RoPE Scaling / YaRN"),
    ("ch13_mla", "2024 · Multi-head Latent Attention (MLA)"),
    ("ch14_mtp", "2024 · Multi-Token Prediction (MTP)"),
    ("ch15_hybrid_attention", "2024–2025 · Delta / KDA / Hybrid Attention"),
    ("ch16_kv_cache", "推理基础 · KV Cache / Prefill / Decode"),
    ("ch17_kv_storage", "2023–2024 推理 · Paged KV / Prefix / Quantization"),
    ("ch18_speculative_decoding", "2023 推理 · Speculative Decoding"),
    ("ch19_sequence_compression", "长上下文支线 · Sequence Compression"),
    ("ch20_qwen_hybrid", "2026 · Qwen3.5 / 3.6 / 3.8 Hybrid"),
    ("ch21_learned_sparse_attention", "2025–2026 · DeepSeek DSA / CSA2 / Qwen QSA"),
    ("ch22_residual_streams", "2025–2026 · mHC / GR / Kimi AttnRes"),
    ("ch23_conditional_memory", "2026 · DeepSeek Engram / Qwen PLE"),
    ("ch24_shared_kv", "2024–2026 · Shared KV / YOCO / DeepSeek CED"),
    ("ch25_multimodal", "2026 · DeepSeek VL / Qwen VL / Omni"),
)


def run_chapter(number: int, device: torch.device) -> dict[str, object]:
    """输入章号 1–25、执行设备；返回 chapter/title/module/result 报告。

    只导入并运行选定章节；不下载权重/数据，不计时，不在 import 时训练。

    Args:
        number: 章号 1–25，整数。
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict：chapter/title/module/result，result为该章数值检查报告。
    """
    if type(number) is not int or not 1 <= number <= len(CHAPTERS):
        raise ValueError(f"chapter must be between 1 and {len(CHAPTERS)}")
    module, title = CHAPTERS[number - 1]
    chapter_module = import_module(f"{__name__}.{module}")
    return {
        "chapter": f"{number:02}",
        "title": title,
        "module": module,
        "result": chapter_module.run(device),
    }
