import pytest

from llms_from_scratch.architecture.specs import (
    DEEPSEEK_V3,
    GPT_OSS_120B,
    LLAMA2_7B,
    LLAMA3_8B,
    MIXTRAL_8X7B,
    QWEN3_235B,
    kv_cache_bytes_per_token,
)


@pytest.mark.parametrize(
    ("spec", "total", "active"),
    [
        (LLAMA2_7B, 6.74e9, 6.74e9),
        (LLAMA3_8B, 8.03e9, 8.03e9),
        (MIXTRAL_8X7B, 46.7e9, 12.9e9),
        (QWEN3_235B, 235e9, 22e9),
        (GPT_OSS_120B, 117e9, 5.7e9),
        (DEEPSEEK_V3, 671e9, 37e9),
    ],
)
def test_counts_match_reported(spec, total, active):
    t, a = spec.count()
    assert t == pytest.approx(total, rel=0.02)
    assert a == pytest.approx(active, rel=0.02)


def test_kv_cache_per_token():
    # LLaMA-2-7B：32 层 × 2 × 32 头 × 128 维 × 2 字节 = 512 KiB
    assert kv_cache_bytes_per_token(LLAMA2_7B) == 512 * 1024
    # GQA 的 8 个 KV 头把它降为 1/4
    assert kv_cache_bytes_per_token(LLAMA3_8B) == 128 * 1024
    # DeepSeek-V3：61 层 × (512 + 64) × 2 字节
    assert kv_cache_bytes_per_token(DEEPSEEK_V3) == 61 * 576 * 2
