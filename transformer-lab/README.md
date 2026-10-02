# Transformer Lab

按模型发展与优化路线学习Transformer，从2017原始Encoder–Decoder到2026年的DeepSeek/Qwen结构。每章有独立代码、shape说明和可执行数值检查；基础计算使用PyTorch的matmul、reshape、transpose、softmax与autograd，不使用nn.Transformer/nn.MultiheadAttention。

## 开始学习

```bash
cd transformer-lab
uv sync --locked
uv run --locked transformer-lab chapters
uv run --locked transformer-lab chapter --chapter 01
uv run --locked transformer-lab chapter --chapter 20 --output runs/qwen-hybrid.json
```

先读[课程规划与统一术语](docs/plan.md)，再按下表进入各章。hidden states统一使用`[B,S,D]`；Attention明确区分`S_q/S_kv`、`H_q/H_kv`和每头宽度。每个源码函数的`Args`/`Returns`说明输入、输出及shape，中间变化写在计算旁。

## 课程目录

| 章 | 模型/优化脉络 | 章节 |
| --- | --- | --- |
| 01 | 2017 · Naive Transformer / Encoder–Decoder | [naive_transformer](docs/chapters/01_naive_transformer.md) |
| 02 | 2018 · Decoder-only / GPT / NTP | [decoder_only](docs/chapters/02_decoder_only.md) |
| 03 | 2018 · Encoder-only / BERT / MLM | [encoder_only](docs/chapters/03_encoder_only.md) |
| 04 | 2019 · Multi-Query Attention (MQA) | [mqa](docs/chapters/04_mqa.md) |
| 05 | 2019–2020 · Pre-Norm / RMSNorm | [normalization](docs/chapters/05_normalization.md) |
| 06 | 2020 · Gated FFN / SwiGLU | [gated_ffn](docs/chapters/06_gated_ffn.md) |
| 07 | 2020 · Sparse / Linear Attention | [efficient_attention](docs/chapters/07_efficient_attention.md) |
| 08 | 2021 · Rotary Position Embedding (RoPE) | [rope](docs/chapters/08_rope.md) |
| 09 | 2021–2024 · Mixture of Experts (MoE) | [moe](docs/chapters/09_moe.md) |
| 10 | 2022 · FlashAttention / Online Softmax / SDPA | [flash_attention](docs/chapters/10_flash_attention.md) |
| 11 | 2023 · Grouped-Query Attention (GQA) | [gqa](docs/chapters/11_gqa.md) |
| 12 | 2023 · RoPE Scaling / YaRN | [rope_scaling](docs/chapters/12_rope_scaling.md) |
| 13 | 2024 · Multi-head Latent Attention (MLA) | [mla](docs/chapters/13_mla.md) |
| 14 | 2024 · Multi-Token Prediction (MTP) | [mtp](docs/chapters/14_mtp.md) |
| 15 | 2024–2025 · Delta / KDA / Hybrid Attention | [hybrid_attention](docs/chapters/15_hybrid_attention.md) |
| 16 | 推理基础 · KV Cache / Prefill / Decode | [kv_cache](docs/chapters/16_kv_cache.md) |
| 17 | 2023–2024 推理 · Paged KV / Prefix / Quantization | [kv_storage](docs/chapters/17_kv_storage.md) |
| 18 | 2023 推理 · Speculative Decoding | [speculative_decoding](docs/chapters/18_speculative_decoding.md) |
| 19 | 长上下文支线 · Sequence Compression | [sequence_compression](docs/chapters/19_sequence_compression.md) |
| 20 | 2026 · Qwen3.5 / 3.6 / 3.8 Hybrid | [qwen_hybrid](docs/chapters/20_qwen_hybrid.md) |
| 21 | 2025–2026 · DeepSeek DSA / CSA2 / Qwen QSA | [learned_sparse_attention](docs/chapters/21_learned_sparse_attention.md) |
| 22 | 2025–2026 · mHC / GR / Kimi AttnRes | [residual_streams](docs/chapters/22_residual_streams.md) |
| 23 | 2026 · DeepSeek Engram / Qwen PLE | [conditional_memory](docs/chapters/23_conditional_memory.md) |
| 24 | 2024–2026 · Shared KV / YOCO / DeepSeek CED | [shared_kv](docs/chapters/24_shared_kv.md) |
| 25 | 2026 · DeepSeek VL / Qwen VL / Omni | [multimodal](docs/chapters/25_multimodal.md) |

01–15为机制发展主线；16–19为推理/长上下文支线；20–25为2026结构专题。支线可以按前置条件提前阅读，不要求把所有机制堆进一个模型。DeepSeek V4/V4.1、Qwen3.5/3.6/3.8、Qwen4-Exp及其他公开方案的结构差异和来源见[2026模型对照](docs/models-2026.md)。

## 核对与实验

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
uv run --locked transformer-lab check --preset classic
uv run --locked transformer-lab check --preset deepseek_v3
uv run --locked transformer-lab check --preset qwen3_hybrid
uv run --locked transformer-lab ledger --length 32768
# 可选：小型合成任务训练；功能核对不要求训练收敛。
uv run --locked transformer-lab train --preset classic --steps 60 --output runs/copy.pt
```

命名配方只代表小规模机制组合，不支持直接加载官方checkpoint；2026专项实现以章节入口为准，不把V3的MLA命名成V4。源代码不在import时训练或下载数据，PyTorch是唯一运行依赖。

## 核心代码与设备

`chapters/`组织课程；`attention/`、`layers/`、`models/`、`cache/`、`training/`、`inference/`提供共享数学与实验实现。目录与接口见[代码组织](docs/architecture.md)，完整公式见[原理](docs/principles.md)，训练/推理成本见[推理分析](docs/inference.md)，检查范围见[覆盖表](docs/coverage.md)。

默认uv.lock安装CPU PyTorch。需要CUDA时按[PyTorch官方安装页](https://pytorch.org/get-started/locally/)选wheel，再用`uv run --no-sync ... --device cuda`避免同步回CPU版本。模型、输入与缓存必须同设备，CPU/CUDA共享实现。

测试默认小张量CPU，CUDA测试在硬件不可用时跳过。2026章节是带检查的机制参考，公开但未实现的模型专属模块在文档中明确列出；不从小样例推断语言能力或GPU吞吐。性能基准需主动运行`transformer-lab benchmark`。
