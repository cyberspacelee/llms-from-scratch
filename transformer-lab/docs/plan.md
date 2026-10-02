# Transformer Lab 课程重构规划

目标：按业界模型与优化发展脉络组织课程，从2017原始Transformer到截至2026-10-02公开的主流结构。每章有独立源码、文档、运行入口与数学检查；不保留旧集中教程、旧步号或兼容包装。

## 统一术语与 tensor 约定

| 符号 | 含义 | 代码中的常用名称 |
| --- | --- | --- |
| B | batch size | batch_size |
| S | sequence length | seq_len |
| D | model dimension / hidden size | dim / hidden_size |
| S_q / S_kv | 当前 query / 可用 KV 长度 | query_len / key_len |
| H_q / H_kv | query / key-value head 数 | heads / effective_kv_heads |
| D_h / D_v | 每头 query/key / value 宽度 | head_dim / value_dim |
| D_ff / V | FFN intermediate size / vocabulary size | ff_dim / vocab_size |
| L_kv / L_q / D_r | MLA latent rank / rotary width | kv_rank / q_rank / rope_dim |
| P / N | cached prefix length / 有效 token 数 | offset / tokens |
| E / K / N_streams | 专家数 / Top-K / 残差流数 | experts / top_k / stream count |

hidden states统一写`[B,S,D]`，token IDs为`long [B,S]`，有效mask为`bool [B,S]`、True=有效/可见。Attention计算布局为`[B,H,S,D_head]`，Cross/Cache明确使用`S_q`与`S_kv`。`D`不必等于`H_q*D_h`，两侧通过独立投影连接；经典章节恰好相等。

字母不是行业强制标准，`T`表示时间步也很常见；本课程统一S避免同一sequence轴在各章换名。正式名称为Transformer、Rotary Position Embedding、Multi-Query Attention、Grouped-Query Attention和Multi-head Latent Attention。

## 发展主线与支线

01–15按代表机制出现/普及年代组织；年代是机制定位，不声称每章完整复现当年模型。不同研究支线有交叠：例如RMSNorm发表于2019，Pre-LN分析发表于2020；MoE研究早于Switch。

16–19为推理与长上下文支线，编号是教学阅读顺序，不能解释为这些优化晚于2025才发明。20–25把已有机制连接到2026公开模型的实际数据流。最新模型名称、资料revision和实现差异见[2026模型对照](models-2026.md)。

| 章 | 定位与机制 | 独立文档 |
| --- | --- | --- |
| 01 | 2017 · Naive Transformer / Encoder–Decoder | [naive_transformer](chapters/01_naive_transformer.md) |
| 02 | 2018 · Decoder-only / GPT / NTP | [decoder_only](chapters/02_decoder_only.md) |
| 03 | 2018 · Encoder-only / BERT / MLM | [encoder_only](chapters/03_encoder_only.md) |
| 04 | 2019 · Multi-Query Attention (MQA) | [mqa](chapters/04_mqa.md) |
| 05 | 2019–2020 · Pre-Norm / RMSNorm | [normalization](chapters/05_normalization.md) |
| 06 | 2020 · Gated FFN / SwiGLU | [gated_ffn](chapters/06_gated_ffn.md) |
| 07 | 2020 · Sparse / Linear Attention | [efficient_attention](chapters/07_efficient_attention.md) |
| 08 | 2021 · Rotary Position Embedding (RoPE) | [rope](chapters/08_rope.md) |
| 09 | 2021–2024 · Mixture of Experts (MoE) | [moe](chapters/09_moe.md) |
| 10 | 2022 · FlashAttention / Online Softmax / SDPA | [flash_attention](chapters/10_flash_attention.md) |
| 11 | 2023 · Grouped-Query Attention (GQA) | [gqa](chapters/11_gqa.md) |
| 12 | 2023 · RoPE Scaling / YaRN | [rope_scaling](chapters/12_rope_scaling.md) |
| 13 | 2024 · Multi-head Latent Attention (MLA) | [mla](chapters/13_mla.md) |
| 14 | 2024 · Multi-Token Prediction (MTP) | [mtp](chapters/14_mtp.md) |
| 15 | 2024–2025 · Delta / KDA / Hybrid Attention | [hybrid_attention](chapters/15_hybrid_attention.md) |
| 16 | 推理基础 · KV Cache / Prefill / Decode | [kv_cache](chapters/16_kv_cache.md) |
| 17 | 2023–2024 推理 · Paged KV / Prefix / Quantization | [kv_storage](chapters/17_kv_storage.md) |
| 18 | 2023 推理 · Speculative Decoding | [speculative_decoding](chapters/18_speculative_decoding.md) |
| 19 | 长上下文支线 · Sequence Compression | [sequence_compression](chapters/19_sequence_compression.md) |
| 20 | 2026 · Qwen3.5 / 3.6 / 3.8 Hybrid | [qwen_hybrid](chapters/20_qwen_hybrid.md) |
| 21 | 2025–2026 · DeepSeek DSA / CSA2 / Qwen QSA | [learned_sparse_attention](chapters/21_learned_sparse_attention.md) |
| 22 | 2025–2026 · mHC / GR / Kimi AttnRes | [residual_streams](chapters/22_residual_streams.md) |
| 23 | 2026 · DeepSeek Engram / Qwen PLE | [conditional_memory](chapters/23_conditional_memory.md) |
| 24 | 2024–2026 · Shared KV / YOCO / DeepSeek CED | [shared_kv](chapters/24_shared_kv.md) |
| 25 | 2026 · DeepSeek VL / Qwen VL / Omni | [multimodal](chapters/25_multimodal.md) |

## 实现原则与验收

章节只引入本章需要的机制。01/02直接展示朴素模型；其他章节复用已经验证的核心数学组件，并用独立projection、state或indexer参考核对变化。配置和可运行检查归本章所有，目录入口不再包含大型机制分支。

每个源码函数写`Args`/`Returns`，覆盖所有参数、返回类型、tensor shape及约束；投影、拆头、转置、旋转、缓存追加、路由、concat、label截短旁注明shape。Loss/auxiliary均明确scalar `[]`，cache明确当前与历史长度。

验收包括全部25章CPU运行、朴素/核心同权重输出与梯度、online-softmax输出/梯度、KV/chunk一致、稀疏选择因果性、多残差流/lookup/VL梯度和旧核心数学检查。使用现有PyTorch、unittest、Ruff，不新增依赖、权重下载或训练框架。

生产kernel和完整官方checkpoint适配不作为小张量课程实现：GPU Flash/Paged/FP4、完整DSpark、完整Engram/PLE、mRoPE/vision/audio encoder、完整V4.1 CED调度都需另有设备/权重/数据流验收；对应章节给出真实差异和修改入口。
