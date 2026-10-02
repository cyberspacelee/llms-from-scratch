# 01 · 2017 · Naive Transformer / Encoder–Decoder

前置：无；先认识张量轴和矩阵乘法。

## 机制与代码

2017 原始 Transformer：Encoder–Decoder、Scaled Dot-Product Attention、MHA、Sinusoidal PE、Post-LayerNorm、残差和逐 token ReLU FFN。先直接读 NaiveAttention 和 NaiveTransformer，再看同权重的核心实现。

入口：[ch01_naive_transformer.py](../../src/transformer_lab/chapters/ch01_naive_transformer.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 01
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| source / decoder IDs | `long [B,S_kv] / [B,S_q]` |
| Embedding + PE | `[B,S,D]，PE[S,D] 按 batch broadcast` |
| Q projection / reshape / transpose | `[B,S_q,D] → [B,S_q,H_q,D_h] → [B,H_q,S_q,D_h]` |
| K/V projection | `[B,S_kv,D] → [B,H_q,S_kv,D_h]` |
| QKᵀ / masked softmax | `[B,H_q,S_q,S_kv]，softmax 沿最后的 key 轴` |
| P@V / merge heads / output | `[B,H_q,S_q,D_h] → [B,S_q,H_q*D_h] → [B,S_q,D]` |
| FFN | `[B,S,D] → [B,S,D_ff] → [B,S,D]` |
| head / teacher forcing | `logits[B,S_q,V]；BOS+targets[:,:-1] 与 targets[B,S_q] 对齐` |

## 检查依据与范围

一次前向/反向/AdamW 更新、因果性、朴素与核心模型同权重输出和 embedding 梯度、多头多层缓存对照。

默认 B=2、D=32、H_q=4、D_h=8、D_ff=64；单层、共享词表，无 padding/dropout，不复刻原论文训练。Self 时 S_q=S_kv；Cross 两侧长度可以不同。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/1706.03762)。2026公开结构汇总见[模型对照](../models-2026.md)。
