# 02 · 2018 · Decoder-only / GPT / NTP

前置：01

## 机制与代码

2018 GPT 路线：保留因果 Decoder，删除 Encoder 和 Cross-Attention；监督改为 next-token prediction。NaiveDecoder 直接展示两条残差分支。

入口：[ch02_decoder_only.py](../../src/transformer_lab/chapters/ch02_decoder_only.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 02
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| IDs / hidden / logits | `[B,S] → [B,S,D] → [B,S,V]` |
| Self scores | `[B,H_q,S,S]，下三角可见` |
| NTP logits / labels | `logits[:,:-1] [B,S-1,V] 对齐 tokens[:,1:] [B,S-1]` |
| loss | `有效 token CE → scalar []` |

## 检查依据与范围

未来输入不改变 earlier logits、所有参数梯度有限、核心完整/缓存前向一致。

使用 Sinusoidal PE 和经典 Block 以便隔离架构变化；不是 GPT 的 learned PE、tokenizer 或原始训练配方。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)。2026公开结构汇总见[模型对照](../models-2026.md)。
