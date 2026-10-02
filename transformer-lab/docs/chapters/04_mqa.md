# 04 · 2019 · Multi-Query Attention (MQA)

前置：02

## 机制与代码

2019 MQA：Q 仍有 H_q 个头，全部 query heads 共享一份 K 和一份 V，H_kv=1。逐头参考循环显式核对映射。

入口：[ch04_mqa.py](../../src/transformer_lab/chapters/ch04_mqa.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 04
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| Q projection | `[B,S,D] → [B,H_q,S,D_h]` |
| K/V projection and cache | `[B,S,D] → [B,1,S,D_h]` |
| temporary expansion | `[B,1,S,D_h] → [B,H_q,S,D_h]，cache仍保留1个头` |
| output | `[B,H_q,S,D_h] → [B,S,D]` |

## 检查依据与范围

各 query head 映射到同一 KV head；输出与逐头参考一致；实际 cache.nbytes 与理论账本一致。

MQA 不是减少 Q 的头数；配置与随机权重不同的 MHA 不要求 logits 相同。这里仍使用绝对 PE，避免提前引入 RoPE。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/1911.02150)。2026公开结构汇总见[模型对照](../models-2026.md)。
