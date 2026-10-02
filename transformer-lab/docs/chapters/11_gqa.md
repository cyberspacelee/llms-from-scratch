# 11 · 2023 · Grouped-Query Attention (GQA)

前置：04、08

## 机制与代码

2023 GQA：在 MHA 的 H_kv=H_q 和 MQA 的 H_kv=1之间选择多个 KV heads。每组连续 query heads共享一个KV head。

入口：[ch11_gqa.py](../../src/transformer_lab/chapters/ch11_gqa.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 11
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| Q | `[B,H_q,S,D_h]` |
| K/V cache | `[B,H_kv,S,D_h]，示例 H_q=4、H_kv=2` |
| group mapping | `[0,0,1,1]；head // (H_q/H_kv)` |
| temporary expansion | `repeat_interleave → [B,H_q,S,D_h]` |
| merge | `[B,H_q,S,D_h] → [B,S,D]` |

## 检查依据与范围

逐头group参考与共享实现一致；cache字节与账本一致。

H_q必须整除H_kv的倍数关系，即 H_q % H_kv=0；本例不实现从MHA checkpoint mean-pooling转换或uptraining。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2305.13245)。2026公开结构汇总见[模型对照](../models-2026.md)。
