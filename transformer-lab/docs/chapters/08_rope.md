# 08 · 2021 · Rotary Position Embedding (RoPE)

前置：06

## 机制与代码

2021 RoFormer/RoPE：不再给 embedding 加 Sinusoidal PE；对每个 head 的 Q/K 坐标对做旋转，让 dot product 编码相对位置。

入口：[ch08_rope.py](../../src/transformer_lab/chapters/ch08_rope.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 08
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| Q/K | `[B,H_q,S,D_h]` |
| coordinate pairing | `[B,H_q,S,D_h/2,2]` |
| frequency / phase | `[D_h/2]；positions[S] → phase[S,D_h/2]` |
| rotate / V | `Q/K shape不变；V不旋转` |

## 检查依据与范围

范数保持、共同位置平移后的 dot product一致、Decoder及Cross cache一致。

相邻坐标配对；官方权重可能使用另一种 pairing 布局。跨 chunk 的 position不能重置；Cross两侧使用各自的坐标。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2104.09864)。2026公开结构汇总见[模型对照](../models-2026.md)。
