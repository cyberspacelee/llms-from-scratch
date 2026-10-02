# 06 · 2020 · Gated FFN / SwiGLU

前置：05

## 机制与代码

2020 GLU variants：ReLU/GELU FFN 与 GLU/GeGLU/SwiGLU 独立对照。SwiGLU 为 SiLU(gate(x))*up(x)，最后 down projection。

入口：[ch06_gated_ffn.py](../../src/transformer_lab/chapters/ch06_gated_ffn.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 06
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| up / gate | `[B,S,D] → [B,S,D_ff]，两个独立投影` |
| elementwise gate | `[B,S,D_ff] * [B,S,D_ff] → [B,S,D_ff]` |
| down | `[B,S,D_ff] → [B,S,D]` |

## 检查依据与范围

所有激活可反向传播；参数数目与 2DD_ff/3DD_ff 账本一致。

固定 D_ff 时 gated FFN 参数更多。公平比较应按参数预算调整 intermediate size；FFN 不交换 token 信息。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2002.05202)。2026公开结构汇总见[模型对照](../models-2026.md)。
