# 13 · 2024 · Multi-head Latent Attention (MLA)

前置：08、11

## 机制与代码

2024 DeepSeek-V2/V3 MLA：压缩 joint KV latent，独立 rotary key。naive展开每头K/V；absorbed把Key up-projection移到query侧，并合并Value/Output映射。

入口：[ch13_mla.py](../../src/transformer_lab/chapters/ch13_mla.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 13
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| KV down / norm | `[B,S_kv,D] → C[B,1,S_kv,L_kv]` |
| decoupled rotary key | `Kr[B,1,S_kv,D_r]` |
| query | `[B,S_q,D] → [B,H_q,S_q,D_h+D_r] → Qc/Qr` |
| naive | `Kc[B,H_q,S_kv,D_h]、V[B,H_q,S_kv,D_v]` |
| absorbed | `Q_latent[B,H_q,S_q,L_kv]；P@C[B,H_q,S_q,L_kv]；W_VO[H_q,L_kv,D]` |
| output | `[B,S_q,D]` |

## 检查依据与范围

同权重naive/absorbed decode一致；全测试额外比较self/cross输出与全部参数梯度。

cache.value字段保存Kr，不是展开V。MLA内容latent不能直接套普通RoPE/QK-Norm。V4的压缩Attention结构已有变化，后续20–25章单独解释。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2405.04434)、[来源 2](https://arxiv.org/abs/2412.19437)。2026公开结构汇总见[模型对照](../models-2026.md)。
