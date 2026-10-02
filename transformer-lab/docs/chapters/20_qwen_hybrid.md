# 20 · 2026 · Qwen3.5 / 3.6 / 3.8 Hybrid

前置：06、11、15、16

## 机制与代码

2026 Qwen3.5/3.6/3.8结构：3:1 Gated DeltaNet/full attention层交替，dense与MoE分支；细化因果短卷积、不同QK/V头数、continuous decay、RMSNorm+SiLU输出门与full-attention sigmoid门。

入口：[ch20_qwen_hybrid.py](../../src/transformer_lab/chapters/ch20_qwen_hybrid.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 20
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| QKV before conv | `[B,S,2H_k*D_k+H_v*D_v] → transpose[B,C,S]` |
| causal conv | `左pad W-1；depthwise conv → [B,S,C]` |
| split / head repetition | `Q/K[B,H_k,S,D_k]、V[B,H_v,S,D_v]；Q/K repeat→H_v` |
| decay / state | `g=-exp(A_log)*softplus(a+dt_bias)[B,H_v,S]；state[B,H_v,D_k,D_v]` |
| output gate | `RMSNorm(context)*SiLU(z) [B,H_v,S,D_v]` |
| full attention gate | `sigmoid gate[B,H_q,S,D_h]，合头前相乘` |

## 检查依据与范围

短卷积不泄漏未来、异构头GatedDelta chunk/full状态与输出一致、有限梯度、四层3:1配方cache一致。

H_q*D_h不一定等于D。核心hybrid配方仍为简化递归；本章独立参考补关键机制，未提供官方权重适配、卷积cache/fused kernel或VL mRoPE。Qwen4-Exp见21–23章。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)、[来源 2](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json)、[来源 3](https://github.com/huggingface/transformers/blob/35924ec379eec682bbdca219886e16eff4df8b09/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。2026公开结构汇总见[模型对照](../models-2026.md)。
