# 15 · 2024–2025 · Delta / KDA / Hybrid Attention

前置：07、11

## 机制与代码

2024–2025 Gated DeltaNet与hybrid：Delta按value预测残差更新记忆，Gated Delta先衰减旧状态，再更新；可以与softmax层交替。

入口：[ch15_hybrid_attention.py](../../src/transformer_lab/chapters/ch15_hybrid_attention.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 15
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| Q/K/V | `[B,H_q,S,D_h/D_v]` |
| beta / alpha | `[B,H_q,S]，每token每head标量` |
| KDA log decay | `[B,H_q,S,D_k]，逐key通道而非每head标量` |
| state | `[B,H_q,D_h,D_v]，不随历史长度增长` |
| token read / merge | `[B,H_q,D_v] → [B,S,H_q*D_v] → [B,S,D]` |

## 检查依据与范围

短/长context的state bytes相同；Delta overwrite方程、chunk/full和hybrid生成一致。

核心通用递归参考没有卷积或复杂output gate；本章另以`channel_delta`实现KDA逐通道衰减，核对full/chunk输出、状态和梯度，不包含完整Kimi模块；Qwen2026细化机制见20章。RoPE/NoPE交替是另一支线，不是旋转半个layer。

术语参照[统一约定](../plan.md)。资料：[Gated DeltaNet](https://arxiv.org/abs/2412.06464)、[Kimi Linear](https://arxiv.org/abs/2510.26692)、[官方Kimi实现](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)。2026公开结构汇总见[模型对照](../models-2026.md)。
