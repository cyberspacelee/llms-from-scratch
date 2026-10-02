# 14 · 2024 · Multi-Token Prediction (MTP)

前置：02、13

## 机制与代码

2024 Multi-Token Prediction：区分独立future heads与DeepSeek顺序future-conditioned模块。本例深度j读真值token x(t+j)，监督x(t+j+1)。

入口：[ch14_mtp.py](../../src/transformer_lab/chapters/ch14_mtp.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 14
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| base hidden / tokens | `[B,S,D] / [B,S]` |
| depth j alignment | `h[B,S-j,D] 与 future embedding[B,S-j,D]` |
| concat / projection / block | `[B,S-j,2D] → [B,S-j,D]` |
| prediction / labels | `logits[B,S-j-1,V] 对齐 tokens[:,j+1:] [B,S-j-1]` |
| joint loss | `NTP + weight*mean(MTP losses) + MoE auxiliary` |

## 检查依据与范围

预测长度与labels对齐、未来目标不泄漏、MTP参数梯度、短序列和padding处理。

训练的真值future embedding在推理时不可直接使用；普通generate仍只调用主head。V4.1的独立DSpark训练路径不等于V3顺序MTP。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2404.19737)、[来源 2](https://arxiv.org/abs/2412.19437)。2026公开结构汇总见[模型对照](../models-2026.md)。
