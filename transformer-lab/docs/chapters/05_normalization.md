# 05 · 2019–2020 · Pre-Norm / RMSNorm

前置：02

## 机制与代码

2019 RMSNorm 与 2020 Pre-LN 分析：分别研究 Norm 类型和残差次序，不同时更换 FFN/位置机制。Post: Norm(x+F(x))；Pre: x+F(Norm(x))。

入口：[ch05_normalization.py](../../src/transformer_lab/chapters/ch05_normalization.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 05
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| hidden | `[B,S,D]` |
| mean / variance / mean-square | `[B,S,1]，只沿最后一维统计` |
| normalized hidden | `[B,S,D]` |
| Norm parameters | `LayerNorm weight/bias[D]；RMSNorm weight[D]` |

## 检查依据与范围

LayerNorm/RMSNorm 与 PyTorch 原生结果一致；四种 Norm/order 配方均通过缓存检查。

Norm 每个 token 独立，不沿 batch/sequence 归一化；Norm 不持有 KV。时间段有交叠，章节按机制首次出现和教学依赖组织。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/1910.07467)、[来源 2](https://arxiv.org/abs/2002.04745)。2026公开结构汇总见[模型对照](../models-2026.md)。
