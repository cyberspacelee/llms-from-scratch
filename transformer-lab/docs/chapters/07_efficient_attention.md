# 07 · 2020 · Sparse / Linear Attention

前置：02；了解 04/06 有助于区分压缩轴。

## 机制与代码

2020 前后效率分支：Local/Sparse 改变读取哪些 token，Linear Attention 则用正特征核与固定矩阵状态改变 Attention 数学。

入口：[ch07_efficient_attention.py](../../src/transformer_lab/chapters/ch07_efficient_attention.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 07
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| sparse indices | `[S_q,M]；selected K/V[B,H_q,S_q,M,D_h/D_v]` |
| sparse scores / context | `[B,H_q,S_q,M] → [B,H_q,S_q,D_v]` |
| linear state / normalizer | `state[B,H_q,D_h,D_v]；z[B,H_q,D_h]` |
| linear read | `q[B,H_q,D_h] 读取 state → y[B,H_q,D_v]` |

## 检查依据与范围

sparse gather 与同 selection 的 dense mask一致；Linear递归与并行特征核一致。

稀疏 mask 本身仍可生成密集 scores；真实 gather 才避免完整 score 分配。Linear 不是精确 softmax Attention，固定推理状态不代表训练内存固定。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2004.05150)、[来源 2](https://arxiv.org/abs/2006.16236)。2026公开结构汇总见[模型对照](../models-2026.md)。
