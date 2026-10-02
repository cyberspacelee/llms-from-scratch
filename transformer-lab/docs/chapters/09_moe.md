# 09 · 2021–2024 · Mixture of Experts (MoE)

前置：06、08

## 机制与代码

MoE 更早已有研究，本章以 2021 Switch 到 2024 DeepSeekMoE 的 Transformer 普及路线为定位。Router→Top-K→Dispatch→Expert→Combine，独立于 MLA。

入口：[ch09_moe.py](../../src/transformer_lab/chapters/ch09_moe.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 09
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| flatten valid tokens | `[B,S,D] → [N,D]` |
| router | `[N,D] → scores[N,E]` |
| top-k | `expert IDs/weights[N,K]` |
| expert and combine | `[N_e,D] → [N_e,D_ff] → [N_e,D]；index_add→[N,D]→[B,S,D]` |
| stats | `counts[E]；auxiliary loss[]` |

## 检查依据与范围

总分派次数=N*K、router梯度、共享专家、bias更新、稀疏派发与 dense参考一致。

总参数与激活参数分开；padding不派发，没有 capacity/token dropping。示例使用 Python expert loop，不能推断 GPU grouped-GEMM 性能。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2101.03961)、[来源 2](https://arxiv.org/abs/2401.06066)。2026公开结构汇总见[模型对照](../models-2026.md)。
