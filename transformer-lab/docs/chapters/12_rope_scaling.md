# 12 · 2023 · RoPE Scaling / YaRN

前置：08、11

## 机制与代码

2023 长上下文频率扩展：Linear interpolation、固定 NTK base调整和 YaRN 分频插值。只改位置频率，不把扩大 max_length当作模型能力提升。

入口：[ch12_rope_scaling.py](../../src/transformer_lab/chapters/ch12_rope_scaling.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 12
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| frequency | `[D_h/2]` |
| phase / rotate | `positions[S] → phase[S,D_h/2]；Q/K[B,H_q,S,D_h] shape不变` |
| cache | `已旋转 K[B,H_kv,S_kv,D_h]，整段会话用固定频率` |

## 检查依据与范围

各缩放范数保持；频率显式输出；完整/缓存前向一致。

这不是动态 NTK；会话中改变频率会使新Q和已缓存K不处于同一坐标系。长上下文质量需要训练与独立评测。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2306.15595)、[来源 2](https://arxiv.org/abs/2309.00071)。2026公开结构汇总见[模型对照](../models-2026.md)。
