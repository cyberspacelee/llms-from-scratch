# 22 · 2025–2026 · mHC / GR / Kimi AttnRes

前置：05、20

## 机制与代码

2025–2026 DeepSeek mHC和Qwen4-Exp GR：多残差流扩展的是residual宽度，内部Attention/FFN仍读D维。mHC约束mixing近似双随机，GR采用低秩elementwise门与注入门。

Kimi AttnRes则沿深度选择此前层/块表示，不需要并行多残差流。`attention_residual`实现normalized keys、learned pseudo-query和depth softmax；没有接入完整Block AttnRes边界调度。

入口：[ch22_residual_streams.py](../../src/transformer_lab/chapters/ch22_residual_streams.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 22
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| expanded residual | `[B,S,D] → streams[B,S,N_streams,D]` |
| dynamic mixer | `flatten[B,S,N*D] → pre/post[B,S,N] + mixing logits[B,S,N,N]` |
| mHC collapse / update | `pre加权→[B,S,D]；mixing@streams+post*branch→[B,S,N,D]` |
| Sinkhorn | `非负matrix[B,S,N,N]，行/列和≈1` |
| GR | `low-rank projection → gates[B,S,N,D]；加权collapse[B,S,D]` |

| AttnRes | `sources[B,S,N_sources,D] → depth weights[B,S,N_sources] → hidden[B,S,D]` |

## 检查依据与范围

双随机行列和、动态mixer梯度、输出保留stream shape，分别展示mHC与GR路径；AttnRes检查零query得到平均、depth权重和为1及source/query梯度。

小规模dynamic reference，非官方初始化/精度策略或Single-Pass fused实现；有限迭代不保证所有极端logits都精确收敛。scalar残差gate不等于HC。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2512.24880)、[来源 2](https://github.com/huggingface/transformers/blob/35924ec379eec682bbdca219886e16eff4df8b09/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)。[AttnRes官方](https://github.com/MoonshotAI/Attention-Residuals)。2026公开结构汇总见[模型对照](../models-2026.md)。

Qwen3.8-Flash-Next的2026报告同样明确描述QSA、四路GR与n-gram embedding；应按具体版本区分dense/MoE、基础hybrid与这些额外优化。资料：[Qwen3.8-Next架构报告](https://arxiv.org/abs/2608.30320)。
