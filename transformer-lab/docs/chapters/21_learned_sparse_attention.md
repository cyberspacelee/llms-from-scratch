# 21 · 2025–2026 · DeepSeek DSA / CSA2 / Qwen QSA

前置：07、19、20

## 机制与代码

2025 DeepSeek DSA到2026 CSA2/Qwen4-Exp QSA：小维度learned indexer选择token或连续块，正式Attention只读取selected keys。区分indexer scores与正式Attention scores。

入口：[ch21_learned_sparse_attention.py](../../src/transformer_lab/chapters/ch21_learned_sparse_attention.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 21
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| indexer projection | `hidden[B,S,D] → Q_i[H_i,S,D_i]、shared K_i[S,D_i]` |
| selection scores | `ReLU(Q_i@K_iᵀ)跨heads汇总→[S,S]` |
| token budget | `indices/valid[S,M]，屏蔽未来与filler` |
| block budget | `完整块score[C] → top blocks → 连续token；尾部不完整块保留` |
| formal gather | `K/V[B,H_q,S,M,D_h/D_v] → scores[B,H_q,S,M] → output[B,H_q,S,D_v]` |

## 检查依据与范围

token/block selection均因果、无重复有效keys、gather与同mask dense一致、未来V不影响过去输出。

one-sequence小参考，indexer评分仍为dense。官方head weights/归一化、indexer蒸馏、压缩缓存、hierarchical候选复用、FP4/FP8 scoring不在本例伪装成已完成。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)、[来源 2](https://github.com/huggingface/transformers/blob/35924ec379eec682bbdca219886e16eff4df8b09/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)。2026公开结构汇总见[模型对照](../models-2026.md)。

Qwen3.8-Flash-Next的2026报告同样明确描述QSA、四路GR与n-gram embedding；应按具体版本区分dense/MoE、基础hybrid与这些额外优化。资料：[Qwen3.8-Next架构报告](https://arxiv.org/abs/2608.30320)。
