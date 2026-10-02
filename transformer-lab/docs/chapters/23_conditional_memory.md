# 23 · 2026 · DeepSeek Engram / Qwen PLE

前置：06、20

## 机制与代码

2026 DeepSeek Engram与Qwen4-Exp PLE：通过因果n-gram查表给特定层加入lexical features，配合context gate和因果卷积；与MoE路由到神经专家不同。

入口：[ch23_conditional_memory.py](../../src/transformer_lab/chapters/ch23_conditional_memory.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 23
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| tokens / bigram | `ids[B,S] + previous[B,S] → hashes[B,S]` |
| lookup | `hash table[T_table,D] → lexical[B,S,D]` |
| context gate | `hidden与projected key点积→scores[B,S,1]` |
| memory feature | `gate*value[B,S,D]` |
| dilated conv | `transpose[B,D,S] + 左pad → [B,S,D]` |

## 检查依据与范围

修改未来IDs/hidden不会影响过去memory，lookup table收到有限梯度。

本例small bigram table有哈希碰撞；官方多阶多head哈希、tokenizer压缩、每层独立布局、增量ngram/卷积cache尚未实现。Engram/PLE仅共享机制，不是相同模型。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2601.07372)、[来源 2](https://github.com/huggingface/transformers/blob/35924ec379eec682bbdca219886e16eff4df8b09/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)。2026公开结构汇总见[模型对照](../models-2026.md)。

Qwen3.8-Flash-Next的2026报告同样明确描述QSA、四路GR与n-gram embedding；应按具体版本区分dense/MoE、基础hybrid与这些额外优化。资料：[Qwen3.8-Next架构报告](https://arxiv.org/abs/2608.30320)。
