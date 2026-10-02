# 03 · 2018 · Encoder-only / BERT / MLM

前置：01；与 02 比较可见性与监督目标。

## 机制与代码

2018 BERT 路线：双向 Encoder-only 和 masked language modeling。输入位置被替换成 MASK ID，只有 selected mask 位置参与 CE。

入口：[ch03_encoder_only.py](../../src/transformer_lab/chapters/ch03_encoder_only.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 03
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| mask / IDs | `bool selected[B,S]；masked_ids[B,S]` |
| Encoder hidden / scores | `[B,S,D]；scores[B,H_q,S,S] 不加因果遮挡` |
| MLM logits | `[B,S,V]` |
| selected CE | `logits[selected] [N_mask,V] 对齐 ids[selected] [N_mask] → []` |

## 检查依据与范围

改动尾部会改变前面输出、MLM loss 有限且可反向传播。

这是双向机制和监督对照；没有 BERT 特有的 token-type/learned-position embeddings、NSP 或 80/10/10 masking 策略。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/1810.04805)。2026公开结构汇总见[模型对照](../models-2026.md)。
