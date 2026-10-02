# 16 · 推理基础 · KV Cache / Prefill / Decode

前置：02、08、11；系统支线可以在11后提前阅读。

## 机制与代码

KV Cache是自回归执行策略，先区分训练、prefill、单token decode与chunk decode；再对照动态Self KV、静态Cross KV和因果Encoder追加。

入口：[ch16_kv_cache.py](../../src/transformer_lab/chapters/ch16_kv_cache.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 16
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| current / prefix | `ids[B,S_q]；old K/V[B,H_kv,P,D_h]` |
| append | `new K/V[B,H_kv,S_q,D_h] → full[B,H_kv,P+S_q,D_h]` |
| query / scores | `Q[B,H_q,S_q,D_h]；scores[B,H_q,S_q,P+S_q]` |
| output / full_valid | `logits[B,S_q,V]；bool valid[B,P+S_q]` |
| Cross | `Encoder memory[B,S_source,D] → static K/V[B,H_kv,S_source,D_h]` |

## 检查依据与范围

完整、逐token、不等长chunk前向/生成一致；静态Cross cache复用；causalEncoder分块编码一致。

cat返回新cache，具有O(S)拷贝成本；mask用绝对k≤q，单行decode不能对本地shape盲用tril。双向Encoder新增source后需要重算。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。2026公开结构汇总见[模型对照](../models-2026.md)。
