# 18 · 2023 推理 · Speculative Decoding

前置：02、14、16

## 机制与代码

2023 speculative decoding：draft提议候选，target一次验证，提交一致前缀、首个修正或bonus token。本例固定greedy，清楚展示对齐。

入口：[ch18_speculative_decoding.py](../../src/transformer_lab/chapters/ch18_speculative_decoding.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 18
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| prompt / proposal | `[1,S_prompt] / [1,K]` |
| target logits | `[1,S_prompt+K,V]` |
| verify rows | `logits[:,S_prompt-1:S_prompt+K].argmax → [1,K+1]` |
| commit | `接受前缀[1,N_accept] + correction/bonus[1,1]` |
| result | `[1,S_prompt+N_generated]` |

## 检查依据与范围

相同/不同draft权重都与target greedy结果一致，接受与提议计数可检查。

不实现随机p/q接受算法、KVrollback或MTP/DSpark专属draft；这里完整前缀重算用于透明验证。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2211.17192)。2026公开结构汇总见[模型对照](../models-2026.md)。
