# 24 · 2024–2026 · Shared KV / YOCO / DeepSeek CED

前置：13、16、19、21

## 机制与代码

2024 YOCO到2026 DeepSeek V4.1：lower causal layers构建global KV，upper层复用。CED把prefill与decode计算预算分开；CSA2再复用compressed KV和稀疏indices。

入口：[ch24_shared_kv.py](../../src/transformer_lab/chapters/ch24_shared_kv.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 24
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| lower stack | `IDs[B,S] → causal hidden[B,S,D]` |
| shared global projection | `[B,S,D] → shared[B,1,S_kv,D_h]，只构建一次` |
| consumer Q | `本层hidden[B,S_q,D] → Q[B,H_q,S_q,D_h]` |
| shared view / visibility | `expand view[B,H_q,S_kv,D_h]；绝对k≤q` |
| consumer output | `[B,S_q,D]，多个consumer各有Q/O但读同一shared KV` |

## 检查依据与范围

两个consumer共享同一tensor，chunk/full结果相同，shared projection与lower embedding均可反向传播。

不是经典独立Encoder/Decoder stack的static Cross语义。示例未复刻V4.1局部窗口、source-layer配置周期、hierarchical indexer、压缩比例、FP4 KV或prefill bypass调度。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2405.05254)、[来源 2](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py)。2026公开结构汇总见[模型对照](../models-2026.md)。
