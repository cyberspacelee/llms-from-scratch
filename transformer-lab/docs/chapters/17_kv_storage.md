# 17 · 2023–2024 推理 · Paged KV / Prefix / Quantization

前置：16

## 机制与代码

2023 PagedAttention启发的页式存储/前缀共享，及2024 KV量化方向。页分配、数学计算和数值精度分别核对。

入口：[ch17_kv_storage.py](../../src/transformer_lab/chapters/ch17_kv_storage.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 17
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| logical KV | `K/V[B,H_kv,S_kv,D_h/D_v]` |
| pages | `[B,H_kv,page_tokens,D_h/D_v]，尾页可不满` |
| quantize | `codes int8[B,H_kv,S_kv,D_h]；scales float[B,H_kv,S_kv,1]` |
| reconstruct | `codes*scales → 原shape/dtype` |

## 检查依据与范围

prefix fork不污染、完整页共享、materialize值一致、误差≤scale/2、实际codes+scale字节。

block_table是对象ID不是设备页表；materialize拷贝全部KV。int8只是存储原语，不是KIVI的完整算法或2026 FP4/FP8 fused Attention。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2309.06180)、[来源 2](https://arxiv.org/abs/2402.02750)。2026公开结构汇总见[模型对照](../models-2026.md)。
