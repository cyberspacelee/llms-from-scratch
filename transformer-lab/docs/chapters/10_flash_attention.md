# 10 · 2022 · FlashAttention / Online Softmax / SDPA

前置：01、07

## 机制与代码

2022 FlashAttention：保持精确 dense softmax 数学，通过分块与 online softmax降低中间存储和 IO。这里先看可微 KV-block reference，再核对 SDPA。

入口：[ch10_flash_attention.py](../../src/transformer_lab/chapters/ch10_flash_attention.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 10
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| Q/K/V | `[B,H_q,S_q,D_h] / [B,H_q,S_kv,D_h] / [B,H_q,S_kv,D_v]` |
| score tile | `[B,H_q,S_q,C]，C为KV块宽` |
| running max / denominator | `[B,H_q,S_q,1]` |
| running numerator | `[B,H_q,S_q,D_v]` |
| final normalize | `numerator/denominator → [B,H_q,S_q,D_v]` |

## 检查依据与范围

online/manual/SDPA 输出一致，online/manual 输入梯度一致；包含不整除的尾块和不同 query/key长度。

不是 GPU FlashAttention kernel；只切 KV、不切 Q、不加mask，不宣称加速。因果 SDPA 对照在核心测试中；SDPA是否使用Flash backend由设备/dtype决定。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2205.14135)、[来源 2](https://arxiv.org/abs/2307.08691)。2026公开结构汇总见[模型对照](../models-2026.md)。
