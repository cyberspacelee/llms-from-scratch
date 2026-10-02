# 19 · 长上下文支线 · Sequence Compression

前置：07、16

## 机制与代码

长上下文压缩支线：区别head轴压缩(MQA/GQA)、channel轴压缩(MLA)和sequence轴pooling。本例保留近期精确KV，只使用已完成旧块摘要。

入口：[ch19_sequence_compression.py](../../src/transformer_lab/chapters/ch19_sequence_compression.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 19
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| original KV | `[B,H_q,S_kv,D_h/D_v]` |
| reshape / pooling | `[B,H_q,C,R,D] → mean over R → [B,H_q,C,D]` |
| summary availability | `ends[C]，只在block结束后可读` |
| selected sources | `旧摘要 + 未完整边界token + local window` |
| output | `[B,H_q,S_q,D_v]` |

## 检查依据与范围

完整与逐token压缩一致、未来值不会进入过去摘要、未完成块保留精确token。

均值池化是近似模型，不是learned压缩；2026 DeepSeek CSA/HCA/CSA2的门控、索引和共享分别在21/24章说明。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://arxiv.org/abs/2606.19348)。2026公开结构汇总见[模型对照](../models-2026.md)。
