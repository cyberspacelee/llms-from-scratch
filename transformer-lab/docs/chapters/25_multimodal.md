# 25 · 2026 · DeepSeek VL / Qwen VL / Omni

前置：20–24；多模态接入支线。

## 机制与代码

2026 DeepSeek VL与Qwen VL/Omni：模态encoder输出经projector进入语言hidden序列。先核对文本/视觉slot融合，再读2D/mRoPE、视觉压缩与Omni Thinker/Talker流。

入口：[ch25_multimodal.py](../../src/transformer_lab/chapters/ch25_multimodal.py)。每个函数都有 `Args` / `Returns`，计算旁标注投影、转置、状态或监督变化。

```bash
uv run --locked transformer-lab chapter --chapter 25
```

## Tensor shape 数据流

| 步骤 | Shape / 对齐关系 |
| --- | --- |
| external encoded features | `[B,N_image,D_in]` |
| projector | `[B,N_image,D_in] → [B,N_image,D]` |
| text embedding / scatter | `ids[B,S] → hidden[B,S,D]；image_slots[B,S]每行恰好N_image个True` |
| causal backbone | `融合序列[B,S,D]经过hybrid layers` |
| head | `logits[B,S,V]` |

## 检查依据与范围

视觉projector梯度非零且有限，未来视觉特征不能影响此前文本输出，slot与feature数在输入边界核对。

输入已编码特征，没有视觉encoder/codec。DeepSeek的视觉token压缩、Qwen的multimodal RoPE、Omni Thinker/Talker和audio-text interleave在2026对照文档中说明；本例不生成音频/视频。

术语参照[统一约定](../plan.md)。资料：[来源 1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)、[来源 2](https://qwen.ai/blog?id=qwen3.8-livetranslate)、[来源 3](https://qwen.ai/blog?id=qwen3.8-omni-flash)。2026公开结构汇总见[模型对照](../models-2026.md)。

Gemma 4 12B展示另一条公开路线：原始图像patch与音频frame直接投影到语言hidden，不经重型独立encoder；shape接口仍为`[B,N_media,D_in] → [B,N_media,D]`，本章用已编码features检查slots，不复现Gemma的patch/frame处理或坐标embedding。来源：[Google开发指南](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/)。
