# 实验覆盖与检查范围

这是原理库的功能覆盖表；默认测试使用小张量 CPU，未来的 GPU 路径需要硬件核对。没有以语言生成样例代替数值检查，没有运行性能基准。

| 用户目标 | 代码/文档 | 功能核对 |
| --- | --- | --- |
| 1. 基础 Transformer 和三类结构 | `attention/softmax.py` / `models/transformer.py` / `training/losses.py` | 手算 SDPA；因果性；双向性；teacher forcing；Encoder memory；greedy cached/naive |
| 2. 位置编码 | `attention/position.py` / `experiments/presets.py` | sin/cos 起点；RoPE 范数与共同平移；Linear Scaling；各 Scaling 的缓存一致 |
| 3. Prefill/Decode 与 Cache | `cache/state.py` / `models/transformer.py` | 完整、逐 token、非等长 chunk 前向；层位置一致；Cross static cache |
| 4. MHA/MQA/GQA/MLA 演进 | `attention/softmax.py` / `attention/mla.py` / `analysis.py` | KV group 手工映射；参数与实际 cache byte 计数 |
| 5. MLA 深入 | `attention/mla.py` | 两种 query rank；self/cross；naive/absorbed 输出和所有参数梯度 |
| 6. Attention pattern | `attention/patterns.py` / `gathered_attention` | 全局/窗口/分块/稀疏缓存；gather 与相同 dense selection 相等 |
| 7. Norm/FFN/Residual | `layers/` / `TransformerBlock` | LayerNorm/RMSNorm 对照；5 种 FFN 参数；Pre/Post-Norm；门控残差配置 |
| 8. MoE | `MixtureOfExperts` | sparse dispatch 与所有专家 reference 相等；router 梯度；shared；bias 更新方向 |
| 9. MTP | `MultiTokenPrediction` / `language_model_loss` | 各深度 Shape/标签；未来目标不泄漏；padding；联合梯度 |
| 10. 长上下文 | `inference/compression.py` / `analysis.py` / `docs/inference.md` | 压缩分块边界；逐 token/完整压缩一致；无未来泄漏；理论成本 |
| 11. Hybrid | `attention/recurrent.py` / `layer_blocks` | Linear 与并行特征 kernel 对照；delta overwrite 方程；hybrid cache |
| 12. 现代 Encoder–Decoder | `encode` / `EncoderMemory` / `causal_seq2seq` | 不同 encoder/decoder 配方；Encoder causal append；aligned causal source |
| 13. 统一框架 | configs / `make_attention` / `TransformerBlock` | 多种小型配方与统一输出/cache 接口 |
| 14. 公式/Shape/成本对应 | `docs/principles.md` / `docs/inference.md` | 参数计数与 tensor 字节可执行核对；FLOPs 标明分析口径 |
| 15. 高性能对照 | SDPA / page/quant/speculation 原语 / `benchmark` | CPU SDPA 输出/梯度已核对；GPU 性能与外部 kernel 未执行 |

运行：

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m compileall -q src tests
uv run --locked python -m unittest discover -s tests -v
```

Ruff 检查 import、Python 常见错误、bugbear 和缺失的函数类型标注；它不是完整静态类型推导器。源码的 Tensor Shape 使用 docstring/文档与运行时检查表达，没有引入专用 shape typing 包。Python `unittest` 使用子用例复用检查，覆盖全部配方，而不是只核对函数能 import。

CPU 验证采用 fp64 做严格数值对照，通常 `atol=1e-9, rtol=1e-7`；单步 optimizer 功能使用小模型。没有大数据集、长上下文或反复拟合测试。当前环境安装 CPU torch，因此 CUDA 检查会跳过；换成 CUDA wheel 后可以运行同一套命令。

只完成原语或研究入口的内容：learned DSA/CSA/HCA indexer/compressor、完整 KDA、Hyper-Connection/mHC、动态扩展 decoder source、MTP 专属 speculative cache、随机 speculative sampling、生产 Prefix 服务、paged allocator 与 GPU paged/quantized/MoE kernels。它们分别需要新数据流、算法或真实硬件；不能把已有相邻功能计为完整实现。

## 由浅入深的入口

`tutorials/mvp.py` 提供固定单头、单层 Encoder–Decoder；`tutorials/evolution.py` 提供 00–13 步可运行核对。新增测试检查 MVP 与统一模型同权重输出/梯度，以及整条演进路线。阅读顺序见 [MVP](mvp.md) → [演进](evolution.md)，包边界见 [子包规划](architecture.md)。
