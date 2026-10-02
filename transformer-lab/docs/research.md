# 官方来源与模型实现边界

检索日期：**2026-10-02**。来源以论文、模型团队的官方代码和模型卡为主；实现独立编写，只移植数学思想，没有下载权重，也没有将第三方复刻当作官方代码。`main` 会变化，下表保存本次实际读取的 revision。

## DeepSeek 的演进

| 来源 | 实际读取 revision | 本项目对应 |
| --- | --- | --- |
| [V3 `inference/model.py`](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py) | `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` | normalized KV latent、query compression、decoupled RoPE、naive/absorbed |
| [V3.2-Exp `inference/model.py`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py) | `87e509a2e5a100d221c97df52c6e8be7835f0057` | indexer 选 key 的研究依据；本项目只提供静态 sparse mask/gather |
| [V4-Pro 官方 HF 推理代码](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py) | `b5968e9190ef611bbf34a7229255be88a0e937c1` | window + token compression + sparse selection 的原理对照 |
| [V4.1-Flash 官方 HF 推理代码](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py) | `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` | 因果非对称计算和跨层共享 compressed KV 的研究入口 |

V3 的 `MLA` 是基础实现最清楚的参照：缓存 normalized joint KV latent 和独立 rotary key；decode 把内容 Key up-projection 移到 Query 一侧。本项目进一步把 Value up-projection 和 Output 合并，保留两条路径的同权重/梯度对照。MoE 使用 routed/shared experts；router 的 selection bias 与最终 combine 权重分离。[V3 技术报告](https://arxiv.org/abs/2412.19437) 还给出顺序 MTP 的训练思路。

V3.2 的 DSA 引入 learned indexer；固定周期 mask 只是用于学习稀疏语义，不能声称复刻 DSA。key selection 代价、indexer 训练和低精度 scoring 是独立课题。本项目 `gathered_attention` 可以接收后续 indexer 选出的 indices。

V4 的公开代码增加 learned gated token pooling、window ring cache、稀疏选择与 mHC。它的 Attention 结构已有变化，不能把 V3 的 KV-up MLA 原封不动命名为 V4。我们的 mean-pooling 压缩原语只验证“序列维压缩”和因果可用时刻。[V4 官方发布](https://deepseek.com/news/v4-preview/) 与 [技术报告](https://arxiv.org/abs/2606.19348) 是结构背景；本项目不复刻 CSA/HCA 的全部细节。

最新检索到的 [V4.1 Flash 官方发布，2026-09-10](https://deepseek.com/news/deepseek-v4-1-flash/) 描述 Causal-Encoder-Decoder 和输入/输出非对称计算。阅读对应推理代码时，注意 `kv_source_layers` / `index_source_layers`、`SharedAttentionRuntime`：某些层构建 compressed KV/index，后续层读取同一组 compressed cache/selection。**不能仅凭发布中的 Encoder–Decoder 名称，就把它理解成经典两个独立 stack 加 Cross-Attention。** 本项目 `causal_seq2seq` 是经典 stack 的因果对照实验，不声称是 V4.1 的实现。后续研究跨层 KV reuse 时应从这些实际数据流出发，而非套名称。

V4.1 官方文件还含 vision、Engram 和 DSpark 相关推理模块；它们超出本次 Transformer 核心实验范围。DSpark forward 也不等于完整接受/拒绝循环。本项目不将这些未实现组件混入已有配方。

## 其他主流架构与小型配方

| 模型/论文与主要来源 | 代码中可复用的核心思想 | 已知差异 |
| --- | --- | --- |
| [原始 Transformer](https://arxiv.org/abs/1706.03762) | `classic`：sinusoidal、Post-LayerNorm、MHA、ReLU、Encoder–Decoder | 小尺寸；Linear 不带 bias；不是原论文训练复刻 |
| [Meta Llama 官方模型代码](https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama4/model.py) | `llama` 的 GQA/RMSNorm/SwiGLU；`irope` 的 RoPE/NoPE 层交替 | 无 inference temperature tuning、官方权重/多模态/完整 MoE 配方 |
| [Gemma 官方 PyTorch](https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py) | `gemma`：Local/Global 交替、QK-Norm、gated GELU | 无官方 ratio、soft capping、额外 norm 和完整局部缓存策略 |
| [Qwen3-Next 官方模型卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | `qwen`：QK-Norm GQA；`qwen_hybrid`：3 个 Gated Delta + 1 个 Attention | 缺少短卷积、output gate、不同 QK/V heads、partial RoPE 和官方 MoE/MTP 配方 |
| [Kimi K2 官方库](https://github.com/MoonshotAI/Kimi-K2) | `kimi`：MLA + routed/shared MoE | 小尺寸，无官方 optimizer/训练/权重 |
| [Kimi Linear 官方库](https://github.com/MoonshotAI/Kimi-Linear) | `kimi_linear`：递归 + softmax hybrid 的实验框架 | 标量 decay 的 Gated Delta 不是 KDA 的细粒度门控 |
| [GPT-OSS 官方 PyTorch](https://github.com/openai/gpt-oss/blob/7b583341fe16729127f6d5b94a7b09ccae97e1a1/gpt_oss/torch/model.py) | `gpt_oss`：局部/完整 Attention、MoE 思想组合 | 未复刻 attention sink、特殊 gated 激活、量化和全部 layer 配方 |

命名配方是可运行的学习起点，不是加载官方权重的 adapter。需要精确复现时，应另写模型专属 config 映射、分词器、参数转换和模型级数值核对；本项目先验证共同原语。

## 数学与实现参照

| 内容 | 一手来源 | 研究重点 |
| --- | --- | --- |
| RoPE | [RoFormer](https://arxiv.org/abs/2104.09864) | 相对 dot product、配对坐标和旋转 |
| GQA | [GQA 论文](https://arxiv.org/abs/2305.13245) | Query/KV head 数与 checkpoint 转换 |
| Scaling | [YaRN](https://arxiv.org/abs/2309.00071) | 分频插值和温度；不是仅延长 tensor |
| Delta/Gated Delta | [Gated Delta Networks](https://arxiv.org/abs/2412.06464) | 遗忘、误差纠正与 chunkwise 训练 |
| MTP | [Multi-token Prediction](https://arxiv.org/abs/2404.19737)、[DeepSeek V3](https://arxiv.org/abs/2412.19437) | 独立未来 heads 与顺序 future-conditioned heads 的区别 |
| 高性能 recurrent | [Flash Linear Attention 官方库](https://github.com/fla-org/flash-linear-attention) | 并行 prefix/chunk 算法与 GPU kernel |
| Flash Attention | [官方实现](https://github.com/Dao-AILab/flash-attention) | 精确 dense softmax 的 IO 优化，不是 sparse mask |
| MoE GPU | [DeepSeek DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[DeepEP](https://github.com/deepseek-ai/DeepEP) | grouped GEMM 与 dispatch/combine 通信的不同职责 |

源码与文档中的公式按本项目实际矩阵布局独立推导。官方规模、质量与速度结论不作为本项目验收结果。

## 残差结构的后续实验

目前有 standard/gated residual。Hyper-Connection 实验可以维护 $X\in\mathbb R^{B\times T\times n\times D}$：

$$
u=H_{pre}X,\qquad X'=H_{res}X+H_{post}F(u).
$$

mHC 将 $H_{res}$ 约束为近似双随机矩阵，核心是对非负矩阵交替归一化行/列的 Sinkhorn 迭代；普通 scalar residual gate 不具有这条多流数据路径。阅读 V4 `Block.hc_pre/hc_post` 和 kernel 的 `hc_split_sinkhorn` 时，应分别检查 pre/post/residual 的 Shape 与约束。本轮保留这个明确的扩展点，没有为未实现的 Hyper-Connection 注册空类或误导性配置。
