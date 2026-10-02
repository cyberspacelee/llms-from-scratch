# 一手来源索引

核对日期2026-10-02。每章页底链接原论文/公开代码；详细版本与revision见[2026模型对照](models-2026.md)。课程从原理独立推导，默认不引入官方kernel或模型依赖。

| 路线 | 一手资料 | 阅读重点 |
| --- | --- | --- |
| 基础结构 | [Transformer](https://arxiv.org/abs/1706.03762)、[BERT](https://arxiv.org/abs/1810.04805) | seq2seq、自回归与双向监督 |
| KV头优化 | [MQA](https://arxiv.org/abs/1911.02150)、[GQA](https://arxiv.org/abs/2305.13245) | 头映射、decode存储与bandwidth |
| Norm/FFN | [RMSNorm](https://arxiv.org/abs/1910.07467)、[Pre-LN](https://arxiv.org/abs/2002.04745)、[GLU](https://arxiv.org/abs/2002.05202) | 稳定性与参数预算 |
| 位置/长度 | [RoFormer](https://arxiv.org/abs/2104.09864)、[YaRN](https://arxiv.org/abs/2309.00071) | 旋转相对位置、频段插值与训练范围 |
| MoE | [Switch](https://arxiv.org/abs/2101.03961)、[DeepSeekMoE](https://arxiv.org/abs/2401.06066) | routed/shared experts、细粒度专家与负载 |
| 高效Attention | [Linear Transformers](https://arxiv.org/abs/2006.16236)、[FlashAttention](https://arxiv.org/abs/2205.14135) | 特征kernel/递归与精确softmax IO优化 |
| MLA/MTP | [DeepSeek V2](https://arxiv.org/abs/2405.04434)、[V3](https://arxiv.org/abs/2412.19437) | latent/解耦RoPE/吸收、future-conditioned顺序MTP |
| Sparse/压缩 | [DeepSeek V4报告](https://arxiv.org/abs/2606.19348)、[V4.1报告](https://arxiv.org/abs/2609.19969) | window、CSA/HCA、hierarchical选择和共享runtime |
| Hybrid | [DeltaNet](https://arxiv.org/abs/2406.06484)、[Gated DeltaNet](https://arxiv.org/abs/2412.06464)、[Kimi Linear](https://arxiv.org/abs/2510.26692) | delta correction、scalar与channel-wise decay |
| Qwen hybrid/扩展 | [Qwen3.8 Next](https://arxiv.org/abs/2608.30320)、[Qwen4-Exp实现](https://huggingface.co/docs/transformers/en/model_doc/qwen4_exp) | 短卷积、gated Attention、QSA/GR/PLE |
| 残差新结构 | [mHC](https://arxiv.org/abs/2512.24880)、[AttnRes官方](https://github.com/MoonshotAI/Attention-Residuals) | 多stream约束与depth source选择的区别 |
| 条件记忆 | [Engram](https://arxiv.org/abs/2601.07372) | hashed n-gram、lookup、context gate |
| 存储/推理 | [PagedAttention](https://arxiv.org/abs/2309.06180)、[KV量化](https://arxiv.org/abs/2402.02750)、[Speculation](https://arxiv.org/abs/2211.17192) | 页所有权、量化误差、接受/拒绝 |
| 官方算子 | [FlashAttention](https://github.com/Dao-AILab/flash-attention)、[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)、[FlashKDA](https://github.com/MoonshotAI/FlashKDA)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[DeepEP](https://github.com/deepseek-ai/DeepEP) | 精确实现、并行递归、GEMM与通信各自职责 |

最新模型按公开结构而非榜单分组：DeepSeek和Qwen为主要路线，另对照Kimi K3、GLM-5.3-Flash、Gemma 4及已有Llama/gpt-oss家族。未公开的闭源内部结构不据产品名字推测。可运行原语、完整core路径和仅有结构阅读的区别均在模型对照表中写明。
