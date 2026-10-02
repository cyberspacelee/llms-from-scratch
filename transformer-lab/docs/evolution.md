# 从最小 Transformer 到现代架构的逐步演进

每一步先说明前一步的限制，再指出本次改变的机制，最后运行小型核对。只改变代码结构时要求输出和梯度一致；改变模型结构时验证因果性、缓存、参数和 Shape，不要求两种模型的随机 logits 相同。

从 [第 00 步 MVP](mvp.md) 开始。后续入口位于 [tutorials/evolution.py](../src/transformer_lab/tutorials/evolution.py)：`classic_config` 定义起点，`lesson_config` 显式写出每一步的配置变化，`run_lesson` 选择示例。

```bash
cd transformer-lab
uv run --locked transformer-lab lesson --step 00
uv run --locked transformer-lab lesson --step 01
# 把步号替换为 02 ... 13，每次只运行一个步骤。
uv run --locked transformer-lab lesson --step 07 --output runs/lesson07.json
```

所有示例是小尺寸 CPU 功能检查；GPU 环境可以使用 `--device cuda`，覆盖 CPU wheel 后使用 `uv run --no-sync`。本路线没有默认 benchmark、完整训练或模型下载。

## 路线总览

| 步号 | 本次学习 | 核对依据 |
| --- | --- | --- |
| 00 | 固定单头、单层 Encoder–Decoder | Shape、因果性、一次 backward/optimizer step |
| 01 | 从 MVP 到配置模型，多头、多层 | 同权重输出/梯度；多头模型 cached/完整前向 |
| 02 | Encoder-only / Decoder-only / Encoder–Decoder | 双向与因果作用范围；Cross memory |
| 03 | Sinusoidal → RoPE | Q/K 位置；Decoder、Encoder/Cross 缓存 |
| 04 | 完整重算 → KV Cache | prefill/decode/chunk 一致；greedy 一致 |
| 05 | Post/Pre-Norm、RMSNorm、QK-Norm、SwiGLU | 分支顺序、逐头归一化、门控 FFN |
| 06 | MHA → MQA → GQA | H/G、参数账本、真实 KV bytes |
| 07 | MLA、Decoupled RoPE、Matrix Absorption | 相同 latent 前缀下 naive/absorbed decode 一致 |
| 08 | Scaling、Local/Global、Sparse、Sequence Compression | 可见矩阵、gather 对照、摘要无未来泄漏 |
| 09 | Dense → Routed/Shared MoE | 分派次数 N*K、router 梯度、总/激活参数 |
| 10 | NTP → 顺序 MTP | future embedding 与标签偏移；多深度梯度 |
| 11 | Linear → Delta → Gated Delta → Hybrid | 固定状态、chunk 一致、混合层；RoPE/NoPE 分支 |
| 12 | 因果 Encoder–Decoder、非对称计算 | Encoder append、memory 复用、Cross 因果性 |
| 13 | SDPA、Paged/Prefix/Quantized、Speculative | 同权重对照、分叉不污染、量化误差、greedy 一致 |

00–07 建立基础 Attention 主线，09–10 继续研究 MLA + MoE + MTP；08 是长上下文分支，11 是递归混合分支，12 是 Encoder–Decoder 分支，13 是推理系统分支。不存在要求所有方案依次叠加的单一“最佳架构”。

## 00：先跑通完整 MVP

只有 `MinimalTransformer(source_ids, decoder_ids) -> logits`。先读 Attention 的两个 matmul，然后读 Encoder 两条分支、Decoder 三条分支和 Teacher Forcing。

默认 `[2,3]` source、`[2,4]` decoder 输入得到 `[2,4,32]` logits；22,848 个参数。`demo` 检查有限梯度、参数更新和未来 decoder 输入不影响早先输出。

本步没有缓存或架构开关。完整公式、Shape、FLOPs 在 [MVP 文档](mvp.md)。

## 01：先重组，再增加多头与多层

固定类便于看清数据流，但难以反复组合实验。先把相同模型表示为 `Transformer(classic_config())`，用 `lift_mvp` 逐项复制权重。相同 source、decoder 输入的输出及关键梯度必须一致。

然后改变：

```python
config = classic_config(heads=4, layers=2)
```

原来 Q/K/V 是 `[B,1,T,32]`，现在是 `[B,4,T,8]`。`H*d=D` 保持不变；MHA 的投影参数仍为 `4D²`，但不同 head 在各自子空间计算不同的 attention map。增加层数增加独立参数，不共享层权重。

阅读：[softmax.py](../src/transformer_lab/attention/softmax.py) 的 reshape/transpose/output，以及 [blocks.py](../src/transformer_lab/models/blocks.py) 的残差顺序。不要把 reshape 成多个 head 理解为增加四倍隐藏宽度。

## 02：拆出三类模型

继续沿用经典 Post-LayerNorm + ReLU + Sinusoidal，只改变：

```python
config = replace(config, architecture="encoder")
config = replace(config, architecture="decoder")
config = replace(config, architecture="encoder_decoder")
```

Encoder-only 的所有位置可读取完整输入，改变尾部可能改变开头；Decoder-only 每个位置只能读取已有前缀，适合 next-token；Encoder–Decoder 通过 Cross 从完整输入条件化输出。三者都可返回 `[B,T,V]` 以便统一实验，Encoder 的词表 head 是教学选择，不代表双向 Encoder 使用自回归训练目标。

本步命令依次检查三类模型。双向 Encoder 不支持追加 KV，因为新增输入会改变旧位置 hidden；其输出可以整体作为 `EncoderMemory` 复用。

阅读：[transformer.py](../src/transformer_lab/models/transformer.py) 的 `encode` / `forward`。后续主线选择 Decoder-only，Encoder–Decoder 在 04、12 步继续研究。

## 03：从输入绝对位置到 Q/K 的 RoPE

Sinusoidal 把位置加到输入 hidden；RoPE 直接旋转每个 head 的 Q/K：

$$q_p'=R(p)q_p,\quad k_s'=R(s)k_s,\quad q_p'^\top k_s'=q_p^\top R(s-p)k_s$$

```python
attention = replace(attention, position=PositionConfig(kind="rope"))
```

Shape 不变，没有可学习参数；计算为 O(BHTd)，不改变 KV 的元素数量。V 不旋转。Encoder 可用 RoPE；Cross 也可给 query/output 与 key/input 各自的位置，但其相对位置语义要按任务定义，不是所有 Encoder–Decoder 都应使用相同 Cross RoPE。

本步同时检查 Decoder-only 和 RoPE Encoder–Decoder；后者覆盖 Encoder、Decoder、Cross 三种位置使用。范数与共同平移的独立不变量见测试。

阅读：[position.py](../src/transformer_lab/attention/position.py)，详细公式与 Scaling 在 [principles.md](principles.md)。

## 04：分开 Prefill 和 Decode

重算整个前缀会重复投影旧 K/V，并重复计算旧位置输出。缓存保存每层历史投影，只计算新 query：

```python
memory = model.encode(source_ids)
prefill = model(prompt, memory=memory, use_cache=True)
decode = model(new_ids, cache=prefill.cache, use_cache=True)
```

| 阶段 | Query 长度 | 可见 Key 长度 | 主要操作 |
| --- | --- | --- | --- |
| 训练 | 完整 T | 完整 T/source S | 并行 causal forward、反向 |
| Prefill | 前缀 P | 前缀 P/source S | 首次投影、创建 Self/Cross 状态 |
| Decode | 通常 1，也支持 chunk | 已有 P + 本次 T/source S | Self append；Cross 重用静态 KV |

RoPE offset 来自完整缓存长度；causal 使用 `key_position <= query_position`，不能对单行 query 套局部 `tril`。本次 `valid[B,T]` 由模型与历史拼接。

示例用 `(3,1,3)` 分块结果与完整前向比较，并比较 cached/重算的 greedy token。经典 MHA 的 KV 字节为 `2*B*H*S*d*bytes`；增加 Cross cache 需要每个 Decoder 层存一份 source 投影。

学习版 `append` 用 cat 分配新状态，方便共享前缀；它并未消除分配与拷贝成本。阅读：[state.py](../src/transformer_lab/cache/state.py) 和 [推理分析](inference.md)。

## 05：现代 Block 的归一化和 FFN

本步在 RoPE MHA Decoder 上改变三个相互独立的轴，研究时也可一次只改一个：

```python
block = replace(block, norm_order="pre", norm="rms", activation="swiglu")
attention = replace(attention, qk_norm=True)
```

Post-Norm 是 `Norm(x+F(x))`，Pre-Norm 是 `x+F(Norm(x))`；Pre-Norm 模型在堆栈末尾额外归一化。LayerNorm 减均值并除标准差，参数 `2D`；RMSNorm 只除均方根，参数 `D`。

QK-Norm 在 split-head 后、RoPE 前沿 head_dim 归一化 Q/K。参考实现按所有 head 共享一组 d 维 scale，增加 `2d` 参数；它不等同于对整段 hidden 做 RMSNorm。

SwiGLU 为 `SiLU(xW_gateᵀ)*(xW_upᵀ)` 再下投影；从两张变成三张矩阵。固定 FFN hidden_dim 时参数由 `2DF` 变为 `3DF`；若需要比较相同参数预算，可把 gated hidden_dim 约缩到普通 FFN 的 2/3。

阅读：[normalization.py](../src/transformer_lab/layers/normalization.py)、[feedforward.py](../src/transformer_lab/layers/feedforward.py)、[blocks.py](../src/transformer_lab/models/blocks.py)。ReLU/GELU/GLU/GeGLU、门控残差也有配置支持；完整功能测试覆盖这些组合。

## 06：MHA → MQA → GQA

保持 `H=4,d=8,D=32`，只改变 KV head 数 G：

| 结构 | G | 每 token K/V 元素 | 投影参数，不含 QK-Norm |
| --- | --- | --- | --- |
| MHA | 4 | `2*4*8=64` | `2D(H+G)d=4096` |
| MQA | 1 | `2*1*8=16` | 2560 |
| GQA | 2 | `2*2*8=32` | 3072 |

Q 仍有 H 个 head；每组 query head 共用对应 KV。缓存保存 G 个头，不保存临时重复后的 H 个头。完整 attention mixing 的计算仍按 H 个 query head 进行，减少 KV 不会自动消除 Q*S 分数计算。

本步输出三个结构的 H/G、理论成本、实际缓存字节与各自的 cached/完整前向核对。报告 `cost_fp32_S1024` 是单层、batch=1、query=1、key=1024、每元素 4 字节；它和 fp64 双 batch 的模型检查具有不同口径。

阅读：`MultiHeadAttention` 与 [analysis.py](../src/transformer_lab/analysis.py)。

## 07：从减少 KV Head 到压缩 KV Latent

GQA 仍保存每组展开后的 K/V。MLA 改存小维度内容 latent C 与共享 RoPE key：

$$C=RMSNorm(XW_{DKV}^\top),\quad K^c=CW_{UK}^\top,\quad V=CW_{UV}^\top$$

$$K^r=RoPE(XW_{KR}^\top),\quad scores=Q^cK^{c\top}+Q^rK^{r\top}$$

本例 `L=8,r=4,H=4`，缓存每 token 为 `L+r=12` 个元素，Shape 分别是 `[B,1,S,8]` 和 `[B,1,S,4]`。Query 先压缩到 q_rank，再上投影拆成内容与位置两部分。

矩阵吸收利用：

$$Q^c(CW_{UK}^\top)^\top=(Q^cW_{UK})C^\top$$

$$P(CW_{UV}^\top)W_O^\top=(PC)(W_{UV}^\top W_O^\top)$$

因此 decode 不必重建所有历史 expanded K/V。位置部分单独旋转，使内容投影可以吸收；这是 Decoupled RoPE 的作用。

示例创建相同权重的 naive/absorbed 模块，以同一 latent prefix 检查新 token 输出一致。现有测试还覆盖训练梯度和 Cross cache。吸收后的 value/output 映射每次从当前权重重新计算，账本包含这项开销；只有实际硬件测试才能比较最终延迟。

阅读：[mla.py](../src/transformer_lab/attention/mla.py)；逐个投影的完整参数公式见 [principles.md](principles.md)。

## 08：长上下文的几种独立变化

RoPE Scaling 改变位置频率，不减少 token 数、KV 或 attention scores；窗口和稀疏改变可见关系；序列压缩合并 token；MLA 压缩每个 token 的通道。要分别测它们改变哪个轴。

本步打印 Global、Sliding、Local、Block Sparse、Token Sparse 的 6×6 因果矩阵。Sliding 读取近期窗口；Local 固定分块；稀疏示例加入周期 global key。`layer_blocks` 可交替 Local/Global。

仅用 dense mask 仍计算/分配 `[B,H,Q,S]`。`gathered_attention` 先按 `indices[Q,M]` 选择 key/value，再计算 `[B,H,Q,M]` 分数；示例与等价 dense mask 的输出对照。重复真实 key 会改变 softmax 分母，因此会报错；被 masked 的 filler 重复允许存在。

序列压缩示例保留精确近期窗口、完整旧块的 mean 摘要及尚不能压缩的边界 token。使用绝对块结束位置，只选结束于窗口之前的摘要，防止 future token 被池化进旧 key。它是近似模型，**不要求与原始 Full Attention 相等**；示例检查改变未来 V 不影响早期输出。

Linear/NTK/YaRN Scaling 都用 factor=4 检查缓存一致；长上下文泛化需要另做训练/评测，本检查不能证明有效上下文变长。

阅读：[patterns.py](../src/transformer_lab/attention/patterns.py)、[compression.py](../src/transformer_lab/inference/compression.py) 和 [长上下文成本](inference.md)。本步没有把窗口旧 KV 从模型 cache 中驱逐，也没有实现 DeepSeek 的 learned compressor/indexer。

## 09：Dense FFN → Mixture-of-Experts

回到 MLA 主线，把每层单个 FFN 换成 4 个 routed experts，Top-K=2，再增加一个 shared expert：

```python
block = replace(block, experts=4, top_k=2, shared_experts=1, router_score="sigmoid", balance="bias")
```

有效 token `[N,D]` 经 router 得 `[N,E]` 分数，选择 `[N,K]` IDs；每个专家只运行分给自己的 token，输出按组合权重 index_add 回原 token 行。Shared Expert 对每个有效 token 始终计算。padding 不进入分派，counts 总和必须为 `N*K`。

若单个专家参数为 P，则总参数 `DE+(E+shared)P`，每 token 激活参数 `DE+(K+shared)P`。Router 仍处理全部专家分数；总参数节省不等于实际速度同比提升，dispatch 小 batch 可能增加开销。

辅助 loss 与 selection bias 是两条负载均衡路径。本例偏置只影响选谁，不改变组合权重；汇总 counts 后在 optimizer.step 之后更新。推理不会更新 bias。

阅读：[moe.py](../src/transformer_lab/layers/moe.py)。本步核对 router 梯度和分派，训练示例中的专家 Python 循环与高性能 grouped GEMM 分别承担原理和性能任务。

## 10：NTP → Multi-Token Prediction

标准 NTP：位置 t 的 logits 预测 `x(t+1)`。第一个额外 MTP 深度把主 hidden 与真值 `x(t+1)` 的 embedding 归一化并融合，经过独立 causal Block，预测 `x(t+2)`；更深一步预测 `x(t+3)`。

```python
config = replace(config, mtp_depth=2)
total, terms, routing = language_model_loss(model, output, tokens)
total.backward()
```

长度 6 的例子：NTP 使用 5 个标签，额外深度分别输出 `[B,4,V]`、`[B,3,V]`。短序列必须丢掉没有未来标签的位置；mask 覆盖从当前位置到标签的全部有效区间。

总损失是 `NTP + 0.3*mean(MTP_depth_losses) + 0.01*MoE_aux`。本步显示每个深度的 labels/Shape 并验证梯度；未来泄漏的不变量由测试进一步核对。

MTP 增加训练参数和计算；普通 greedy generate 只调用主模型，不会自动获得多 token 加速。MTP 训练使用真值 future embedding，推理时只能使用候选 token。第 13 步的 draft/target 示例独立验证接受算法，尚未接入 MTP 草稿 head。

阅读：[mtp.py](../src/transformer_lab/training/mtp.py) 和 [losses.py](../src/transformer_lab/training/losses.py)。

## 11：固定状态的递归与混合架构

这是从 GQA/RMSNorm/SwiGLU Decoder 分出的路线，暂不叠加 MLA/MoE。换掉 Self-Attention，使历史状态不随 S 增长：

$$Linear:\quad S_t=S_{t-1}+\phi(k_t)v_t^\top,\quad z_t=z_{t-1}+\phi(k_t)$$
$$y_t=\phi(q_t)^\top S_t/\max(\phi(q_t)^\top z_t,10^{-6})$$

$$Delta:\quad S_t=S_{t-1}+\beta_t k_t(v_t-k_t^\top S_{t-1})^\top$$

Gated Delta 先将 `S` 乘以 `alpha_t`，然后按残差更新。Delta 的 Q/K 单位归一化；本实现 alpha/beta 是每 head、每 token 的标量。状态 `[B,H,d,v]`，Linear 另存 z；softmax KV 则是 `[B,G,S,d]`。

本步检查 S=16 和 S=4096 的递归 state bytes 相同，并运行 `gated_delta → GQA` 的两层混合。每层使用自己的缓存类型，模型级 offset 保持一致。训练需保存反向过程，推理固定 state 不意味着训练内存固定。

还运行 `RoPE → NoPE` 交替层，帮助研究 iRoPE 类设计；这表示整层的位置机制不同，而不是简单旋转一半维度。具体主流模型的配置周期与其他改动见 [官方资料](research.md)。

阅读：[recurrent.py](../src/transformer_lab/attention/recurrent.py)。当前是顺序原理版本，完整 KDA、卷积前处理、chunkwise/fused kernel 均需进一步实现。

## 12：因果 Encoder–Decoder 与输入/输出非对称

返回 Encoder–Decoder，把 Encoder 改成 causal，并设置 1 个输入层、2 个输出层；Cross 也设置按输入/输出绝对位置对齐的因果约束：

```python
config = replace(
    config, architecture="encoder_decoder", encoder_causal=True, cross_causal=True, encoder_layers=1
)
```

双向 Encoder 新增 source 会影响旧 hidden，不能 append；因果 Encoder 可以把 source 分块编码、追加层状态，拼回完整 memory。本步比较一次编码与两段编码，再比较直接 source_ids 与准备好的 memory 两条模型路径。

输入计算可以多次用于不同输出 prompt，Encoder memory 与每层静态 Cross KV 分别复用。多模态/长输入可把输入侧表示、长度和计算预算与输出侧分开研究；当前共享 hidden_dim 和词表，尚不包含图像/音频编码器。

`cross_causal=True` 假设位置 p 的输出只允许读取输入 0..p，适合特定对齐/流式实验。它不是通用 seq2seq 的默认语义。DeepSeek V4.1 官方代码中的 CausalEncoderDecoder 是跨深度共享 source-layer 压缩 KV 的设计，**本例不是其完整复现**，对应区别见 [research.md](research.md)。

阅读：[EncoderMemory/encode/forward](../src/transformer_lab/models/transformer.py)。

## 13：在原理正确之后对比高性能路径

本步仍不计时。先给 manual 与 PyTorch SDPA 加载相同权重，比较同一 causal forward 输出。SDPA 由 PyTorch 根据设备/dtype/Shape 选择实现，不能仅凭 API 名称认定用了 FlashAttention。

分页存储把序列拆成 pages；prefix fork 共享前缀，对部分页采用 copy-on-write。示例核对 append 后两分支长度不同、完整页仍共享、还原 K 与原始 K 一致。当前 materialize 会拼成连续 Tensor，因此还没有生产级 paged kernel 的收益。

量化使用 per-vector `scale=max(abs(x))/127`，误差界 `scale/2`；报告同时统计 int8 codes 与 scales 的字节。计算 Attention 前仍需反量化，不等同于直接使用 int8 KV 的 fused kernel。

Greedy draft/target 由 draft 提议块，target 一次 forward 验证，遇到首个不匹配提交 target 修正，后续 proposal 丢弃；全接受可以追加 bonus token。示例与 target greedy generate 比较。随机采样需要 p/q 接受算法，KV 回滚、MTP 草稿以及系统调度均未集成。

阅读：[storage.py](../src/transformer_lab/cache/storage.py)、[speculative.py](../src/transformer_lab/inference/speculative.py)。FlashAttention、DeepGEMM、DeepEP 等官方项目与后续对比入口在 [research.md](research.md)；只有用户主动运行 `benchmark` 才会计时。

## 如何继续实验

先从 `lesson_config` 导出配置，用 `dataclasses.replace` 一次改一个机制，使用 `transformer-lab check` 或对应的不变量。理论成本使用 `analysis.attention_cost` / `ffn_cost`，注明 batch、query/key 长度、dtype 和计算范围。训练与性能研究再使用独立数据、预算和硬件，不能把小样例的功能通过写成模型效果或吞吐结论。

每个模块的完整公式、Shape、参数与复杂度查 [principles.md](principles.md)；不同阶段的内存、带宽和计算查 [inference.md](inference.md)；实现覆盖和研究边界查 [coverage.md](coverage.md)。
