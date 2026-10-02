# 数学、Tensor Shape 与源码

本文公式对应本项目源码；不是某个官方 checkpoint 的逐参数复刻。核心算子使用基础 PyTorch，`sdpa` 分支只用于实现对照。共同符号：batch 为 $B$，模型宽度 $D$，query 长度 $Q$，可见 key 长度 $S$，query heads 为 $H$，KV heads 为 $G$，单头 Q/K 宽度 $d$，V 宽度 $v$，FFN 中间宽度 $F$。忽略 bias 时，一层 Linear 的参数为 $d_{in}d_{out}$，matmul FLOPs 为 $2N d_{in}d_{out}$。

## 1. Scaled Dot-Product Attention

$$
P=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}+M\right),\qquad O=PV.
$$

`scaled_dot_product_attention` 在 [attention.py](../src/transformer_lab/attention.py) 中实现。矩阵形状为 `Q[B,H,Q,d]`、`K[B,H,S,d]`、`V[B,H,S,v]`、`scores/P[B,H,Q,S]`、`O[B,H,Q,v]`。mask True 为可见，转换为 $0/-\infty$；全遮挡行先替换为有限分数，再乘零，避免 NaN 和 NaN 梯度。fp16/bf16 的 softmax 统计升为 fp32，fp64 保持 fp64。

此原语无可训练参数。QK 和 PV 的 matmul FLOPs 合计 $2BHQ S(d+v)$；显式 scores 的元素数为 $BHQ S$。训练保留计算图并允许 Attention dropout；prefill 的 $Q$ 可以很大；decode 的 $Q=1$，仍读取历史 $K/V$。dropout 后的权重不再是逐行和严格为 1 的概率。

## 2. MHA → MQA → GQA

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\quad
Y=\operatorname{Concat}(O_1,\ldots,O_H)W_O.
$$

`MultiHeadAttention` 把投影 `[B,Q,H*d]` reshape 为 `[B,H,Q,d]`。MHA 取 $G=H$，MQA 取 $G=1$，GQA 取 $1<G<H$ 且 $H\bmod G=0$；query head $h$ 使用 KV group $\lfloor h/(H/G)\rfloor$。源码为便于阅读，在计算时 `repeat_interleave` KV；持久缓存仍只有 $G$ 个头。

| 项目 | 一般 MHA/MQA/GQA，$v=d$ |
| --- | --- |
| 参数 | $2D(H+G)d$；QK-Norm 再加 $2d$ |
| Query / Output 投影 FLOPs | $4BQDHd$ |
| 新增 $N$ 个 KV token 的投影 FLOPs | $4BNDGd$ |
| Attention mixing FLOPs | $4BHQ S d$ |
| 每层 KV 元素 | $2BG S d$ |

减少 $G$ 主要减少 KV 投影、持久缓存和理想读取量；相同 $H$ 下，显式 QK/PV 计算没有随 $G$ 同比减少。本项目的临时 KV 展开还会抬高实际峰值内存。

训练一次处理全部 token；prefill 同时输出全部 query 并建立 cache；decode 投影新增 token，追加其 K/V，再读旧缓存。三条路径调用同一个 `forward`。

## 3. Encoder、Decoder、Cross-Attention

模型组装位于 [model.py](../src/transformer_lab/model.py)。

| 结构 | Query 来源 | Key/Value 来源 | 可见性 |
| --- | --- | --- | --- |
| Encoder Self-Attention | Encoder hidden | 同层 Encoder hidden | 通常双向 |
| Decoder Self-Attention | Decoder hidden | 同层 Decoder hidden | $k\le q$ |
| Cross-Attention | Decoder hidden | Encoder 最终 hidden | 通常全源可见；可选时间线对齐的 $k\le q$ |

经典 Encoder：Self-Attention → residual/norm → FFN → residual/norm。经典 Decoder：Causal Self-Attention → residual/norm → Cross-Attention → residual/norm → FFN → residual/norm。Encoder-only / Decoder-only 都省去 Cross-Attention；Encoder–Decoder 使用独立的两个 stack。

经典输入为 $\sqrt D\,\mathrm{Embedding}(ids)+PE$。输入/输出 embedding 可共享，词表参数计一次 $VD$；关闭共享时计两次。Encoder 与 Decoder 可设不同层数、FFN、KV heads 和 position config，但当前共享 vocabulary 和 hidden width $D$。不同宽度或多模态接口可以先投影到 `EncoderMemory.hidden[B,S,D]`。

Cross-Attention 的 query 长度和 source 长度不同；其 FLOPs 是 $O(BHQSd)$。prefill 计算 source K/V 一次；decode 仅重算 Query，不 append source K/V。缓存保存 Encoder hidden、source valid，以及每层不同投影得到的 Cross K/V。跨层投影参数不同，不能直接用同一组 Cross K/V。

双向 Encoder 输出可以重复使用；不能追加后假设旧 hidden 不变。因果 Encoder 可用 `encode(..., past=..., use_cache=True)` 追加，与完整 Encoder 前向相等。当前 decoder 会话绑定固定 source；扩展 source 后应开新 decoder 会话，动态扩展 Cross cache 尚未接入。

Teacher forcing 在 [objectives.py](../src/transformer_lab/objectives.py) 中：targets 为 `[y0,y1,...,EOS]`，输入为 `[BOS,y0,y1,...]`。因果 mask 确保并行训练不会看到当前位置的目标。自回归生成只给已生成前缀，逐步选择下一个 token；`generate` 用 greedy argmax，便于对比缓存与完整重算。

## 4. Absolute PE、RoPE 和 Scaling

[position.py](../src/transformer_lab/position.py) 实现相邻坐标配对。Sinusoidal：

$$
PE(p,2i)=\sin(p\,10000^{-2i/D}),\quad
PE(p,2i+1)=\cos(p\,10000^{-2i/D}).
$$

`positions[T] → phase[T,ceil(D/2)] → PE[T,D] → X[B,T,D]`，参数量为零，计算/临时内存为 $O(TD)$。支持奇数模型宽度，最后一个 cosine 截掉。训练和 prefill 从位置零开始，decode 从 cache.length 开始，不能每个 chunk 重置位置。

RoPE 在 Attention 投影后作用于 Q/K，V 不旋转：

$$
R(p,\omega)\begin{bmatrix}a\\b\end{bmatrix}=
\begin{bmatrix}a\cos(p\omega)-b\sin(p\omega)\\a\sin(p\omega)+b\cos(p\omega)\end{bmatrix},
\quad \omega_i=\theta^{-2i/d}.
$$

于是 $(R(p)q)^\top(R(s)k)=q^\top R(s-p)k$。Shape 保持 `[B,H,T,d]`，无参数，计算 $O(BHTd)$，只缓存已经旋转的 K。Cross-Attention 独立旋转 decoder 位置 $p$ 和 encoder 位置 $s$；这样的相对位置关系是一个实验选择，需要任务本身存在合适的坐标关系，不能自动当成所有 seq2seq 的默认最佳方案。

| Scaling | 本项目频率修改 | 注意事项 |
| --- | --- | --- |
| none | $\omega_i$ | 不变 |
| linear | $\omega_i/f$ | 相当于位置 $p/f$ |
| ntk | $\theta' = \theta f^{d/(d-2)}$ | 固定 base scaling，$d>2$；不是随长度更新的 dynamic NTK |
| yarn | $(1-r_i)\omega_i+r_i\omega_i/f$ | $r_i$ 按 correction range 线性插值，低频更多插值 |

`attention_factor` 对最终 logits 乘平方因子；默认 1，未自动选择某个模型的温度策略。可以显式设置常见实验值 `1 + 0.1*log(factor)`。整个缓存会话保持频率与温度固定；改变 Scaling 要重新 prefill。能算更长位置不代表模型已经具备长上下文能力，后者需要训练与任务评估。

`layer_blocks` 可在 RoPE 和 NoPE 间交替，见 `irope` 配方；它实现分层位置编码交替这个核心思想，不包含 Llama 4 的 temperature tuning、完整训练配方或多模态 MRoPE。RoPE 的“相邻维度交错”与“RoPE/NoPE 层交错”是两个概念。

## 5. MLA：特征维压缩与矩阵吸收

[attention.py](../src/transformer_lab/attention.py) 的 `MultiHeadLatentAttention` 用内容维 $c$（配置 `head_dim`）、rotary 维 $r$、KV latent rank $L$、query rank $L_q$ 和 value 维 $v$。

$$
C^{KV}=\mathrm{RMSNorm}(XW_{DKV}),\qquad
K_h^C=C^{KV}W_{UK,h},\quad V_h=C^{KV}W_{UV,h}.
$$

$$
C^Q=\mathrm{RMSNorm}(XW_{DQ}),\quad
[Q_h^C,Q_h^R]=C^QW_{UQ,h},\quad
K^R=R(s)(XW_{KR}),\quad Q_h^R\leftarrow R(p)Q_h^R.
$$

当 `q_rank=0`，Query 直接从 $X$ 投影。$K^R$ 是所有 heads 共享的独立位置支路。

| 数据 | Shape |
| --- | --- |
| 输入 $X$ | `[B,T,D]` |
| 规范化 KV latent | `[B,1,S,L]` |
| 共享 rotary key | `[B,1,S,r]` |
| 内容/位置 Query | `[B,H,Q,c]` / `[B,H,Q,r]` |
| 朴素展开 K/V | `[B,H,S,c]` / `[B,H,S,v]` |
| 吸收后 Query | `[B,H,Q,L]` |
| Attention probabilities | `[B,H,Q,S]` |
| latent context | `[B,H,Q,L]` |
| 输出 | `[B,Q,D]` |

分数使用 $\sqrt{c+r}$，不是吸收后的 $\sqrt L$：

$$
s_{h,p,s}=\frac{Q^C_{h,p}(K^C_{h,s})^\top+Q^R_{h,p}(K^R_s)^\top}{\sqrt{c+r}}.
$$

吸收 Key 上投影：$\widetilde Q_h=Q_h^C W_{UK,h}^\top$（此处矩阵按 PyTorch weight `[out,in]` 记），直接与 latent 点积。将 Value 上投影与 head 对应的输出投影合并：

$$
Z_h=P_h C^{KV},\quad
Y=\sum_h Z_h\underbrace{W_{UV,h}^\top W_{O,h}}_{W_{VO,h}\in\mathbb R^{L\times D}}.
$$

普通 RoPE 不可任意穿过内容上投影矩阵，$R(p)W\ne WR(p)$；把位置支路解耦后，内容投影才可以按上述代数移动。MLA 对 Cross-Attention 也使用这个分离定义；source latent 和 rotary key 可静态复用。

参数数目（包含本实现的 RMSNorm scale）：

$$
P_Q=\begin{cases}DH(c+r),& L_q=0,\\
DL_q+L_qH(c+r)+L_q,& L_q>0,
\end{cases}
$$

$$
P_{MLA}=P_Q+D(L+r)+L+LH(c+v)+HvD.
$$

两条路径共用参数和 latent cache，便于严格对照。`naive` 每次重建所有历史 K/V；`absorbed` 直接读取 latent。训练/prefill/decode 都可选择两种路径，训练梯度也应相等；prefill 未必选择 absorbed 更快。本实现每次从最新权重重算 $W_{VO}$，因此不会在 optimizer.step 后复用陈旧矩阵，但增加 $2HLvD$ 的吸收开销。推理专用的预吸收权重可在 `eval` 冻结之后另做优化。

缓存每层 $BS(L+r)$ 个元素，不随 $H$ 成比例增加。但如果 rank 设得过大，它也可能比 MQA/GQA cache 更大；“压缩”应和具体尺寸比较。更多 FLOPs 与 bandwidth 账本见 [inference.md](inference.md)。

## 6. Full、Sliding、Local、Sparse

[masks.py](../src/transformer_lab/masks.py) 以绝对位置计算可见性：

| pattern | 规则，最后还与 causal $k\le q$ 相交 |
| --- | --- |
| global | 所有有效 key；causal 后即完整因果注意力 |
| sliding | 因果时 $0\le q-k<W$；双向时 $|q-k|<W$ |
| local | $\lfloor q/W\rfloor=\lfloor k/W\rfloor$，分块局部 |
| block_sparse | 相邻 KV block 或周期性 global block |
| token_sparse | 近邻 token 或周期性 global token |

周期性 global key 是可被读的锚点，当前不是“global query 可以读所有 token”的双向 Longformer 规则。Local/Global 层交替使用 `layer_blocks`，不是在每个 query 上随机切换规则。

这些 mask 先验证 Attention pattern 的语义，默认路径仍计算 dense `[B,H,Q,S]`。真正少算分数的独立原语 `gathered_attention` 使用 `indices[Q,M]` 和 `valid[Q,M]`，只形成 `[B,H,Q,M]` 分数与 `[B,H,Q,M,d]` gather 临时量；代价为 $O(BHQ M(d+v))$，无参数。测试核对同一 selection 的 dense mask 和 gather 输出，不要求稀疏输出与 full Attention 相同。

序列压缩原语 `compressed_causal_attention` 保留最近窗口和未完成旧块，已完成旧块按均值汇聚。在 query $p$ 处仅可用 block end $\le p-W$ 的旧摘要，避免 pooling 未来 token。它减少历史 token 数，MLA 则减少特征维度；两者是不同压缩轴。本实现按需重算摘要并返回 Attention 输出，还没有持久化 compressed-cache 模型路径，也没有复刻 DeepSeek CSA/HCA 的 learned gate/indexer。

## 7. Block、Normalization 与 FFN

[layers.py](../src/transformer_lab/layers.py)：

$$
\mathrm{LN}(x)=\gamma\frac{x-\mu}{\sqrt{\operatorname{mean}[(x-\mu)^2]+\epsilon}}+\beta,
\qquad
\mathrm{RMS}(x)=\gamma\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}.
$$

Shape 均保持 `[B,T,D]`。LayerNorm 参数 $2D$，RMSNorm 参数 $D$，统计计算 $O(BTD)$，无 KV。QK-Norm 对每个 head 的 Q/K 最后一维做 RMSNorm，发生在 RoPE 之前；scale 跨 heads 共享，增量参数 $2d$。MLA 不应用展开后的 QK-Norm，改用 KV/Query latent norm，保持直接矩阵吸收条件。

Pre-Norm 为 $x'=x+F(\mathrm{Norm}(x))$；Post-Norm 为 $x'=\mathrm{Norm}(x+F(x))$。`TransformerBlock` 的每个 Attention/Cross/FFN 子层有独立 norm；Pre-Norm stack 最后再做 norm，Post-Norm 无额外 final norm。残差保持 token/hidden 维度。可选 `residual="gated"` 用一个 sigmoid scalar 乘各子层分支，增量参数为子层数；这是学习用门控残差，不是 mHC。

$$
\mathrm{FFN}(x)=\phi(xW_{up})W_{down},\quad
\mathrm{GLU}(x)=[\phi(xW_{gate})\odot(xW_{up})]W_{down}.
$$

非 gated 的 $\phi$ 是 ReLU/GELU，GLU/GeGLU/SwiGLU 的 gate 分别是 sigmoid/GELU/SiLU。中间张量 `[B,T,F]`，输出 `[B,T,D]`。普通 FFN 参数 $2DF$、matmul FLOPs $4BTDF$；gated FFN 参数 $3DF$、FLOPs $6BTDF$。相同 hidden $F$ 下 gated 的参数更多，公平实验可按参数预算减少 $F$。训练保留激活，prefill/decode 都逐 token 做同样的 FFN，无 KV cache。

真正 Hyper-Connection 需要 residual stream `[B,T,n,D]`、pre/post mixing 和 $n\times n$ 的 residual mixing；mHC 还需要 Sinkhorn 约束。应单独扩展 block 的残差流，不用一个 `x+gate*branch` 标识成 Hyper-Connection。该结构的实验方程和官方代码入口在 [research.md](research.md)。

## 8. MoE：Router、Dispatch、Combine 与 Balance

$$
p_{t,e}=\operatorname{softmax}(x_tW_R)_e\ \text{或}\ \sigma(x_tW_R)_e,
\quad I_t=\operatorname{TopK}(p_t+b),
$$

$$
w_{t,e}=\frac{p_{t,e}}{\sum_{j\in I_t}p_{t,j}},\qquad
y_t=\sum_{e\in I_t}w_{t,e}E_e(x_t)+\sum_{s\in shared}E_s(x_t).
$$

`MixtureOfExperts` 将 `[B,T,D]` flatten 成 `[N,D]`；router scores `[N,E]`；indices/weights `[N,K]`；每个专家只读分给它的 tokens；最后 `index_add` 合并同一 token 的多个专家输出。padding 不 dispatch。没有 capacity 截断，没有 token dropping。shared expert 总是激活。

若每个专家参数 $P_e$，router 参数 $DE$，shared 数 $E_s$：总参数 $DE+(E+E_s)P_e$；每 token 激活参数近似 $DE+(K+E_s)P_e$；matmul FLOPs 近似 $2N[DE+(K+E_s)P_e]$。计数不含 topk、dispatch/combine、激活/通信的费用，因此激活参数少不保证端到端 latency 更低。

Aux balance：$L_{bal}=E\sum_e f_e\bar p_e$，$f_e=count_e/(NK)$。`balance="aux"` 将它交给 loss；`"bias"` 用选择偏置更新 $b_e\leftarrow b_e+\eta\operatorname{sign}(\overline{count}-count_e)$ 并居中；`"none"` 不均衡。bias 只影响选择 indices，combine weights 仍取原始 $p$。为避免 repeated forward 改模型行为，更新显式发生在 optimizer.step 之后。

```python
output = model(tokens)
loss, terms, routing = language_model_loss(model, output, tokens)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
model.update_router_bias(routing)  # 包含主干及 MTP 的每个 MoE
```

训练 router 得到选中组合权重的梯度；Top-K 的离散 indices 不可微。prefill 对许多 token dispatch，decode 可能每 expert 只有极少 token，效率受 GEMM 粒度限制。当前循环是原理实现，没有 expert parallel all-to-all；多卡时应先汇总每个专家的 count，再更新 balance。

## 9. NTP 与顺序 MTP

NTP：$L_{NTP}=\sum_t\mathrm{CE}(head(h_t),x_{t+1})$，实现使用有效 token 的平均。MTP 第 $j$ 个额外深度：

$$
h_t^{(j)}=\mathrm{Block}_j\left(W_j[\mathrm{RMS}(h_t^{(j-1)});\mathrm{RMS}(Emb(x_{t+j}))]\right),
$$

$$
L_j=\mathrm{CE}(head(\mathrm{RMS}(h_t^{(j)})),x_{t+j+1}),
\quad L=L_{NTP}+\lambda_{MTP}\operatorname{mean}_j L_j+\lambda_{bal}L_{bal}.
$$

`MultiTokenPrediction` 和 `language_model_loss` 位于 [objectives.py](../src/transformer_lab/objectives.py)。基干 hidden `[B,T,D]`；depth 1 的 future embedding 为 `tokens[:,1:]`，MTP 状态长度 `T-1`，可用于监督的 logits 长度 `T-2`，labels 为 `tokens[:,2:]`。每加一个深度缩短一个 token；不同深度 mask 要覆盖从起点到目标的全部有效位置。短序列不会产生空监督的 CE/NaN。

每个深度有三套 RMSNorm（$3D$）、投影 `[2D,D]`（$2D^2$）和一个无 Cross 的 TransformerBlock；embedding/head 与主干共享，不额外计词表参数。新增 FLOPs 是各深度的投影、block、head，监督序列逐层变短。训练可能改善表示，但需实测任务质量。

标准 prefill/decode/`generate` 只走主干，MTP 不强行增加生成开销。MTP 的 teacher-forced 未来 embedding 不能直接拿来声称一次生成多个未知 token。要用于投机推理，需要自回归 draft 状态、接受/拒绝与缓存回滚。本项目已实现独立 draft/target 的 greedy 验证原语；MTP 专属 draft cache 和随机接受算法是后续扩展。

## 10. Linear、DeltaNet、Gated DeltaNet 与 Hybrid

[recurrent.py](../src/transformer_lab/recurrent.py) 使用状态 $S_t\in\mathbb R^{d\times v}$，实际 `[B,H,d,v]`。Linear Attention 用正特征 $\phi(x)=elu(x)+1$：

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,\quad z_t=z_{t-1}+\phi(k_t),
\quad y_t=\frac{\phi(q_t)^\top S_t}{\max(\phi(q_t)^\top z_t,\epsilon)}.
$$

DeltaNet 的 unit-norm Q/K 使用误差纠正：

$$
S_t=S_{t-1}+\beta_t k_t(v_t-k_t^\top S_{t-1})^\top,\quad y_t=q_t^\top S_t.
$$

Gated DeltaNet 先遗忘，再写入：

$$
\bar S_t=\alpha_t S_{t-1},\quad
S_t=\bar S_t+\beta_t k_t(v_t-k_t^\top\bar S_t)^\top,
\quad \alpha_t,\beta_t\in(0,1).
$$

当前 $\alpha/\beta$ 是每 token/head 的 scalar sigmoid；没有 Qwen 的短卷积、output gate，或 Kimi KDA 的细粒度通道门控。`delta` 只有 beta，`gated_delta` 增加 alpha。状态与 cache 大小不依赖历史长度；但随着 $d,v$ 增大，状态是 $dv$，并非免费。

每个 mixer 的 Q/K/V/O 参数 $2DH(d+v)$，delta 另加 $H(D+1)$，gated delta 再加一份。Linear 持久状态元素 $BH(dv+d)$，delta 为 $BHdv$。忽略投影后，每 token 递归成本 $O(BHdv)$。训练/prefill 当前顺序循环，保留完整计算图；decode 只一步，没有 `[Q,S]` 分数。生产中要做 chunkwise parallel recurrence，才能改善 prefill/训练吞吐。

Hybrid 用同一个 `TransformerBlock` 混排 recurrent 和 softmax mixer。递归状态善于用固定容量聚合历史，完整 Attention 保留显式 token 检索通道；二者组合是能力与成本的权衡。没有声称有限状态对所有长上下文任务都足够。
