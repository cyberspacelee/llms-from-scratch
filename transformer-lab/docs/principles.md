# 数学与 Tensor Shape

统一符号见[课程规划](plan.md)：B为batch，S为sequence，D为model dimension；Attention使用S_q、S_kv、H_q、H_kv、D_h/D_v，词表为V，FFN中间宽度为D_ff。PyTorch Linear的weight布局为`[out,in]`。各章都有输入输出表，源码docstring包含每个参数及返回值。

## Scaled Dot-Product Attention

$$
P=\operatorname{softmax}(QK^\top/\sqrt{D_h}+M),\qquad O=PV.
$$

[softmax.py](../src/transformer_lab/attention/softmax.py)：Q`[B,H_q,S_q,D_h]`，K`[B,H_q,S_kv,D_h]`，V`[B,H_q,S_kv,D_v]`，scores/P`[B,H_q,S_q,S_kv]`，context`[B,H_q,S_q,D_v]`。bool mask的True表示可见；全遮挡行返回零，避免NaN。半精度softmax统计升fp32，fp64保留。

MHA取H_kv=H_q，MQA取H_kv=1，GQA取1<H_kv<H_q且H_q可被H_kv整除。query head h对应KV group `h//(H_q//H_kv)`。源码为易读在计算时展开KV；持久cache仍为`[B,H_kv,S_kv,D_h]`。投影与合头宽度可以不同于D，Output projection负责回到D。

## 三类结构与监督

| 结构 | Hidden | 可见性与输出 |
| --- | --- | --- |
| Encoder-only | `[B,S,D]` | 双向Self；词表head为`[B,S,V]`，03章只监督选中的MASK |
| Decoder-only | `[B,S,D]` | 因果Self；NTP将logits`[:,:S-1]`对齐tokens`[:,1:S]` |
| Encoder–Decoder | Encoder`[B,S_kv,D]`，Decoder`[B,S_q,D]` | Decoder因果Self，再以Encoder hidden为Cross K/V |

01章直接写经典多头、单层、Sinusoidal、Post-LayerNorm、ReLU计算；02章拆出自回归路径；03章转为双向路径。核心模型支持多层组装。Teacher forcing将`[y0,...,EOS]`右移成`[BOS,y0,...]`，每个位置只看已知前缀。

普通Cross每层有自己的K/V投影，source cache静态；24章的共享联合KV让多个consumer使用同一张量，另有自己的Q/O投影。这是两条不同的数据流。

## 位置编码

$$
PE(p,2i)=\sin(p\,10000^{-2i/D}),\quad PE(p,2i+1)=\cos(p\,10000^{-2i/D}).
$$

[位置算子](../src/transformer_lab/attention/position.py)：positions`[S]`→phase`[S,ceil(D/2)]`→PE`[S,D]`，加到embedding`[B,S,D]`。奇数D截掉最后一个cosine。

RoPE旋转Q/K的成对坐标，V不旋转，shape保持`[B,H,S,D_h]`；频率为`theta**(-2i/D_h)`。性质为`<R(p)q,R(s)k>=<q,R(s-p)k>`，08章检查范数与共同位置平移。Linear Scaling除频率以factor；固定NTK调整base；YaRN按频段插值。12章的频率在一个cache会话中固定，不实现dynamic NTK。改变频率需要重新prefill。

## Normalization、FFN与MoE

$$
\mathrm{LN}(x)=\gamma(x-\mu)/\sqrt{\operatorname{mean}((x-\mu)^2)+\epsilon}+\beta,\quad
\mathrm{RMS}(x)=\gamma x/\sqrt{\operatorname{mean}(x^2)+\epsilon}.
$$

均保持`[...,D]`。Pre-Norm为`x+F(Norm(x))`，Post-Norm为`Norm(x+F(x))`。QK-Norm对每个head最后一维操作，在RoPE前完成。

FFN：`down(phi(up(x)))`；gated FFN：`down(phi(gate(x))*up(x))`。Hidden`[B,S,D]`→up/gate`[B,S,D_ff]`→output`[B,S,D]`。GLU/GeGLU/SwiGLU分别使用sigmoid/GELU/SiLU；同D_ff下gated比普通FFN多一个投影。

[MoE](../src/transformer_lab/layers/moe.py)将有效hidden展平为`[N,D]`，router得到`[N,E]`，Top-K的ids/weights为`[N,K]`；按expert gather各自token，再以index_add合并`[N,D]`并还原`[B,S,D]`。shared专家总是激活。selection bias只影响选择，combine用原评分归一化；bias更新显式放在optimizer.step之后。Top-K离散索引本身不可微；被选中权重有梯度。本库支持softmax/sigmoid评分，不把它们当作所有DeepSeek版本的同一router。

## MLA与顺序MTP

[MLA](../src/transformer_lab/attention/mla.py)：`X[B,S_kv,D]`→normalized joint latent`C[B,1,S_kv,L_kv]`，另存rotary key`Kr[B,1,S_kv,D_r]`。朴素路径展开content K`[B,H_q,S_kv,D_h]`、V`[B,H_q,S_kv,D_v]`。吸收路径将content Query映射为`[B,H_q,S_q,L_kv]`，直接与C计算分数，并合并Value上投影/Output投影。

$$
s=(Q^C(K^C)^\top+Q^R(K^R)^\top)/\sqrt{D_h+D_r},\quad
\widetilde Q_h=Q_h^C W_{UK,h},\quad
Y=\sum_h(P_h C)W_{UV,h}^\top W_{O,h}^\top.
$$

weight按PyTorch布局，`W_UK,h[D_h,L_kv]`，`W_UV,h[D_v,L_kv]`，`W_O,h[D,D_v]`。RoPE独立支路使内容矩阵可以吸收；不能任意交换位置旋转与上投影。13章核对两路径的输出及所有参数梯度，不只比shape。

[MTP](../src/transformer_lab/training/mtp.py)的第j个额外深度（j从1开始）将先前hidden与已知future embedding归一化、拼接：`[B,S-j,2D]`→projection`[B,S-j,D]`→Block→head。监督logits`[B,S-j-1,V]`对齐tokens`[:,j+1:]`；BOS/embedding/head共享，不新增词表。训练目标为NTP + weighted MTP + MoE balance。teacher-forced future embedding不能当作未知未来token的合法推理输入。

## 稀疏、压缩与高效计算

07章的规则mask仍可走dense计算；gather原语选择indices`[S_q,M]`，形成K/V`[B,H_q,S_q,M,D_h/D_v]`与scores`[B,H_q,S_q,M]`。21章先用小维learned投影评分，再因果Top-K或选完整blocks；评分目前仍dense，所有batch/head共享selection，未训练官方indexer。

19章只pool因果可用的完整旧块，保留近期窗口和不完整尾部。序列压缩减少token轴；MLA减少特征轴。压缩比例固定时，复杂度仍可能为二次除以比例。10章用KV分块online softmax维护running max、normalizer和weighted sum；数学仍是精确dense Attention，无GPU kernel。

## 递归与2026结构

状态`state[B,H,D_k,D_v]`。Linear：累加`phi(k) outer v`与normalizer；DeltaNet：写入`beta*k outer (v-k^T state)`；Gated Delta先衰减再纠错。15章还实现KDA逐key通道衰减`log_decay[B,H,S,D_k]`，与每token/head标量衰减不同，检查full/chunk状态一致。

20章补充Qwen短因果depthwise convolution、连续log decay、非对称QK/V头、gated RMS output。Q/K`[B,H_k,S,D_k]`扩展为H_v个头后，状态为`[B,H_v,D_k,D_v]`；核心四层配方用于缓存核对，完整卷积cache与正式chunkwise训练不在该配方中。

22章包含三种残差机制：mHC/GR维护`[B,S,N_streams,D]`，branch输入/输出为`[B,S,D]`；mHC mixing`[B,S,N_streams,N_streams]`经Sinkhorn近似双随机化，GR使用coordinate gates；AttnRes在depth sources`[B,S,N_sources,D]`上计算weights`[B,S,N_sources]`，沿深度聚合为`[B,S,D]`。N_streams与N_sources含义不同。

23章提供hashed bigram lookup与context gate/causal conv；24章核对共享KV消费者的绝对因果位置；25章将模态features`[B,N_image,D_in]`投影为`[B,N_image,D]`并写入hidden slots。模型对应关系、官方特有机制和公开来源见[2026模型](models-2026.md)。
