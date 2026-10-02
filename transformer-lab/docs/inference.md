# 推理、计算量与内存账本

当前仅有 CPU。本项目完成小张量功能核对，性能内容使用公式分析，**没有将 CPU 耗时外推为 GPU 吞吐，也没有 GPU 性能排名**。以下乘加按 2 FLOPs 计，参数与状态按实际配置计算。

## 1. Training、Prefill、Decode

| 路径 | 输入 | 可并行的维度 | 保存内容 |
| --- | --- | --- | --- |
| Training | 完整序列 | batch、head、token；递归参考实现按 token 循环 | autograd 激活与梯度；默认不返回推理 cache |
| Prefill | 完整 prompt 或新 chunk | query 维可并行 | 每层已处理前缀的 KV/latent/state，以及 valid |
| Decode | 通常一个新增 token | batch、head | append 一个 KV/latent，或更新一个有限状态 |

prefill 和 decode 不是两个不相关的模型，都是 `forward(..., use_cache=True)`。Decoder 输出为 `[B,Q,Vocab]`；cache 中各层的 `length` 必须一致。因果 chunk mask 用 $k\le offset+q$，不能只对 `[Q,Q]` 做三角 mask，也不能让 chunk 中早期 token 看见较晚 token。

调用范式：

```python
model.eval()
with torch.no_grad():
    first = model(prompt, use_cache=True)  # prefill
    next_output = model(next_ids, cache=first.cache, use_cache=True)
```

缓存本身不 detach，方便研究 autograd；实际推理用 `no_grad()`，否则完整旧前缀的图仍被持有。标准等价性是在 `eval()` 下比较，训练 dropout 和逐块 MoE balance loss 不要求与整序列相同。

## 2. KV 容量和 bandwidth

$b$ 为每个元素的字节数，$N_l$ 为层数。MHA/MQA/GQA 的缓存字节：

$$
M_{KV}=2bBN_lGSd.
$$

MLA：$M_{latent}=bBN_lS(L+r)$。Linear：$bBN_lH(dv+d)$；Delta/Gated Delta：$bBN_lHdv$。Decoder 多一层 Cross-Attention 时，还加 static source cache；Encoder hidden 本身为 $bBSD$。`ModelCache.kv_nbytes` 只统计各层 KV/latent/recurrent state，**不包含 hidden memory、valid mask、临时分数或参数**。

一个规模示例，$B=1,S=32768,N_l=32,H=32,d=128,b=2$：

| 结构 | 每层 KV | 32 层 KV |
| --- | --- | --- |
| MHA，$G=32$ | 512 MiB | 16 GiB |
| GQA，$G=8$ | 128 MiB | 4 GiB |
| MQA，$G=1$ | 16 MiB | 512 MiB |
| MLA，$L=512,r=64$ | 36 MiB | 1.125 GiB |

这组 MLA 小于 MHA/GQA，但大于 MQA；它们的表达能力与投影结构也不同，不能仅凭字节数选架构。理论上单 token decode 读完整历史一次，所读 KV 字节与上表相近；GQA kernel 的 KV group 复用、cache residency、head tiling 和临时展开都会改变真实访存次数。本项目账本的 `decode_cache_read_bytes` 是逻辑读取量，不是 HBM counter。

粗略 roofline：$time\ge\max(FLOPs/throughput,bytes/bandwidth)$；还要考虑 launch、通信和串行依赖。prefill 通常更容易利用大 GEMM，decode 的小 batch GEMM 与历史读取容易受带宽限制。没有 GPU 时可推导成本随 $S$、$G$、rank 的变化，不能给出 tokens/s。

缓存追加目前 `cat` 新旧 tensor，会复制旧前缀；token decode 累积复制成本为二次量级。Sliding mask 不自动驱逐旧 KV，当前容量仍线性增长。优化顺序是预分配/分页、窗口 ring buffer、grouped kernel，而非根据教学实现的 CPU 时间断言某架构更慢。

## 3. 投影与 Attention FLOPs

[analysis.py](../src/transformer_lab/analysis.py) 的 `attention_cost` 明确区分 query 长 $Q$、可用 key 长 $S$ 和新投影 KV 数 $N$。例如 Cross cache decode 传 `new_kv_tokens=0`，首次 source 投影则传 source 长度。

MHA/MQA/GQA：投影 $4BQDHd+4BNDGd$；mixing $4BHQ S d$。MLA 记内容维 $c$、rotary 维 $r$、latent rank $L$、value 维 $v$。忽略两条路径共同的 Query/KV down projection：

| MLA 路径 | 新增 matmul 项 |
| --- | --- |
| naive | KV up: $2BSLH(c+v)$；Attention: $2BHQ S(c+r+v)$；输出: $2BQHvD$ |
| absorbed | Q 吸收: $2BQHcL$；Attention+latent context: $2BHQ S(2L+r)$；latent 输出: $2BQHLD$；权重吸收: $2HLvD$ |

吸收后的维度未必更小，例如 $L>c$；节省来自避免反复展开历史 K/V，以及实际 kernel 和访存布局。当前每次计算 $W_{VO}$，并未缓存冻结后的吸收权重。把 absorption 与 mixing 单列，便于下一阶段研究预计算权重如何改变 decode 成本。

这些 FLOPs 不含 norm、softmax、dropout、activation、Top-K、mask 构建、Python 循环和 gather/cat。`score_elements` 只统计一个显式分数 tensor 的元素数，不代表训练 peak memory；autograd 还保存许多激活。

```bash
uv run transformer-lab ledger --length 32768 --output runs/ledger.json
```

## 4. 长上下文方案比较

| 方法 | 理想 mixing 复杂度 | 持久状态 | 信息变化 | 本项目实现 |
| --- | --- | --- | --- | --- |
| Full softmax | prefill $O(T^2d)$；decode $O(Td)$ | $O(TGd)$ | 完整 token 检索 | 默认路径 |
| RoPE Scaling | 不改变 Attention 复杂度 | 同原 KV | 改变位置频率 | 固定 Linear/NTK/YaRN |
| Sliding/Local | 特定 kernel 下 $O(TWd)$ | 驱逐后 $O(WGd)$ | 丢弃窗口之外的直接读取 | mask；当前仍 dense compute/full cache |
| Local/Global 交替 | 各层求和，global 层仍二次 | 各层不同 | 随深度传播更远信息 | `layer_blocks` |
| Sparse gather，每 query $M$ keys | $O(TMd)$，另加 selection/indexing | 不一定减少完整 KV | 仅选中 token 参与当前 Attention | 独立 gather 原语 |
| MLA | naive/absorbed 不同；仍有 $Q\times S$ | $O(T(L+r))$ | 共享压缩特征 | 全模型路径 |
| 序列压缩，比例 $R$ | $O(T(W+T/R)d)$ | 实际缓存可约 $O((W+T/R)d)$ | 历史块变为摘要，通常近似 | completed-block mean pooling 原语 |
| Recurrent/Delta | $O(Tdv)$ | $O(dv)$ | 固定容量记忆 | sequential reference |
| Attention/Recurrent hybrid | 两类层成本相加 | 部分 KV + 部分 state | 保留有限比例的显式检索 | 全模型路径 |

固定比例的序列压缩仍是 $O(T^2/R)$，不是自动降为线性。Sparse 的 indexer 如果先扫全历史，selection 本身可能有较大开销。Window + compressed summary + sparse selection 结合时，还需计算 summary 的因果可用时刻、更新状态和位置编码；DeepSeek V4/V4.1 的差异见 [research.md](research.md)。

## 5. 分页、Prefix 和量化原语

`PagedKVCache` 在 [cache/storage.py](../src/transformer_lab/cache/storage.py) 中存储 tensor pages。`fork()` 共享已有 pages；追加不满页时复制该页，满页继续共享。`block_table` 用 tensor 对象标识说明映射，不是设备地址或可传给真实 kernel 的页表。`materialize()` 拼回连续 K/V，方便复用普通 Attention 验证数值；其拷贝说明这还不是 paged kernel 加速。

普通 `ModelCache` 的 append 也返回新 tensor，旧 prefix 不会被原地修改，允许安全分叉；Tensor 内容仍需由调用者保持只读。生产 Prefix Cache 还要实现内容 hash、模型权重/版本/dtype/position/padding/source 身份、引用计数、驱逐和跨请求隔离，本项目只验证 prefix ownership 与复用原语。

`QuantizedTensor` 对最后一维做对称 int8：

$$
s=\max_i|x_i|/127,\quad q_i=\operatorname{clip}(\operatorname{round}(x_i/s),-127,127),\quad\hat x_i=s q_i.
$$

全零向量取 scale 1。codes 为 int8、scales 为 fp32（fp64 输入时为 fp64）；实际字节数包括 scales，没有假装 3/4-bit 打包。本原语可以编码 K/V/latent，再 decode 成普通 `KVCache` 使用；原缓存误差每坐标不超过 $s/2$，Attention 输出误差仍需独立评估。

数据流为 float cache → quantized codes/scales → restored cache → reference Attention。当前 append 还没有直接在量化存储上进行，没有 GPU fused dequant-Attention kernel，因此不声称 decode 更快。

## 6. 投机与 MTP

[inference/speculative.py](../src/transformer_lab/inference/speculative.py) 提供 greedy draft/target 验证：draft 生成长度 $K$ 的候选；target 对 `prefix+draft` 做一个因果前向；逐个比较对应 greedy token；遇到第一次不匹配，保留已经接受的 draft 并提交 target 修正；全接受时可提交一个 bonus token。

它严格保持 target greedy 输出，测试覆盖相同 draft 的全接受，以及不同 draft 的拒绝。当前完整重算 prefix 以展示原理；没有 decoder KV rollback，没有动态 batching，也没有把减少 target forward 次数直接称为速度提升。

随机采样需不同接受规则 $\min(1,p(x)/q(x))$，拒绝后从归一化 $(p-q)_+$ 采样。直接沿用 greedy 的“token 相等”判据会改变分布；本项目不提供该随机推理循环。MTP 专属 speculative 路径需处理各 head 的条件状态，不能拿 teacher-forced logits 当合法 draft 分布。

## 7. 高性能实现如何对照

| 技术 | 需要核对的功能 | 硬件实测重点 | 当前状态 |
| --- | --- | --- | --- |
| PyTorch SDPA | output/gradient、mask、GQA group | 具体 backend、dtype、长度 | 已接入，并在 CPU 核对 |
| FlashAttention | 与显式 softmax 的容差；mask/dropout 契约 | HBM traffic、峰值、prefill latency | 经 CUDA SDPA 可由 PyTorch 选择；未单独接入外部包 |
| Paged KV | 页表顺序、跨页追加、共享页读写 | page fragmentation、连续/分页 kernel | tensor pages 与 COW 原语 |
| Prefix Cache | prefix 身份、复用范围 | 命中率、prefill saved work | 只读 prefix fork 原语 |
| Quantized KV | 误差、metadata、位置支路 | 显存 savings、dequant 开销 | int8 encode/decode 原语 |
| Speculation | token/distribution 等价、拒绝回滚 | 接受率、draft 开销 | greedy reference；MTP cache 路径后续 |
| MoE kernel | 同路由 combine/gradient、load balance | grouped GEMM、all-to-all、小 batch | Python dispatch reference |

SDPA 是 API，FlashAttention 是可能的执行 backend，二者不能互相当同义词。设备、dtype、head 维、mask 与版本都影响选择。[PyTorch SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 说明 backend dispatch 与语义。[FlashAttention 官方库](https://github.com/Dao-AILab/flash-attention) 提供专用实现；当前不将其列为默认依赖。

可选 `benchmark` 入口只测小型单步 decode，报告同步后的 wall-clock median、KV 字节和 CUDA peak allocated；独立随机模型只能比较代码路径，不能比较训练质量。获得 GPU 后，先核对功能，再用 CUDA events、实际 kernel 名称、warmup、同权重和同工作负载做正式性能实验。当前没有执行该入口。
