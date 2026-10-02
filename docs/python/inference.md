# 推理、计算量与内存

统一shape符号见[统一契约](../PYTHON_DEVELOPMENT.md)。乘加按2 FLOPs计。成本由[analysis.py](../../src/llms_from_scratch/analysis.py)计算；本轮仅运行小张量CPU功能检查，未测GPU吞吐。

## Training、Prefill、Decode

| 路径 | 输入 | 输出 | 状态 |
| --- | --- | --- | --- |
| Training | IDs`[B,T]` | logits`[B,T,V]` | autograd激活，默认不返回推理cache |
| Prefill | IDs`[B,T_q]`，可为新chunk | logits`[B,T_q,V]` | 追加T_q个KV或更新递归state |
| Decode | IDs`[B,1]` | logits`[B,1,V]` | 读取前缀，处理一个新token |

```python
model.eval()
with torch.no_grad():
    first = model(prompt, use_cache=True)
    next_output = model(next_ids, cache=first.cache, use_cache=True)
```

前缀长度为P时，mask为`key_position <= P+local_query_position`；完整KV长度T_kv=P+T_q。每层length必须一致，RoPE/PE从P继续。cache本身不detach，实际推理用no_grad避免保存旧计算图。完整/chunk等价性在eval下核对，不要求dropout或分块MoE auxiliary loss完全相同。

普通Cross cache静态，K/V由每层自己的投影生成；source不变时复用。双向Encoder追加新token会改变旧hidden，不能沿用旧memory；因果Encoder支持追加。当前decoder会话绑定固定source，source扩展后需重建会话。24章独立演示共享联合KV，未接入官方CED调度。

## 持久状态与逻辑读取量

令每元素字节为b，层数为N_l；普通Attention的D_v=D_h：

| 结构 | 每层状态shape | N_l层字节 |
| --- | --- | --- |
| MHA/MQA/GQA | K/V各`[B,H_kv,T_kv,D_h]` | `2*b*B*N_l*H_kv*T_kv*D_h` |
| MLA | latent`[B,1,T_kv,L_kv]`、rotary`[B,1,T_kv,D_r]` | `b*B*N_l*T_kv*(L_kv+D_r)` |
| Linear | matrix`[B,H,D_k,D_v]`、normalizer`[B,H,D_k]` | `b*B*N_l*H*(D_k*D_v+D_k)` |
| Delta/Gated Delta/KDA | matrix`[B,H,D_k,D_v]` | `b*B*N_l*H*D_k*D_v`，完整模型另有conv状态 |

`ModelCache.kv_nbytes`只计Attention状态，不含Encoder hidden、mask、权重、scores或分配器保留内存。固定递归推理state也不意味着训练总内存固定。

例如B=1、T_kv=32768、N_l=32、H_q=32、D_h=128、b=2：MHA32个KV头为16 GiB，GQA8头为4 GiB，MQA1头为512 MiB；MLA若L_kv=512/D_r=64则为1.125 GiB。MLA不必比MQA更省，具体尺寸必须计算。

单token逻辑读取量近似缓存容量；真实HBM流量还受KV group复用、tiling、cache residency影响。粗略roofline为`time >= max(FLOPs/throughput, bytes/bandwidth)`，另有launch和通信开销。小型CPU参考耗时不能外推GPU性能。

## 投影与Attention成本

`attention_cost`区分T_q、T_kv和新投影KV数N；static Cross decode令N=0。普通MHA/MQA/GQA参数为`2*D*D_h*(H_q+H_kv)`；QK-Norm另加2D_h。

| 普通Attention matmul | FLOPs |
| --- | --- |
| Q/O投影 | `4*B*T_q*D*H_q*D_h` |
| 新K/V投影 | `4*B*N*D*H_kv*D_h` |
| QK/PV mixing | `4*B*H_q*T_q*T_kv*D_h` |

减少H_kv不会同比减少QK/PV mixing。当前repeat_interleave还会展开临时KV，不具有优化GQA kernel的内存访问特性。

MLA的naive路径展开所有历史K/V；absorbed路径在L_kv轴计算，并重新计算Value/Output吸收权重。若L_kv大于content head维，其算术量不保证更小；本库每次重算吸收矩阵保证optimizer更新后的正确性。冻结权重后可预吸收，但本轮不保存第二套可失效权重。

普通FFN参数2D*D_ff，gated FFN参数3D*D_ff；对应N个token的matmul为4N*D*D_ff与6N*D*D_ff。MoE总专家数决定参数，Top-K+shared决定激活近似成本；Top-K、dispatch/combine、norm/softmax与通信不在matmul账本里。

```bash
uv run --locked llms-from-scratch ledger --length 32768 --output runs/ledger.json
```

## 长上下文路线

| 方法 | 理想计算/状态变化 | 当前实现边界 |
| --- | --- | --- |
| RoPE Scaling | 改频率，不降低Attention复杂度 | 固定Linear/NTK/YaRN |
| Flash/online softmax | 精确dense结果，减少scores驻留 | 10章KV tiling；SDPA对照，无外部GPU kernel |
| Sliding/Local | 专用kernel可O(S*W)，驱逐后窗口state固定 | 核心mask仍dense，旧cache未驱逐 |
| Sparse gather | mixing O(T_q*M)，另加indexer成本 | gather算子；21章indexer先计算dense scores |
| MLA | 压缩特征轴，scores仍T_q*T_kv | naive/absorbed全模型路径 |
| Sequence compression | 历史块摘要，固定比例仍可二次 | 19章mean pooling；无持久compressed cache |
| Recurrent/hybrid | 递归层O(S*D_k*D_v)，softmax层另计 | 顺序参考；无生产chunkwise kernel |
| Shared KV | 减少重复source投影/存储 | 24章独立consumer，不实现官方lower/upper调度 |

## Cache存储与投机

17章[分页与量化](../../src/llms_from_scratch/cache/storage.py)：`fork()`共享已有页，不满页追加采用copy-on-write；`materialize()`重新拼成普通KV，所以不是paged kernel。普通KV的cat追加会复制前缀，逐token累积拷贝为二次量级。生产服务需要预分配、页引用管理、prefix身份与驱逐策略。

int8对最后一维做对称量化：`scale=max(abs(x))/127`，codes为round/clip到[-127,127]；零向量scale取1。scales占真实字节，fp64输入保留fp64 scales，否则fp32。还原值每坐标误差最多scale/2，Attention输出误差需单独评估；没有FP4/FP8布局或fused dequant算子。

18章[greedy speculation](../../src/llms_from_scratch/inference/speculative.py)用draft提议K个候选，target并行核对；接受连续相同greedy token，首次不符提交target修正，全接受可加bonus。结果与target greedy相等；当前重算prefix，未接KV rollback。随机采样需要min(1,p/q)接受率和(p-q)+修正分布，不能沿用token相等判据。14章MTP是训练目标，未将teacher-forced未来当合法draft。

## 性能验证口径

[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)是API，[FlashAttention](https://github.com/Dao-AILab/flash-attention)是可能的backend/独立kernel；设备、dtype、head维与mask影响选择。`benchmark`显式测median、cache bytes和CUDA peak；本轮没有执行性能排名。先核对同权重结果与梯度，再在目标硬件上验证backend、同步、warmup、延迟与峰值。
