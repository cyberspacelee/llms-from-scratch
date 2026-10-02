独立 Python review，2026-10-02。审查者此前只编辑教材 MDX，没有参与 `src/`、`examples/`、配置或测试的结构实施。本轮按输入边界和独立数值对照检查，没有重跑完整 59 个 CPU 实验。

审查结论：发现并修复了蒸馏损失的 NaN、生成接口静默忽略源输入、reference exporter 内部容差降级三个问题。当前检查范围没有未修复的 Python 行为问题；CUDA 硬件正确性仍需要实际设备验收。

| 级别 | 发现与根因 | 修复与回归 |
| --- | --- | --- |
| 高 | `examples/post_training/distillation.py` 对 float32 one-hot teacher 的零概率执行 `clamp_min(1e-300).log()`，阈值下溢为零，出现 `0 * -inf`，loss 为 NaN。先计算所有位置再取 mask 时，忽略的 NaN 行仍产生 NaN student 梯度。 | 先选择有效位置，再调用原生 `F.kl_div`，支持 teacher 部分 `-inf` 表示零概率。有效学生 logits 经温度缩放后必须有限；teacher 拒绝 NaN、+inf、整行 -inf；忽略行可以非有限。新增真实 float32 one-hot 与交叉熵对照、有效/忽略行梯度、空贡献、非法有效分布和温度溢出的回归。 |
| 中 | `Transformer.generate` 对 decoder 的 `source_ids` 静默忽略；seq2seq 同时传 `source_ids` 与 `memory`，或 prepared memory 配额外 `source_valid`，同样被忽略，与 `forward` 的严格语义不同。 | 生成入口与 `forward` 保持同一来源契约；冲突抛 `ValueError`。新增 decoder 源输入、seq2seq 冲突与 `max_new_tokens=0` 的回归。修正 memory 文档为真实 `EncoderMemory` 类型。 |
| 中 | `scripts/export_site_traces.py` 的 online/dense 比较在 `.item()` 后调用无 dtype 的 `torch.tensor`，实际降为 float32 并使用默认宽松容差；JSON 却宣称 float64、1e-12。 | 明确 float64、`atol=1e-12, rtol=0`。扩充 BOS/A 主分布、正/零/负 repetition penalty、稳定排序 tie、past=3 的 rectangular mask，54 个 Python/TypeScript bridge cases 全通过。 |
| 低 | CI lint 未包含新增 exporter；Python 推理指南仍给出已删除的 `transformer-lab` CLI。 | CI 与开发指南的 Ruff 范围纳入 `scripts`，exporter 已格式化并通过 Ruff。命令更新为 `llms-from-scratch`，实际运行 ledger 成功。 |

统一实现与配置审查覆盖了根 `pyproject.toml`、锁定环境、`ModelConfig`、`AttentionConfig`、`BlockConfig`、`PositionConfig`、`Transformer`、`ModelOutput`、Cache 与损失。`uv sync --frozen --group dev` 成功。CPU PyTorch index 与可选 plots/gpu 依赖分开；README 的 GPU 独立环境使用 `uv pip`，没有声称 CPU 锁定环境能运行 CUDA。

基础 Decoder 明确采用 learned position、Pre-LayerNorm、MHA、GELU、有 bias、未绑定词表权重；现代配置采用 Pre-RMSNorm、GQA、RoPE、SwiGLU、默认词表绑定。两者都调用同一个 `Transformer` 并返回 `.logits/.cache`。P4 `ReferenceDecoder` 只用于独立同权重 oracle，现有测试检查 logits、embedding/位置/V 梯度与分块 cache；examples 中 BigramLM、TinyWords、ScalarModel 和单算子参考分别承担基线或局部机制教学，没有另一个通用训练模型。

Cache 审查核对完整 prefix validity、统一绝对 offset、每层长度、静态 cross KV 与会话源内存、dtype/device、GQA 存储头数、recurrent matrix state，以及旧状态只读/追加返回新对象的契约。实际运行 cached/full matrix、encoder memory/asymmetry、padding/batch isolation、非法 cache 与 generation/speculation 五项现有测试。没有把普通 `cat` KV 或分页 materialize 的教学实现描述成生产 paged kernel。

损失审查核对 `loss_sum + valid_count` 聚合、DDP 的 world_size/global_count 缩放、空贡献可微零与空 mean 拒绝。SFT、文本预训练的可变有效目标更新在 count 为零时跳过整个 optimizer step，避免 AdamW weight decay/状态更新。固定非空目标的示例没有装作支持任意空数据。评估采用有效 token 加权均值，局部 mean-or-zero 仅作为教学便利函数。

恢复边界验证使用两个原创训练文档：先更新 2 步保存，再更新 3 步；恢复后的 3 个 loss、所有 model 参数、Torch/Python RNG 与连续运行完全一致。恢复包含 optimizer、tokenizer、config、数据 fingerprint、cursor/order；篡改 fingerprint 被拒绝。检查了原子保存及 `weights_only=True` 加载，没有把这个 CPU checkpoint 例子描述成分布式分片恢复。

后训练分别实际运行 SFT、LoRA、distillation、DPO、reasoning RL。Teacher/reference 在 eval、no_grad 与 `requires_grad_(False)` 约束下保持冻结；LoRA 只更新 A/B、基座权重不变、adapter 保存恢复与合并后的完整模型 logits 相同。当前合并例子是无 bias projection，`merged_weight()` 仅返回权重，带 bias 的调用者必须保留原 bias。RL 的 old log-probs/advantages 固定，clip 与 exact KL 更新有有限梯度；训练奖励从 0.059623 到 0.700805，而 held-out 从 0.059930 到 0.034654，示例诚实报告了泛化未改善。

CI 覆盖清单核对得到 59 个 CPU curriculum 模块，无遗漏的顶层 runnable entry；六个 math 模块与实际非绘图文件一一对应。`unittest discover` 另有 25 个 chapter 的入口检查；build/check 都重新生成并核对 reference traces。可选绘图、CUDA C++、Triton kernel、真实 GPU profiling 与 CPU compile 的显式扩展模式不属于“59 个 CPU 实验”，CI 没有冒充这些设备检查。

Reference oracle 独立性：TS 的 sampling、RoPE/cache、online attention、masked loss、vector autograd 都执行各自的算式，读取 Python JSON 仅作为 expected，不读取 expected 构造 actual。Python exporter 调用真实 canonical 算子与 F4 autograd；它是跨语言数值 oracle，包内部正确性另外由手算、dense/online、manual/SDPA、ReferenceDecoder 等测试支持。文件包含 source hash，`--check` 同时验证源码身份和重新计算值。当前 1e-12 绝对容差适合这 54 个小 float64 案例，不表示全站交互、全部模型、任意大范围参数或低精度计算已经获得跨语言覆盖。

CUDA tile16/tile32 静态审查：row/col 与 shared A/B 的索引符合 row-major GEMM；M/K/N 的越界 load 都补零；没有 barrier 前的线程提前返回；读写 shared 的两处 `__syncthreads()` 完整；每个有效输出恰有一个 writer。tile32 使用 1024 threads/8192 shared bytes，host 检查设备限制；两组 event timing 使用同一确定性输入，10 warmup、31 repeats、同 stream、stop event 同步，计时范围不含 host-device 传输。另用 CPU 逐 tile guarded-load/index 模拟核对两种 tile 的 5×7×6、17×19×13、32×32×32、33×35×37，全部与 float64 GEMM 在 host 验收容差下吻合并且每个输出写一次。环境没有 `nvcc`，`torch.cuda.is_available()` 为 False；这些证据不能替代 GPU 编译、执行、sanitizer 和性能测量。

本轮验证：5 个 curriculum tests（其中 2 个新回归）、5 个有针对性的现有 cache/model tests、五个后训练 CLI、精确恢复/篡改边界、59 个入口的静态覆盖清单、tile 索引模拟、54 个跨语言 cases、frozen sync、完整 Python Ruff 范围、当前 ledger CLI，均通过。没有修改 MDX/UI，也没有重跑全量 CPU curriculum。

补充边界复审（Python实施者另行核查，与上面的独立reviewer分开）：root指出采样入口可能存在三项输入契约问题，均已在真实调用中复现。空logits之前直到max()才抛RuntimeError；history=[0.5]被torch.long转换为ID0；top_k=True利用bool是int子类而被当成1。统一distribution现要求非空有限一维词表，history逐项检查Python整数及词表范围后才构造索引，top_k要求确切int类型；布尔ID同样拒绝。历史可迭代输入先固定为tuple，避免验证消耗生成器后丢掉惩罚项。温度、候选过滤和CTRL惩罚的合法公式未改。

在现有test_curriculum增加一项最小回归，覆盖上述三项、布尔/越界history，以及合法top-k分布与重复ID惩罚/迭代history的独立手算值。该文件6项测试全部通过，既有examples.principles.generation完整分布/停止协议检查通过，两处修改文件的Ruff及format通过。此次只改sampling.py、现有测试和本报告，未重跑59个CPU实验，也未修改MDX或组件；source hash由root最后重生成并验跨语言bridge。
