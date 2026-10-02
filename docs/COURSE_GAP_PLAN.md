> 重构前的规划或审查记录。当前结构与验证结果见 [实施记录](MDX_REBUILD_STATUS.md)，教学约定见 [当前课程设计](CHAPTER_BLUEPRINTS.md)。

# 课程对照与补充实施计划

研究日期：2026-09-28。仓库基线：`c929dbe`，53 章正文与 7 篇导读。

状态：现代 decoder-only 方案已实施。课程对照沿用 CS336，用户已确认现代版本并要求全部实施。下列差距矩阵记录的是实施前基线；当前交付见 [课程验收](CURRICULUM_PROGRESS.md) 的 CS336 补充部分。正文现为 58 章与 7 篇导读。

## 1. 已确定范围

1. 课程身份：最初请求写作 CS339。检索到的 Stanford [CS339](https://web.stanford.edu/class/cs339/syllabus.html) 是 Eigenvalue Computations，包含 SVD、扰动、QR 迭代、Lanczos 等。本次根据 LLM 仓库与上下文采用 [CS336：Language Modeling from Scratch](https://cs336.stanford.edu/) 作为对照。
2. 完整 Transformer：用户已明确选定现代 decoder-only LM。模型与文本训练、缓存、分布式和 RL 实验均使用同一个实现。

本次选定现代 decoder-only，不增加原论文 encoder-decoder 分支。完整模型为 `ModernDecoder`，训练、消融、分布式与 RL 复用同一实现。

## 2. 一手来源与证据边界

| 来源 | 核查内容 | 用途 |
| --- | --- | --- |
| [CS336 2026 官方课程](https://cs336.stanford.edu/) | 讲次安排、先修与五份作业的目标 | 课程覆盖矩阵与实作深度要求 |
| [CS336 2025 官方课程](https://cs336.stanford.edu/spring2025/) | 上一年度课程安排 | 区分版本差异，避免混用讲次 |
| [2026 Lecture 9 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf) | 已下载并提取正文：幂律、模型与数据联合缩放、Chinchilla 三种拟合方式、isoflop、重复数据与外推边界 | scaling laws 的实际讲义深度 |
| [Lecture 14 官方源码讲义](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) | 转换、过滤、Jaccard/MinHash/LSH、数据混合、合成数据 | 数据章节的算法缺口 |
| [Lecture 7 官方源码讲义](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) | 集合通信、多进程数据并行、张量并行与流水线示例 | 分布式训练的实作缺口 |
| [CS339 官方大纲](https://web.stanford.edu/class/cs339/syllabus.html) | 特征值计算课程主题 | 课程编号歧义与数学补充边界 |

其余讲次在本轮主要依据官方安排与作业说明分类，未逐页审计所有 PDF。表中的补充建议是结合仓库读者定位作出的判断，不代表 Stanford 官方课程要求。已固定讲义仓库快照 `de53a9f979a6ee35f7d13a5e1aadee5ea1afc58e`（2026-09-28）；新增章节的 Lecture 7/9/14 引用使用该 revision。

## 3. 当前覆盖与真实缺口

“缺章”与“有章但实验不足”必须分开判断。仓库已补齐旧计划中不少缺口：不能再说没有 tokenizer、完整基础 Decoder、训练恢复、初始化、混合精度或 GRPO。

| CS336 主题 | 当前覆盖 | 缺口与建议 | 优先级 |
| --- | --- | --- | --- |
| 数学先修 | M1-M7：概率、CE/KL、矩阵、微积分、矩阵求导与反向传播 | 谱分解、SVD、条件数和误差传播缺独立桥接；按下节补充 | 中 |
| Tokenization，Lecture 1 / 作业 1 | P1 与 `tokenization.py` 已有字节 BPE、预切分与 UTF-8 核对 | 特殊 token 的保留识别、tokenizer 保存/加载与文本训练的集成不足；扩展 P1/T3 | 高 |
| PyTorch、资源账，Lecture 2 | F1-F7、S1、T3 的真实 FLOP 核对 | 主链已覆盖；einops 不是必需的新依赖 | 保留 |
| 架构，Lecture 3 / 作业 1 | P3/P4 完整基础 Decoder；P7 RoPE；P8 现代模块 | P8 脚本只核对模块，没有 RoPE+GQA+RMSNorm+SwiGLU 的完整可训练模型；新增整合章 | 最高 |
| MoE 与注意力替代，Lecture 4 | A1/A6/A7/A8 | 已有机制与小实验；全模型训练属于后续进阶 | 保留 |
| GPU / Kernels，Lecture 5-6 / 作业 2 | G1-G4、S2/S3、CUDA/Triton 示例 | 完整 LM 的 profiler 与注意力优化前后对照不足；GPU 实测作为可选实验 | 中 |
| Parallelism，Lecture 7-8 / 作业 2 | S9 有 TP/PP/collectives/ZeRO/FSDP 账本 | 以推理和代数示意为主，缺多进程训练、梯度同步与全局目标归约；新增训练章 | 高 |
| Scaling laws，Lecture 9/11 / 作业 3 | T3 已有 6ND、Chinchilla 与预算介绍 | 缺联合损失拟合、isoflop 最优点与外推检验；新增实作章 | 高 |
| Inference，Lecture 10 | P9/P10、S1-S10 | 生成、缓存、调度、分页、量化、投机与服务指标已有主线 | 保留 |
| Evaluation，Lecture 12 | T4 的 NLL/PPL、任务评分、样本不确定性、污染与消融 | 整合模型后补统一逐样本报告与受控消融，不另开重复章 | 中 |
| Data，Lecture 13-14 / 作业 4 | T1 已有文档划分、精确去重、packing、来源与 token 混合 | 缺原始网页到文本、可核对过滤、近重复算法和混合重采样实验；新增数据工程章 | 高 |
| Mid/post-training，Lecture 15-16 / 作业 5 | T5/T6、A3/A5：SFT、LoRA、DPO、策略梯度与 GRPO | 缺奖励模型/PPO 的完整流程解释；现有 RL 示例未构成真实 LM rollout→reward→update 闭环 | 中 |
| Multimodality，2026 Lecture 17 | A9 | 机制与形状已覆盖；真实视觉编码器训练属于进阶 | 保留 |

现状的关键边界：`code/principles/decoder.py` 已有多层基础模型；`code/training/pretraining.py` 已训练并核对恢复，但语料是四 ID 的合成循环，checkpoint 在内存中序列化。因此“完整基础模型存在”与“自然文本的可用训练程序尚不足”同时成立。

## 4. 数学补充：补到后文能用

建议新增 M8「谱分解、低秩与数值稳定性」，放在 M7 后作为分支先修，不要求读者先学完它才能进入 tokenization。

| 内容 | 推导与算例 | 后续连接 |
| --- | --- | --- |
| 特征值、特征向量、对称矩阵正交分解 | 二维对角/旋转矩阵，正定与二次型 | M4 Hessian、T2 学习率和曲率 |
| 奇异值分解与低秩近似 | 2×3 矩阵分解、秩与截断重建误差 | T6 LoRA、A2 潜在表示；明确 LoRA 不等于每步做 SVD |
| 矩阵算子范数、条件数 | 对角矩阵受小扰动后的解误差 | 梯度尺度与数值敏感性 |
| 浮点与归约误差 | 大小数相加、归约顺序、稳定 logsumexp | T2 混合精度、P8 RMSNorm、S3 在线 softmax |

复用 M2 的 CE/KL/logsumexp、M7/T2 的方差与初始化，不重复展开。scaling 拟合需要的对数坐标、最小二乘与残差就地放入新的训练章节；T4 就地补配对 bootstrap；A5 就地解释重要性采样与 baseline。

若用户确认对照确实为 Stanford CS339，需另立数值线性代数分支覆盖 QR、幂法、扰动界、Lanczos 等；上述 M8 只能覆盖其中部分先修。

## 5. 必须新增的完整 Transformer 章

建议新增 P11「从零组装一个完整现代 Transformer」，位于 P10 后、进入 T 路线前。保留 P4 的基础模型，便于比较学习式位置与现代结构。默认章名明确 decoder-only，防止读者以为包含原论文 encoder-decoder。

### 模型边界

- 输入 token ID，经 embedding、逐层 Pre-RMSNorm、RoPE+GQA 因果注意力、残差、SwiGLU、末层 RMSNorm 与 LM head，输出所有位置 logits。
- 配置只保留实际使用项：词表、层数、隐藏/FFN 维度、查询/KV 头、头维度、位置上限、RoPE 底数、epsilon、是否绑定词表权重。头维度单独定义，避免把 `d = n_q*d_h` 当成所有现代模型的约束。
- 普通前向用于训练；缓存前向用于 prefill 与增量 decode。缓存保存旋转后的 K 和 V，明确位置偏移与每层形状。
- 训练目标复用 `sequence_loss`；优化器复用 PyTorch AdamW；生成复用既有采样规则。
- 第一版采用独立等长窗口，不宣称支持 ragged/padding 或文档隔离 packing。T1 的隔离 mask 在模型明确接入前不能直接用于该训练程序。
- “从零”指显式实现核心 attention、norm、FFN 与模型组装；允许 PyTorch 张量、线性层与自动微分。不调用 `nn.Transformer` 或 Hugging Face 模型替代推导。

### 章节组织

1. 固定小配置与全部张量轴，画出从文本到 logits、loss、更新的路径。
2. 模块到完整块，再到多层模型；每一步给公式、形状与代码对应。
3. 完整前向与反向、参数账及绑定权重的两路梯度。
4. 整段、逐 token 与分块缓存前向的等价性。
5. 用固定小语料进行一次可运行训练和生成；正式数据/恢复教程链接 T3。
6. 常见错误、练习与可执行核对。

### 实施文件与复用

- 扩展 `code/principles/modern_decoder.py` 为实际模型，保留现有模块核对；优先复用 `position_encoding.py` 中适合真实张量输入的 RoPE 运算。
- 新增 `site/src/content/lessons/principles/complete-transformer.mdx`，引用同一个模型文件。
- 更新 P 路线导读和导航来源，保持既有 P1-P10 编号稳定；正文用统一模型，避免给 T3 再复制一个实现。

### 验收

- 未来 token 改动不影响更早 logits；不同批样本互不影响。
- GQA 与逐头参考计算一致，MHA/MQA 两个边界配置可运行。
- 每层缓存形状正确；整段、单 token、多 token chunk 的 logits 在声明精度容差内相同。
- 各参数梯度有限，固定批可过拟合；小配置参数数与显式账本一致。
- 保存后加载输出一致，恢复下一次更新与连续训练一致。
- 无效 token/config/超长输入给出清楚错误。

若选择原论文 encoder-decoder：另增 P12「完整序列到序列 Transformer」，实现源 embedding/位置编码、双向 encoder self-attention、decoder causal self-attention、cross-attention、FFN、归一化与输出。配复制/逆序等合成任务，核对源 padding、目标 padding、因果性、标签移位与过拟合；与现代 decoder-only 不共用一套含混的 mask 语义。

## 6. 训练路线新增与扩展

保持 T1-T6 的已有编号，新增章节用 T7-T9；导读写明 T3 后可选读这些分支，阅读顺序不机械等于编号。

| 位置 | 内容 | 最小可交付实验 |
| --- | --- | --- |
| 扩展 T3 | 自编可分发的短文本，训练/验证先划分，固定 tokenizer，训练/验证/保存/恢复/生成入口 | 磁盘 checkpoint 保存 config、tokenizer、模型/优化器、RNG 与数据游标；使用 P11 的同一模型 |
| T7 缩放规律与计算预算 | `L(N,D)=E+A/N^alpha+B/D^beta`，拟合、isoflop、固定 C 下的 N/D 分配，重复数据与外推误差 | 已知参数的合成点核对拟合与最优点；少量 CPU 小模型测量单独报告，不冒充大规模经验定律 |
| T8 预训练数据工程 | 原始样本→文本→过滤→近重复→来源混合；shingle、Jaccard、MinHash/LSH 的含义 | 固定小样本上精确 Jaccard 作为参考，核对候选筛选与去重；训练/验证同重复组不得分散 |
| T9 分布式训练与状态分片 | 数据并行、有效 token 加权、同步时机、梯度累积、ZeRO/FSDP 生命周期与通信账 | CPU Gloo 两进程的一次更新，对照单进程全局 batch；FSDP/NCCL 实测标为可选 GPU 分支 |
| 扩展 A5 | RLHF 奖励模型/PPO/价值估计流程，与 GRPO/RLVR 的关系；实际 LM rollout | 固定小任务上的采样、奖励、旧策略 log-prob、loss 与一次更新；奖励提升结论必须来自独立评估 |
| 扩展 T4、G4/S3 | 配对实验报告、模型 profiler、注意力执行对照 | 复用 P11，质量与吞吐分别记录；无 GPU 时只交正确性结果 |

T7 的最佳点按拟合参数推导，不固定成“每参数 20 token”；重复抽到的训练 token 与唯一语料量分开。T8 首版小样本可用精确算法作基线，不新建爬虫平台。T9 的 CPU 检查不声称验证 GPU 通信性能或 FSDP 实际显存。

## 7. 批次与完成条件

| 批次 | 工作 | 完成条件 |
| --- | --- | --- |
| 0 | 确认课程与 Transformer 结构，固定来源版本 | 本文两项范围已确认，实施清单无歧义 |
| 1 | P11 完整现代 Transformer；若选择则加 P12 | 完整模型与 CPU 核对通过，正文公式/轴/代码一致 |
| 2 | 扩展 T3 与 P1/T4 的集成 | 文本训练→验证→磁盘保存→恢复→生成可独立运行 |
| 3 | M8 与 T7 | 先修链接明确，分解/误差/拟合实验通过 |
| 4 | T8 与 T9 | 数据流程与两进程训练有可执行证据，规模边界写明 |
| 5 | 后训练与 GPU 可选实作，更新课程导读及进度 | 已承诺范围完成，CPU/GPU 结果分别报告 |

每批遵循 `docs/CONVENTIONS.md`：结论推导、手算或图示、可执行核对、误区与练习；代码沿用现有 NumPy/PyTorch，不先引入训练框架。执行受影响脚本，最后跑仓库要求的全量脚本、`pnpm check`、`pnpm build` 与链接检查。

本次新增 5 章（M8、P11、T7-T9），正文由 53 章增至 58 章，并扩展现有 P1/P8/T3/T4/A5/G4/S3。RAG/Agent、完整爬虫平台与大规模模型复现不纳入此次基础课程补充。

## 8. 实施对应

| 交付 | 文件 | 结果 |
| --- | --- | --- |
| 完整现代模型 | `modern_decoder.py`，P11 | RoPE/GQA/RMSNorm/SwiGLU、独立头宽、绑定权重、训练和逐层缓存 |
| Tokenizer 与文本恢复 | `tokenization.py`、`text_pretraining.py`，P1/T3 | 特殊项 opt-in、JSON、磁盘原子替换、冻结语料、RNG/游标与精确下一次更新 |
| 数学与缩放 | `spectral.py`、`scaling.py`，M8/T7 | 分解/低秩/条件数、联合拟合、isoflop、CPU 九点 pilot |
| 数据与分布式 | `data_engineering.py`、`distributed_training.py`，T8/T9 | HTML/过滤/MinHash/LSH/混合、真实两 rank DDP 与 token 加权 |
| 评估与后训练 | `evaluation.py`、`ablation.py`、`reasoning_rl.py`，T4/A5 | 配对 bootstrap、两 seed 消融、奖励模型/GAE、LM rollout 与裁剪更新 |
| 模型测量 | `model_profile.py`，G4/S3 | manual/SDPA 输出与梯度核对、CPU profiler、可选 CUDA event 与显存报告 |

实测验收与硬件边界记录在 [课程进度](CURRICULUM_PROGRESS.md)。全部默认验证只需 CPU；CUDA/FSDP 的硬件性能不由这些小实验作出结论。
