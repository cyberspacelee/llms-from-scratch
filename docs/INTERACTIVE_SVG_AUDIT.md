> 重构前的规划或审查记录。当前结构与验证结果见 [实施记录](MDX_REBUILD_STATUS.md)，教学约定见 [当前课程设计](CHAPTER_BLUEPRINTS.md)。

# 全站交互 SVG 审视

日期：2026-10-01。范围：61 章正文、7 篇路线导读、首页和 404，共 70 页。

以下清单保留实施前的逐页评估与接入顺序。用户随后要求全量实施，新增、复用和增强工作已按清单落实；交付与验收记录见文末。清单中的“现有表达”描述实施前状态。

## 实施前现状与结论

- 仓库有 45 个实验组件，其中 42 个直接绘制 SVG。公共容器 `Lab.tsx` 不计为实验。
- 正文挂载了 36 个不同的实验，覆盖 24 章；其中 33 个绘制 SVG，覆盖 21 章。缓存索引、调度器、分页缓存目前使用 DOM 展示交互状态。
- 9 个已有实验尚未挂载。优先接入其中 6 个，剩余 3 个都是 GPU 线程布局扩展，需控制 G1 的交互密度。
- 概率主线 M1、M2、M4、M5、P2，以及系统 S3、S6、S9、S10，是新增图解最有价值的区域：需要读者追踪分母、状态更新、索引映射或事件时序。
- GPU 路线和 P3、P6 已有较密集的实验。优先完善现有实验，保持公式、例题与操作的阅读顺序。

## 参考文章中可借鉴的做法

[SGLang Unified Radix Cache](https://www.sglang.io/blog/unified-radix-cache) 把缓存树、组件复用条件、分配映射和淘汰过程拆成独立图解。图中可以逐步前进、后退或重置，切换固定场景，并高亮同一个请求或状态在不同视图中的位置。

适合迁移到本课程的设计：

1. **单步状态回放**：当前步骤显示操作前后的状态和数值。例如 KV 写入、在线 softmax、all-reduce、checkpoint 恢复。
2. **同一输入下的对照**：正确与错误分母、整段与缓存计算、行切分与列切分、连续与中断训练并排比较。
3. **跨视图关联高亮**：选择一个 token，同时标明它的逻辑位置、物理槽位、注意力可见集合与损失目标。
4. **少量固定场景**：正常路径、一个反例、一个边界条件。初次进入页面时展示正文案例，额外参数随后展开。
5. **可核对的读数**：图下保留分子、分母、状态向量、元素数或输出误差，读者能够代回公式。

树形图适用于树结构；本课程的概率路径、计算图、归约和资源账应采用各自更直接的图形。参考文章的性能数字和具体混合缓存实现不作为本站教学实验的基准数据。

## 第一批接入顺序

### 先接入已有实验

这些组件已有计算实现，接入前仍需对齐正文案例并检查布局。

| 顺序 | 页面与位置 | 复用组件 | 接入前必须处理 |
| --- | --- | --- | --- |
| 1 | F4「下一次反向：梯度缓冲与图的寿命」 | `AutogradBranchLab` | 正文主例与实验 `x²+3x` 不同，优先沿用正文计算图；若保留小例，明确它用于隔离梯度累积。保持“重新前向”与“保留同一张图”的区别。 |
| 2 | F5「从类别轴到有效目标平均」 | `MaskedLossLab` | 当前实验为三类别，正文为四类别；对齐 logits、目标与忽略位置。全部忽略时 mean 的原始分母为零，明确教学保护策略。P2 引用此实验，避免重复挂载。 |
| 3 | P9「温度」「Top-k 与 top-p」 | `SamplingLab` | 当前八项词表改为正文词表或明确扩展示例；先展示温度、top-k、top-p，min-p 与重复惩罚放进选读。保留确定性的候选分布展示。 |
| 4 | P10「分块追加：两行查询怎样读五个键」 | `CachedAttentionLab` | 对齐正文的历史、新增长度和位置编号；保留错误左上角三角 mask 的对照，显示当前查询对应的绝对位置。 |
| 5 | P7「二维旋转：怎样让共同起点消失」 | `DotProductLab` | 用正文 q、k 和频率；先调整共同位置偏移，再调整相对位移。M6 尚未准备旋转先修，不直接挂载完整版本。 |
| 6 | M10「反向传播：每个参数收到什么信号」 | `SoftmaxLossLab` | 默认温度固定为 1，对齐正文 logits。当前梯度条按固定比例绘制，低温时可能超出画布，需调整坐标范围；手机上概率与梯度上下排列。M5 可引用其选读入口。 |

### 再新增关键图解

| 顺序 | 页面 | 一幅图要回答的问题 | 最小交互与验收 |
| --- | --- | --- | --- |
| 7 | M1 随机事件与条件概率 | 条件已知后，哪些记录成为新的分母？ | 保留四条记录；选择事件 A、B，突出交集和条件集合。并列显示 `P(A)`、`P(A∩B)`、`P(A\|B)`；零概率条件显示未定义。 |
| 8 | M2 贝叶斯与序列概率 | 联合路径怎样求和，再反查来源？ | 两层概率树沿用 M1 记录；逐步突出“来源→结果”，显示全概率求和与后验归一化。链式法则用独立的序列路径小图，避免把贝叶斯误当先修。 |
| 9 | M4 信息量与熵 | 同一个分布如何得到信息量、熵、交叉熵与 KL？ | 选择结果显示 `−log p`；调整模型 q 显示逐类贡献和 `CE=H+KL`。条件熵另用来源切换展示加权平均；声明对数底与零概率边界。 |
| 10 | S3 FlashAttention | 后来的块出现更大分数时，之前累积的值怎样修正？ | 按正文六个键单步读取块；显示旧/新 `m,l,o`、重缩放系数和输出。对比整行 softmax，不把教学状态回放标成 GPU 性能模拟。 |
| 11 | S6 引擎执行器与 CUDA Graph | 一个请求的 token 如何进入 packed 数组和物理槽位？ | 点选正文三个请求之一，联动高亮 token、position、slot mapping、block table。第二场景展示 capture/replay 的固定地址与有效行。 |
| 12 | T9 分布式训练与状态分片 | 两个 rank 的有效目标数不同，怎样得到同一次全局更新？ | 沿用 5/9 个目标；逐步显示局部和、缩放、all-reduce 与更新。切换错误局部均值平均，对比单进程参考梯度。 |

第一批完成后，再做 P2 的序列概率与 PPL 图、T3 恢复对照、T4 滑窗评分、S9 通信和 S10 服务事件。下面的“先做”表示教学收益高，实际排期仍按上表推进。

## 逐页清单

“先做”：补上目前缺少的关键计算解释，或接入已有组件。“后做”：有明确收益，但优先级较低。“保留”：现有表达已足够，先维护现有实验。

### 数学：11 章

| 页面 | 现有表达 | 可加位置与交互 | 建议 |
| --- | --- | --- | --- |
| [M1 随机事件与条件概率](../site/src/content/lessons/math/events.mdx) | 四记录表与公式 | 「联合概率」「条件概率」：记录集合选区，条件集合成为分母；独立与互斥采用固定反例。 | 先做，新建事件图 |
| [M2 贝叶斯与序列概率](../site/src/content/lessons/math/bayes.mdx) | 来源与结果表 | 「全概率」「贝叶斯」：路径求和、后验归一化；「概率链式法则」：展开依赖历史的序列路径。 | 先做，新建分步概率树 |
| [M3 概率与统计](../site/src/content/lessons/math/probability.mdx) | batch 均值静态图 | 「batch 均值」：固定随机种子或固定预生成样本，改变 batch 大小，比较样本均值分布与理论标准误。标明独立抽样假设。 | 后做，替换同义静态图 |
| [M4 信息量与熵](../site/src/content/lessons/math/information.mdx) | 信息量静态图 | 「交叉熵」「KL」：目标 P 固定、模型 Q 可调，显示逐类代价及分解；条件熵按来源切换并展示平均。 | 先做，新建概率代价图 |
| [M5 分布与预测损失](../site/src/content/lessons/math/distributions.mdx) | 正态分布静态图、分类数值表 | 「类别分布」「NLL」：三项 logits→softmax→目标概率→损失；加共同常数时结果不变。MLE 用一小批观测比较连乘和负对数求和。梯度留给 M10。 | 先做，复用 softmax 计算，简化呈现 |
| [M6 向量与矩阵](../site/src/content/lessons/math/linear-algebra.mdx) | 点积、范数、矩阵轴静态图 | 「矩阵乘法」「batch」：选择一个输出格，联动输入行、权重行和逐项乘积；沿用正文矩阵。 | 后做，新建矩阵元素图 |
| [M7 微积分与梯度](../site/src/content/lessons/math/calculus.mdx) | `DescentLab` 与下降静态图 | 先完善实验的差商与局部线性预测显示，保留原有学习率与下降对照。 | 保留，增强现有实验 |
| [M8 矩阵求导](../site/src/content/lessons/math/matrix-calculus.mdx) | 线性梯度静态图 | 「三条记录共用权重」：选择一个 W 元素，逐条累积外积贡献；切换 sum/mean，联动梯度矩阵。 | 先做，新建贡献累积图 |
| [M9 链式法则与反向传播](../site/src/content/lessons/math/backprop.mdx) | `SharedNodeLab` | 沿现有共享节点图高亮反向路径与相加位置；避免再加一幅同义分支图。 | 保留，增强现有实验 |
| [M10 神经网络训练](../site/src/content/lessons/math/training.mdx) | 训练、激活静态图 | 「反向传播」接 `SoftmaxLossLab`；随后可单步显示原参数、梯度、更新参数与新损失，严格沿用正文网络。 | 先做，先复用再扩展 |
| [M11 谱分解、低秩与数值稳定性](../site/src/content/lessons/math/spectral.mdx) | `LowRankLab`、`SpectralLab` 与配套静态图 | 已覆盖方向、谱、低秩与误差；优先检查缩放与手机字号。 | 保留 |

### 数组与框架：7 章

| 页面 | 现有表达 | 可加位置与交互 | 建议 |
| --- | --- | --- | --- |
| [F1 NumPy 数组与存储](../site/src/content/lessons/frameworks/numpy-arrays.mdx) | `NumpyStorageLab` | 在原视图/副本实验中保持同一存储地址身份；如增加修改操作，联动原数组和选区。 | 保留，增强现有实验 |
| [F2 NumPy 的轴与计算](../site/src/content/lessons/frameworks/numpy-computation.mdx) | `NumpyAxesLab`、`NumpyContractionLab` | 现有轴与收缩实验已覆盖主问题，避免另做矩阵乘面板。 | 保留 |
| [F3 PyTorch 张量与轴](../site/src/content/lessons/frameworks/torch-tensors.mdx) | `TorchStrideLab` | 在原实验中跟踪一个元素拆头、换轴、合头的坐标与 stride；共享存储与求导关系分开标注。 | 保留，增强现有实验 |
| [F4 自动微分与梯度状态](../site/src/content/lessons/frameworks/autograd.mdx) | 计算图寿命静态图 | 「下一次反向」接 `AutogradBranchLab`，显示图内贡献相加与跨调用 `.grad` 累积。 | 先做，复用 |
| [F5 模块、参数与损失](../site/src/content/lessons/frameworks/modules-and-losses.mdx) | 模块树静态图 | 「有效目标平均」接 `MaskedLossLab`；模块树可后续点击查看参数名、共享身份与 train/eval 模式。 | 先做，先接损失实验 |
| [F6 数据、更新与训练状态](../site/src/content/lessons/frameworks/data-and-training.mdx) | `BatchStreamLab` | 现有批流实验保留；恢复完整状态的对照集中在 T3，F6 给出入口。 | 保留 |
| [F7 torch.compile](../site/src/content/lessons/frameworks/torch-compile.mdx) | Mermaid 流程 | 「第二次调用」：依次提交相同输入、改 shape、触发图中断；回放 guard 检查、复用与重新编译。时间只用已声明的教学记录。 | 先做，新建执行状态图 |

### 模型原理：11 章

| 页面 | 现有表达 | 可加位置与交互 | 建议 |
| --- | --- | --- | --- |
| [P1 文本、token 与 embedding](../site/src/content/lessons/principles/tokenization.mdx) | `TokenMergeLab` | 在现有合并实验之后点选 token，联动 ID 与 embedding 行；特殊 token 仍遵循正文规则。 | 后做，增强现有实验 |
| [P2 下一 token 预测](../site/src/content/lessons/principles/language-modeling.mdx) | 移位表、概率与损失公式 | 「标签移位」「PPL」：点选 `[BOS,A,B,EOS]` 的预测行，显示历史、目标、概率、NLL；汇总有效位置后显示 PPL。F5 承接完整损失归约接口。 | 先做，新建序列账目图 |
| [P3 注意力与多头](../site/src/content/lessons/principles/attention.mdx) | `CausalMaskLab`、`HeadPatternLab`、`MultiHeadLab` | 已覆盖可见集合、权重模式与多头；保持它们各自对应正文问题。 | 保留 |
| [P4 从注意力到基础 Decoder](../site/src/content/lessons/principles/decoder.mdx) | Decoder、归一化、可见性静态图 | 「两次残差」：按步骤显示原状态、分支输出与相加结果；词表头接已有输出读数。先只追踪正文一个位置。 | 后做，新建数值残差图 |
| [P5 注意力从哪里得到顺序](../site/src/content/lessons/principles/attention-order.mdx) | 受控对照表 | 交换两个输入；切换位置与 mask，联动 token 身份、分数矩阵、可见集合和输出。区分输入重排与固定槽位。 | 先做，新建重排对照图 |
| [P6 多频率 sin/cos 位置编码](../site/src/content/lessons/principles/sinusoidal.mdx) | `RotationLab`、`FrequencyLab`、`EncodingMatrixLab` | 已有完整实验链，不再增加旋转面板。 | 保留 |
| [P7 相对位置与 RoPE](../site/src/content/lessons/principles/rope.mdx) | RoPE、二维分块静态图 | 「共同起点消失」接 `DotProductLab`；固定相对位置，改变共同偏移，核对点积不变。 | 先做，复用 |
| [P8 常见 Decoder 结构选择](../site/src/content/lessons/principles/modern-decoder.mdx) | Norm、注意力、FFN 变体静态图 | 「GQA」选择 Q 头高亮共享的 K/V；切换 KV 头数，显示实际缓存元素。SwiGLU 数值门控后续单独做。 | 后做，新建头映射图 |
| [P9 生成、采样与对话格式](../site/src/content/lessons/principles/generation.mdx) | 候选分布表 | 「温度」「Top-k/top-p」接 `SamplingLab`，同时显示原分布、被删除项与重新归一化分布。 | 先做，复用 |
| [P10 KV cache](../site/src/content/lessons/principles/kv-cache.mdx) | `CacheIndexLab`，使用 DOM | 「分块追加」接 `CachedAttentionLab`；关联逻辑位置、正确 mask、分数与输出，不重复原槽位索引实验。 | 先做，复用 |
| [P11 可训练、可缓存的 Decoder](../site/src/content/lessons/principles/complete-transformer.mdx) | `TransformerFlowLab` 与 Mermaid | 已有阶段、形状、KV 头数、历史/新增长度控制；可增加训练、prefill、decode 预设及当前有效位置高亮。 | 保留，增强现有实验 |

### 训练：9 章

| 页面 | 现有表达 | 可加位置与交互 | 建议 |
| --- | --- | --- | --- |
| [T1 数据与样本构造](../site/src/content/lessons/training/data.mdx) | Packing 静态图 | 「移位与窗口」「三种 mask」：同一文档逐步切窗、移位、packing；选择位置区分可读历史与有效监督。文档身份保持可见。 | 先做，新建样本构造图 |
| [T2 优化与训练稳定性](../site/src/content/lessons/training/optimization.mdx) | `TokenWeightLab` | 保留目标加权实验；「一步 AdamW」「第二步」显示同一参数的梯度、m、v、偏差修正与解耦衰减。 | 后做，新增优化器状态图 |
| [T3 从零训练](../site/src/content/lessons/training/pretraining.mdx) | Checkpoint 静态图 | 「恢复检验」：连续与暂停恢复使用相同事件序列；选择漏存优化器、RNG 或游标，显示从哪一步开始偏离。 | 先做，新建恢复对照图 |
| [T4 模型评估与受控实验](../site/src/content/lessons/training/evaluation.mdx) | 滑窗评分静态图 | 「有限窗口」单步显示上下文与新增评分目标，累计每个目标恰好一次；「合并 NLL」对比 token 加权与错误文档均值。 | 先做，新建评分回放 |
| [T5 指令微调](../site/src/content/lessons/training/instruction-tuning.mdx) | SFT mask 静态图 | 「移位与 mask」联动角色、input、target、loss mask；选择 prompt 位置区分直接监督与通过后续位置传来的梯度。 | 先做，新建会话对齐图 |
| [T6 LoRA](../site/src/content/lessons/training/lora.mdx) | LoRA 静态图 | 「初始与首步」「合并」逐步显示冻结 W 与可训练 A/B，观察首步梯度；切换两路径与合并矩阵，核对同一输出。 | 先做，新建参数与梯度路径图 |
| [T7 缩放规律与计算预算](../site/src/content/lessons/training/scaling-laws.mdx) | `ScalingBudgetLab` | 已覆盖预算与分配；保持理想拟合和真实 pilot 的区别。 | 保留 |
| [T8 预训练数据工程](../site/src/content/lessons/training/data-engineering.mdx) | 重复组静态图 | 「重复关系组」调阈值，显示相似边、连通分量和整组划分；用 A-B-C 链展示同组不等于任意两篇相似。 | 后做，新建分组图 |
| [T9 分布式训练与状态分片](../site/src/content/lessons/training/distributed-training.mdx) | 加权 DDP 静态图 | 「DDP」「梯度累积」回放局部和、目标计数、缩放与归约，比较单进程参考；状态分片另用可选资源布局。 | 先做，新建归约回放 |

### GPU：4 章

| 页面 | 现有表达 | 可加位置与交互 | 建议 |
| --- | --- | --- | --- |
| [G1 GPU 的执行模型](../site/src/content/lessons/gpu/execution-model.mdx) | `ThreadIndexLab`、`WarpBranchLab`、`BlockScheduleLab` | 「迁移到矩阵坐标」可接 `MatrixThreadLab`；`WarpLayoutLab`、`BlockVolumeLab` 放到二维/三维选读。先整合原有线程实验，避免同时堆叠六个面板。 | 后做，选读复用 |
| [G2 内存与同步](../site/src/content/lessons/gpu/memory-and-sync.mdx) | `CoalescingLab`、`TransposeLab`、`BankConflictLab` | 在现有转置实验中强调写入完成、屏障、交换读取、再次覆盖四阶段；需要时增加错误提前读取预设。 | 保留，增强现有实验 |
| [G3 分块与融合 kernel](../site/src/content/lessons/gpu/kernel-design.mdx) | `TileReuseLab`、`KernelLifecycleLab` | 在现有生命周期图中突出融合前后的中间写入与最终写入；保持地址和 FLOP 可核对。 | 保留，增强现有实验 |
| [G4 怎样测量 GPU kernel](../site/src/content/lessons/gpu/measurement.mdx) | `OccupancyLab`、`StreamTimelineLab` | 在现有时间线中切换 CPU 提交、设备区间、端到端计时范围；不能把教学时间线当实测。 | 保留，增强现有实验 |

### 推理系统：10 章

| 页面 | 现有表达 | 可加位置与交互 | 建议 |
| --- | --- | --- | --- |
| [S1 Prefill 与 decode 资源估算](../site/src/content/lessons/systems/ledger.mdx) | 资源账静态图 | 「KV 容量账」「运算账」改变正文 B/T，分开显示权重、KV 容量、当步读取字节和 FLOP；各项可展开核算。 | 后做，新建资源账图 |
| [S2 单卡计算与带宽上限](../site/src/content/lessons/systems/accelerator.mdx) | Roofline 静态图与 Mermaid | 「屋顶线」沿用 S1 的负载点，改变 batch/长度，显示强度、转折点和理论下界。设备规格固定并注明来源。 | 后做，新建交互 roofline |
| [S3 FlashAttention](../site/src/content/lessons/systems/flash-attention.mdx) | Mermaid 与状态手算 | 「第二块重缩放」单步读块，突出旧状态衰减和新贡献，核对输出与朴素注意力一致。 | 先做，新建在线 softmax 回放 |
| [S4 请求生命周期与连续批处理](../site/src/content/lessons/systems/engine-runtime.mdx) | `SchedulerLab`，使用 DOM | 已有预算、队列、单步与重置；增加 token 身份高亮，区分刚采样输出与已写入 KV，取消/抢占作为后续固定场景。 | 后做，增强现有实验 |
| [S5 分页 KV cache 与前缀复用](../site/src/content/lessons/systems/paged-cache.mdx) | `PagedKvLab`，使用 DOM | 已有共享、COW、引用计数与退出；在原实验增加实际物理槽内容与空闲块高亮。「前缀身份」展示整块哈希链的反例。 | 后做，增强现有实验 |
| [S6 引擎执行器与 CUDA Graph](../site/src/content/lessons/systems/engine-execution.mdx) | Mermaid 与 packed 数组表 | 「Prefill」「KV 写入」联动三个请求、packed 数组、逻辑位置和物理槽。Graph 另用同一批数据回放地址稳定与有效行。 | 先做，新建索引映射图 |
| [S7 量化](../site/src/content/lessons/systems/quantization.mdx) | 数值表与资源公式 | 「对称/非对称量化」调位宽、分组或异常值，显示刻度、整数码、重建值、误差与 scale 存储成本。计算真实字节，避免只显示压缩倍数。 | 先做，新建量化刻度图 |
| [S8 推测解码](../site/src/content/lessons/systems/speculative-decoding.mdx) | `SpeculativeLab` | 已有单位置接受质量与残差分布；新增正文两步固定接受/拒绝轨迹，显示提交 token 与 KV 回滚边界。 | 后做，增强现有实验 |
| [S9 多卡推理](../site/src/content/lessons/systems/cluster.mdx) | Mermaid 与两卡手算 | 「列切分」「行切分」同一矩阵单步分发、局部计算、拼接或求和；同时显示完整参照输出与通信元素数。 | 先做，新建通信回放 |
| [S10 服务评估与容量规划](../site/src/content/lessons/systems/serving-evaluation.mdx) | Mermaid 与事件记录 | 「输出事件」点选同一请求的到达、首输出、后续输出、取消；联动 TTFT、ITL、TPOT。切换 SLO 阈值显示合格产出与分母。 | 先做，新建服务事件时间线 |

### 进阶模型：9 章

| 页面 | 现有表达 | 可加位置与交互 | 建议 |
| --- | --- | --- | --- |
| [A1 MoE](../site/src/content/lessons/advanced/moe.mdx) | `MoeRouteLab` 与 Mermaid | 现有实验覆盖单 token 路由；「三个 token 分给专家」增补 dispatch、专家计算、加权 gather，保持原行身份；容量冲突放后续预设。 | 后做，增强现有实验 |
| [A2 MLA](../site/src/content/lessons/advanced/mla.mdx) | Mermaid 与矩阵手算 | 「内容键」「值」切换显式展开与投影吸收，逐步显示缓存对象、矩阵形状与同一输出；位置键单独保留。 | 先做，新建两路径对照 |
| [A3 DPO](../site/src/content/lessons/advanced/dpo.mdx) | DPO 损失静态图 | 「响应似然」「参考差」点选 chosen/rejected 的回复 token，累计 log-prob，显示 policy/ref 差与 logistic 损失；提示不直接计入回复和。 | 后做，新建偏好账目图 |
| [A4 长上下文与窗口](../site/src/content/lessons/advanced/long-context.mdx) | `WindowCacheLab` | 原实验保留；频率方案变化时新旧 cache 不一致可作为额外预设，不混入窗口主问题。 | 保留，增强现有实验 |
| [A5 可验证奖励训练](../site/src/content/lessons/advanced/reasoning-rl.mdx) | 四回复表与 GRPO 公式 | 「组内比较」「受限更新」回放奖励→优势→概率比→裁剪；选择正/负优势，展示梯度方向和零优势场景。 | 先做，新建策略更新图 |
| [A6 状态空间与递归模型](../site/src/content/lessons/advanced/state-space.mdx) | 状态衰减静态图 | 「四个输入」单步显示旧状态衰减、新输入写入和读出；「扫描」切换串行递推与仿射组合，比较同一结果。 | 先做，新建状态回放 |
| [A7 递归式线性注意力与混合层](../site/src/content/lessons/advanced/hybrid-attention.mdx) | `DeltaStateLab` | 现有实验覆盖 gate/delta；后续展示不同层分别保存 KV 或递归状态。混合缓存安全边界作为明确的新扩展主题。 | 后做，增强现有实验 |
| [A8 稀疏注意力](../site/src/content/lessons/advanced/sparse-attention.mdx) | 稀疏选择静态图 | 「固定集合」「漏读」选择键，联动 mask、重新归一化权重、被丢弃质量与输出误差；indexer 分数与主注意力权重分开显示。 | 先做，新建选择与误差图 |
| [A9 视觉语言模型](../site/src/content/lessons/advanced/multimodal.mdx) | 多模态展开静态图 | 「占位展开」点选像素块，联动视觉特征、连接器输出、展开位置、回答目标和因果可见集合；梯度路径随后回放。 | 先做，新建跨模态对齐图 |

### 导读、首页与 404：9 页

| 页面 | 现状与候选 | 建议 |
| --- | --- | --- |
| [M0 数学导读](../site/src/content/lessons/math/index.mdx) | 已有完整概率链表；可将章节依赖转成可点击 SVG 路径，突出主线与谱分解分支。 | 后做，导航图无需数值实验 |
| [F0 框架导读](../site/src/content/lessons/frameworks/index.mdx) | 章节列表与六个数的学习路径清楚。 | 保留 |
| [P0 模型导读](../site/src/content/lessons/principles/index.mdx) | 已拆训练、顺序、生成路线；可选依赖图展示 P4 与 P11 的区别。 | 后做，复用章节导航数据 |
| [T0 训练导读](../site/src/content/lessons/training/index.mdx) | 已说明阅读顺序与实际数据处理执行顺序不同；如增加图，采用两条可点击路径。 | 后做 |
| [G0 GPU 导读](../site/src/content/lessons/gpu/index.mdx) | 已有总体模型静态图；点击 CPU/SM/block/thread 可连接相应章节，不增加硬件调度动画。 | 后做，关联导航 |
| [S0 系统导读](../site/src/content/lessons/systems/index.mdx) | 请求的资源、调度、缓存、执行、评估路径清楚；可选可点击请求流程图。 | 后做 |
| [A0 进阶导读](../site/src/content/lessons/advanced/index.mdx) | 按问题分支已有明确入口。 | 保留 |
| [首页](../site/src/pages/index.astro) | 已有路线导航，无需新增装饰动画。实验数量的 glob 包含 `Lab.tsx`，显示 46；真实实验文件为 45，已挂载为 36，统计口径需修正。 | 保留布局，先修正数量口径 |
| [404](../site/src/pages/404.astro) | 已有各路线与首页链接。 | 保留，不增加 SVG |

## 现有资源与实现边界

### 尚未挂载的实验

6 个优先复用组件已列入第一批；其余为 `MatrixThreadLab`、`WarpLayoutLab`、`BlockVolumeLab`，归入 G1 的矩阵/多维选读。完整源码位于 [labs](../site/src/components/labs/)。

还有 3 个未挂载的静态图组件：`NanoOverview.astro`、`Qwen3Layer.astro`、`LaunchTimeline.astro`。可分别评估用于 S4–S6 总览、P8/P11 真实模型对照、S6 Graph 发射对照。它们目前不提供交互，也不是直接可用的手机布局；`LaunchTimeline` 的条带示意不能证明 CUDA Graph 消除了所有 GPU 空隙。

### 缓存图的语义

S5 当前主线是固定 nano-vLLM 实现的完整块哈希链复用。参考文章的 radix tree 和 FULL/SWA/MAMBA 安全边界涉及额外机制。若加入树形回放，应明确建立“SGLang 混合缓存扩展”小节并补充正文和来源；不要把它画成当前 nano-vLLM 的内部结构。

PagedKvLab 已有 COW、引用计数、退出回收；SchedulerLab 已有单步执行；TransformerFlowLab 已有阶段选择与缓存规模。新增工作应补充缺少的关联和数值，不重复已有功能。

### 接入前的实际缺口

1. `SoftmaxLossLab` 的温度范围包含 0.25，此时梯度绝对值可接近 4；当前固定绘图比例可能将梯度条及读数画出 SVG 范围。需先调整坐标尺度。
2. 多个未挂载实验的数据与目标章节不同。必须对齐正文案例，或清楚标明独立的扩展示例。
3. 固定宽度 SVG 在手机上整体缩小可能导致文字难读。特别是概率/梯度并排图，应切为上下布局并保持独立的读数区域。
4. 首页实验计数目前统计文件而非可使用的实验，并把公共容器计入。应明确采用“已上线实验”或“仓库实验组件”的口径。

## 实现与验收要求

- 沿用 [Lab.tsx](../site/src/components/labs/Lab.tsx) 的容器、颜色、控件与读数，计算优先复用现有 `site/src/lib/*-model.ts`。继续采用 React 状态与内联 SVG，无需增加图形或动画依赖。
- 每个实验先落实一个问题和一个正文案例。已有静态图若被交互完整替代，保留必要图注，删除同义重复图。
- 回放使用有限状态序列与当前步骤即可；出现实际重复代码后再提取公共部分。默认停在可阅读的初始状态，手动单步可完成全部教学任务。
- 原生 range、select、checkbox 与按钮均可键盘操作；SVG 选择动作提供键盘或表单等价操作。触屏无需 hover 即能获取读数；颜色之外保留文本或线型标识。
- 布局保留稳定尺寸；桌面横向关联图在手机上改成纵向或分段视图，避免整图缩小导致文字不可读。重要公式和数值可在 SVG 外换行。
- 新计算保留一个独立的可运行核对，沿用配套 Python 案例作为数值参照；需要验收零分母、零概率、全部忽略、尾块、拒绝与取消等实际边界。
- 集成后运行 `pnpm check`、`pnpm build`，检查正文入口与链接；浏览器覆盖桌面和手机、键盘、关键状态、读数变化与截图。SSR 初始图和 hydration 后状态应一致。
- 性能、随机采样与 GPU 时间线注明证据类型。教学状态图证明索引、公式和给定案例结果，实际吞吐与延迟仍需设备测量。

## 本轮核对记录

- 枚举 68 个 MDX 页面，并逐页记录标题、章节、图、实验与核心案例；另读首页和 404，形成上述 70 页清单。
- 根据实际挂载而非历史规划统计 45 个实验组件、36 个已挂载实验、9 个未挂载实验；检查未挂载候选及缓存、调度、MoE、混合层、完整模型等已有实现。
- 参考文章检查了缓存树分步回放、复用边界、分配映射与淘汰对照；本轮概率主线的浏览器检查覆盖桌面与手机代表页面。
- 此段记录审视阶段；全量实现及验收如下。

## 全量实施交付

全部建议已按各章主问题落实。保留项继续使用已经完整覆盖问题的实验，并完成数值与浏览器核对；新增图和正文沿用本章案例，独立的扩展示例明确声明假设。

| 范围 | 实际交付 |
| --- | --- |
| M1–M11 | 事件集合、Bayes/序列连乘、精确 batch 分布、信息量/条件熵/CE/KL、softmax/NLL、矩阵元素与梯度贡献、差商、反向阶段、XOR 完整更新与局部梯度；谱实验保留核对。 |
| F1–F7 | 存储身份与修改、轴收缩、拆头元素追踪、梯度缓冲累积、模块参数身份与六行四类 CE、批流与恢复入口、compile guard 回放。 |
| P1–P11 | 查表、序列概率/PPL、注意力、数值残差、置换/槽位对照、位置编码、RoPE、GQA/SwiGLU、采样、分块缓存注意力、完整模型调用预设。 |
| T1–T9 | Packing、AdamW 历史、三种漏存反例、滑窗覆盖/加权 PPL、SFT 因果梯度探针、LoRA 首步/合并、预算、重复组整组划分、DDP 与状态分片。 |
| G1–G4 | 接入二维索引、warp 布局、三维切片；增加发布/回收屏障反例、融合写回数值/字节账、四种计时范围；核对已有访存与资源实验。 |
| S1–S10 | 资源账、roofline、在线 softmax、调度 token/KV 状态、真实 COW 槽位、前缀身份、packed/Graph 映射、量化、推测验证/回滚、通信与服务事件。 |
| A1–A9 | 多 token MoE、MLA 两路径、DPO 账目、频率与旧缓存、GRPO、递推/仿射组合、混合缓存共同安全边界、稀疏选择误差、视觉位置/监督对齐。 |
| 7 篇导读及首页 | 共用课程内容数据的可点击 SVG 路线图；训练导读区分数据处理顺序，进阶导读保持独立分支，GPU 导读增加执行层次图。首页改用准确的路线数量，移除组件文件数统计。 |
| 404 | 保留有效的各路线与首页入口，完成浏览器检查。 |

原有 9 个未挂载实验全部接入；3 个静态图组件接入 S6 并修正手机布局与语义。新增图未增加依赖，沿用 React、内联 SVG 和现有实验容器；新增计算检查由 `scripts/check-interactive-models.mjs` 汇总并纳入 `pnpm check`。

SGLang 扩展位于 A7，明确采用独立 FULL/SWA/递归快照布局和逐组件投票，不声称复现参考文章的 radix tree、分配器或性能。S5 的 nano-vLLM 主线仍准确描述完整块哈希链。

### 最终验收

- 61 章正文全部提供交互实验；7 篇导读和首页提供交互路线图，共 101 处实验挂载，比审视时新增 65 处。404 的导航入口保留并核对。
- `pnpm check`：138 个文件，0 错误、0 警告、0 提示；既有模型检查和新增五组教学计算检查全部通过。
- `pnpm build`：70 个页面，3662 条站内链接与锚点可解析。构建仍有既有 MDX head 指令与包体积警告。
- Playwright 覆盖全部 70 页的 390/1440 两种宽度，共 140 次成功访问、588 次控件操作；检查 hydration、键盘操作、公式、SVG 非空、文字边界、图片和页面横向溢出，并检查代表性截图。
- 浏览器核对使用构建预览；另在开发服务器检查导读与语言模型实验。开发命令显式设置 development，避免生产 JSX 运行时导致交互挂载失败。
