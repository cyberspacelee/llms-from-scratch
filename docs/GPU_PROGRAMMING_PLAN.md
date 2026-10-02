> 重构前的规划或审查记录。当前结构与验证结果见 [实施记录](MDX_REBUILD_STATUS.md)，教学约定见 [当前课程设计](CHAPTER_BLUEPRINTS.md)。

# GPU 编程支线：大纲、资料与实施拆分

研究日期：2026-09-27。仓库基线：`6008442`。状态：G1–G4 正文、导读、图与代码已实施，交互由初版七项扩充为十三项，包含二维/三维索引、完整 GPU 模型与调度、转置和分块生命周期；CPU、类型、构建与浏览器验收通过。真实 GPU 验证待具备设备后执行。实施结果见 [GPU 编程验收](GPU_PROGRAMMING_PROGRESS.md)。

## 定位与阅读路径

现有 S2《单卡的上限》解释屋顶线、分块和测量，但 thread、warp、block、grid 只在一段中出现。S3 的 FlashAttention 和 S6 的 GPU 执行器已经使用这些概念。补充内容应让读者能回答三个问题：谁负责哪段数据，谁可以交换结果，凭什么判断实现更快。

建议新增 `gpu/` 路线，编号 G1–G4，另有一篇导读；在系统路线旁展示。保留现有 S1–S10 的编号和 URL。入门读者从 Python 数组进入 G1；已读系统路线的读者从 S2 转入 G1–G3，再回 S3；需要测量时读 G4，再回 S6。

以 NVIDIA CUDA 为具体实现对象，第一次出现时解释中文术语和英文名称。CUDA C++ 展示线程语义，Triton 展示数据分块语义；使用其他 GPU 的读者先迁移概念，再核对厂商的线程组、执行宽度和内存模型。每章按现有规范写约 5k–9k 字，正文、数值、图与运行说明构成完整单元。

主线心智模型：

1. 把数组工作拆成带编号的独立任务。
2. 把有共享数据需求的任务放进协作组。
3. 将逻辑任务映射到有限的硬件资源，允许分批执行。
4. 看一组线程实际访问的地址，再讨论搬运效率。
5. 用明确的同步边界建立先写后读关系。
6. 先核对结果，再用 GPU 时间与工具报告检验性能假设。

## G1：一次 kernel 怎样覆盖整个数组

拟定路径：`/gpu/execution-model/`。

核心问题：同一个函数执行很多次，怎样保证每个输出都有唯一负责人？

- 从向量加法的 CPU 循环进入 host、device、kernel、launch；说明调用返回与 GPU 完成是不同事件。
- 软件视图：grid 包含 block，block 包含 thread；线程有自己的索引和寄存器状态。
- 硬件视图：GPU 包含多个 SM，SM 接收 block，warp scheduler 发射可运行 warp 的指令。图中分别画软件组织与硬件承载。
- 一维索引 `i = blockIdx.x * blockDim.x + threadIdx.x`、向上取整、尾部 mask；用长度 100、每块 64 线程手算两个 block、128 个逻辑线程和 28 个无效位置。
- 二维索引与行主序；block 内线性线程号以 x 维最快变化，再说明 warp 和 lane 的编号。
- warp 为 32 个线程的分组；末尾不完整 warp、条件分支的活跃 lane、SIMT 与独立线程调度。不能依赖隐含的 warp 同步。
- block 调度没有可依赖的全局顺序，一个 SM 可以驻留多个 block；普通 kernel 不靠跨 block 自旋建立全局屏障。
- grid-stride loop：逻辑线程数与数据规模可以不同，一个线程可以负责多个元素。
- 回扣 LLM：逐元素激活、RoPE、每行归一化为什么需要不同的工作划分。

交付：线程索引交互、warp 分支交互、一张软件/硬件映射 SVG；CPU 索引覆盖验证；完整 CUDA 向量加法入口。

练习：100 个元素的尾部；二维 block 的第 32 号线程；改变 block 大小是否改变结果；为什么 block 不能假设按编号执行。

## G2：线程怎样搬运数据并安全协作

拟定路径：`/gpu/memory-and-sync/`。先修 G1。

核心问题：相同的加法次数，为什么地址布局和共享方式会改变数据搬运？

- global memory、L2、L1、shared memory、register 的作用域、生命周期与用途；CUDA local memory 是线程私有地址空间，不能当作片上寄存器的同义词。
- 用一条 warp 的地址表解释合并访问。教学模型限定 32 个 lane、float32、32-byte 对齐段，并注明适用条件。
- 连续地址、偏移一元素、步长访问：计算覆盖的段集合。对齐连续访问覆盖 4 段，偏移 4 字节覆盖 5 段；不据此承诺 4/5 的实测速度。
- 行主序矩阵：读一行和读一列为什么不同；转置用 shared memory 改变写回布局。
- shared memory 的 bank、不同地址冲突和同地址读取广播；用 32-bit 元素与 32-bank 模型解释 `32×32` 与 `32×33` 布局。
- 协作的三个阶段：装载、同步、消费；复用缓冲区前还要保护上一轮读取。边界线程用中性值参与，不能把 block 屏障随意放进分歧分支。
- `__syncthreads()`、`__syncwarp(mask)`、原子操作与跨 block 的 kernel 边界分别保证什么。参与集合与退出线程规则按官方同步语义讲解。
- 从共享内存树形求和进入 block reduction，再用第二个 kernel 合并 block 的部分结果。
- 回扣 LLM：RMSNorm 与 softmax 的归约和同一行数据复用。

交付：地址合并交互、bank 映射交互、装载/同步/读取 SVG；CPU 地址与归约验证；可编译的共享内存转置与归约例子；由地址模型生成的热力图 PNG。

练习：偏移与 stride 的段数；广播为何不是多路冲突；重复使用 tile 为什么需要两个同步边界；浮点求和顺序为何影响末位。

## G3：从正确 kernel 到分块与融合

拟定路径：`/gpu/kernel-design/`。先修 G1、G2；矩阵乘回读 M3，softmax 回读 P3。

核心问题：怎样把数学算子映射成可验证的 GPU 实现，再减少重复搬运？

- 明确输入输出的形状、dtype、stride、边界与别名约束；一个 kernel 的正确性先于 launch 参数优化。
- CUDA 向量加法与 Triton 向量加法对照：线程标量程序与一段数据的程序；`BLOCK_SIZE` 是元素数，不能直接当作 CUDA threads/block，`num_warps` 另行定义。
- 每行 stable softmax：先最大值，再指数和，再归一化；明确全屏蔽行策略与非连续张量支持范围。
- 分块矩阵乘：一个输出 tile、K 维循环、协作装载、寄存器累加、边界补零、写回；从 5×7 乘 7×6 的非整除形状开始。
- 对照朴素与 tiled CPU 模型；数学 FLOP 相同，重复访问计数不同。矩阵乘教学 kernel 不宣称达到 cuBLAS 性能。
- 融合偏置/激活减少中间写回，也可能增加寄存器压力；Tensor Core 的类型、布局与指令约束作概念桥梁。
- 用一段 NVIDIA CUDA、Triton、库调用的选择表指导读者；CUTLASS、TMA、warpgroup/cluster 的专门优化放在延伸阅读。
- 在分块强度之后定义屋顶线（假想设备，不引用产品规格）；S2 再代入真实硬件。链接 S3 的在线 softmax，避免重复推导 FlashAttention。

交付：tile 消费与复用交互、计算阶段 SVG、由真实小矩阵生成的 tile 图片；CPU softmax/分块矩阵乘验证；CUDA tiled GEMM 和 Triton 向量加法/softmax 可选示例。

练习：非整除边界；每个输出的写入唯一性；融合前后理论字节；为什么 Triton 的 1024 元素 tile 不表示 1024 个硬件线程。

## G4：异步执行、资源约束与测量

拟定路径：`/gpu/measurement/`。先修 G1–G3；屋顶线回读 G3，真实硬件规格见 S2。

核心问题：为什么看起来更并行的代码不一定更快，怎样得到可信的测量？

- latency、throughput 与隐藏延迟：调度可运行 warp 不等于单次访存变快。
- occupancy 的分母与硬件限制：线程、warp、block、register、shared memory 一起约束驻留。概念模型保留设备参数，真实分配粒度和架构限制用 API/工具核对。
- 一个寄存器较多的 kernel 可能减少驻留数量，也可能增加 ILP、减少 spill；occupancy 不作为唯一优化目标。
- host/device 异步、stream 内顺序、event 依赖和默认 stream 差异；多 stream 的并发、拷贝重叠都受依赖、内存与硬件条件限制。
- 预热、JIT 编译、重复次数、CUDA event、CPU 端到端计时、同步位置；区分 kernel 时间与包含传输的应用时间。
- 有效带宽按明确的读写字节计算，不能把它等同于 profiler 记录的物理 HBM 流量。
- Nsight Systems 查时间线，Nsight Compute 查单 kernel；用一次可证伪的问题组织指标。
- memcheck、racecheck、initcheck、synccheck 的职责；错误日志与性能报告不能用人工伪造图片代替。
- 回扣 S6 的 CUDA Graph：减少重复发射开销，不删除真实计算和依赖。

交付：驻留资源交互、stream/event 时间线交互、测量边界 SVG；CPU 教学资源与依赖验证；可选 GPU event 计时脚本和 profiler 命令。

练习：资源中哪个约束先达到；漏同步的计时测到了什么；跨 stream 依赖；为何 100% occupancy 不能保证最快。

## 原始材料与用途

以下均于研究日期访问。写作时再次核对引用段落、工具版本和适用架构；定义与数值保留链接，教学例子和图自行制作，不整段翻译文档。

| 原始资料 | 分配给章节 | 需要核对的内容 |
| --- | --- | --- |
| [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) | G1 | host/device、grid/block、SM 映射、warp 与 lane |
| [Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) | G1、G3 | 完整主机入口、内存分配、启动、错误处理 |
| [Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) | G1、G2 | 索引、shared memory、同步语义 |
| [Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html) | G1、G4 | warp 调度、独立线程调度、资源驻留 |
| [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | G2、G4 | 合并访问、bank、occupancy、计时；访问页面标注版本 13.4 |
| [Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) | G4 | stream、event、依赖、执行重叠的条件 |
| [Triton Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) | G3 | program、元素 tile、mask、launch grid |
| [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | G3 | 行归约与融合；说明示例支持边界 |
| [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | G3 延伸 | tile、stride、尾部处理、调优 |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | G2、G4 | 检查越界、shared-memory hazard、初始化和同步 |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | G4 | 程序时间线、NVTX、CLI、测量区间 |
| [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | G4 | 指标、重放带来的测量边界、占用率与内存分析 |

## 图、交互与代码约定

- 七个交互按章节分配：G1 索引/分支，G2 段/bank，G3 tile，G4 驻留/stream。每个交互改变一个可解释的量，输出手算能核对的数字；不把模拟计数标成毫秒或实测吞吐。
- 复用 React + SVG 的 `LabFrame`、`Range`、`Readout`、`pen` 与 `Formula`。模式用原生单选/选择控件，图中上下标用 SVG `tspan`，正文和结果公式用 KaTeX。
- 静态 SVG 只画具体结构和阶段，主题映射、图注和可访问标签沿用现有组件。PNG 由数值脚本生成地址或 tile 热力图，不使用装饰性芯片照片代替编程结构。
- CPU 代码放 `code/gpu/`，只依赖已有 stdlib/NumPy/PyTorch，固定种子，用 `assert` 验证覆盖、段数、bank、归约与数学结果；加入现有 CI 的运行集合。
- 真实 GPU 示例放 `code/gpu/examples/`，给出完整入口、安装前提、编译/执行命令、边界检查和结果对照。独立运行，不能混入 CPU CI，也不因为导入 Triton 让无 GPU 环境失败。
- CUDA 使用 float32，CPU 教学参考使用 float64；说明比较容差。矩阵用非整除维度和固定数据，补充零长度/单元素等有意义边界。
- 示例查询设备属性，不把一个架构的 SM 数、寄存器和 shared-memory 上限当成所有 GPU 的固定值。

## 多 agent 编写拆分

在本大纲确定后实施，采用现有四个并发槽：主 agent 负责集成，三个子 agent 负责章节与自身资产。

| 执行者 | 第一批负责内容 | 文件所有权 |
| --- | --- | --- |
| Agent A | G1 正文、索引/分支交互、CPU 覆盖验证、CUDA 向量加法 | `execution-model.mdx`，G1 专属组件、SVG 与脚本 |
| Agent B | G2 正文、段/bank 交互、CPU 访存/归约验证、CUDA 转置与归约 | `memory-and-sync.mdx`，G2 专属组件、SVG 与脚本 |
| Agent C | G3 正文、tile 交互、CPU 算子验证、CUDA GEMM 与 Triton 示例 | `kernel-design.mdx`，G3 专属组件、SVG 与脚本 |
| 主 agent | G4 正文及资产、导读、路线入口、导航、CI 和整体验收 | `measurement.mdx`、共享元数据和配置 |

先发放统一记号、来源和算例约束，再并行写作；每个 agent 不改共享配置或其他章节文件。初稿完成后交叉审查：A 检查 G2 同步，B 检查 G3 访存，C 检查 G1 映射；主 agent 检查全路线概念一致性并消除重复。

## 验收与阶段边界

当前环境未找到 `nvidia-smi` 和 `nvcc`。这次规划不假设本机可运行 CUDA；实施时再次探测设备与工具。

1. 规划阶段：交付本大纲、原始资料索引、章节归属、交互与代码清单。
2. 编写阶段：多 agent 交付四章、一篇导读、图与代码；主 agent 完成入口和 CI 集成。
3. CPU 与浏览器验收：运行新增 CPU 脚本、类型检查、构建和所有链接检查；320/390/1440 宽度、明暗主题、滑块与模式变化、公式、非空 SVG、键盘可访问性。
4. GPU 验收：有真实 NVIDIA GPU 时执行 CUDA/Triton 对照与 Sanitizer，再采集实际 timing/profiler 数据；无设备时明确标记 GPU 示例尚未运行，正文不给不存在的测量结果。
5. 发布阶段：汇总变更与验证边界，再按用户后续发布指令提交和部署。
