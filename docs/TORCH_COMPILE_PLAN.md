> 重构前的规划或审查记录。当前结构与验证结果见 [实施记录](MDX_REBUILD_STATUS.md)，教学约定见 [当前课程设计](CHAPTER_BLUEPRINTS.md)。

# torch.compile：大纲与实施

设计日期：2026-09-28。沿用 [讲义规范](CONVENTIONS.md)，正文放在数组与框架路线 F7 `frameworks/torch-compile`，先设计本大纲，再按表中顺序补充正文。

## 放置位置

F3 已解释 shape、stride 与 device，F4 解释自动微分，F5/F6 解释模块与训练状态。F7 在这些接口之上回答：同一段张量程序怎样被捕获、求导、优化和重复执行。GPU 性能细节回读 G3/G4，LLM 应用连接 P8 的现代 Decoder、P10 的 KV cache 与 S6 的执行器。基础读者可以先完成 F1–F6，等读完 G3/G4 再回到 F7 的性能部分。

新增一章足以建立完整调用链；不另建编译器路线，不重讲自动微分、Triton 线程布局或 CUDA Graph 缓冲区管理。原有编号与 URL 保留，导航由 frontmatter 自动生成。

## 正文大纲

| 顺序 | 要回答的问题 | 原理与技术细节 | 核对方式 |
| --- | --- | --- | --- |
| 1 | 为什么少量算术也可能有较大开销 | eager 的解释、dispatch、launch、中间张量；融合前后流量 | 四个逐元素算子的 8N 与 2N 元素读写模型 |
| 2 | compile 调用后发生什么 | 惰性编译、Dynamo、FX、AOTAutograd、Inductor、设备代码与外部库 | 编译管线 Mermaid 图；三个 backend 的定位 |
| 3 | Python 怎样变成可优化的图 | frame/字节码、符号执行、FakeTensor 元数据、guards 与缓存 | 自定义计数 backend 观察捕获与复用 |
| 4 | 反向传播也能编译吗 | functionalization、ATen 分解、联合图、保存/重算与前后向分区 | 多项式前向与解析梯度，aot_eager 对照 |
| 5 | 图节点怎样成为 kernel | 循环 IR、融合、依赖、归约、布局、矩阵乘库与 autotune | 连接 G3，明确一张图不等于一个 kernel |
| 6 | 为什么编译区域被切开 | graph break、数据分支、日志、disable 与 fullgraph | 显式禁用函数；分段成功与完整捕获失败 |
| 7 | 为什么同一个函数反复编译 | guard failure、静态/符号 shape、dynamic 参数、形状分支与 bucket | 静态/动态尺寸、dtype 改变的捕获次数 |
| 8 | LLM 应该编译哪里 | RMSNorm/SwiGLU、SDPA 边界、训练前后向、prefill/decode、KV 地址与 padding mask | 连接现代 Decoder、缓存与 nano-vLLM |
| 9 | 怎样选参数与定位错误 | backend 分层、mode、TORCH_LOGS、output_code、数值容差 | 可直接运行的诊断命令和最小复现步骤 |
| 10 | 怎样判断收益 | 冷启动/稳态、完整训练预热、同步、编译摊销与端到端收益 | 30 秒编译、每次省 2 ms 的回本手算 |
| 11 | 如何复习和验证 | 完整 CPU 脚本、误区、折叠练习、官方资料 | CPU 默认检查；可选真实 CPU Inductor 检查 |

## 示例与边界

使用已有 PyTorch，固定 CPU float64；贯穿例子为四个逐元素操作和多项式 `x*x+3*x`。默认脚本用自定义 backend 和 `aot_eager`，分别检验 Dynamo 缓存行为与 AOTAutograd，避免把 CI 的系统 C++ 编译器当作基础依赖。`--inductor` 选项额外运行真实 CPU 代码生成，需要兼容的本地 C++ 工具链；默认检查不证明 Inductor 性能。

不提供无硬件依据的加速倍数。GPU 计时按 G4 的同步边界执行，版本、输入、dtype、模式、缓存和编译成本必须记录。graph break 与重编译分开解释，不保证 dynamic=True 只生成一份代码，不把 fullgraph=True 等同于单 kernel，不把 compile 等同于模型导出。

## 资料与验收

资料读取日期 2026-09-28；以 PyTorch 2.14 官方文档核对，并注明 2.x 版本差异。

- [API](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)：参数、模式、缓存与完整图捕获。
- [Dynamo](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_dynamo_overview.html)、[核心概念](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/compile/programming_model.dynamo_core_concepts.html)：捕获、guards、graph break。
- [动态形状](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html)：符号尺寸、特殊化与约束。
- [编译器概览](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html)、[入门教程](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)：编译流程、后端与训练。
- [排障](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_troubleshooting.html)、[完整实验](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html)：日志、正确性与冷启动/稳态测量。

验收：默认 CPU 脚本与可选 Inductor 检查；站点类型检查；构建和内部链接；移动/桌面正文、公式和 Mermaid 渲染；git diff --check。结果补记于本文件末尾。

## 实施与验收结果

2026-09-28 完成 F7 正文约 6800 汉字、两张 Mermaid 图、五道折叠练习与一份 CPU 验证脚本；框架导读、F6 衔接、GPU 导读、S6 及 README 已接入。导航和 CI 使用已有内容集合与脚本通配符，无需新增配置。

- PyTorch 2.14.0+cpu：默认与 `--inductor` 检查通过；静态捕获 2 份、动态捕获 1 份，dtype guard、分段捕获、完整图失败及前后向数值均通过。
- `pnpm check`：98 个文件，0 errors、0 warnings、0 hints，既有教学模型检查通过。
- `pnpm build`：62 页，3134 条内部链接有效。构建仍提示 MDX head-inject 指令与部分大体积 chunk；没有构建错误。
- 浏览器：390/1440 宽度、浅色/深色四种组合通过；无横向页面溢出或公式错误，两张图渲染成功，折叠练习支持鼠标与 Enter，导读包含新章入口。截图已目视核对。
- `git diff --check` 通过。未运行 GPU，不提供 GPU 实测加速结论。

验收时本地预览使用 `http://127.0.0.1:4322/llms-from-scratch/frameworks/torch-compile/`。用户于 2026-09-28 要求提交、推送并停止后台预览，按仓库流程推送 master，由 CI 验证后部署 GitHub Pages。
