# GPU 编程实施与验收

依据：[GPU 编程规划](GPU_PROGRAMMING_PLAN.md) 与 [讲义规范](CONVENTIONS.md)。日期：2026-09-27。

## 正文与资产

| 章节 | 正文与核心内容 | 配套资产 |
| --- | --- | --- |
| G1 执行模型 | 约 5900 汉字；host/device、thread/block/grid、warp/lane、SM、二维索引与 grid-stride | 索引与分支两项 SVG 交互、执行映射 SVG、CPU 覆盖核对、CUDA 向量加法 |
| G2 内存与同步 | 约 5700 汉字；地址段、bank、广播、归约、装载/消费/复用同步 | 地址段与 bank 两项 SVG 交互、生命周期 SVG、地址热力图 PNG、CPU 计数、CUDA 转置与归约 |
| G3 Kernel 设计 | 约 5900 汉字；输入契约、CUDA/Triton、masked softmax、非整除 tiled GEMM、融合 | tile SVG 交互、阶段 SVG、矩阵 PNG、CPU 数值核对、CUDA GEMM 与 Triton 示例 |
| G4 测量与优化 | 约 5700 汉字；延迟隐藏、驻留资源、stream/event、计时边界、工具职责 | 驻留与 stream 两项 SVG 交互、测量边界 SVG、CPU 资源/依赖核对、GPU event 脚本 |

新增一篇导读；G 路线位于训练与系统之间。S2、S3、S6 和系统导读有明确回读入口；原有章节编号与 URL 保留。首页、导航、README 与 CPU CI 已接入。

静态 SVG 与交互使用原有主题和框架。两张 PNG 由地址表/小矩阵数值生成，生成脚本在 `code/gpu/plots/`。代码组件支持直接读取 `.py` 与 `.cu`，避免正文抄录后漂移。

## 编写与交叉审查

三个子 agent 分别负责 G1–G3；主 agent 负责 G4 与共享配置。完成 G1→G2 同步、G2→G3 访存、G3→G1 索引，以及 G1→G4 测量的交叉审查。

修正了归约校验对 NaN 的遗漏、Triton launch 的输入设备上下文、event 首次初始化计入采样的问题，以及将两块输入 tile 称为 double buffering 的术语混淆。图中按原生 SVG 排版上下标，React 公式复用 KaTeX。

集成检查还修正了开发服务器子路径下 PNG 的地址，以及共用滑块标签被 `<output>` 占用的问题；标签现在显式关联输入框。warp 偶数分支模式下禁用无效的阈值输入。

## 验收清单

- 四个 CPU 脚本：索引覆盖和维度展开、地址段/bank/归约、稳定 masked softmax/分块 GEMM、驻留资源/stream 依赖/字节换算。
- 浏览器数值模型：运行 `site/scripts/check-gpu-models.mjs`，覆盖资源瓶颈、零驻留、部分 warp 与缺少显式 event 的情形；纳入 `pnpm check`。
- 类型与构建：`pnpm check`、`pnpm build`、内部链接和章节编号。
- 页面与导航：320/390/768/1024/1440 宽度下的新导读、四章和首页；无整页横向溢出，数学解析无错误，图片加载成功。
- 七项交互：键盘操作所有滑块、模式、容量输入与 event 切换；核对默认值和边界状态；390/1440、明暗主题截图复核 SVG 与文本。
- 静态资产：非空 SVG、图片真实尺寸与图注、边界及同步含义；公式使用数学排版。

最终结果：四个 CPU 脚本与浏览器模型断言均通过；`pnpm check` 检查 57 个文件，零错误、零警告；`pnpm build` 生成 54 页，2553 条内部链接全部有效。首页、导读和四章在五种宽度下均无整页横向溢出、KaTeX 错误或图片缺失。七项交互在 390/1440 宽度通过键盘与边界数值检查，明暗主题共 28 张交互截图完成复核；浏览器运行错误为零，SVG 文字越界为零。八份新增 Python 文件均已通过语法解析，`git diff --check` 通过。

## GPU 验证边界

当前环境未发现 NVIDIA GPU 工具、`nvcc` 或可用 CUDA 设备。CUDA 与 Triton 示例已作源码审查，Python 示例已解析语法，尚未在真实 GPU 编译/运行；没有 Sanitizer、Nsight 或实际 kernel 时间报告。

CPU 断言证明教学算法或模型关系。驻留参数、时间线单位、地址段与 bank 模型不是实际设备的完整模拟；有效带宽算例使用明示的假设时间。可选 GPU event 脚本会在读者的实际设备上生成记录，不提供编造的示例性能。
