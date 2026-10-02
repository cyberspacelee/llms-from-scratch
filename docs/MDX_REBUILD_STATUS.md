# MDX 与 Python 统一重构

目标：逐篇改进内容与阅读顺序，统一图表与交互，合并章节代码和 Transformer Lab；不保留旧目录、旧接口或重复模型。

重构前的问题与逐篇设计见 [全量审查](MDX_FULL_REVIEW.md)。当前教学边界见 [课程设计](CHAPTER_BLUEPRINTS.md)，写作和组件约定见 [排版规范](CONVENTIONS.md)，Python 用法见 [开发说明](PYTHON_DEVELOPMENT.md)。

## 已实施的结构

- 唯一根 `pyproject.toml` / `uv.lock`，唯一 `src/llms_from_scratch/Transformer`。`examples/` 保存章节实践与独立参考；原 `code/` 与 `transformer-lab/` 删除。
- 基础与现代模型作为同一实现的明确配置；组装、真实文本训练、SFT、LoRA、DPO、分布蒸馏、RL 和生成共享模型接口。
- 原理分开归一化、头共享、门控 FFN，并增加经典 encoder/cross-attention 分支。完整二维 trace 在组装章选读。
- 谱方向与稳定步长、SVD/低秩/条件数拆开；checkpoint 从文本训练主线分出；后训练独立 H 路线。
- 混合状态恢复边界与 ring 存储从架构原理移至系统选读；MTP、序列压缩、多残差流、条件记忆、共享 KV 接入 MDX 研究选读。
- GPU 主线保留四篇计算链，三维索引、归约、Triton softmax 与模型 profiling 分出实践。
- 学习路径与目录顺序分开；每篇声明先修和选读属性，校验不存在断链或循环。前后导航按目标路径生成，避免跨所有学科的一条强制链。

## 共享视觉与源码

- `Lab` 统一输入、选择、按钮与读数；`DataViews` 复用矩阵、token 与计算账；`SvgCanvas` / `FigureShell` 使用可读字号与局部滚动。
- 大型实验聚合文件拆为单实验模块。目录使用静态 DOM，完整代码折叠，正文从真实函数或 region 摘录，支持 Python AST 与 CUDA region。
- 同内容的手机/桌面 SVG 共用一个图注和编号；SVG 实例 ID 独立；明暗配色与 Markdown/公式排版采用统一 token。
- Python 参考结果导出到 `site/src/data/reference-traces.json`，TypeScript 对照采样、RoPE/矩形缓存、online softmax、有效损失和向量 autograd；构建前同时检查实际结果与源码哈希，阻止漂移。

## 独立复审

作者按目录实施，随后轮换审查未参与编写的领域，按实际页面与源码复核。每篇结论记录在以下报告，发现的问题修正后复测：

- [内容复审](review-rounds/content-review.md)：逐篇内容、数值、先修与实际源码摘录。
- [Python 复审](review-rounds/python-review.md)：统一接口、有效目标、缓存、恢复、后训练与独立参考。
- [视觉与 GPU 复审](review-rounds/visual-gpu-review.md)：九篇 GPU 页面、共享组件与图表规范。
- [实际控件复核](review-rounds/browser-controls-review.md)：25 项真实 UI 操作与开发服务器并发验证。
- [全站浏览器验收](review-rounds/browser-final-review.md)：所有页面的三种屏宽、明暗主题与交互加载。

已发现并修正的问题包括：旧 SFT/checkpoint 数值及编号、`ModelOutput` 被当作 tensor、F4 实验与正文损失不一致、蒸馏忽略非有限行导致 NaN 梯度、float32 教师零概率导致 NaN，以及生成接口静默忽略不合法 source 参数。

## 验证记录

全部实施与独立复审完成。当前为 86 篇 MDX（78 篇正文、8 篇主题索引），五条学习路径，91 个独立实验模块、94 处 MDX 实验挂载。发现项均已修正并复核。

| 检查 | 最终结果 |
| --- | --- |
| Python 单元测试 | 26 项：25 通过，1 项 CUDA 跳过 |
| 章节 CPU 实验 | 59 项全部通过 |
| Python 静态检查 | Ruff 通过；140 个文件格式检查通过；uv 锁文件有效 |
| Python / TypeScript 数值对照 | 54 项通过，包含真实计算结果与源码哈希检查 |
| 站点检查 | `pnpm check` 通过，Astro 零错误、零警告；各实验模型检查通过 |
| 生产构建与链接 | 88 个页面构建通过；4112 个内部链接有效 |
| 全站实际浏览器 | 87 个入口 × 三种屏宽，共 261 轮，每轮检查明暗主题；零发现项 |
| 实际实验控件 | 25 项带具体数值预期的 UI 断言全部通过 |

GPU 环境缺少 CUDA/nvcc；CUDA/Triton 的编译、设备执行与性能未验证。本次通过的是相关源码、CPU 参考和索引/资源模型检查。
