# 讲义写作与排版规范

正文唯一来源是 `site/src/content/lessons/` 下的 MDX。页面外壳、目录、编号、上一章/下一章都由站点生成，正文只写内容。

章节的教学顺序、贯穿案例、术语取舍和逐页验收见 [课程章节重写规范](EDITORIAL_STANDARD.md)。本文件主要约定内容格式、链接、公式和技术排版。

## 路线与章节

- 各路线占一个目录：`math/`（M）、`frameworks/`（F）、`principles/`（P）、`training/`（T）、`post-training/`（H）、`gpu/`（G）、`systems/`（S）、`advanced/`（A）。主题元数据在 `site/src/lib/tracks.ts`；按目标选择的学习路径统一定义在 `site/src/lib/curriculum.ts`。
- 每条路线的 `index.mdx` 是导读，`order: 0`；章节 `order` 从 1 连续编号，章节代码（M3、P2、S4）由路线字母和 `order` 生成。新增或调整章节只改 frontmatter，不维护任何目录表。
- 一章讲清一个问题，篇幅由该问题需要的推导与练习决定。若存在两个独立的先修链或验收目标，就按问题拆开；两章不重复推导同一个结论，后出现的一章链接到先出现的那一节。

frontmatter：

```yaml
title: 相对位置与 RoPE              # 短标题，出现在 h1、侧栏和导航卡片
question: 怎样让注意力分数只通过位置差依赖位置   # 这一章回答的问题，出现在章节列表
description: 一两句摘要，作为页面导语和 meta description
order: 3
code:                                # 可选，本章对应的验证脚本
  - examples/principles/position_encoding.py
prerequisites:
  - principles/attention-order
optional: false                       # 研究与专项实践分支设为 true
```

## 章节结构

- 开篇用一段话从具体计算问题进入，交代先修和本章主线；避免重复页面摘要。
- 二级标题写成“名称：要解决的问题”，**不写编号**，页面按顺序生成 01、02……。三级标题解释一个具体疑问，不重复泛泛的“直觉/公式/应用”标签。
- 每个核心概念按读者的计算障碍引入，并回到同一贯穿案例；定义、推导、具体数值和用途须构成连续的推理，公式上下都有完整解释。
- 数学路线首次使用核心概念时给出定义与成立条件；`<Definition>` 只用于确实需要读者停下来辨认的定义，不按概念数量机械添加。来源链接指向具体资料，正文用自己的话解释，不复制条目；涉及库行为还须核对官方 API 文档。
- 手算例跟随相应推导；进阶证明置于主问题得到回答之后，并以“选读”标明。读者仅扫标题与例题，也应能掌握主线。
- 章末依次是“常见误区”（没有值得单列的误区时可省略）和“练习与核对”，之后一句话衔接下一章。引用集中放在最后的“原始资料”或“引用与边界”。
- 练习用 `<details>` 与下一行的 `<summary>问题</summary>` 折叠答案，答案包含推理，不只报数字；两个标签在 MDX 中必须分行。
- 对话中的修改记录不进入读者正文。

## 链接

- 站内链接一律写成从站点根开始的路径：`[M6 向量与矩阵](/math/linear-algebra/)`、`[P6 sin/cos](/principles/sinusoidal/#固定偏移sincos-两个坐标怎样一起变化)`。构建时自动加上部署前缀。
- 提到其他章节时用章节代码加链接，不写“第 06 章”“上一篇”这类会随结构变化失效的说法。
- `pnpm build` 会检查所有站内链接和锚点，断链时构建失败。

## 正文组件

以下组件无需 import，直接在 MDX 中使用；交互实验和系统图按需从 `src/components/labs`、`src/components/diagrams` 导入，并加 `client:visible`。

| 组件 | 用途 |
| --- | --- |
| `<Figure src="math/04-matrix.svg" alt="…">图注</Figure>` | `src/assets/figures/` 中的 SVG。页面自动编号“图 N”，内联后跟随明暗主题 |
| `<KeyEq label="…">$$…$$</KeyEq>` | 一章围绕的少数关键公式；普通行间公式不加框 |
| `<Definition term="…" source="…">定义</Definition>` | 数学路线的核心定义、适用条件与来源 |
| `<Callout tone="note \| warn" title="…">` | 路线提示、证明边界、运行限制；一般旁注直接用 Markdown 引用块 |
| `<CodeFile path="examples/…py" symbol="函数名" />` | 从 `examples/` 或 `src/llms_from_scratch/` 读取实际源码。正文用 `symbol`（支持数组）摘录 Python 函数/类，或用 `region` 摘录标记块；完整文件留在折叠区。CUDA 使用 `// region NAME` 标记 |
| `<Steps>`、`<StatGrid>`、`<Panels>` + `<Panel>` | 横向流程、少量关键数字、并列对比 |
| `<SourceNote>` | 指向上游源码具体行的锚点 |
| `<ChapterList track="…" />` | 路线导读中的章节列表，由内容集合生成 |

## 图

- 图放在所解释概念旁，图后有正文读图说明；图注用中文说明怎样读轴、线、箭头。
- SVG 不写标题或编号，也不画整幅背景：标题由图注承担，背景和编号由页面提供。颜色只用固定调色板（见 `Figure.astro` 中的映射），构建时会换成主题色；新增颜色要同时加进映射。
- 函数图由 `examples/math/plot_style.py` 统一中文字体、配色和字号，输出到 `site/src/assets/figures/math/`。重新作图需要系统有中文字体，例如 Noto Sans CJK SC。
- Mermaid 图写在 ```` ```mermaid ```` 代码块里，由浏览器按当前主题绘制；节点多时优先 `flowchart LR` 并控制在一屏宽度内。

## 样式

站点只用 Tailwind CSS 一套样式系统。颜色、字号、间距 token 定义在 `site/src/styles/global.css` 的 `@theme` 中，组件和页面在标记里写 utility class；只有 Markdown、KaTeX、Shiki、Mermaid 生成的、无法写 class 的标记，才在 `global.css` 中用 `@apply` 补规则。不要新增独立的 CSS 文件，也不要在 token 之外使用颜色和字号。

## 术语与记号（数学基础路线，其他路线沿用）

- 统一写“正态分布（高斯分布）”“前向传播”“反向传播”“偏导数”“梯度”“雅可比矩阵”“海森矩阵”。“均值”必须说明对象：样本均值是观测平均，总体均值是分布期望。
- 首次出现的术语解释其作用，之后采用 [术语规范](EDITORIAL_STANDARD.md#术语与语言) 中的默认写法，不要求把常用英文术语强译后反复使用。API 和代码标识符保留原文。sigmoid、softmax、ReLU 为函数名称，首次解释其含义。
- sigmoid 记作 $s(z)$，标准差用 $\sigma$；类别数 $C$；稳定 softmax 的最大分数记 $a$；海森矩阵写作 $\mathcal H$，隐藏激活保留 $H$。
- 标量、向量小写（默认列向量），矩阵大写，不强制粗体；维度与数量允许使用约定的大写字母 $B,T,C$。高阶张量沿用大写，必须说明各轴。随机变量用 $U,V$，避免与批输入 $X$ 混淆。行向量显式写转置；形状与元素个数分开表达。
- 单样本 $x\in\mathbb R^d$，批输入 $X\in\mathbb R^{B\times d}$ 按行存放样本；$W\in\mathbb R^{m\times d}$，$z=Wx+b$，$Z=XW^\top+b$，与 `torch.nn.Linear` 一致。模型原理路线沿用注意力论文的 $Q=XW_Q$、$W_Q\in\mathbb R^{d\times d_h}$，两者互为转置布局，换算时先对齐形状。
- 梯度与被求导参数同形状；$J_{ij}=\partial f_i/\partial x_j$，反向传播为 $J^\top g$。
- 损失单样本 $\ell$，批平均 $L=(1/B)\sum\ell_i$；学习率 $\eta$；$\ln$ 或 $\log$ 都指自然对数。
- 概率 $P(U=u)$，密度 $p(u)$，期望 $\mathbb E[U]$，方差 $\operatorname{Var}(U)$；$\mathcal N(\mu,\sigma^2)$ 第二参数是方差。明确 `ddof=0` 与 `ddof=1`。
- 位置下标用 $i,j,p$，虚数单位写作正体 $\mathrm i$；导数的撇号写 `f'`，不写 `f\prime`。
- 公式用 `$...$` 和独立的 `$$...$$`，转置 `\top`，逐元素乘 `\odot`。
- 跨路线维度：批大小 $B$、序列长度 $T$、历史缓存长度 $T_{\mathrm{past}}$、本次新增长度 $T_{\mathrm{new}}$；隐藏维度 $D$、头维度 $D_h$、前馈维度 $D_{\mathrm{ff}}$；查询头数 $H_q$、KV 头数 $H_{\mathrm{kv}}$、层数 $L$、词表大小 $V$。代码中的 `H,D` 等轴名在正文中映射到这些记号，不改外部 API。
- 块大小 $T_b$、每元素字节数 $b_{\mathrm{elem}}$、张量并行度 $n_{\mathrm{TP}}$；频率底数 $\beta$、二维分块编号 $r$、角频率 $\omega_r$、位置转角 $p\omega_r$；采样温度 $\tau$。RoPE 原论文的 $\theta_r$ 对应本文 $\omega_r$。局部临时变量可以保留，但不要占用跨路线记号而不解释。
- 向量欧氏范数保留下标 $\|x\|_2$；矩阵 Frobenius 范数用 $\|W\|_F$。单位化、RMSNorm、LayerNorm、softmax 分别说明实际运算，不用“归一化”代替定义。
- 文字下标和上标用 `\mathrm{...}`，函数名用 `\operatorname{...}`；公式使用短数学符号，旁边说明代码字段映射。拼接等非算术操作必须定义符号。
- 正文统一“查询、键、值、隐藏状态”；batch、logits、token、RoPE、KV cache、prefill、decode 等按 [术语规范](EDITORIAL_STANDARD.md#术语与语言) 使用并首次解释；源码、API、论文名称保持原文。
- 性能计量区分 FLOP（运算量）、FLOP/s（速率）和 FLOP/byte（算术强度）；一次融合乘加（FMA）计 2 FLOP。Adam 状态写“一阶矩、二阶原始矩”，不将梯度平方的滑动平均直接称为方差。服务指标区分首 token 延迟 TTFT、逐次输出间隔 ITL、请求级平均每输出 token 时间 TPOT，说明测量边界与聚合方式。

## 代码与验证

- 每章的关键推导配可在 CPU 上独立运行的 NumPy 或 PyTorch 示例：固定随机种子，float64，用 assert 核对推导。代码紧接对应推导，说明每个关键操作对应哪条公式、预期结果是什么。
- 模型只有 `src/llms_from_scratch/` 一套实现；手算与独立数值核对放在 `examples/`，由 frontmatter 的 `code` 字段和 `<CodeFile>` 引用；使用 `uv sync --locked` 建立唯一根环境，命令为 `uv run python -m examples.<主题>.<模块>`；CI 运行所有 CPU 实验，GPU 编译和性能须在有硬件的环境另行实测。
- 系统路线引用外部数字（产品规格、论文数据）时写明出处与读取日期；没有实测数据的图注明“只示意形状”。

## 交互实验与共享图表

- SVG 使用 `FigureShell`/`SvgCanvas`；文字在移动端保持可读，宽图在自身视口滚动。外层页面不得横向溢出。图形颜色取主题 token，操作控件复用 `Lab` 中的控件，表格/向量复用 `DataViews`。
- 一章一个交互组件文件，导入具体文件；索引路线图使用静态 `CourseMap`，避免为目录加载 React。
- 章节实验的初始值、公式和 Python 必须对应。核心数值由 `scripts/export_site_traces.py` 从唯一实现导出到 `site/src/data/reference-traces.json`；TS 检查与源哈希共同在构建前阻止漂移。更新公式先运行导出，再执行 `pnpm check` 和 `pnpm build`。
- 移动/桌面图若只是同一内容的两种排布，用 `Figure mobileSrc`，只生成一个编号与图注。
