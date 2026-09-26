# 讲义写作与排版规范

正文唯一来源是 `site/src/content/lessons/` 下的 MDX。页面外壳、目录、编号、上一章/下一章都由站点生成，正文只写内容。

## 路线与章节

- 三条路线各占一个目录：`math/`（M）、`principles/`（P）、`systems/`（S）。路线元数据在 `site/src/lib/tracks.ts`。
- 每条路线的 `index.mdx` 是导读，`order: 0`；章节 `order` 从 1 连续编号，章节代码（M3、P2、S4）由路线字母和 `order` 生成。新增或调整章节只改 frontmatter，不维护任何目录表。
- 一章讲清一个问题，篇幅以 5k–9k 字正文为宜。过长就按问题拆开，过短就并入相邻章节；两章不重复推导同一个结论，后出现的一章链接到先出现的那一节。

frontmatter：

```yaml
title: 相对位置与 RoPE              # 短标题，出现在 h1、侧栏和导航卡片
question: 怎样让注意力分数只通过位置差依赖位置   # 这一章回答的问题，出现在章节列表
description: 一两句摘要，作为页面导语和 meta description
order: 3
code:                                # 可选，本章对应的验证脚本
  - code/principles/position_encoding.py
```

## 章节结构

- 开篇从读者能理解的具体计算问题进入，交代先修与本章结果；不要只列“本章学习目标”。
- 二级标题写成“名称：要解决的问题”，**不写编号**，页面按顺序生成 01、02……。三级标题解释一个具体疑问，不重复泛泛的“直觉/公式/应用”标签。
- 每个核心概念按“为什么需要 → 定义与符号含义 → 分步推导 → 具体数值 → 深度学习或系统中的用途”展开，公式上下都有完整解释。
- 章末依次是“常见误区”（没有值得单列的误区时可省略）和“练习与核对”，之后一句话衔接下一章。引用集中放在最后的“原始资料”或“引用与边界”。
- 练习用 `<details><summary>问题</summary>` 折叠答案，答案包含推理，不只报数字。
- 对话中的修改记录不进入读者正文。

## 链接

- 站内链接一律写成从站点根开始的路径：`[M3 向量与矩阵](/math/linear-algebra/)`、`[P2](/principles/sinusoidal/#极坐标把旋转变成相位相加)`。构建时自动加上部署前缀。
- 提到其他章节时用章节代码加链接，不写“第 06 章”“上一篇”这类会随结构变化失效的说法。
- `pnpm build` 会检查所有站内链接和锚点，断链时构建失败。

## 正文组件

以下组件无需 import，直接在 MDX 中使用；交互实验和系统图按需从 `src/components/labs`、`src/components/diagrams` 导入，并加 `client:visible`。

| 组件 | 用途 |
| --- | --- |
| `<Figure src="math/04-matrix.svg" alt="…">图注</Figure>` | `src/assets/figures/` 中的 SVG。页面自动编号“图 N”，内联后跟随明暗主题 |
| `<KeyEq label="…">$$…$$</KeyEq>` | 一章围绕的少数关键公式；普通行间公式不加框 |
| `<Callout tone="note \| warn" title="…">` | 路线提示、证明边界、运行限制；一般旁注直接用 Markdown 引用块 |
| `<CodeFile path="code/…py" />` | 从 `code/` 读入完整脚本，正文与文件不会漂移 |
| `<Steps>`、`<StatGrid>`、`<Panels>` + `<Panel>` | 横向流程、少量关键数字、并列对比 |
| `<SourceNote>` | 指向上游源码具体行的锚点 |
| `<ChapterList track="…" />` | 路线导读中的章节列表，由内容集合生成 |

## 图

- 图放在所解释概念旁，图后有正文读图说明；图注用中文说明怎样读轴、线、箭头。
- SVG 不写标题或编号，也不画整幅背景：标题由图注承担，背景和编号由页面提供。颜色只用固定调色板（见 `Figure.astro` 中的映射），构建时会换成主题色；新增颜色要同时加进映射。
- 函数图由 `code/math/plot_style.py` 统一中文字体、配色和字号，输出到 `site/src/assets/figures/math/`。重新作图需要系统有中文字体，例如 Noto Sans CJK SC。
- Mermaid 图写在 ```` ```mermaid ```` 代码块里，由浏览器按当前主题绘制；节点多时优先 `flowchart LR` 并控制在一屏宽度内。

## 样式

站点只用 Tailwind CSS 一套样式系统。颜色、字号、间距 token 定义在 `site/src/styles/global.css` 的 `@theme` 中，组件和页面在标记里写 utility class；只有 Markdown、KaTeX、Shiki、Mermaid 生成的、无法写 class 的标记，才在 `global.css` 中用 `@apply` 补规则。不要新增独立的 CSS 文件，也不要在 token 之外使用颜色和字号。

## 术语与记号（数学基础路线，其他路线沿用）

- 统一写“正态分布（高斯分布）”“前向传播”“反向传播”“偏导数”“梯度”“雅可比矩阵”“海森矩阵”。“均值”必须说明对象：样本均值是观测平均，总体均值是分布期望。
- 首次出现写中文名称与英文：小批量（mini-batch）、批大小（batch size）、未归一化分数（logits）、独热编码（one-hot）、自动微分（automatic differentiation）。后文使用中文；API 和代码标识符保留原文。sigmoid、softmax、ReLU 为函数名称，首次解释其含义。
- sigmoid 记作 $s(z)$，标准差用 $\sigma$；类别数 $C$；稳定 softmax 的最大分数记 $a$；海森矩阵写作 $\mathcal H$，隐藏激活保留 $H$。
- 标量、向量小写（默认列向量），矩阵大写；随机变量用 $U,V$，避免与批输入 $X$ 混淆。
- 单样本 $x\in\mathbb R^d$，批输入 $X\in\mathbb R^{B\times d}$ 按行存放样本；$W\in\mathbb R^{m\times d}$，$z=Wx+b$，$Z=XW^\top+b$，与 `torch.nn.Linear` 一致。模型原理路线沿用注意力论文的 $Q=XW_Q$、$W_Q\in\mathbb R^{d\times d_h}$，两者互为转置布局，换算时先对齐形状。
- 梯度与被求导参数同形状；$J_{ij}=\partial f_i/\partial x_j$，反向传播为 $J^\top g$。
- 损失单样本 $\ell$，批平均 $L=(1/B)\sum\ell_i$；学习率 $\eta$；$\ln$ 或 $\log$ 都指自然对数。
- 概率 $P(U=u)$，密度 $p(u)$，期望 $\mathbb E[U]$，方差 $\operatorname{Var}(U)$；$\mathcal N(\mu,\sigma^2)$ 第二参数是方差。明确 `ddof=0` 与 `ddof=1`。
- 位置下标用 $i,j,p$，虚数单位写作正体 $\mathrm i$；导数的撇号写 `f'`，不写 `f\prime`。
- 公式用 `$...$` 和独立的 `$$...$$`，转置 `\top`，逐元素乘 `\odot`。

## 代码与验证

- 每章的关键推导配可在 CPU 上独立运行的 NumPy 或 PyTorch 示例：固定随机种子，float64，用 assert 核对推导。代码紧接对应推导，说明每个关键操作对应哪条公式、预期结果是什么。
- 完整验证脚本放在 `code/`，由 frontmatter 的 `code` 字段和 `<CodeFile>` 引用；CI 每次推送都运行全部脚本。
- 系统路线引用外部数字（产品规格、论文数据）时写明出处与读取日期；没有实测数据的图注明“只示意形状”。
