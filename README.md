# LLMs From Scratch

一组理解大语言模型原理与推理实现的中文讲义，通过数学推导、小维度算例、交互图示和 PyTorch 数值验证串起概念与代码。

当前内容围绕两条相连主线展开：**数学基础 → 位置编码 → KV cache 正确性 → 推理引擎实现**，以及 **AI Infra：一次生成的账单 → 架构与负载 → 加速器上限 → kernel 与测量 → 推理引擎 → 集群**。适合具备 Python 和基础深度学习知识的读者；数学系列从概率、矩阵和微积分开始补齐训练与推理所需工具。AI Infra 的正文是按这条路线重写的手算和图，定义和数字指向论文、厂商文档与公开模型配置。

## 阅读路线

| 顺序 | 讲义 | 核心问题 |
| --- | --- | --- |
| 数学导读 | [术语、公式与张量形状](foundations/mathematics.html) | 八章的先修关系、符号、损失与代码约定。 |
| 数学 01 | [概率与统计](foundations/math-01.html) | 样本均值、期望、方差、协方差与批量统计。 |
| 数学 02 | [分布与信息量](foundations/math-02.html) | 正态分布、最大似然、交叉熵与 KL。 |
| 数学 03 | [向量与矩阵](foundations/math-03.html) | 向量、矩阵乘法、转置、广播与线性层。 |
| 数学 04 | [微积分与梯度](foundations/math-04.html) | 导数、积分、偏导、方向导数与优化。 |
| 数学 05 | [矩阵求导](foundations/math-05.html) | 梯度形状、批量线性层、手工与自动求导。 |
| 数学 06 | [链式法则与反向传播](foundations/math-06.html) | 计算图、梯度累加与自动微分。 |
| 数学 07 | [神经网络训练](foundations/math-07.html) | 激活函数、两层网络、反向传播与 XOR。 |
| 数学 08 | [旋转与位置编码](foundations/math-08.html) | 正交变换、复数、矩阵指数与 RoPE。 |
| 01 | [位置编码：从第一性原理推导](training/position-encoding.html) | 注意力如何表示顺序？sin/cos 和 RoPE 如何引入位置关系？ |
| 02 | [位置编码与 KV cache](inference/position-encoding-and-kv-cache.html) | 为什么历史 K/V 可以复用？位置、掩码和缓存如何保持一致？ |
| 03 | [nano-vLLM：从一条请求到一个推理引擎](inference/nano-vllm-from-zero-to-mastery.html) | 调度、分页缓存、张量并行与 CUDA Graph 如何协作？ |
| AI Infra 导读 | [从一次请求到一簇机器](astro-site/src/content/lessons/ai-infra.mdx) | 八章为什么按记账、架构、硬件、引擎、集群来拆？ |
| AI Infra 01 | [一次生成的三本账](astro-site/src/content/lessons/ai-infra-01.mdx) | 时间、容量和数据搬移各记在哪里？ |
| AI Infra 02 | [架构改写哪一笔账](astro-site/src/content/lessons/ai-infra-02.mdx) | GQA、分数矩阵和专家结构怎样改写容量？ |
| AI Infra 03 | [同一模型的两套账单](astro-site/src/content/lessons/ai-infra-03.mdx) | prefill、decode 和训练为什么不是同一笔账？ |
| AI Infra 04 | [算力、带宽与片上存储](astro-site/src/content/lessons/ai-infra-04.mdx) | 一张加速器先给出哪些上限？ |
| AI Infra 05 | [让读进来的数据被复用](astro-site/src/content/lessons/ai-infra-05.mdx) | 分块、融合和 Tensor Core 怎样少访问显存？ |
| AI Infra 06 | [先写下可证伪的假设](astro-site/src/content/lessons/ai-infra-06.mdx) | 时间线、单 kernel 和编译产物各自回答什么？ |
| AI Infra 07 | [把空转填上](astro-site/src/content/lessons/ai-infra-07.mdx) | 调度、分页 KV、量化和投机解码各改变哪一笔账？ |
| AI Infra 08 | [从一卡到一簇](astro-site/src/content/lessons/ai-infra-08.mdx) | 并行、网络、训练和端边云怎样为容量支付通信？ |

数学篇提供独立的分章推导和图解；位置编码与 KV cache 篇从原理推到数值验证；推理系统篇沿源码调用链展开，并提供调度与缓存交互实验。nano-vLLM 讲义基于固定源码快照，具体版本见页面标注。

## 打开讲义

下载或克隆仓库后，直接用浏览器打开上表中的 HTML 文件，无需构建或启动服务器。页面支持桌面和手机阅读，公式、代码高亮及 Mermaid 图的完整渲染需要联网加载 CDN 资源。

在 GitHub 上阅读时，可直接查看 Markdown 原文：[数学基础](foundations/mathematics.md)、[位置编码](training/position-encoding.md)、[KV cache](inference/position-encoding-and-kv-cache.md)。HTML 中的交互需在浏览器打开本地文件后使用。

AI Infra 八章在 `astro-site` 中。进入该目录执行 `pnpm install` 与 `pnpm dev`，浏览器打开本地地址后从导读进入。页面里的 Mermaid 图在构建后仍由浏览器加载 CDN 上的 Mermaid 绘制；公式同样依赖 CDN。图中的 SVG 随站点静态提供。

## 运行数值验证

需要 Python、NumPy 和 PyTorch，CPU 即可运行。数学绘图另需 Matplotlib；依赖见 `foundations/requirements.txt`。在仓库根目录执行：

```sh
python "foundations/probability.py"
python "foundations/linear_algebra.py"
python "foundations/calculus.py"
python "foundations/neural_network.py"
python "training/position_encoding.py"
python "inference/position_encoding_cache.py"
```

- **数学验证**：均值与方差、信息量、线性层梯度、有限差分、计算图反传，以及两层网络手工梯度和 XOR 拟合。
- **位置编码验证**：置换等变、sin/cos 平移、相对位置核、RoPE 范数与梯度。
- **缓存验证**：比较整段、逐 token、prefill 后解码和分块计算，并用错误位置编号作负面对照。

这些脚本验证教学算例与单层注意力性质；数学 07 包含四点 XOR 训练，不包含大模型训练或端到端文本生成。运行 nano-vLLM 本身所需的 GPU、模型权重和依赖，见第三篇的“最小入口”。

## 内容维护

```text
foundations/    数学静态讲义、NumPy / PyTorch 验证与绘图脚本
training/       位置编码讲义与 PyTorch 实现
inference/      KV cache 讲义、验证代码与 nano-vLLM 专题
assets/         页面样式、交互脚本、数学 SVG 图解与讲义生成器
astro-site/     数学八章、模型原理与 AI Infra 的站点和数学正文源文件。在该目录执行 pnpm dev 或 pnpm build 后用浏览器阅读
```

### 原理讲义

数学正文统一编辑 `astro-site/src/content/lessons/mathematics.mdx`（导读）及 `math-01.mdx` 至 `math-08.mdx`；术语规范见 [CONVENTIONS.md](foundations/CONVENTIONS.md)。位置编码与 KV cache 仍编辑对应 `.md` 和 `.py`。然后在仓库根目录重新生成静态讲义：

```sh
node "assets/build.mjs"
```

生成器仅依赖 Node.js 标准库，从数学 MDX 同步生成 `foundations/` 的 Markdown 和 HTML，并转换图片与章节链接；位置编码和 KV cache 继续嵌入完整 Python 源码。数学图片的源文件位于 `astro-site/public/assets/math/`，生成时同步到 `assets/math/`。不要直接编辑生成的数学 Markdown 或 HTML，避免两套正文漂移。

在 `astro-site/` 执行 `pnpm check` 和 `pnpm build` 验证类型、MDX 与公式。数学图像由 `foundations/plot_style.py` 统一中文字体、配色和字号；已生成的 SVG 内嵌字形，阅读无需安装字体。重新作图需要系统具备中文字体（例如 Noto Sans CJK SC、苹方或微软雅黑）。数学图像可复现：

```sh
python "foundations/plot_probability.py"
python "foundations/plot_calculus.py"
python "foundations/neural_network.py" --plot
node "assets/build.mjs"
```

完整阅读顺序为 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08；只读 RoPE 可从 03 直接进入 08。

### nano-vLLM 专题与样式

直接编辑 `inference/nano-vllm-from-zero-to-mastery.html`，该页面不经过生成器。

| 修改内容 | 文件入口 |
| --- | --- |
| 全站配色、排版与页壳 | `astro-site/src/styles/site.css`。改完后在 `astro-site` 执行 `pnpm css`，生成 `assets/site.css` |
| 三篇原理讲义的交互 | `assets/lesson.js`、`assets/reader.js` |
| 三篇原理讲义的页面模板与导航链接 | `assets/build.mjs` |
| 数学基础的逐章 SVG 图解 | `astro-site/public/assets/math/`；构建后同步到 `assets/math/` |
| nano-vLLM 正文、导航与专用交互 | `inference/nano-vllm-from-zero-to-mastery.html` |

修改正文后检查公式与链接；修改样式或交互后检查桌面和手机布局；修改 Python 示例后运行对应数值验证。
