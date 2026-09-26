# LLMs From Scratch

一组理解大语言模型原理与推理实现的中文讲义：每个结论先推导，再给能手算的小算例、图解或交互实验，最后用只需 CPU 的 PyTorch / NumPy 脚本核对。

**在线阅读：<https://cyberspacelee.github.io/llms-from-scratch/>**

## 三条路线

| 路线 | 章节 | 回答的问题 |
| --- | --- | --- |
| [数学基础](https://cyberspacelee.github.io/llms-from-scratch/math/) | 导读 + M1–M7 | 统计、分布、矩阵、微积分与反向传播，怎样组成一次完整的神经网络训练 |
| [模型原理](https://cyberspacelee.github.io/llms-from-scratch/principles/) | 导读 + P1–P4 | 注意力为什么需要位置，sin/cos 与 RoPE 怎样写进点积，KV cache 为什么与整段计算相等 |
| [推理系统](https://cyberspacelee.github.io/llms-from-scratch/systems/) | 导读 + S1–S5 | 一次生成的字节与强度、单卡上限与 kernel、nano-vLLM 的调度与分页缓存、多卡与集群 |

三条路线按依赖排好，也可以从首页的捷径直接进入某一章。适合具备 Python 基础的读者；数学基础从概率与矩阵开始，不要求先学过深度学习。

## 运行验证脚本

需要 Python 3.10+，CPU 即可：

```sh
python -m venv .venv
.venv/bin/pip install -r code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python code/math/probability.py
.venv/bin/python code/math/linear_algebra.py
.venv/bin/python code/math/calculus.py
.venv/bin/python code/math/neural_network.py
.venv/bin/python code/principles/position_encoding.py
.venv/bin/python code/principles/kv_cache.py
```

脚本用 float64 与 assert 核对教学算例：梯度与自动微分、有限差分一致；RoPE 只依赖相对位置；分块解码与整段注意力相等，并用一个故意重置位置的实现作负面对照。它们不包含大模型训练或端到端文本生成；运行 nano-vLLM 本身所需的 GPU 与权重见 S3。

## 本地预览站点

```sh
cd site
pnpm install
pnpm dev      # 本地预览
pnpm check    # 类型检查
pnpm build    # 构建并检查所有站内链接与锚点
```

公式由 KaTeX 在构建时渲染，Mermaid 图在浏览器里绘制，所有依赖随站点打包，不需要 CDN。

## 仓库结构

```text
site/                       Astro 站点，唯一的正文来源
  src/content/lessons/      讲义正文（MDX），按路线分目录：math/ principles/ systems/
  src/components/           Figure、KeyEq、Callout 等正文组件，交互实验与系统图
  src/assets/figures/       SVG 图解；构建时内联并映射到站点配色
  src/styles/global.css     唯一的样式入口：Tailwind 主题 token 与生成内容的规则
code/                       验证脚本与数学作图脚本
docs/CONVENTIONS.md         写作、排版与组件约定
```

修改正文前先读 [docs/CONVENTIONS.md](docs/CONVENTIONS.md)。CI 在每次推送时运行全部验证脚本、类型检查、构建与链接检查，通过后部署到 GitHub Pages。
