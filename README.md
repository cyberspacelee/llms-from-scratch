# LLMs From Scratch

一组理解大语言模型原理与推理实现的中文讲义：每个结论先推导，再给能手算的小算例、图解或交互实验，最后用只需 CPU 的 PyTorch / NumPy 脚本核对。

**在线阅读：<https://cyberspacelee.github.io/llms-from-scratch/>**

## 主线、GPU 编程与进阶模型

| 路线 | 章节 | 回答的问题 |
| --- | --- | --- |
| [数学基础](https://cyberspacelee.github.io/llms-from-scratch/math/) | 导读 + M1–M7 | 统计、分布、矩阵、微积分与反向传播，怎样组成一次完整的神经网络训练 |
| [数组与框架](https://cyberspacelee.github.io/llms-from-scratch/frameworks/) | 导读 + F1–F7 | NumPy、Torch 张量与求导、模块、训练状态，以及 torch.compile 编译原理 |
| [模型原理](https://cyberspacelee.github.io/llms-from-scratch/principles/) | 导读 + P1–P10 | 从 token 与目标搭建 Decoder，解释位置、现代结构、生成与完整模型缓存 |
| [训练与评估](https://cyberspacelee.github.io/llms-from-scratch/training/) | 导读 + T1–T6 | 数据、优化、完整训练恢复、评估、指令微调与 LoRA |
| [GPU 编程](https://cyberspacelee.github.io/llms-from-scratch/gpu/) | 导读 + G1–G4 | thread/warp/block/grid、SM、访存、同步、CUDA/Triton kernel 与测量 |
| [推理系统](https://cyberspacelee.github.io/llms-from-scratch/systems/) | 导读 + S1–S10 | 计算账本、内核、调度、分页、量化、投机、多卡与服务指标 |
| [进阶模型](https://cyberspacelee.github.io/llms-from-scratch/advanced/) | 导读 + A1–A9 | MoE、MLA、DPO、长上下文、推理训练、状态空间、混合与稀疏注意力、多模态 |

53 章正文与 7 篇导读按依赖组织，也可以从首页的捷径进入。适合具备 Python 基础的读者；数学基础不要求先学过深度学习。大纲与逐章设计见 [课程计划](docs/CURRICULUM_PLAN.md)、[设计卡](docs/CHAPTER_BLUEPRINTS.md)、[数组与框架规划](docs/ARRAY_FRAMEWORK_PLAN.md)、[GPU 编程规划](docs/GPU_PROGRAMMING_PLAN.md) 和 [torch.compile 规划](docs/TORCH_COMPILE_PLAN.md)。

## 运行验证脚本

需要 Python 3.10+，CPU 即可：

```sh
python -m venv .venv
.venv/bin/pip install -r code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python code/math/probability.py
.venv/bin/python code/math/linear_algebra.py
.venv/bin/python code/math/calculus.py
.venv/bin/python code/math/neural_network.py
for script in code/frameworks/*.py code/principles/*.py code/training/*.py code/gpu/*.py code/systems/*.py code/advanced/*.py; do
  .venv/bin/python "$script" || exit 1
done
```

脚本核对梯度、分词、标签移位、完整 Decoder 缓存、采样、训练恢复以及系统与进阶算法。T3 包含小语言模型训练、生成和断点恢复，全部教学验证只需 CPU；不复现大型模型的训练质量与 GPU 内核性能。运行 nano-vLLM 所需 GPU 与权重见 S4。

G1–G4 的 CPU 脚本检查索引覆盖、地址段、bank、归约、分块 GEMM、资源账与依赖。`code/gpu/examples/` 是独立的真实 GPU 示例：CUDA C++ 需 NVIDIA GPU 与 `nvcc`，Triton 和 event 计时需相容的 CUDA-enabled PyTorch，安装与运行见各章。无 GPU 的 CI 不运行这些示例，教学图和 CPU 数字不代表 GPU 实测。

F1–F7 位于数学与模型之间，解释常用 NumPy/PyTorch API 的形状、共享存储、梯度与状态契约，以及 torch.compile 的捕获、求导、代码生成、图中断和重编译。七份 CPU 脚本验证真实 API 行为；F7 默认检查 Dynamo 与 AOTAutograd，运行 `python3 code/frameworks/torch_compile.py --inductor` 额外核对真实 CPU 代码生成，需要兼容的本地 C++ 编译器。浏览器交互配有独立教学模型断言，使用已有依赖。本轮运行版本为 NumPy 2.5.2、PyTorch 2.14.0+cpu。

## 本地预览站点

```sh
cd site
pnpm install
pnpm dev      # 本地预览
pnpm check    # 类型检查与数组/张量/GPU 教学模型核对（Node 22.18+）
pnpm build    # 构建并检查所有站内链接与锚点
```

公式由 KaTeX 在构建时渲染，Mermaid 图在浏览器里绘制，所有依赖随站点打包，不需要 CDN。

## 仓库结构

```text
site/                       Astro 站点，唯一的正文来源
  src/content/lessons/      讲义正文：math/ frameworks/ principles/ training/ gpu/ systems/ advanced/
  src/components/           Figure、KeyEq、Callout 等正文组件，交互实验与系统图
  src/assets/figures/       SVG 图解；构建时内联并映射到站点配色
  src/styles/global.css     唯一的样式入口：Tailwind 主题 token 与生成内容的规则
code/                       验证脚本与数学作图脚本
docs/CONVENTIONS.md         写作、排版与组件约定
```

修改正文前先读 [docs/CONVENTIONS.md](docs/CONVENTIONS.md)。CI 在每次推送时运行全部验证脚本、类型检查、构建与链接检查，通过后部署到 GitHub Pages。
