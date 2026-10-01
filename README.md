# LLMs From Scratch

这套中文讲义从数学、数组和模型结构讲到训练、GPU 编程与推理系统。关键结论配有推导、可手算的例子和 CPU 上可运行的 NumPy / PyTorch 核对脚本。

**在线阅读：<https://cyberspacelee.github.io/llms-from-scratch/>**

## 主线、GPU 编程与进阶模型

| 路线 | 章节 | 回答的问题 |
| --- | --- | --- |
| [数学基础](https://cyberspacelee.github.io/llms-from-scratch/math/) | 导读 + M1–M11 | 从随机事件、贝叶斯、信息量与预测损失到梯度训练、谱分解与数值稳定性 |
| [数组与框架](https://cyberspacelee.github.io/llms-from-scratch/frameworks/) | 导读 + F1–F7 | NumPy、Torch 张量与求导、模块、训练状态，以及 torch.compile 编译原理 |
| [模型原理](https://cyberspacelee.github.io/llms-from-scratch/principles/) | 导读 + P1–P11 | 从 token 到完整现代 Transformer，连接 RoPE/GQA/RMSNorm/SwiGLU、生成与逐层缓存 |
| [训练与评估](https://cyberspacelee.github.io/llms-from-scratch/training/) | 导读 + T1–T9 | 文本训练与磁盘恢复、评估与适配、缩放规律、数据工程和多进程训练 |
| [GPU 编程](https://cyberspacelee.github.io/llms-from-scratch/gpu/) | 导读 + G1–G4 | thread/warp/block/grid、SM、访存、同步、CUDA/Triton kernel 与测量 |
| [推理系统](https://cyberspacelee.github.io/llms-from-scratch/systems/) | 导读 + S1–S10 | 计算账本、内核、调度、分页、量化、投机、多卡与服务指标 |
| [进阶模型](https://cyberspacelee.github.io/llms-from-scratch/advanced/) | 导读 + A1–A9 | MoE、MLA、DPO、长上下文、推理训练、状态空间、混合与稀疏注意力、多模态 |

61 章正文与 7 篇导读按先修关系排列，也可以从首页直接进入感兴趣的章节。读者需要具备 Python 基础，无需预先学过深度学习。课程设计见 [课程计划](docs/CURRICULUM_PLAN.md)、[设计卡](docs/CHAPTER_BLUEPRINTS.md)、[CS336 对照与补充](docs/COURSE_GAP_PLAN.md)、[数组与框架规划](docs/ARRAY_FRAMEWORK_PLAN.md)、[GPU 编程规划](docs/GPU_PROGRAMMING_PLAN.md) 和 [torch.compile 规划](docs/TORCH_COMPILE_PLAN.md)。

## 运行验证脚本

需要 Python 3.10+，CPU 即可：

```sh
python -m venv .venv
.venv/bin/pip install -r code/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python code/math/probability.py
.venv/bin/python code/math/linear_algebra.py
.venv/bin/python code/math/calculus.py
.venv/bin/python code/math/neural_network.py
.venv/bin/python code/math/spectral.py
for script in code/frameworks/*.py code/principles/*.py code/training/*.py code/gpu/*.py code/systems/*.py code/advanced/*.py; do
  .venv/bin/python "$script" || exit 1
done
```

脚本核对梯度、分词、标签移位、完整现代模型缓存、采样、磁盘恢复、缩放拟合、近重复、多进程 DDP 与真实 LM 的 RL 更新。全部默认验证只需 CPU；T9 需允许本机进程创建与 Gloo 回环通信。小实验不复现大型模型质量或 GPU 性能。运行 nano-vLLM 所需 GPU 与权重见 S4。

训练现代模型并恢复、生成：

```sh
.venv/bin/python code/training/text_pretraining.py --checkpoint runs/prose.pt --steps 80
.venv/bin/python code/training/text_pretraining.py --checkpoint runs/prose.pt --resume --steps 20
.venv/bin/python code/training/text_pretraining.py --checkpoint runs/prose.pt --generate-only --prompt "the cat"
.venv/bin/python code/training/scaling.py --measure --output runs/scaling-pilot.json
.venv/bin/python code/training/ablation.py
```

默认语料为自编短文；新训练可指定 `--train-text` 和 `--validation-text` 两个 UTF-8 文件，每行一篇文档。checkpoint 内保存冻结文档、tokenizer、配置、权重、优化器、游标与 RNG；`--steps` 是本次额外更新数。恢复不替换语料。少量生成不保证可读，验证报告与恢复等价性是教学验收对象。

G1–G4 的 CPU 脚本检查索引覆盖、地址段、bank、归约、分块 GEMM、资源账与依赖。`code/gpu/examples/` 是独立的真实 GPU 示例：CUDA C++ 需 NVIDIA GPU 与 `nvcc`，Triton 和 event 计时需相容的 CUDA-enabled PyTorch，安装与运行见各章。无 GPU 的 CI 不运行这些示例，教学图和 CPU 数字不代表 GPU 实测。

`python code/gpu/model_profile.py` 默认核对同一个现代模型的显式注意力/SDPA 输出与梯度，并采集 CPU profiler；加 `--trace model-trace.json` 导出时间线。`--device cuda` 才执行 CUDA event 与峰值显存测量，需要真实 GPU。A5 的 RL 实验分别报告训练题与未见题奖励，不将训练奖励上升当作泛化证据。

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
