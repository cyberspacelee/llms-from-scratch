# LLMs From Scratch

这套中文讲义从数学和张量计算讲到模型、训练、GPU 编程与推理系统。正文、交互和本地实验围绕可核对的教学案例组织。

**在线阅读：<https://cyberspacelee.github.io/llms-from-scratch/>**

| 路线 | 回答的问题 |
| --- | --- |
| 数学基础 | 概率、信息量、梯度、反向传播与谱分解 |
| 数组与框架 | NumPy/PyTorch 的轴、存储、求导、模块和训练状态 |
| 模型原理 | 从 token 到同一个完整 Decoder，解释位置、归一化、头共享和 FFN |
| 训练与评估 | 文本处理、训练、评估、checkpoint 精确恢复与分布式更新 |
| 后训练 | SFT、LoRA、蒸馏、DPO 与可验证奖励 |
| GPU 编程 | 线程映射、访存、同步、kernel 编写与真实测量 |
| 推理系统 | 资源账、调度、分页、量化、投机、多卡与服务指标 |
| 高级结构 | MoE、MLA、长上下文、递归、稀疏注意力与多模态 |

网站是完整教程的唯一正文。`src/llms_from_scratch/` 是统一的可训练模型实现；`examples/` 是按课程命名的运行实验和独立参考计算。原 Transformer Lab 的有效机制检查已并入同一包、环境和测试入口，不再维护平行课程目录或旧命令。

## 运行 Python 实验

需要 Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)。默认环境使用 CPU PyTorch，不需要 GPU：

```sh
uv sync --locked --group dev
uv run python -m unittest discover -s tests -v
uv run python -m examples.verify
uv run ruff check src examples tests scripts
```

单独运行一章：

```sh
uv run python -m examples.math.probability
uv run python -m examples.principles.decoder
uv run python -m examples.principles.modern_decoder
uv run python -m examples.principles.complete_transformer
uv run python -m examples.post_training.distillation
```

`examples.verify` 执行全部默认 CPU 核对，包括真实 PyTorch API、完整模型输出/梯度/缓存、文本训练和磁盘恢复、多进程 Gloo、参考算法与原 25 个机制实验的配套检查。原 25 个机制实验本身由 `tests/test_lab.py` 执行；也可指定运行：

```sh
uv run llms-from-scratch chapter --chapter 13
uv run llms-from-scratch check --preset deepseek_v3
uv run llms-from-scratch ledger --length 128
```

这些小实验验证实现与协议，不复现大型 checkpoint 的质量。多进程例需要允许创建本机进程与回环通信。GPU 测试在硬件不可用时明确跳过。

## 用同一文本工件训练、恢复和生成

```sh
uv run python -m examples.training.text_pretraining --checkpoint runs/prose.pt --steps 80
uv run python -m examples.training.text_pretraining --checkpoint runs/prose.pt --resume --steps 20
uv run python -m examples.training.text_pretraining --checkpoint runs/prose.pt --generate-only --prompt "the cat"
uv run python -m examples.training.scaling --measure --output runs/scaling-pilot.json
uv run python -m examples.training.ablation
```

默认语料为自编短文。新训练可传 `--train-text` 与 `--validation-text` 两个 UTF-8 文件，每行一篇文档；恢复使用 checkpoint 中冻结的语料与 tokenizer。checkpoint 保存模型配置、权重、优化器、游标、数据指纹、Python/Torch RNG；`--steps` 表示本次额外更新数。少量训练的生成文本不保证可读，验证 NLL、缓存等价性和恢复轨迹是实验验收对象。

## GPU 和作图选项

CPU 核对不会 import Triton。`examples/gpu/examples/` 下的 CUDA/Triton 示例需要真实 NVIDIA GPU，CUDA C++ 还需要 `nvcc`。安装 GPU 依赖可使用独立环境，避免 CPU 锁定入口覆盖已安装的 CUDA PyTorch：

```sh
uv venv .venv-gpu
uv pip install --python .venv-gpu/bin/python -e '.[gpu]'
.venv-gpu/bin/python -m examples.gpu.model_profile --device cuda --trace runs/model-trace.json
```

这里的 pip 安装使用 PyPI；先核对 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 中驱动与 PyTorch/CUDA 构建的匹配关系。无 GPU 时 `uv run python -m examples.gpu.model_profile` 仅核对 CPU manual/SDPA 输出和梯度，并采集本机 CPU profiler。CPU 教学数字不代表 GPU 实测。

重建图像使用 `uv sync --locked --group dev --extra plots`，并安装中文字体（例如 Noto Sans CJK SC）；作图入口在 `examples/math/plot_*.py` 和 `examples/gpu/plots/`。F7 可运行 `uv run python -m examples.frameworks.torch_compile --inductor` 核对真实 CPU 代码生成，需要可用的 C++ 编译器。

## 本地站点

源码符号摘录需要系统 `python3`（只用标准库）。跨端固定案例检查需要先执行上面的 `uv sync --locked --group dev`：

```sh
cd site
pnpm install
pnpm dev
pnpm check
pnpm build
```

公式由 KaTeX 在构建时渲染，Mermaid 和实验在浏览器执行；依赖随站点打包。CI 使用同一 Python 锁文件运行测试和全部 CPU 实验，再执行站点检查、构建和内部链接验证。

## 仓库结构

```text
pyproject.toml / uv.lock       统一 Python 环境、可选依赖与锁定入口
src/llms_from_scratch/         模型、层、attention、cache、损失与机制实验
examples/                     按课程组织的运行实验与独立参考计算
tests/                        不变量、独立参考、输出/梯度/cache 与合并契约核对
site/src/content/lessons/      唯一完整教学正文
site/src/components/          图表、实验与公共阅读组件
site/src/assets/figures/       站点图像
site/src/styles/global.css     统一主题和正文样式
docs/                         编辑规范、审查基线与开发说明
```

修改正文前阅读 [编辑约定](docs/CONVENTIONS.md)。Python 配置、损失、缓存与研究范围见 [Python 开发说明](docs/PYTHON_DEVELOPMENT.md)。
