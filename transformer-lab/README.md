# Transformer Lab

面向学习、实验和原理验证的现代 Transformer 代码库。以 PyTorch 的 `matmul`、`reshape`、`transpose`、`softmax` 和自动求导为基础，不使用 `nn.Transformer` / `nn.MultiheadAttention`。默认 CPU，模型和张量可迁移到 CUDA；CPU 功能核对与 GPU 性能研究分开进行。

## 从最小 MVP 开始

```bash
cd transformer-lab
uv sync --locked
uv run --locked python -m transformer_lab.tutorials.mvp --device cpu
# 接下来把步号依次改成 01 ... 13；每步只运行小型功能核对。
uv run --locked transformer-lab lesson --step 01
```

先读 [最小 MVP：公式、Shape 与完整数据流](docs/mvp.md)，再按 [00–13 步演进路线](docs/evolution.md) 运行示例。MVP 固定为单头、单层 Encoder–Decoder；第一步用相同权重验证它与统一模型的输出、梯度一致，然后才增加多头与层数。后续按机制演进，而不是一开始启用全部现代配置。

路线：MVP → 多头/多层 → 三类模型 → RoPE → KV Cache → 现代 Block → MQA/GQA → MLA。之后分支研究长上下文、MoE/MTP、递归 Hybrid、因果 Encoder–Decoder 与推理存储。每一步说明新增机制、代码入口、Shape 和核对依据。

## 使用 uv

```bash
cd transformer-lab
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -v
uv run --locked transformer-lab check --preset classic
uv run --locked transformer-lab check --preset deepseek
uv run --locked transformer-lab check --preset qwen_hybrid
uv run --locked transformer-lab ledger --length 32768
```

`pyproject.toml` / `uv.lock` 默认安装 CPU PyTorch，让无 GPU 的学习环境保持简单。需要 NVIDIA GPU 时，先按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择与驱动相容的 CUDA wheel，再覆盖当前环境的 CPU wheel：

```bash
# CUDA_INDEX 替换为官方页面给出的索引，不硬编码某个 CUDA 版本。
uv pip install --python .venv/bin/python --index-url "$CUDA_INDEX" torch
# --no-sync 避免 uv 按 CPU lock 恢复 torch。
uv run --no-sync transformer-lab check --preset deepseek --device cuda
```

CPU/CUDA 共享实现，没有 `.cuda()` 或固定设备分配。`model.to(device)`、输入和缓存必须处于同一设备；dtype 由 `model.to(dtype=...)` 控制。默认 fp32；CLI 的 `check` 使用 fp64 做数值核对。CUDA/fp16/bf16 的容差需要根据设备和算子单独选择，本次无 GPU，不声称已完成 GPU 验证。AMD/macOS 等设备也可通过 PyTorch 使用；尚未验证。

## 一个接口，三种架构

```python
import torch
from transformer_lab import Transformer, ModelConfig, BlockConfig, AttentionConfig

model = Transformer(
    ModelConfig(
        architecture="encoder_decoder",
        block=BlockConfig(attention=AttentionConfig(kind="gqa", kv_heads=2)),
    )
)
source = torch.tensor([[1, 2, 3, 4]])
decoder_input = torch.tensor([[0, 5, 6]])
training = model(decoder_input, source_ids=source)

model.eval()
with torch.no_grad():
    memory = model.encode(source)  # 可复用 Encoder 输出
    prefill = model(decoder_input, memory=memory, use_cache=True)
    decode = model(torch.tensor([[7]]), cache=prefill.cache, use_cache=True)
    generated = model.generate(decoder_input, 4, memory=memory)
```

`architecture="encoder"` 是双向 Encoder-only；`"decoder"` 是因果 Decoder-only；`"encoder_decoder"` 使用双向 Encoder、因果 Decoder 和 Cross-Attention。`encoder_causal=True` 允许因果 Encoder 流式追加；`cross_causal=True` 表示输入/输出位置对齐的因果 Cross-Attention，需与业务时间线相符。

`valid` / `source_valid` 为 boolean `[B,T]`，**True 表示有效**。缓存追加时只传本次 decoder chunk 的 `valid`，模型拼接完整历史。全遮挡行返回有限的零 Attention 分支。生成示例仅接受无 padding 的等长 prompt；使用 padding 训练时，loss 同时过滤无效标签。

## 模块与实验

| 子包 / 模块 | 职责 |
| --- | --- |
| `attention/` | 基础 SDPA、MHA/MQA/GQA、MLA、递归注意力、位置编码与可见关系 |
| `layers/` | LayerNorm/RMSNorm、Dense/Gated FFN、Router/Expert/Dispatch/Combine |
| `models/` | Transformer Block、三类模型、Encoder Memory、模型级缓存与生成 |
| `cache/` | KV/Latent/Recurrent 状态、分页、prefix fork、int8 存储 |
| `training/` | Teacher Forcing、NTP/MTP 与损失对齐 |
| `inference/` | 因果序列压缩、greedy draft/target 推测解码 |
| `experiments/` | 统一 CLI、思想预设、数值检查、理论账本与可选基准 |
| `tutorials/` | 最小 MVP，以及复用核心实现的逐步演进示例 |
| `config.py` / `analysis.py` | 配置检查；参数、激活参数、FLOPs、KV 和访存账本 |

完整文件树、依赖方向、接口和修改入口见 [子包规划](docs/architecture.md)。例如位置编码在 `attention/position.py`，MLA 在 `attention/mla.py`，MoE 在 `layers/moe.py`；常用 `from transformer_lab import Transformer` 接口保持不变。

```bash
# 可选训练示例；检查功能不需要运行这些训练或性能实验。
uv run transformer-lab train --preset classic --steps 60 --output runs/copy.pt
uv run transformer-lab train --preset deepseek --steps 60 --output runs/mtp.pt

# 有 GPU 时再测。SDPA 并不保证使用 FlashAttention kernel。
uv run --no-sync transformer-lab benchmark --device cuda --length 4096
```

组合示例：`classic`、`llama`、`gemma`、`qwen`、`deepseek`、`kimi`、`gpt_oss`、`qwen_hybrid`、`kimi_linear`、`irope`、`causal_seq2seq`。这些名字表示教学配方；例如 Kimi Linear 配方使用 scalar gated delta 方程帮助入门，**不是完整 KDA**；GPT-OSS 配方不包含 attention sink 或专用激活。每个差异见 [官方资料与实现边界](docs/research.md)。

源码不做 import 时的训练/下载；默认不请求模型权重和数据集。PyTorch 是唯一运行依赖；测试使用 Python `unittest`，Ruff 作为 uv 的开发依赖检查和格式化源码。所有函数均有参数与返回值类型标注，Tensor Shape 另由 docstring 和运行时检查表达。

## 学习路径与验收

依次阅读 [最小 MVP](docs/mvp.md) → [逐步演进](docs/evolution.md)，再查 [数学、Shape 与实现对照](docs/principles.md)、[推理与成本分析](docs/inference.md)、[来源与模型对照](docs/research.md) 和 [实验覆盖表](docs/coverage.md)。

基础 Attention → Encoder/Decoder → Cross-Attention → Encoder–Decoder → 因果 Encoder–Decoder；MHA → MQA → GQA → MLA → Sparse/Compressed → Hybrid；Absolute PE → RoPE → Scaling → Decoupled RoPE → RoPE/NoPE 分层交替；FFN → GLU/SwiGLU → MoE；完整重算 → KV → Latent KV → 分页/量化原语 → MTP/投机。

CPU 检查只覆盖数学与功能：缓存/完整前向一致、MVP/核心模型同权重输出与梯度一致、00–13 步示例可运行、MLA 双路径输出/梯度一致、稀疏 gather 对照、padding/因果性、MTP 标签对齐和参数账本。它们不验证语言能力、长上下文泛化或吞吐。CUDA kernel、分页 allocator、Prefix 检索服务和高性能 MoE 通信需要真实硬件与独立系统工程。
