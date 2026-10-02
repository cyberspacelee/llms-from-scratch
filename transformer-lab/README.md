# Transformer Lab

面向学习、实验和原理验证的现代 Transformer 代码库。以 PyTorch 的 `matmul`、`reshape`、`transpose`、`softmax` 和自动求导为基础，不使用 `nn.Transformer` / `nn.MultiheadAttention`。默认 CPU，模型和张量可迁移到 CUDA；CPU 功能核对与 GPU 性能研究分开进行。

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

| 模块 | 实现 |
| --- | --- |
| `config.py` | 位置、Attention、Block、Model 的显式 dataclass 配置与检查 |
| `position.py` | Sinusoidal、RoPE、固定 Linear/NTK/YaRN Scaling |
| `masks.py` | Global/Causal、Sliding、分块 Local、Block/Token Sparse mask |
| `attention.py` | MHA/MQA/GQA、MLA、显式 SDPA、真正 gather 的稀疏原语 |
| `recurrent.py` | 归一化 Linear Attention、DeltaNet、Gated DeltaNet 的递归方程 |
| `layers.py` | LayerNorm/RMSNorm、ReLU/GELU/GLU/GeGLU/SwiGLU、Top-K MoE |
| `model.py` | Pre/Post-Norm、标准/门控残差、三类模型、Encoder/Cross/Self 缓存 |
| `objectives.py` | Teacher Forcing、NTP、顺序 MTP 和联合训练目标 |
| `cache.py` | KV/Latent/Recurrent 状态、分页与 prefix fork、int8 量化、序列池化 |
| `inference.py` | 因果序列压缩 Attention、draft/target greedy 投机验证 |
| `analysis.py` | 参数、激活参数、matmul FLOPs、KV 字节与理想访存账本 |
| `presets.py` | 主流模型核心思想的小型组合，不兼容官方权重 |
| `experiments.py` | 数值核对、可选小任务训练、理论账本、可选基准入口 |

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

从 [数学、Shape 与实现对照](docs/principles.md) 开始，再阅读 [推理与成本分析](docs/inference.md)、[来源与模型对照](docs/research.md) 和 [实验覆盖表](docs/coverage.md)。

基础 Attention → Encoder/Decoder → Cross-Attention → Encoder–Decoder → 因果 Encoder–Decoder；MHA → MQA → GQA → MLA → Sparse/Compressed → Hybrid；Absolute PE → RoPE → Scaling → Decoupled RoPE → RoPE/NoPE 分层交替；FFN → GLU/SwiGLU → MoE；完整重算 → KV → Latent KV → 分页/量化原语 → MTP/投机。

CPU 检查只覆盖数学与功能：缓存/完整前向一致、MLA 双路径输出/梯度一致、稀疏 gather 对照、padding/因果性、MTP 标签对齐和参数账本。它们不验证语言能力、长上下文泛化或吞吐。CUDA kernel、分页 allocator、Prefix 检索服务和高性能 MoE 通信需要真实硬件与独立系统工程。
