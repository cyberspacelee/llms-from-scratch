# 统一 Python 实现与开发检查

完整教程在 `site/src/content/lessons/`。核心包 `src/llms_from_scratch/` 维护一个可训练 `Transformer`；`examples/` 按教学任务调用它，独立小型参考只负责核对。没有 `code/`、`transformer-lab/` 的兼容入口。

## 环境和执行

```sh
uv sync --frozen --group dev
uv run ruff check src examples tests scripts
uv run python -m unittest discover -s tests -v
uv run python -m examples.verify
```

CPU 环境通过根 `pyproject.toml` 与 `uv.lock` 锁定；`numpy`、`torch` 是共同依赖，`plots` 与 `gpu` extras 是可选依赖。GPU kernel 与真实性能另外验收，CPU CI 不用模拟延迟充当 GPU 证据。所有默认实验用 `uv run python -m examples.<track>.<module>` 运行，不修改 `sys.path`。

## 模型 API 与教学配置

```python
import torch
from llms_from_scratch import Transformer, decoder_config, token_loss_sum

model = Transformer(decoder_config(vocab_size=8))
ids = torch.tensor([[0, 1, 2, 3]])
output = model(ids[:, :-1])
objective = token_loss_sum(output.logits, ids[:, 1:])
objective.mean().backward()
prefill = model(ids[:, :2], use_cache=True)
decode = model(ids[:, 2:], cache=prefill.cache, use_cache=True)
```

`Transformer.forward` 返回 `ModelOutput`，包含 `.logits`、`.hidden`、`.cache`、`.auxiliary_loss` 和显式路由统计。输入、目标由实验提前对齐，模型不会自行移位。

- `basic_decoder_config`：学习式位置表、Pre-LayerNorm、MHA、GELU、带 bias 的投影和独立 head；用于基础配置对照。默认 LayerNorm epsilon 为 1e-5，手算时明确改成 1。
- `decoder_config`：Pre-RMSNorm、GQA、相邻维 RoPE、SwiGLU、绑定 embedding/head；现代主线使用同一配置创建函数。
- 任意研究结构通过 `ModelConfig/BlockConfig/AttentionConfig/PositionConfig` 的真实配置组合，独立 `layer_blocks` 指定各层；不能用未实现的配置开关宣称支持一种模型。
- 同一 forward 可以接入显式 `inputs_embeds[B,T,D]`；视觉 slots 替换 embedding 后仍调用完整骨干，不能自行复制模型层循环。

checkpoint 将 `dataclasses.asdict(model.config)` 存成普通数据，读取用 `ModelConfig.from_dict` 恢复嵌套配置。`examples.training.text_pretraining` 另存 tokenizer、数据身份、优化器、游标和两个 RNG；恢复后核对后续更新的权重和损失逐项一致。

## 损失与空目标

`token_loss_sum(logits, targets, valid)` 返回 `TokenLoss(loss_sum, valid_count)`。`logits[B,T,V]` 与 `targets[B,T]` 已对齐，`valid=True` 表示参与监督。无有效行时返回可微零 sum 与零 count，ignored targets 不必属于词表。`TokenLoss.mean()` 在 count=0 时抛错，评估均值不能报告为零。

先跨微批/rank 聚合 sum 与 count，再除以总 count。DDP 默认平均梯度，因此 rank 上的未归约 sum 要乘 `world_size/global_valid_count`。局部空 rank 仍能参与同步；全局 count=0 且没有其他目标时跳过整个 optimizer step，包括 weight decay、动量、step 计数和学习率调度。

`token_loss` 是小型本地实验的 mean-or-zero convenience；零结果不授权调用 optimizer.step。真实文本训练通过 sum/count 检查更新资格。蒸馏同样把局部空 mask 视为零贡献，并 detach 教师；涉及多个独立目标的调用者分别决定有效数量和权重。

## 状态、符号与独立参考

统一维度使用 `B,T,D,H_q,H_kv,D_h,D_ff,V`，不同 query/key 长度用 `T_q/T_kv`，历史前缀用 `P`。行批计算为 `X @ W.T`，`nn.Linear.weight` 为 `[out,in]`。bool mask 的名字和 API 含义必须明确；允许读取的 mask 与要填掉的位置不能直接混用。

`ModelCache` 持有每层 self/cross 状态与有效前缀。完整、单 token、非等长 chunk 的输出必须一致；cached forward 只输入新 chunk。学习式位置和 RoPE 使用绝对 offset。双向 encoder 不可追加缓存；encoder memory 和各层静态 cross KV 属于不同状态。分页、量化、混合递归状态保留各自所有权和恢复边界。

`examples.principles.decoder.ReferenceDecoder` 是仅用于检验的独立基础 oracle，不被训练脚本 import。`tests/test_curriculum.py` 将其参数映射到统一模型，核对输出、关键梯度、学习式位置、分块缓存。现代通用 Decoder 类已删除，`modern_decoder.py` 只保留逐数手算和统一模型检查。朴素 attention、MLA absorbed/dense、online softmax 等参考不能全部 import 被测函数，否则失去独立核对价值。

原 25 个机制实验保留在 `llms_from_scratch.chapters`，用途是可运行配方和机制检查；历史顺序不会决定网站学习顺序。CLI `chapter --chapter N` 是开发实验入口。其章级 prose 已归入 MDX，不再维护第二套完整课程目录。机制覆盖、源码边界和来源资料见 [代码组织](python/architecture.md)、[推理边界](python/inference.md)、[研究来源](python/research.md) 与 [模型对照记录](python/models-2026.md)。

## 真实 GPU 候选测量

`examples/gpu/examples/tiled_gemm.cu` 同时提供 tile16 和 tile32。默认检查非整倍数边界与 double CPU oracle；`--benchmark` 用相同 256×256×256 输入，先 warmup，再报告 31 次 CUDA event 的 min/median/max。记录 GPU、能力、SM 数和 runtime；事件范围只包括预分配 buffer 上的 kernel，不包括 H2D/D2H。两种配置都要先算对；固定先测 tile16 后测 tile32，严谨性能结论还应交错测量/重复整批实验评估时钟变化。

```sh
nvcc -O2 examples/gpu/examples/tiled_gemm.cu -o /tmp/tiled_gemm
/tmp/tiled_gemm --benchmark
```

本次环境没有 `nvcc` 或 CUDA hardware/build，以上 GPU 编译、数值与延迟未实测。CPU full suite、模型不变量与手算对照是独立验收项。
