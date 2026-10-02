# 检查覆盖

每章有可运行的`run(device)`与assert，所有25章由unittest调用；这不是官方大模型质量评估。

| 章节 | 保留的检查 |
| --- | --- |
| 01–03 | 朴素/核心同权重结果及embedding梯度；多头多层；NTP因果性与MLM双向性 |
| 04–06 | MQA逐head参考、KV容量；Pre/Post-Norm；FFN各激活梯度与参数 |
| 07–08 | sparse gather与同selection dense结果；linear特征对照；RoPE范数/平移 |
| 09–12 | MoE dispatch/router/shared/bias；online softmax输出及输入梯度；GQA映射；Scaling/cache |
| 13–15 | MLA naive/absorbed所有参数梯度；MTP对齐与未来泄漏；hybrid cache、KDA full/chunk状态 |
| 16–19 | 非等长chunk、static Cross、causal Encoder；分页COW/int8误差；draft接受/拒绝；压缩因果边界 |
| 20–22 | 短卷积因果性、异构头递归；token/block selection；Sinkhorn行列和、mHC/GR形状、AttnRes深度权重/梯度 |
| 23–25 | hashed memory因果性/梯度；共享KV的chunk绝对位置/梯度；模态slots与projector梯度 |

原有数值不变量测试仍覆盖全部小型配方、padding/all-masked rows、teacher forcing、cache非法输入、analysis实际参数/字节对照、SDPA输出/梯度和联合loss。章节自检与这些测试都使用小张量，无权重下载、数据集训练或语言质量基准。fp64对照通常使用atol=1e-9/rtol=1e-7；新独立原语按各自数值精度检查。

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m compileall -q src tests
uv run --locked python -m unittest discover -s tests -v
```

每个源码函数/方法（包括构造函数、property和内部函数）都有Args/Returns；tensor参数、返回和关键reshape/transpose/gather/state写入都说明shape。Ruff核对imports、Python错误、bugbear与函数类型标注；不把它当完整shape类型系统。本轮使用CPU Torch，CUDA测试跳过。

官方模型的完整训练、checkpoint mapping、分布式专家通信、chunkwise recurrence、持久压缩cache、Single-Pass残差、conv/lookup增量状态、低比特GPU算子与音视频encoder未实现；具体对应见[2026结构边界](models-2026.md)。入口为[25章课程](../README.md)、[规划](plan.md)、[源码边界](architecture.md)。
