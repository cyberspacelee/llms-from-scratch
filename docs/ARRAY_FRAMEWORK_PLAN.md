# 数组与框架：大纲、资料与实施

设计日期：2026-09-27。基线：`b2d763c`。状态：已按大纲完成正文、配套实验与验收，见 [完成记录](ARRAY_FRAMEWORK_PROGRESS.md)。

## 放置位置与目标

新增 `frameworks/` 路线，章节 F1–F6 与一篇导读，默认顺序为数学 M → 数组与框架 F → 模型原理 P → 训练 T → GPU G → 系统 S → 进阶 A。现有章节代码与 URL 保留。已熟悉 NumPy/PyTorch 的读者可跳过 F；数学初学者可在 M3 后先读 F1–F3，在 M6 后读 F4，其余章节在 M7 后阅读。

现有 M3/M5/M6 用数组、广播和自动微分核对数学；P1–P4 直接使用嵌入、`reshape/transpose`、`nn.Module` 与交叉熵；T1–T3 使用数据迭代、优化器与恢复。新路线把这些操作的输入形状、输出、共享存储、梯度和状态契约明确写出来。数学推导留在 M，模型结构留在 P，训练算法与实验协议留在 T；F 负责把公式可靠地写成 API 调用。

每章按讲义规范写完整的约 5k–9k 汉字正文：具体问题、定义、数值、短代码、图后读图、完整 CPU 脚本、误区、折叠练习与原始资料。API 汇总表用于回查；正文围绕计算展开，不做函数名罗列。

## 六章大纲

| 章与路径 | 核心问题 | API 与边界 | 图与验证 |
| --- | --- | --- | --- |
| F1 `numpy-arrays` | 数字怎样成为有轴、有类型、有存储关系的数组 | `array/asarray/arange/linspace/zeros/ones/empty`，`shape/ndim/size/dtype`，切片/高级索引/布尔索引，`reshape/transpose/copy`，`shares_memory` | 2×3 数组索引与视图/副本联动；CPU 断言形状、别名、修改传播 |
| F2 `numpy-computation` | 怎样按正确的轴批量计算 | 广播、`sum/mean/var` 与 `axis/keepdims/ddof`，`*` 对照 `@/matmul/einsum`，`where/clip/argmax/take_along_axis`，`stack/concatenate`，`default_rng`、稳定 softmax | 广播/归约与收缩轴交互，NumPy 数值热力图；CPU 核对手算与显式循环 |
| F3 `torch-tensors` | NumPy 到 Torch 后哪些语义保持，哪些增加 | `tensor/as_tensor/from_numpy`，`dtype/device/to`，`long/float64`，`reshape/view/permute/transpose/contiguous`，`unsqueeze/squeeze`，`expand/repeat`，`cat/stack`，`gather` 与 `masked_fill` | 存储与 stride、B/T/H/dh 轴重排图；CPU 核对别名、连续性、形状与 gather |
| F4 `autograd` | 哪些操作被记录，梯度怎样留在参数上 | `requires_grad/grad_fn/is_leaf/grad`，`backward` 与向量 VJP、梯度累积/清空，`autograd.grad`，`detach/clone/no_grad/inference_mode`，原地修改与图释放 | 分支计算图与梯度累计交互；CPU 手算、Torch 自动求导、有限差分核对 |
| F5 `modules-and-losses` | 数学层怎样成为可保存、可训练的模块 | `nn.Module/Parameter/ModuleList/Sequential`，`Linear/Embedding/LayerNorm/Dropout`，`F.relu/gelu/softmax/log_softmax/cross_entropy`，class 轴/目标 dtype/ignore_index，`train/eval/parameters/state_dict` | 模块参数树、logits 到 masked CE 交互；CPU 登记、嵌入重复梯度、线性权重布局、CE 与 train/eval 核对 |
| F6 `data-and-training` | 怎样让批次、梯度、更新和恢复各自有边界 | `TensorDataset/Dataset/DataLoader`，默认 collate/末尾小批次/`drop_last`，`Generator/manual_seed`，`optim.SGD/AdamW`、`zero_grad/step`，`clip_grad_norm_`，`save/load/load_state_dict/map_location/weights_only` | 数据到更新/恢复状态图与批次计数交互；CPU 最小训练、梯度清空、末尾权重和续训等价验证 |

## 贯穿例子与取舍

数组例子从 `arange(6).reshape(2,3)` 开始。计算章使用明确数值的批输入和权重；张量章逐轴解释 `(B,T,H,dh)` 重排。模块章用小词表和显式目标核对 logits/交叉熵。F6 用五个样本和小线性模型演示 batch size 2 的末尾小批次与保存恢复。

所有数学核对优先 CPU float64、固定局部随机生成器；token 索引使用 int64/long，精度与设备说明不假装 float64 是大型模型默认。图是教学计算或实际数组值，不模拟真实内存分配器或 GPU 性能。可视化计算用现有 React/SVG/KaTeX；图表可由已有 NumPy/Matplotlib 生成，不引入浏览器 NumPy、Torch 或新绘图库。

F6 checkpoint 只读写脚本自己在临时目录创建的文件。显式 `weights_only=True`，状态仅包含兼容张量、数值、字符串与容器；不加载第三方 pickle，不把权重恢复等同于优化器与随机状态恢复。恢复等价范围为本章固定 CPU、固定批次、没有 dropout 的教学实验；数据位置与 RNG 另行演示、说明。

不在 F 重讲 BPE、注意力结构、AdamW 推导、GPU kernel、分布式训练或大型框架。这些分别链接 P/T/G/S。F4 主线不使用 `retain_graph=True` 掩盖错误；需要复用已释放计算图时先说明重做前向。

## 官方资料与检索结论

以下资料于设计日读取，stable 文档会更新。实际运行版本在验收记录中列出；示例覆盖仓库现有 NumPy 2.x/PyTorch 2.x 依赖范围中的基础接口，不依赖刚新增的功能。

- [NumPy 初学者](https://numpy.org/doc/stable/user/absolute_beginners.html)：先解释 shape/axis/dtype，再讲批量运算；`empty` 不是清零。
- [NumPy copies/views](https://numpy.org/doc/stable/user/basics.copies.html)：基础切片通常是视图，高级索引取值是副本，reshape 是否共享要核对实际布局。
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)：从尾轴对齐，兼容条件是相等或为 1；不保证任何广播中间结果都省内存。
- [NumPy einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)：显式保留轴和收缩轴，与 matmul 的语义分别说明。
- [PyTorch Tensor Views](https://docs.pytorch.org/docs/2.14/tensor_view.html)：转置可非连续，`view` 有 stride 条件，`reshape` 可能复制。
- [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html)：叶子参数、梯度模式、图生命周期与原地操作。
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)：传 logits，类别索引/概率目标契约不同，索引目标 ignore_index 与 mean 分母必须说明。
- [PyTorch data](https://docs.pytorch.org/docs/stable/data.html)、[optim](https://docs.pytorch.org/docs/stable/optim.html)：批次组织与参数更新是不同步骤。
- [PyTorch saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)：state_dict、优化器状态与恢复边界。

## 实施拆分与验收

| 负责人 | 独占范围 |
| --- | --- |
| NumPy agent | F1、F2、相关交互/图/CPU 代码与可运行模型检查 |
| Tensor agent | F3、F4、相关交互/图/CPU 代码与可运行模型检查 |
| Module agent | F5、相关交互/图/CPU 代码与可运行模型检查 |
| 主 agent | F6、导读、导航/入口/CI、集成与交叉审查 |

验收依次为：六份 CPU API 验证脚本；可视化模型断言；类型检查；构建、章节顺序与内部链接；320/390/768/1024/1440 宽度；明暗主题、真实图片、公式与非空 SVG；键盘操作和模式/边界检查。七条路线导航需无溢出且各屏宽可达。

用户先要求设计并实施，验收完成后于 2026-09-27 明确要求提交并发布；按仓库流程推送 master，由 CI 验证后部署 GitHub Pages。
