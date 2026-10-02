独立视觉与 GPU 内容复审，2026-10-02。审查者没有实施本轮共享视觉组件和 GPU 教材改写；本轮阅读源码、九篇实际 MDX、Python/CUDA/Triton 参考，并独立复算变式。全页浏览器、移动端、主题与水合后的实际测量由 root 另行验收，本文不冒充浏览器结果。

最终结论：GPU 四章主线与四篇实践的边界明确，CLI 和源码摘录有效；本轮发现均已修复并独立复核关闭。G3 明确手算与真实 CUDA 数据切换，无 SVG 的栅格图滚动区获得独立 overflow 测量，CourseMap 按目标路线保留后续顺序，G4 CLI 已核对并报告正文的新 tile16/32 案例。

| 页面 | 主要目标、先修与案例 | 独立审查结论 |
| --- | --- | --- |
| GPU index | 从唯一输出负责人、内存协作、tile，到真实设备时间；四项实践作为分支。 | 顺序可解释，明确 Python 数组下标与矩阵乘入口。CourseMap 的 kernel route 仅包含四章主线，实践不强制穿插。图中的 SM/block 数量明确不当作规格；CPU/GPU 证据分界合理。 |
| G1 execution-model | numpy 数组先修；100 元素、64-thread block 的唯一归属。 | 商余数证明、尾部 28 个位置、warp lane、独立 block 调度、grid-stride 到二维越界反例连续。40-thread 尾 warp 与“存在但被 mask”区分清楚。owners/stride_owners 摘录和 CUDA add region 有效，当前 CLI 通过；257 元素/40 threads/3-block stride 新变式覆盖一次。 |
| G2 memory-and-sync | G1 后；35×67→67×35 转置，32×8 threads、32×33 shared。 | lane 地址段→交换写入者→bank→屏障→3×3 边缘完整连贯。32-byte sector 与 32-bit bank 假设注明，广播和竞争没有混淆。转置 producer/consumer 的 j=0/8/16/24 与 CUDA 一致。33×65 新形状逐输出验证通过。栅格地址图现在不依赖 SVG，owner 与 child 的 ResizeObserver、load 事件均触发 overflow/focus 测量，缺失项关闭。 |
| G3 kernel-design | G2 与矩阵乘；5×7·7×6、t=4，C00=20+92=112。 | 所有权、K 分块、三种边界、发布/回收、局部强度、epilogue 依次推进同一矩阵乘。修正正文：图内坐标需要代回公式；后续生命周期显示累计值。明确真实 CUDA 只实例化 tile16/32，host 改用正负周期数值，不以112验收。3×5·5×4、tile3 变式得到 C23=-15、逻辑读取50，独立 NumPy oracle 通过。 |
| G4 measurement | G3 后；相同256³、float32、tile16与tile32。 | 0.03355 GFLOP、输入请求数2097152/1048576、shared2/8KiB、32/64 registers 驻留、同 stream event、三种时间与端到端边界正确。10 warmup、31 repeats、设备限制与相同输入真实存在于 CUDA；计时固定顺序偏差及无设备证据明确。Python CLI 新增 tile16/2KiB→8blocks、tile32/8KiB→2blocks 核对，实际重新运行打印两者100% occupancy，旧16KiB/75%明确标为附加变式，案例同步项关闭。512threads/40regs/12KiB 新变式得到3blocks、75%，无event而碰巧同结束时刻仍被判不具顺序。 |
| G5 indexing-practice | optional，G1 后；三个64-thread布局与8×4×2双射。 | 一个问题、明确逆变换，未把坐标当GPU核心位置。Node入口和 Python demo 摘录有效。独立4×3×5变式枚举60个唯一编号；59→(3,2,4)、warp1/lane27，与直接商余数核对一致。 |
| G6 reduction-practice | optional，G2 后；八整数三层树归约→1003元素两kernel。 | 36的逐层合并、无效21槽中性值、全thread屏障、跨block阶段与浮点结合律说明充分。reduce region 与同stream第二次launch对应；host零长度不启动grid。长度1003的全负输入求和=-3009通过，最大值无效槽0的反例合理。没有把CPU拷贝模拟当并发正确性证明。 |
| G7 triton-softmax | optional，G3与信息论后；(1000,1001,1002)稳定softmax。 | 地址mask与语义mask、空行零策略、正stride、float32、宽1–4096与输出连续契约符合源码。被mask也必须有限的限制写明，合法分布与工程零行明确。变式只允许第0/2项得到(sigmoid(-2),0,sigmoid(2))，全mask行零。真正Triton执行仍需要GPU，未虚报设备运行。 |
| G8 model-profiling | optional，G4与完整Transformer后；同权重两层现代Decoder、manual/SDPA。 | 输出与全部参数梯度先验收，采集前排除反向，CPUfloat64和CUDAfloat32容差合理。CLI实际生成CPU Chrome trace并通过logits/所有参数梯度对照；本文不将一次profiler时间写成速度结论。CUDA30次event均值与峰值增量描述匹配，明确不是独立样本中位数或端到端时间。 |

共享系统逐项审查：

| 系统 | 结论与后续 |
| --- | --- |
| `Figure` / `FigureShell` | 所有实际 SVG hex 色均被语义 palette 映射，没有游离色；热图色阶保留顺序。内联去掉绝对宽高、font-family，id与url/href引用加唯一前缀；静态图、numbered caption、主题边框共用FigureShell。最终源码为所有owner注册overflow observer及图片load事件，无SVG的栅格图也可按overflow获得键盘region；问题关闭。 |
| `SvgCanvas` / `diagram-viewport` | 使用真实CTM与字体像素计算最小可读宽度，React与静态图统一12px门槛；静态路径另覆盖Matplotlib路径字形与Mermaid foreignObject。超宽局部滚动、窄布局重排与实际字体下限的设计合理。旧Observer注册用WeakSet，避免重复注册；浏览器仍须检查hidden/mobile双图、尺寸变化后的focus与observer行为，代码审查不能替代实测。 |
| `DataViews` | TraceTable、TokenSequence、MatrixGrid公共表示成立：状态有文字/数字补充，不只靠颜色；表格有caption、行列scope，超宽可聚焦局部滚动。没有必要把小矩阵再画成独立SVG。 |
| `Lab` | LabFrame、Controls、Range、Select、Toggle、Button、StepControls、Readout、pen、Arrow和useWidth覆盖共用职责。标签采用useId，marker/clip用唯一实例id，动态readout aria-live。没有组件私有style或硬编码黑白/旧palette样式残留。 |
| `CodeFile` | Python AST定义含decorator与多行签名，源码行号真实；region支持CUDA和嵌套边界，缺失/歧义/错误闭合拒绝。全文折叠与主摘录分离，显式run保持可复制。GPU20项symbols/regions逐个运行extractor均通过；extractor的边界自检也通过。 |
| `CourseMap` | 实际使用learningPaths，首页按目标展开真实steps，轨道页提供先修链接。最终源码将后续拆为 `main.slice(6)` 的“路线后续”与不属于main的“其他目标与选读”两个details，目录order不会再重排route；问题关闭。当前原理主路线明确纳入generation/cache，13步的必要先修顺序也通过独立核对。 |
| 全站入口与CSS | 实际91个独立Lab模块、94处MDX挂载（“92个*Lab.tsx”包含共享Lab.tsx）。每个Lab都使用LabFrame，所有SVG绘图都经SvgCanvas；没有raw svg入口、旧聚合或Qwen3Layer引用/资产残留。只有全局CSS入口，字体、浅深色、prose、Shiki、KaTeX、Mermaid和overflow共用语义token。此项是源码遍历结果，不表示94个水合入口都经浏览器测试。 |

至少三项独立变式已经扩大为十项检查：

- 257元素、40threads，直接映射与3-block grid-stride覆盖完全一致；256归block6/thread16。
- 17个活跃lane、stride3/offset5覆盖7个sector；shared跨度35无冲突，跨度34/17lane最多两地址同bank。
- 33×65转置右下输入(32,64)与输出(64,32)的覆盖和地址一致。
- 3×5·5×4/tile3得到C23=-15、有效请求50，与独立NumPy矩阵乘一致。
- 512threads/40regs/12KiB驻留3blocks、48/64warps；A2/B7/C3没有event时虽恰好10结束，仍无依赖保证。
- partial-mask softmax的两点分布与显式指数公式一致；全mask零行；1003个-3归约=-3009。
- TS4×3×5编号双射、tile3右下C33=28、row4/col5的C45=-84且融合/独立epilogue均输出0。C33初次预期误用了另一坐标的符号，逐项复算为-12-10-6+0+8+18+30=28，确认模型正确。

原始资料复核采用 [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory) 的访问/bank与同步条件，以及 [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) 的稳定化、power-of-two地址padding及program级归约。网页代码的教学约束比官方通用例子窄，正文已明确其支持范围，没有推断生产加速比。

运行证据：四个主线CPU CLI、GPU indexing/lifecycle/transpose三个Node检查、七组Python独立边界变式、三个TS独立变式、20个GPU摘录、extractor自检、模型CPU profile/trace均通过。CUDA tile16/32的静态屏障与四shape guarded-load索引模拟沿用本次独立Python复审证据；无nvcc、CUDA设备，没有真实kernel、Triton、Sanitizer或GPU event性能结论。

最终关闭复核额外检查了实际 `learningPaths` 与所有路线 MDX 的 frontmatter 先修。文本训练主路线为 T1 data → T7 data-engineering → T2 optimization → T3 pretraining → T4 evaluation → T5 checkpoint；T7 的必要先修 T1 在前，数据身份与目标先于数据工程，真实原料处理再先于更新和恢复。五条路线的所有内部必要先修均早于使用它的步骤，文件存在。`lessons.ts` 新校验使用当前route的路径集合与completed集合，缺页、重复主路线归属、路线内先修倒置都会在构建时拒绝，同时保留全库先修DAG检查。该检查不会把跨路线先修强行插入当前路线，页头仍负责给出这些必要入口。
