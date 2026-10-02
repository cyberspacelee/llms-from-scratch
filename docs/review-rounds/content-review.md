# 逐篇独立内容复审

审查范围：数学、框架、训练、后训练、系统和已有七篇进阶章节及各路线索引。原理十五页及新五篇研究章在本轮后续交叉复审；GPU 由另一位独立 reviewer 覆盖。依据当前 MDX、统一 Python 实现与 CPU 实验真实输出；此文件记录复审期间的发现，修正后追加状态，不把旧版本问题当作最终现状。

## 数学

| 页面 | 结论与核对 | 具体修改/摘录建议 |
| --- | --- | --- |
| index | 需修导读。主线 M1–M10 合理，但新 M12 尚未进入分支说明。 | 将谱分解与低秩拆为 M11/M12；跨路线记号表映射 D/H_q/H_kv/V。已通知 root。 |
| events | 通过。1/3 条件概率与 1/4 联合概率区分明确，样本空间和权重变式都有定义。 | 主例可从 probability.py 的事件核对区块摘录；避免把全部概率验证函数展开。 |
| bayes | 通过。链式概率 0.252、证据分母、零概率前缀边界正确；练习定义完整。 | probability.py 联合表/条件化区块。 |
| probability | 通过。均值 2、方差 3、B=16/B=60 的 Chebyshev 界与有放回采样条件一致。 | probability.py 样本均值方差区块。 |
| information | 通过。H=0.693147、条件熵=0.477386、CE=0.836988、正/反向 KL 的数值与定义一致。证明内容较长但层级合理。 | 可将额外 KL 非负证明折叠；概率文件的 CE/KL 区块。 |
| distributions | 内容通过。logits=(0,ln2,ln3) 与 softmax/梯度正确，连续密度不冒充概率。 | 元数据缺正文明确要求的 math/information 先修。概率文件稳定 softmax 区块。 |
| linear-algebra | 通过。X/W/b 形状、逐行线性映射与广播陷阱一致；秩与可训练低秩增量有区分。 | linear_algebra.py 对应 matmul/broadcast 区块。 |
| calculus | 内容通过。L0=4、梯度=(2,4)、η=0.2 后 L=1.36，η=0.6 反例成立。 | prerequisites 为空但正文需要 M6 的向量内积；尾部 M11“谱分解与低秩”名称应按拆章更新。 |
| matrix-calculus | 通过。143/24 损失、dW=[[-.5,-1],[19/3,22/3]]、db=(-.5,1) 与归约轴对应正确。 | linear_algebra.py 线性层手算梯度区块。 |
| backprop | 通过。两层共享分支主例前向/全部梯度、batch 偏置归约、不等长微批权重与脚本一致。 | 目前正文有重复内嵌可运行代码，改为 backprop.py 的 same_network 区块。 |
| training | 内容通过。XOR 四点、两隐藏单元、初值与一次更新来源明确，训练拟合与泛化分开。 | prerequisites 应包含正文要求的 math/distributions；neural_network.py 初值及单步区块。 |
| spectral | 通过。Q=(.8,.6) 旋转、λ=(1,9)、η<2/9 与 GUI/独立计算相符。 | 当前 symbol=verify 会展开跨两章的大验证函数；应仅摘录 eigen 主例区块。 |
| low-rank | 通过。W 的输入/输出方向、秩一误差、条件数3和最坏方向扰动正确。 | 当前同样摘录整 verify；拆 svd 区块。精确恢复链接建议直达 training/checkpoint；M12虽写无需二次损失，但首段依赖 spectral 的 Q，已在正文给出完整值，无隐含计算缺口。 |

数学整改状态：索引 M11/M12 和跨路线记号已更新；distributions、calculus、training 的显式先修已补。全部十二篇改为短函数/region 摘录。M8 原脚本只有另一组两行例，已增加正文三行例的 143/24 及全部梯度断言；M12 原脚本仅验证对角例，已增加正文旋转矩阵的秩一误差、方向与扰动断言。反传内嵌代码改为同源摘录，并修正原先 retain_grad 的失效指代。

## 框架

| 页面 | 结论与核对 | 整改与实际摘录 |
| --- | --- | --- |
| index | 通过。F1–F7 的 storage→求导→训练→compile 次序合理；compile 是可选扩展。 | 索引不需源码。 |
| numpy-arrays | 通过。六值、view/copy、byte stride=(24,8)、重复高级索引行为一致。 | storage region。 |
| numpy-computation | 通过。Z=[[-2,4],[-2,13]]、行归约、总体方差2/3、稳定 softmax 正确。 | softmax 函数。 |
| torch-tensors | 通过。相同六值被明确重新解释，四维拆头/错误 reshape、三角轴验证正确。 | head_axes region。 |
| autograd | 通过。向量主例 u=(3,-1)、L=8、dw=(40,-12)、db=32；VJP 和存储边界正确。 | branch region。 |
| modules-losses | 通过。TinyWords20参数、eps=1手算、四目标0.801667、mask按有效数一致。 | 修旧 helper 名为 reference_token_loss；摘录该函数。 |
| data-and-training | 通过。五样本2+2+1加权均值9、错误11.66667；SGD=(.7,-.4)、AdamW恢复一致。 | 精确恢复链接改T5，旧环境版本标为历史记录；update函数。 |
| torch-compile | 通过。多项式输出/梯度、静态2图/动态1图、断图均有实际计数。 | 修统一模型 .logits 访问；polynomial函数。 |

## 训练

| 页面 | 结论与核对 | 整改与实际摘录 |
| --- | --- | --- |
| index | 通过。数据→优化→文本训练→评估→精确恢复，扩展分支分明。 | 索引不需源码。 |
| data | 通过。四记录三唯一、train2/valid1；连续打包5目标、文档隔离4目标与mask一致。 | pack_documents函数。 |
| optimization | 通过。均值.625、梯度(.5,-.25)、不等微批反例、AdamW(.88,-1.86)、clip/AMP顺序一致。 | manual_adamw函数。 |
| pretraining | 通过。六篇训练/两篇验证、BPE16merge、24/24/2尾窗；Q宽32与残差D24有意分开。 | 验证描述改直接loss_sum/count聚合；已有 corpus/new_run/train_steps/evaluate 摘录。 |
| checkpoint | 通过。四ID模型作为协议探针，18+12恢复与30次完整轨迹；文本20+8随机恢复边界明确。 | 改旧NLL为1.386083→.593010，固定探针1.355465→.003959；P12缓存链接；train_steps函数。 |
| evaluation | 通过。手算2.5/2.2 NLL、16目标各计一次、成对bootstrap/Wilson边界一致。 | T3标题更新；target_windows函数，ablation改matched_branches短区块。 |
| scaling | 通过。16格/holdout与N*=18.929、D*=5.283；pilot日志不冒充独立样本。 | M12低秩链接与T5 FLOP探针标签修正；optimal_allocation函数。 |
| data-engineering | 通过。八记录筛五、Jaccard.4、MinHash.410与band概率、ABC分组、重复量和clip界一致。 | 补数据与概率先修；duplicate_groups函数。 |
| distributed-training | 通过。两rank3+2样本/14目标，梯度-32/14、DDP R/N补偿和no_sync范围正确。 | T5精确恢复链接；worker函数。 |

## 后训练

| 页面 | 结论与核对 | 整改与实际摘录 |
| --- | --- | --- |
| index | 通过。H1–H5按目标分支，DPO不是RLVR必修先修。 | 索引不需源码。 |
| instruction-tuning | 通过。四有效目标行4/5/9/10；回复mask不切prompt梯度。 | 修旧NLL为2.115084→.006339、旧回复[[7,6],[7,5]]；prompt_gradient region。 |
| lora | 通过。rank2 20适配参数/24base、零B首步梯度、手算1.555142→1.179401成立。 | 修H1/M12标签、局部d/m公式、六类例与H1词表区别；LoRALinear函数。 |
| distillation | 通过。tau²KL方向、teacher detach、空局部mask和上层skip界；三类KL=.296794与梯度可核对。 | 既有distillation_loss短函数。 |
| dpo | 通过。p=.44/.18、q=.30/.20、margin=.097671、loss=.645504、回复mask与梯度方向正确。 | T5→H1标签；response_log_prob函数。 |
| reasoning-rl | 通过。四候选期望.5、分组adv与clip、old/ref区别、24轮实际Transformer路径清楚；保留heldout退化结果。 | rollout_update region。 |

## 系统

| 页面 | 结论与核对 | 整改与实际摘录 |
| --- | --- | --- |
| index | 通过。账本→硬件→IO→调度/缓存→部署→指标，混合/窗口缓存为分支。 | 索引不需源码。 |
| ledger | 通过。372参数、744bytes、2432/704线性FLOP、64/80cachebytes；8B模型值有固定配置来源。 | 修部分小写维度与P12链接；ledger函数。 |
| accelerator | 通过。H100SXM稠密半精度989.5TF/s、3.35TB/s用于理论下界，不宣称测得延迟。 | 修注意力公式D；roofline_time函数。 |
| flash-attention | 通过。两块max2/5、归一化量和5.43293输出；长方形offsetmask、全遮挡处理正确。 | 修T_q D_v符号；online_attention函数。 |
| engine-runtime | 通过。A/B/C四轮计划与KV lag对应，preempt保输出/重算缓存，固定策略界明确。 | Scheduler.step函数。 |
| paged-cache | 通过。块4、共享六位置、A写slot10/B写slot6、refs与释放一致；prefix身份非局部ID等价。 | Pages.append函数。 |
| engine-execution | 通过。cuQ=(0,3,5,9)、cuK=(0,3,9,13)、槽与context一致；graph哑行slot=-1。 | prefill/decode函数。 |
| quantization | 通过。三位全局/逐行/逐组误差、packed理论bytes与int8实际存储有区分。 | symmetric/affine函数。 |
| speculative-decoding | 通过。接受质量(.5,.1,.2)、残差仅ID1、错误重抽分布、首次拒绝与bonus缓存边界正确。 | 修sum_xD(x)大小写；compensation/speculative_step函数。 |
| cluster | 通过。矩阵分片输出与部分和、通信bytes、pipeline空泡和带宽下界不冒充NCCL实测。 | verify函数（38行一个算例）。 |
| serving-evaluation | 通过。TTFT1/.5/.6、TPOT .25/.10、取消/一token空值、nearest-rank与goodput分母正确。 | measure函数。 |
| hybrid-cache | 通过。FULL/SWA/递归三字段联合恢复；b4/b5/b8只命中部分字段的反例正确。 | ch15 state_size region。 |
| window-cache | 通过。窗口3逻辑位置与tags/物理槽分开，chunk写入覆盖反例、完整窗口模型等价边界正确。 | long_context ring_cache region。 |

## 已有进阶

| 页面 | 结论与核对 | 整改与实际摘录 |
| --- | --- | --- |
| index | 通过。A1–A7机制与A8–A12研究支线明确，章节实验号另行标识。 | 索引不需源码。 |
| moe | 通过。三token/top2输出、hits=(2,1,2,1)、均衡1.198394lambda、16总/8活跃参数正确。 | 修失效P8锚点为P10独立FFN、补先修、D公式；route/dispatch函数。 |
| mla | 通过。两头latent与RoPE分支、显式/吸收两路相等、输出(1.584319,3.539461)、4 vs10/12缓存元素正确。 | P12缓存先修、路线尾部改A3；absorbed_decode region。 |
| long-context | 通过。插值p8→旧p4、换频率需重算旧K、窗口5位置均值4而全历史2.5；环形与相同窗口参照相等。 | 尾部改A4；rotate函数。 |
| state-space | 通过。固定3.125/选择4.25、仿射组合(.125,4.25)、结合但不交换、扫描结构条件正确。 | 修D_s公式；compose函数。 |
| hybrid-attention | 通过。三个delta矩阵、门先衰减、eta=1替换单位key、第二层(33/19,1)等可独立算出。 | 修D_k/D_v公式；delta_step函数。 |
| sparse-attention | 通过。全520/12、选集340/7、漏质量5/12及误差界；indexer和主attention打分分开。 | 补KV先修、修D_I/D_h公式；gathered_attention函数。 |
| multimodal | 通过。四patch均值、6→8 connector、九输入位置与监督6/7正确；文本目标不屏蔽视觉梯度。 | 补H1先修；原脚本仅手写attention，已改唯一Transformer.inputs_embeds，验证connector梯度与7→9缓存；visual_prefix region。 |

## 原理路线交叉复审

本节由未编写这批 MDX 的 Python agent 逐篇阅读；原理作者同时独立复审 Python。十五页包含一个索引和十四章。

| 页面 | 结论与核对 | 修正/源码边界 |
| --- | --- | --- |
| index | 通过。四词表基础模型→位置/三个结构单元→生成/cache→2748参数完整模型；P14选读分支独立。 | first-model已包含P11/P12，未跳过完整模型实际先修。 |
| tokenization | 通过。猫UTF-8三字节、两次并列BPE→ID256/257、特殊258/259、V260、共享行梯度(4,1)正确。 | 修embedding参数VD；fit/encode摘录有效。 |
| language-modeling | 通过。三移位目标、序列概率.252、NLL1.378326/3、PPL1.583190；零表SGD概率.317501正确。 | 空局部损失与全局跳更新边界清楚；两种损失实现不混接口。 |
| attention | 通过。Q/K人为表、位置1/2输出、多头与batch轴、全遮挡查询、练习均有定义。 | 修线性成本TD²；单头参考与完整模型边界明确。 |
| decoder | 通过。74参数/268参数两个配置分开；手算三行loss1.0370、Q/K当前零梯度、随机模型有限梯度核对正确。 | 明确随机模型eps=1e-5，手算验证eps=1；统一块数L。 |
| attention-order | 通过。标量(1,2,3)、无位置等变/固定causal反例、同步搬动mask和位置恢复等变，四变式正确。 | 显式位置与因果前缀作用未混淆。 |
| sinusoidal | 通过。π/8手算p0/4/8、A方向约定、D8频率、sin-only与投影反例；指数证明已折叠选读。 | 共享实现奇数尾维和偶数证明边界分开。 |
| rope | 通过。i3/j1、点积√3/2、四维(√3+1)/2、QK Norm不可交换与缓存位置反例正确。 | 修局部k=(c,d)被误改D、QK Norm前后指代。 |
| normalization | 通过。LN(2,0)=(1,-1)/√2、RMS=(2,0)/√3、平移/epsilon缩放反例与Pre/Post雅可比正确。 | 另用共享RMSNorm eps1实际核对主例；ch05随机D32与手算不混。 |
| head-sharing | 通过。两查询p=.669762、共享输出(2p,0)、24/32权重、8/16缓存元素、4/2组身份正确。 | 134行整类改grouped_read region；摘录保持持久KV头与临时展开区别。 |
| gated-ffn | 通过。单位投影SiLU(1)=.731059、负输入正.268941反例、门零保残差、同预算30→20正确。 | 单算子主例与完整组合独立。 |
| generation | 通过。温度平方、top-k/p/min-p、惩罚BOS/A顺序、greedy序列反例与EOS/预算分开；代码固定表+真实模型核对。 | 长上下文标签A4→A3。 |
| kv-cache | 通过。RoPE频率1、C行2.061223、3/1/1与3/2矩形mask、逐层归纳、两层窗口间接历史例正确。 | 长上下文标签A3、块数L；原始token裁剪不冒充相同窗口整模型重算。 |
| complete-transformer | 通过。2748参数、四目标、实际NLL2.138576→.002258、EOS序列五token/缓存四token；48参数oracle不同配置已折叠。 | 修绑定参数VD与块数L；缓存128为启用cache容量账，普通训练未返回cache已明确。 |
| encoder-cross-attention | 通过。三mask身份、Q2×K3输出(4/3,2/3)/(1,1)、错tril反例、两种teacher forcing入口正确。 | 经典PostLN/ReLU/sincos、基础、现代配置明确区分；encoder-only MLM不冒充NTP。 |

## 新五篇研究章交叉复审

| 页面 | 结论与核对 | 实际计算验证 |
| --- | --- | --- |
| multi-token-prediction | 通过。顺序模块消费真实中间token、偏移r+1、5/4/3标签、链valid与按深度平均有定义。 | ch14 CPU实际标签[2…6]/[3…6]/[4…6]、额外长度4/3；训练真值分支不宣称推理草稿已实现。 |
| sequence-compression | 通过。摘要结束点与精确尾部防重复/未来；候选[1,5,8,10,12]均值7.2，dense6。 | 共享函数实际输出p6=7.2、p5=6；改变未来V不改变旧输出。当前保完整KV、未省进程内存边界诚实。 |
| residual-streams | 通过。双随机混合保平均不保各stream，注入后平均可变；Sinkhorn近似与depth-source轴区别明确。 | 共享sinkhorn得到[[.75,.25],[.25,.75]]，mhc_update得[[2,1],[2,3]]，零query深度均值(1,1)。未接完整模型边界明确。 |
| conditional-memory | 通过。hash=(1,3,0)、碰撞、归一化gate区间、dilation2读取t/t−2/t−4、24参数账正确。 | 设置正文表行/单位投影实际复现1.462118与.537882，卷积中tap输出(.731059,.731059)，参数数24。未实现增量hash/conv cache明确。 |
| shared-kv | 通过。lower必须因果、K=V额外假设、consumer独立Q/O、绝对query_offset和48/96元素账正确。 | 共享读取函数实际复现1.880797/1.119203；随机整段/逐行核对另保留，不宣称端到端lower增量已测。 |

## 验证和导航闭环

- 逐篇复审共77页：57页六条原路线+15页原理+5页新增研究；GPU另有独立报告。
- 100处显式CodeFile符号/region已用构建的同一AST提取器逐个验证，全部存在且没有超过70行的摘录。
- 新增M8/M12正文主例与A7统一视觉接线均CPU运行通过；G4新增正文tile16/32的2/8KiB资源核对通过；ruff check通过。
- 新研究五篇的固定手算例与共享函数独立实算通过，结果见上表；未将随机章节实验输出冒充固定例数值。
- first-model的13步已按实际先修排列，H/S路线通过跨路线先修入口保持单一页脚导航。复审发现text-training将需T1的T7放在T1之前，root已调整为T1→T7→T2→T3→T4→T5；该前向先修问题已闭环。

末轮补充：checkpoint的真实train_steps摘录已移到更新步骤旁，并解释token_loss导入别名sequence_loss；F4实际branch摘录已移到backward核对旁。Python剩余数学旧编号输出已改M6/M8等当前章号。最终25项单元检查24通过、1项CUDA因无硬件跳过；日志 `/tmp/python-content-review-tests.log`。139个Python文件格式检查通过（site内提取器由root单独格式化）。
