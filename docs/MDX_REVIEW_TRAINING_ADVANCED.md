> 重构前的规划或审查记录。当前结构与验证结果见 [实施记录](MDX_REBUILD_STATUS.md)，教学约定见 [当前课程设计](CHAPTER_BLUEPRINTS.md)。

# Training 与 Advanced MDX 逐页整改记录

本批审阅 20 页，按 training、advanced 两批修改。修改前的问题已交给主代理，汇总见 [全站审计](MDX_READABILITY_AUDIT.md)。本记录对照逐页实际内容及现有交互控件，不以句长统计代替阅读。

采用接近 ASD-STE100 的清晰写作原则：一句完成一个判断，一个段落解释一个主题，操作说明明确控件、顺序和应观察的结果。这是中文教学改写，并非 ASD-STE100 合规认证。原有公式、推导、脚本代码块、数值条件和练习保留。

以下“代理复述/变式”是编辑代理依据正文完成的读者检查，不是真实新手试读，也不是独立盲测。

## 第一批：Training

| 页面 | 修改前的具体问题 | 已完成的处理 | 代理复述/变式核对 |
| --- | --- | --- | --- |
| [index.mdx](../site/src/content/lessons/training/index.mdx) | “阅读路线”首段合并 T1–T3 的数据边界、梯度归约与恢复能力，读者难定位每章验收对象。 | 每章各用一段，分别说明得到什么能力；保留 CourseMap 和实际执行顺序提醒。 | 变式：已有可用 checkpoint，该从何处进入？T4 可以独立做受控评估；准备新语料仍先执行 T8 的清理，再应用 T1 划分。 |
| [data.mdx](../site/src/content/lessons/training/data.mdx) | Tokenizer 节末把 ID/embedding 对应、恢复协议及验证选择写在同一段。 | 分为“ID 决定查哪一行”和“验证结果参与选择”两段；保留原有 packing 图与具体行任务。 | 变式：验证未反向传播，但据验证结果修改词表，是否仍独立？不是，验证信息已参与选择；另留测试集才能做最终报告。 |
| [optimization.mdx](../site/src/content/lessons/training/optimization.mdx) | TokenWeightLab 默认值为 2/6 个目标、NLL 1/3，与正文平方损失的 1/3 划分不同；说明仅要求“设不同值”。 | 明确它是独立归约教具；指定默认值算出 2.5 与 2，再改成等目标数核对两结果重合。 | 变式：两个均值不同，但各有 2 个有效目标，为什么可以等权？每批权重都是 2/4。相等只说明此输入下正确，不能证明实现普遍正确。 |
| [pretraining.mdx](../site/src/content/lessons/training/pretraining.mdx) | CheckpointReplayLab 后一段包含参数、RNG、保存边界及三种遗漏，观察路径被埋在条件中。 | 将探针设定、按菜单查首偏离、Decoder 脚本验收分开。 | 变式：漏 RNG 时加载后前向相同，能否宣布恢复成功？不能；随机状态要在下一次 shuffle 才改变样本，须核对后续更新。 |
| [evaluation.mdx](../site/src/content/lessons/training/evaluation.mdx) | 实验说明同时讲三窗、四篇合并、去掉 D4 和改变步幅，读者缺明确次序。 | 先核对覆盖，再取消 D4 比较 A/B，再改变步幅；每次声明读数来源。 | 变式：去掉 D4，A/B 的新 NLL 为 22/10=2.2、20.8/10=2.08；PPL 必须重新取指数。改步幅不会让固定 s=2 的观测自动成为新分数。 |
| [instruction-tuning.mdx](../site/src/content/lessons/training/instruction-tuning.mdx) | ChatMaskLab 后同段介绍有效行、SYS 反向、标量探针、第二轮和空截断。 | 按行 4→行 0、只第二轮、截断安排检查；单独声明数值梯度来自标量探针。 | 变式：只监督第二轮，分母为 2；SYS 仍在两行的因果历史中。截到第一个 assistant 头后没有回复目标，不能执行空平均。 |
| [lora.mdx](../site/src/content/lessons/training/lora.mdx) | LoraTraceLab 说明把初始化、首步、零零反例及合并挤在一段；“输出类别”易被当作训练目标。 | 分成三次操作；明确输出类别只选择读数，训练目标固定下标 2。 | 变式：两因子均为零时推进两次，数据梯度仍为零；单边零时 B 先动，下一次 A 有梯度。合并比较必须使用同一更新数的 A/B。 |
| [scaling-laws.mdx](../site/src/content/lessons/training/scaling-laws.mdx) | 实验只说“拖动 N”，没有与离散候选比较相连的操作任务。 | 固定 K=100、α=.6、β=.4，指定 N=16→24，对照 D、两项误差与总损失。 | 变式：N 增大并不保证总损失下降；此条件下总损失从约 2.2188 增至 2.2243，数据减少的代价超过容量收益。 |
| [data-engineering.mdx](../site/src/content/lessons/training/data-engineering.mdx) | 八阶段处理顺序只在一条长箭头串中，训练侧拟合与验证侧应用未直接画出。 | 用 Mermaid 展示流程与划分后的两分支；文字跟踪 H 的排除位置和验证侧冻结 tokenizer。 | 变式：H 与 A 逐字相同，仍不能继承许可；验证 D 可以使用冻结 tokenizer 编码，但不能参与拟合。 |
| [distributed-training.mdx](../site/src/content/lessons/training/distributed-training.mdx) | StateAllocationLab 的说明只说增加 rank，没有要求同时检查单 rank 与全局总账。 | 指定 4→8 个 rank，分别选择 DDP 和 ZeRO-3，核对每 rank 及总量；峰值范围另成段。 | 变式：八 rank 时 DDP 每 rank 16 GB、总计 128 GB；ZeRO-3 每 rank 2 GB、总计 16 GB。每 rank 下降不能当作峰值已被验证。 |

## 第二批：Advanced

| 页面 | 修改前的具体问题 | 已完成的处理 | 代理复述/变式核对 |
| --- | --- | --- | --- |
| [index.mdx](../site/src/content/lessons/advanced/index.mdx) | 五条路线项目混合问题、入口、分支与跨章先修，不便比较 MoE/MLA 和递归/稀疏路线。 | 改为按具体问题选择的路线表；表后说明两对路线改变的不同对象，保留 CourseMap。 | 变式：只想减少历史读取，可以进入 A8；它保留逐位置状态，再选择集合。A6–A7 先把历史汇总进递归状态。 |
| [moe.mdx](../site/src/content/lessons/advanced/moe.mdx) | MoeRouteLab 只要求“改变分数”，未给入选集合变化的明确操作。 | 指定专家 0 的 4→0，再把专家 1 的 3→2，分别观察换专家和固定集合内权重变化。 | 变式：分数变为 (0,2,1,0)，仍选专家 1、2；两个专家接收完整 x，输出按其 softmax 权重相加。 |
| [mla.mdx](../site/src/content/lessons/advanced/mla.mdx) | 正文显式缓存每位置 12 元素，交互显示 10 元素；位置键是否跨头复制未说明。交互说明也未给逐头操作。 | 用表明确 4=MLA 潜变量+共享位置键、10=显式内容 K/V+一份位置键、12=每头完整拼接键和值；指定逐头、逐步骤切换路径。 | 变式：三位置 MLA 为 12 元素；显式共享位置键为 30 元素；按头复制位置键为 36 元素。10/12 布局算同一注意力，GQA 比较仅是形状参照。 |
| [dpo.mdx](../site/src/content/lessons/advanced/dpo.mdx) | DpoLedgerLab 被放在原始推导“选读”标题内，主线似然和 mask 检查被误归为进阶内容。 | 移到选读标题前；分别检查四项似然、四行 mask、去掉 EOS 和改 λ 的作用。 | 变式：去掉 EOS 后相对差为 ln(11/6)，不是 ln(44/27)；说明改变了回复 span。调 λ 不改变四项回复概率。 |
| [long-context.mdx](../site/src/content/lessons/advanced/long-context.mdx) | WindowCacheLab 说明主要给初态和等价条件，未直接要求观察 slot 覆盖；易把地址图当作 logits 检验。 | 给位置 5→6 的覆盖任务；单独说明交互只算地址，前向结果由脚本检验。 | 变式：位置 6 可见 4/5/6，slot 为 1/2/0；slot 0 的 tag 是 6，RoPE 位置也为 6。物理槽不会把逻辑位置重置。 |
| [reasoning-rl.mdx](../site/src/content/lessons/advanced/reasoning-rl.mdx) | 同一密集段解释旧策略、参考策略、KL 与长度归约，三份策略的职责难追踪。 | Mermaid 展示采样、验证、当前重算、裁剪及参考 KL 的路径；固定范围与长度归约另写。 | 变式：同批样本重复更新时旧 log-prob 不重算成当前值；参考模型是长期起点；仅当前策略更新。验证器奖励不需要对模型参数求导。 |
| [state-space.mdx](../site/src/content/lessons/advanced/state-space.mdx) | StateScanLab 位于 Mamba 节，实际演示前一节的递推与仿射组合；说明没有逐步检查任务。 | 移到扫描推导末；固定 t=3 核对 3.125→4.25，再切换组合核对 (.125,4.25)。 | 变式：零初态下累计输出为 B，但 A=.125 仍表示非零初态的传递系数；组合满足结合律，时间顺序不能交换。 |
| [hybrid-attention.mdx](../site/src/content/lessons/advanced/hybrid-attention.mdx) | DeltaStateLab 只说两控制作用不同，没有用边界值暴露差别。 | 指定 (.5,1)→(.5,0)→(0,1)，核对旧预测与更新关联。 | 变式：η=0 只保留衰减状态 (1,.5)；a=0、η=1 清除旧状态后仍写入 (0,2)。a 和 η 控制不同项。 |
| [sparse-attention.mdx](../site/src/content/lessons/advanced/sparse-attention.mdx) | SparseSelectionLab 泛泛要求删除项，但没有分母/输出可复算目标。 | 取消键 3，核对 7→5 与 340/7→56，再全部读取回 130/3；空集合与误差界另写。 | 变式：选 {0,7} 时位置 3 的指数 2 退出分母，所以不能沿用 7；未来查询仍可能读位置 3，因此本次未选不等于可永久释放。 |
| [multimodal.mdx](../site/src/content/lessons/advanced/multimodal.mdx) | 交互说明混合 patch 身份、因果监督、特征范围和梯度，读者缺操作顺序。 | 先选 patch 0，再选预测行 6，再上下交换图像；每步核对均值、输入位置或目标，单独限定图的计算范围。 | 变式：交换后 patch 0 均值为 10.5/15，视觉行仍占输入位置 1；正确答案改变，旧 KV 不可复用。监督仍落在 logits 行 6/7。 |

## 局部核对

- 完整阅读 20 页、相邻章节衔接、`EDITORIAL_STANDARD.md`、`CONVENTIONS.md` 和相关 chapter blueprints。
- 逐项核对 `TrainingTraceLabs.tsx`、`AdvancedLabs.tsx` 与五个独立实验的真实控件范围。所有新增操作值可由现有控件选择。
- 使用 Node `--experimental-strip-types` 直接导入现有训练、进阶和预算模型，核对新增观察值：去掉 D4 的 NLL、SFT 有效数、LoRA 首步损失/梯度、八 rank 状态账、两个预算候选、MLA 分数、DPO 去 EOS、选择性递推、稀疏输出与换图均值。断言全部通过。
- `git diff --check` 通过。已有非 Mermaid 代码块、公式推导和章节标题不改；DPO、状态空间移动实验但保留标题，因此已有章节锚点稳定。
- 站点构建、链接及全站检查由主代理统一执行；本批未新增依赖或修改公共组件。

数学/框架代理另行独立复核本批 20 页，核对真实控件、计算结果与操作范围。按其建议，T2 补明控件序号 0 对应恢复后第 1 步。统一检查、构建、链接与浏览器结果见[统一验收](MDX_READABILITY_AUDIT.md#验收)。
