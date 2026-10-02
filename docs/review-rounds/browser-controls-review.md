浏览器与实验控制复核，2026-10-02。执行者负责共享视觉组件与 GPU 内容实施；本报告是实际浏览器验证记录，独立源码复审见 `visual-gpu-review.md`，全站最终浏览器验收另由 root 执行。

本轮对 23 个入口以 390、768、1440px 各检查一次，共 69 次页面检查。范围是 principles 目录的 15 个入口（含索引），以及 GPU 四篇主线和 indexing-practice、reduction-practice、triton-softmax、model-profiling 四篇实践。它不是全站验收。逐个打开包含实验的折叠区，滚动实验实际子元素，并等待对应 `astro-island` 的 `ssr` 属性消失，再读取渲染后的 SVG 文本字号、页面横向溢出和浏览器异常。69 次均无 hydration 超时、页面异常、低于 12px 的可见 SVG text 或整页横向溢出。首次仅滚动 `display:contents` 的 island 并不能证明已激活，最终检查已改正这一方法。原始逐页结果见 [局部浏览器记录](../review-assets/browser-local.json)。

另外在持续运行的开发服务器上用真实页面控件完成 25 项有具体预期的交互断言，全部通过且没有 pageerror；没有直接调用模型函数代替 UI 操作。原始结果见 [控件断言记录](../review-assets/browser-controls.json)；执行脚本在实施环境 `/tmp/principles-controls.mjs`。

| 章节 | 实际操作与核对 | 断言数 |
| --- | --- | --- |
| generation | 惩罚 θ=2、含 BOS/A 时 B≈0.45（分母0.67）；排除 BOS 后 B≈0.39（分母0.76）；复原惩罚并设 min-p=0.6 时只留 A/B。 | 3 |
| language-modeling | 取消 B 后均值0.510826、PPL1.666667；取消全部真实目标后平均 NLL/PPL 未定义。 | 2 |
| attention-order | 无 mask/位置时置换最大差0；固定 causal mask 后差非零；mask 随身份移动后恢复0。 | 3 |
| head-sharing | KV 头数从1改为2，正文二维例的缓存由8变为16元素。 | 1 |
| kv-cache | P=3、U=2、首行输出(2.061,0)；错误左上 tril 后输出0但概率和仍1；第二行绝对位置4。 | 3 |
| complete-transformer | 参数2748且训练不创建缓存；decode P=2/U=1 两层缓存96、每层分数12；缓存阶段使用 KVCache/append 真实接口；MHA、T=5 时参数3132、缓存320，较原配置增加384参数。 | 4 |
| decoder | 输入位置1、末阶段正确目标概率约0.3878，复用正文手算参数。 | 1 |
| tokenization | 两次合并后词表258；添加 BOS/EOS 后查表序列形状(7,2)。 | 2 |
| sinusoidal | Δ=4、p分别0/4/8，点积均0；Δ=8时点积-1。 | 4 |
| rope | 改共同旋转后点积仍0.866；内容差30°与额外旋转-30°抵消后点积1。 | 2 |

复核修正了完整现代块训练预设把推理缓存容量称作实际缓存的读数：训练现在明确不创建缓存，推理预设才显示 cache 元素。缓存代码显示当前统一包的不可变 KVCache.append；多头与缓存形状的文本符号统一为 D、H_q、H_kv、D_h。剩余旧原生 select 及实验动态 checkbox 已收归现有 Select/Toggle；Checklist 因为包含持久化自评内容，保留自己的原生 checkbox 布局。

开发服务器曾在并发 check/build 后出现 `_jsxDEV is not a function`。相关进程的 NODE_ENV 确认为 development，而服务中的共享预构建 react_jsx-dev-runtime 文件包含 production 实现（jsxDEV 为 undefined），服务 URL hash 与磁盘依赖 metadata hash 不同。Vite 官方优化器源码以 NODE_ENV 分支选择 runtime；原因是长运行 dev 与其他命令共用依赖 cache。astro.config 现按 dev/check/build 分离 cacheDir。生产 build 与 dev 并行后的真实激活检查、以及后续 check 与 dev 并行的上述 25 项交互均正常；dev cache 文件明确包含 react-jsx-dev-runtime.development.js 和 jsxDEVImpl。该修复没有依赖删除缓存，也未改变 NODE_ENV 来掩盖故障。

无 SVG 的 G2 地址栅格图在390px下实际横向溢出，复核确认其 FigureShell viewport 获得 tabIndex=0、role=region 和“左右滚动查看”的可访问标签。所有静态图框先独立测量 overflow；SVG 字号测量只是附加责任。Source excerpt helper 经 ruff 格式化后，装饰器、多行定义、摘录边界、歧义与 region 自检通过。pnpm check 的源码/模型/54项 Python–TypeScript 交叉参考检查通过；全站生产构建与最终三宽、深色验收由 root 串行执行，本文不把旧的局部检查充作最终全站结果。CUDA/Triton 实际设备执行和性能未验证，环境没有 CUDA 设备。
