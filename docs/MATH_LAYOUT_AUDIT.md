# 数学公式排版排查

## 修复结果

- KaTeX 样式纳入 `components` 层，公式字号 `1.05em`、普通公式上下间距 24px 和关键公式框内部间距 8px 均已在浏览器中确认生效。
- 显示公式允许在 KaTeX 原有断点处折行；超长推导、方框中的多个梯度和并列等式改为显式分行。长代码字段使用已说明的数学记号或多行代码表示。
- 表格中的短形状表达式保持完整，宽表和不可拆分的矩阵继续在自身容器中滚动。
- 重建后检查全部 19 个章节与导读页面、1,752 个公式，其中显示公式仍为 163 个。四种宽度下均未发现数学解析错误、表格公式拆行、显示公式左侧不可恢复的裁切或整页横向溢出。

| 宽度 | 修复前超宽显示公式 | 修复后超宽显示公式 |
| --- | ---: | ---: |
| 320px | 115 | 1 |
| 390px | 87 | 0 |
| 768px | 6 | 0 |
| 1440px | 4 | 0 |

320px 下剩余的一处是 KV cache 章节的完整因果掩码矩阵；保留矩阵结构，已验证横向滚动可查看全部内容。表格通过键盘方向键验证滚动，长张量形状的代码块在 320px 下也无需横向滚动。

构建、715 个内部链接检查和 Astro 类型检查通过；复核了手机、桌面和暗色主题的代表截图。

以下为修复前的排查记录，数据与源码行号基于 `ad608d5`，当前源码行号可能已变化。

## 范围与方法

- 执行 `cd site && npm run build`，21 个页面构建成功，715 个内部链接检查通过。
- 对本地构建产物的全部 19 个章节与导读页面运行 Chromium/Playwright 检查，共 1,736 个公式，其中显示公式 163 个、行内公式 1,573 个。
- 视口宽度：320、390、768、1440 像素。等待字体加载，逐个测量公式、滚动容器和页面边界；同时检查展开答案后的状态。
- 全量自动测量，截图复核概率推导、矩阵求导、训练公式、KV 形状表和推理引擎超宽公式等代表场景。未修改站点代码。
- 报告中的超宽数量表示公式需要内部横向滚动，不等同于不可恢复的裁切；行内跨行是候选检查项，不一概认定为错误。

## 结论

### 1. 共用公式样式受到 CSS 层级覆盖

`site/src/styles/global.css:1` 将 KaTeX CSS 导入为无层样式，而 `:141` 开始的站点覆盖规则位于 `@layer utilities`。正常声明中，无层样式优先于所有命名层，所以更具体的站点选择器也不能覆盖 KaTeX 默认值。

- `:172` 设置 `.katex { font-size: 1.05em }`，浏览器实际使用 KaTeX 默认 `1.21em`。17px 正文中的公式实际为 20.57px，预期为 17.85px，大约多 15.2%。
- `:173` 设置的 `my-6` 同样受到默认 `margin: 1em 0` 覆盖；实测上下边距 17px，预期 24px。KeyEq 中的 `my-2` 也有同一覆盖风险。
- `overflow-x-auto` 没有对应的 KaTeX 默认冲突，所以横向滚动规则仍然生效。
- 在浏览器临时注入无层的 `1.05em` 规则进行因果验证：390px 下超宽显示公式从 87 个减少到 71 个，1440px 下从 4 个减少到 1 个。

建议先统一调整 KaTeX 导入与站点覆盖的 CSS 层级关系，然后重新统计需要分行的公式。

### 2. 显示公式禁止自动折行，多个推导没有显式分行

KaTeX 默认 `.katex-display > .katex { white-space: nowrap }`，站点采用横向滚动。MDX 中普通换行在 TeX 里只是空白，不代表显示换行；这本身是正常语义，不能据此把每个分行源码都判为错误。

原始布局在 390px 下有 87/163 个显示公式需要滚动，1440px 下仍有以下 4 个：

| 位置 | 原因 |
| --- | --- |
| [principles/rope.mdx:212](../site/src/content/lessons/principles/rope.mdx#L212) | 连续三步等式写在同一显示行 |
| [systems/engine-execution.mdx:124](../site/src/content/lessons/systems/engine-execution.mdx#L124) | 含多个长标识符的乘积 |
| [systems/engine-runtime.mdx:186](../site/src/content/lessons/systems/engine-runtime.mdx#L186) | 两个含长标识符的映射公式并排 |
| [systems/engine-runtime.mdx:197](../site/src/content/lessons/systems/engine-runtime.mdx#L197) | 带多个长字段名的张量形状 |

浏览器实验中，在恢复字号后允许显示公式顶层折行，390px 下仍有 12 个超宽公式；说明单独修改 CSS 无法覆盖所有结构。此实验不是最终修复方案，断点与等号对齐仍需视觉复核。

### 3. aligned、boxed 和长括号内部仍需内容分行

`aligned` 会保留作者的显式行，但不会自动把其中一个超长行再拆开；`boxed`、部分长括号和分数也是整体排版结构。以下 12 个公式在上述字号与顶层折行实验后仍需处理：

| 位置 | 建议 |
| --- | --- |
| [math/matrix-calculus.mdx:165](../site/src/content/lessons/math/matrix-calculus.mdx#L165) | 把 boxed 内的多个梯度改为多行 aligned |
| [math/probability.mdx:86](../site/src/content/lessons/math/probability.mdx#L86) | 在超长行内继续拆分推导，调整对齐点 |
| [math/training.mdx:147](../site/src/content/lessons/math/training.mdx#L147) | 在超长行内继续拆分推导，调整对齐点 |
| [principles/rope.mdx:131](../site/src/content/lessons/principles/rope.mdx#L131) | 在超长行内继续拆分推导，调整对齐点 |
| [principles/rope.mdx:190](../site/src/content/lessons/principles/rope.mdx#L190) | 拆分长表达式，必要时定义中间记号 |
| [principles/sinusoidal.mdx:146](../site/src/content/lessons/principles/sinusoidal.mdx#L146) | 在超长行内继续拆分推导，调整对齐点 |
| [principles/sinusoidal.mdx:187](../site/src/content/lessons/principles/sinusoidal.mdx#L187) | 在超长行内继续拆分推导，调整对齐点 |
| [principles/sinusoidal.mdx:207](../site/src/content/lessons/principles/sinusoidal.mdx#L207) | 在超长行内继续拆分推导，调整对齐点 |
| [principles/sinusoidal.mdx:277](../site/src/content/lessons/principles/sinusoidal.mdx#L277) | 在超长行内继续拆分推导，调整对齐点 |
| [systems/engine-execution.mdx:128](../site/src/content/lessons/systems/engine-execution.mdx#L128) | 先定义剩余可用内存量，再写块数公式 |
| [systems/engine-runtime.mdx:197](../site/src/content/lessons/systems/engine-runtime.mdx#L197) | 张量字段列表改用代码或分行表示 |
| [systems/ledger.mdx:37](../site/src/content/lessons/systems/ledger.mdx#L37) | 在超长行内继续拆分推导，调整对齐点 |

### 4. 表格中的短形状表达式在加号后被拆开

`principles/kv-cache.mdx:50` 的 `[B,H,L+1,D]` 和 `[B,H,1,L+1]` 在 390px 下分别跨成两行。KaTeX 行内公式会在二元运算符或关系符后提供断行点，而表格没有保护这些应整体阅读的形状。截图确认出现 `L+` 与 `1,D]` / `1]` 分离的情况。

表格已有横向滚动容器；建议对表格内的短形状表达式禁止内部折行，让表格滚动完整呈现。不要直接对全部行内公式禁止折行，否则长公式可能撑出正文。

在 390px 正文中，另有 51 个行内公式跨行；很多是正常的等式断行，完整候选清单见下方，修复时应区分短元组与长推导。

## 统计与因果实验

| 宽度 | 原始超宽显示公式 | 恢复 1.05em 后 | 再允许顶层折行后 | 原始行内跨行 |
| --- | ---: | ---: | ---: | ---: |
| 320px | 115 | 97 | 19 | 84 |
| 390px | 87 | 71 | 12 | 53 |
| 768px | 6 | 2 | 0 | 8 |
| 1440px | 4 | 1 | 0 | 10 |

未发现 `.katex-error`、显示公式左侧超出其滚动容器起点、整页被数学公式撑出视口。KV 表和微积分表中的部分公式起初在屏幕右侧，但处于可横向滚动的表格容器中，不能将它们算成不可恢复的裁切。`aligned` 的换行结构在构建产物和浏览器中均被保留；没有证据表明 Markdown 插件吞掉了 `\`。

## 全量超宽位置清单

按 320px 基线列出所有 115 个超宽显示公式。各列为原始布局下是否需要横向滚动；“是”表示需要，“否”表示不需要。

| 公式位置 | 320px | 390px | 768px | 1440px |
| --- | --- | --- | --- | --- |
| [math/backprop.mdx:33](../site/src/content/lessons/math/backprop.mdx#L33) | 是 | 是 | 否 | 否 |
| [math/backprop.mdx:108](../site/src/content/lessons/math/backprop.mdx#L108) | 是 | 是 | 否 | 否 |
| [math/backprop.mdx:117](../site/src/content/lessons/math/backprop.mdx#L117) | 是 | 是 | 否 | 否 |
| [math/backprop.mdx:125](../site/src/content/lessons/math/backprop.mdx#L125) | 是 | 是 | 否 | 否 |
| [math/backprop.mdx:173](../site/src/content/lessons/math/backprop.mdx#L173) | 是 | 是 | 否 | 否 |
| [math/calculus.mdx:19](../site/src/content/lessons/math/calculus.mdx#L19) | 是 | 是 | 否 | 否 |
| [math/calculus.mdx:78](../site/src/content/lessons/math/calculus.mdx#L78) | 是 | 是 | 否 | 否 |
| [math/calculus.mdx:125](../site/src/content/lessons/math/calculus.mdx#L125) | 是 | 是 | 否 | 否 |
| [math/calculus.mdx:134](../site/src/content/lessons/math/calculus.mdx#L134) | 是 | 是 | 否 | 否 |
| [math/calculus.mdx:176](../site/src/content/lessons/math/calculus.mdx#L176) | 是 | 否 | 否 | 否 |
| [math/calculus.mdx:210](../site/src/content/lessons/math/calculus.mdx#L210) | 是 | 是 | 否 | 否 |
| [math/calculus.mdx:225](../site/src/content/lessons/math/calculus.mdx#L225) | 是 | 是 | 否 | 否 |
| [math/distributions.mdx:25](../site/src/content/lessons/math/distributions.mdx#L25) | 是 | 是 | 否 | 否 |
| [math/distributions.mdx:33](../site/src/content/lessons/math/distributions.mdx#L33) | 是 | 是 | 否 | 否 |
| [math/distributions.mdx:52](../site/src/content/lessons/math/distributions.mdx#L52) | 是 | 是 | 否 | 否 |
| [math/distributions.mdx:63](../site/src/content/lessons/math/distributions.mdx#L63) | 是 | 否 | 否 | 否 |
| [math/distributions.mdx:105](../site/src/content/lessons/math/distributions.mdx#L105) | 是 | 是 | 否 | 否 |
| [math/distributions.mdx:132](../site/src/content/lessons/math/distributions.mdx#L132) | 是 | 否 | 否 | 否 |
| [math/distributions.mdx:140](../site/src/content/lessons/math/distributions.mdx#L140) | 是 | 否 | 否 | 否 |
| [math/distributions.mdx:153](../site/src/content/lessons/math/distributions.mdx#L153) | 是 | 否 | 否 | 否 |
| [math/distributions.mdx:165](../site/src/content/lessons/math/distributions.mdx#L165) | 是 | 是 | 否 | 否 |
| [math/distributions.mdx:174](../site/src/content/lessons/math/distributions.mdx#L174) | 是 | 是 | 否 | 否 |
| [math/distributions.mdx:195](../site/src/content/lessons/math/distributions.mdx#L195) | 是 | 是 | 否 | 否 |
| [math/distributions.mdx:208](../site/src/content/lessons/math/distributions.mdx#L208) | 是 | 是 | 否 | 否 |
| [math/distributions.mdx:219](../site/src/content/lessons/math/distributions.mdx#L219) | 是 | 否 | 否 | 否 |
| [math/linear-algebra.mdx:43](../site/src/content/lessons/math/linear-algebra.mdx#L43) | 是 | 否 | 否 | 否 |
| [math/linear-algebra.mdx:62](../site/src/content/lessons/math/linear-algebra.mdx#L62) | 是 | 否 | 否 | 否 |
| [math/linear-algebra.mdx:110](../site/src/content/lessons/math/linear-algebra.mdx#L110) | 是 | 否 | 否 | 否 |
| [math/linear-algebra.mdx:138](../site/src/content/lessons/math/linear-algebra.mdx#L138) | 是 | 是 | 否 | 否 |
| [math/linear-algebra.mdx:152](../site/src/content/lessons/math/linear-algebra.mdx#L152) | 是 | 否 | 否 | 否 |
| [math/linear-algebra.mdx:207](../site/src/content/lessons/math/linear-algebra.mdx#L207) | 是 | 是 | 否 | 否 |
| [math/matrix-calculus.mdx:19](../site/src/content/lessons/math/matrix-calculus.mdx#L19) | 是 | 是 | 否 | 否 |
| [math/matrix-calculus.mdx:42](../site/src/content/lessons/math/matrix-calculus.mdx#L42) | 是 | 是 | 否 | 否 |
| [math/matrix-calculus.mdx:50](../site/src/content/lessons/math/matrix-calculus.mdx#L50) | 是 | 否 | 否 | 否 |
| [math/matrix-calculus.mdx:61](../site/src/content/lessons/math/matrix-calculus.mdx#L61) | 是 | 是 | 否 | 否 |
| [math/matrix-calculus.mdx:94](../site/src/content/lessons/math/matrix-calculus.mdx#L94) | 是 | 是 | 否 | 否 |
| [math/matrix-calculus.mdx:102](../site/src/content/lessons/math/matrix-calculus.mdx#L102) | 是 | 是 | 否 | 否 |
| [math/matrix-calculus.mdx:116](../site/src/content/lessons/math/matrix-calculus.mdx#L116) | 是 | 是 | 否 | 否 |
| [math/matrix-calculus.mdx:148](../site/src/content/lessons/math/matrix-calculus.mdx#L148) | 是 | 是 | 否 | 否 |
| [math/matrix-calculus.mdx:158](../site/src/content/lessons/math/matrix-calculus.mdx#L158) | 是 | 否 | 否 | 否 |
| [math/matrix-calculus.mdx:165](../site/src/content/lessons/math/matrix-calculus.mdx#L165) | 是 | 是 | 否 | 否 |
| [math/matrix-calculus.mdx:183](../site/src/content/lessons/math/matrix-calculus.mdx#L183) | 是 | 是 | 否 | 否 |
| [math/probability.mdx:42](../site/src/content/lessons/math/probability.mdx#L42) | 是 | 否 | 否 | 否 |
| [math/probability.mdx:48](../site/src/content/lessons/math/probability.mdx#L48) | 是 | 是 | 否 | 否 |
| [math/probability.mdx:61](../site/src/content/lessons/math/probability.mdx#L61) | 是 | 是 | 否 | 否 |
| [math/probability.mdx:86](../site/src/content/lessons/math/probability.mdx#L86) | 是 | 是 | 否 | 否 |
| [math/probability.mdx:128](../site/src/content/lessons/math/probability.mdx#L128) | 是 | 是 | 否 | 否 |
| [math/probability.mdx:135](../site/src/content/lessons/math/probability.mdx#L135) | 是 | 是 | 否 | 否 |
| [math/probability.mdx:149](../site/src/content/lessons/math/probability.mdx#L149) | 是 | 是 | 否 | 否 |
| [math/probability.mdx:156](../site/src/content/lessons/math/probability.mdx#L156) | 是 | 否 | 否 | 否 |
| [math/probability.mdx:170](../site/src/content/lessons/math/probability.mdx#L170) | 是 | 否 | 否 | 否 |
| [math/probability.mdx:176](../site/src/content/lessons/math/probability.mdx#L176) | 是 | 是 | 否 | 否 |
| [math/probability.mdx:187](../site/src/content/lessons/math/probability.mdx#L187) | 是 | 是 | 否 | 否 |
| [math/probability.mdx:224](../site/src/content/lessons/math/probability.mdx#L224) | 是 | 是 | 否 | 否 |
| [math/training.mdx:25](../site/src/content/lessons/math/training.mdx#L25) | 是 | 是 | 否 | 否 |
| [math/training.mdx:35](../site/src/content/lessons/math/training.mdx#L35) | 是 | 是 | 否 | 否 |
| [math/training.mdx:76](../site/src/content/lessons/math/training.mdx#L76) | 是 | 否 | 否 | 否 |
| [math/training.mdx:86](../site/src/content/lessons/math/training.mdx#L86) | 是 | 否 | 否 | 否 |
| [math/training.mdx:100](../site/src/content/lessons/math/training.mdx#L100) | 是 | 否 | 否 | 否 |
| [math/training.mdx:133](../site/src/content/lessons/math/training.mdx#L133) | 是 | 是 | 否 | 否 |
| [math/training.mdx:147](../site/src/content/lessons/math/training.mdx#L147) | 是 | 是 | 否 | 否 |
| [principles/attention-order.mdx:27](../site/src/content/lessons/principles/attention-order.mdx#L27) | 是 | 是 | 否 | 否 |
| [principles/attention-order.mdx:38](../site/src/content/lessons/principles/attention-order.mdx#L38) | 是 | 是 | 否 | 否 |
| [principles/attention-order.mdx:51](../site/src/content/lessons/principles/attention-order.mdx#L51) | 是 | 是 | 否 | 否 |
| [principles/attention-order.mdx:65](../site/src/content/lessons/principles/attention-order.mdx#L65) | 是 | 是 | 否 | 否 |
| [principles/attention-order.mdx:74](../site/src/content/lessons/principles/attention-order.mdx#L74) | 是 | 是 | 否 | 否 |
| [principles/attention-order.mdx:81](../site/src/content/lessons/principles/attention-order.mdx#L81) | 是 | 是 | 否 | 否 |
| [principles/attention-order.mdx:90](../site/src/content/lessons/principles/attention-order.mdx#L90) | 是 | 是 | 否 | 否 |
| [principles/kv-cache.mdx:21](../site/src/content/lessons/principles/kv-cache.mdx#L21) | 是 | 是 | 否 | 否 |
| [principles/kv-cache.mdx:39](../site/src/content/lessons/principles/kv-cache.mdx#L39) | 是 | 否 | 否 | 否 |
| [principles/kv-cache.mdx:76](../site/src/content/lessons/principles/kv-cache.mdx#L76) | 是 | 否 | 否 | 否 |
| [principles/kv-cache.mdx:93](../site/src/content/lessons/principles/kv-cache.mdx#L93) | 是 | 是 | 否 | 否 |
| [principles/kv-cache.mdx:111](../site/src/content/lessons/principles/kv-cache.mdx#L111) | 是 | 是 | 否 | 否 |
| [principles/kv-cache.mdx:142](../site/src/content/lessons/principles/kv-cache.mdx#L142) | 是 | 是 | 否 | 否 |
| [principles/kv-cache.mdx:165](../site/src/content/lessons/principles/kv-cache.mdx#L165) | 是 | 否 | 否 | 否 |
| [principles/kv-cache.mdx:185](../site/src/content/lessons/principles/kv-cache.mdx#L185) | 是 | 否 | 否 | 否 |
| [principles/rope.mdx:39](../site/src/content/lessons/principles/rope.mdx#L39) | 是 | 是 | 否 | 否 |
| [principles/rope.mdx:53](../site/src/content/lessons/principles/rope.mdx#L53) | 是 | 否 | 否 | 否 |
| [principles/rope.mdx:60](../site/src/content/lessons/principles/rope.mdx#L60) | 是 | 是 | 否 | 否 |
| [principles/rope.mdx:77](../site/src/content/lessons/principles/rope.mdx#L77) | 是 | 是 | 否 | 否 |
| [principles/rope.mdx:109](../site/src/content/lessons/principles/rope.mdx#L109) | 是 | 是 | 是 | 否 |
| [principles/rope.mdx:120](../site/src/content/lessons/principles/rope.mdx#L120) | 是 | 否 | 否 | 否 |
| [principles/rope.mdx:131](../site/src/content/lessons/principles/rope.mdx#L131) | 是 | 是 | 否 | 否 |
| [principles/rope.mdx:149](../site/src/content/lessons/principles/rope.mdx#L149) | 是 | 否 | 否 | 否 |
| [principles/rope.mdx:165](../site/src/content/lessons/principles/rope.mdx#L165) | 是 | 是 | 否 | 否 |
| [principles/rope.mdx:190](../site/src/content/lessons/principles/rope.mdx#L190) | 是 | 是 | 否 | 否 |
| [principles/rope.mdx:201](../site/src/content/lessons/principles/rope.mdx#L201) | 是 | 是 | 否 | 否 |
| [principles/rope.mdx:212](../site/src/content/lessons/principles/rope.mdx#L212) | 是 | 是 | 是 | 是 |
| [principles/sinusoidal.mdx:37](../site/src/content/lessons/principles/sinusoidal.mdx#L37) | 是 | 否 | 否 | 否 |
| [principles/sinusoidal.mdx:61](../site/src/content/lessons/principles/sinusoidal.mdx#L61) | 是 | 是 | 否 | 否 |
| [principles/sinusoidal.mdx:85](../site/src/content/lessons/principles/sinusoidal.mdx#L85) | 是 | 是 | 否 | 否 |
| [principles/sinusoidal.mdx:99](../site/src/content/lessons/principles/sinusoidal.mdx#L99) | 是 | 是 | 否 | 否 |
| [principles/sinusoidal.mdx:146](../site/src/content/lessons/principles/sinusoidal.mdx#L146) | 是 | 是 | 否 | 否 |
| [principles/sinusoidal.mdx:187](../site/src/content/lessons/principles/sinusoidal.mdx#L187) | 是 | 是 | 否 | 否 |
| [principles/sinusoidal.mdx:207](../site/src/content/lessons/principles/sinusoidal.mdx#L207) | 是 | 是 | 否 | 否 |
| [principles/sinusoidal.mdx:216](../site/src/content/lessons/principles/sinusoidal.mdx#L216) | 是 | 是 | 否 | 否 |
| [principles/sinusoidal.mdx:231](../site/src/content/lessons/principles/sinusoidal.mdx#L231) | 是 | 是 | 是 | 否 |
| [principles/sinusoidal.mdx:256](../site/src/content/lessons/principles/sinusoidal.mdx#L256) | 是 | 是 | 否 | 否 |
| [principles/sinusoidal.mdx:265](../site/src/content/lessons/principles/sinusoidal.mdx#L265) | 是 | 否 | 否 | 否 |
| [principles/sinusoidal.mdx:277](../site/src/content/lessons/principles/sinusoidal.mdx#L277) | 是 | 是 | 否 | 否 |
| [principles/sinusoidal.mdx:289](../site/src/content/lessons/principles/sinusoidal.mdx#L289) | 是 | 是 | 否 | 否 |
| [principles/sinusoidal.mdx:297](../site/src/content/lessons/principles/sinusoidal.mdx#L297) | 是 | 是 | 否 | 否 |
| [systems/accelerator.mdx:32](../site/src/content/lessons/systems/accelerator.mdx#L32) | 是 | 是 | 否 | 否 |
| [systems/accelerator.mdx:43](../site/src/content/lessons/systems/accelerator.mdx#L43) | 是 | 是 | 否 | 否 |
| [systems/engine-execution.mdx:105](../site/src/content/lessons/systems/engine-execution.mdx#L105) | 是 | 是 | 否 | 否 |
| [systems/engine-execution.mdx:124](../site/src/content/lessons/systems/engine-execution.mdx#L124) | 是 | 是 | 是 | 是 |
| [systems/engine-execution.mdx:128](../site/src/content/lessons/systems/engine-execution.mdx#L128) | 是 | 是 | 否 | 否 |
| [systems/engine-execution.mdx:163](../site/src/content/lessons/systems/engine-execution.mdx#L163) | 是 | 是 | 否 | 否 |
| [systems/engine-runtime.mdx:186](../site/src/content/lessons/systems/engine-runtime.mdx#L186) | 是 | 是 | 是 | 是 |
| [systems/engine-runtime.mdx:197](../site/src/content/lessons/systems/engine-runtime.mdx#L197) | 是 | 是 | 是 | 是 |
| [systems/engine-runtime.mdx:209](../site/src/content/lessons/systems/engine-runtime.mdx#L209) | 是 | 是 | 否 | 否 |
| [systems/ledger.mdx:37](../site/src/content/lessons/systems/ledger.mdx#L37) | 是 | 是 | 否 | 否 |
| [systems/ledger.mdx:52](../site/src/content/lessons/systems/ledger.mdx#L52) | 是 | 是 | 否 | 否 |
| [systems/ledger.mdx:79](../site/src/content/lessons/systems/ledger.mdx#L79) | 是 | 是 | 否 | 否 |
| [systems/ledger.mdx:142](../site/src/content/lessons/systems/ledger.mdx#L142) | 是 | 否 | 否 | 否 |

## 行内公式跨行候选

390px 下的 53 个候选，其中 KV 表中的 2 个是短形状被拆开的明确问题；其余需要结合正文语境选择保留自然断行或转成显示公式。重复位置表示同一源码行中有多个公式。

| 位置 | 公式源码 |
| --- | --- |
| [math/backprop.mdx:71](../site/src/content/lessons/math/backprop.mdx#L71) | `J_{ij}=\partial z_i/\partial x_j` |
| [math/backprop.mdx:81](../site/src/content/lessons/math/backprop.mdx#L81) | `z_1+2z_2` |
| [math/backprop.mdx:112](../site/src/content/lessons/math/backprop.mdx#L112) | `b_1=(0.5,-0.5)^\top` |
| [math/calculus.mdx:165](../site/src/content/lessons/math/calculus.mdx#L165) | `L(x_1,x_2)=\tfrac12(x_1^2+4x_2^2)` |
| [math/calculus.mdx:165](../site/src/content/lessons/math/calculus.mdx#L165) | `\partial L/\partial x_1=x_1` |
| [math/calculus.mdx:192](../site/src/content/lessons/math/calculus.mdx#L192) | `x_2=x_1^2` |
| [math/calculus.mdx:220](../site/src/content/lessons/math/calculus.mdx#L220) | `x_{t+1}=x_t-\eta\nabla L(x_t)` |
| [math/calculus.mdx:261](../site/src/content/lessons/math/calculus.mdx#L261) | `1-\eta\lambda` |
| [math/distributions.mdx:67](../site/src/content/lessons/math/distributions.mdx#L67) | `P(a\le U\le b)=\int_a^b p(u)\,du` |
| [math/distributions.mdx:71](../site/src/content/lessons/math/distributions.mdx#L71) | `\varphi(v)=e^{-v^2/2}/\sqrt{2\pi}` |
| [math/distributions.mdx:109](../site/src/content/lessons/math/distributions.mdx#L109) | `p=n_1/(n_1+n_0)` |
| [math/distributions.mdx:160](../site/src/content/lessons/math/distributions.mdx#L160) | `I(u)=-\log P(U=u)` |
| [math/linear-algebra.mdx:34](../site/src/content/lessons/math/linear-algebra.mdx#L34) | `f_1=(1,1)^\top,f_2=(1,-1)^\top` |
| [math/linear-algebra.mdx:47](../site/src/content/lessons/math/linear-algebra.mdx#L47) | `\&#124;W\&#124;_F=\sqrt{\sum_{k,j}W_{kj}^2}` |
| [math/linear-algebra.mdx:71](../site/src/content/lessons/math/linear-algebra.mdx#L71) | `x-cy` |
| [math/linear-algebra.mdx:213](../site/src/content/lessons/math/linear-algebra.mdx#L213) | `Q=\left[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right]` |
| [math/linear-algebra.mdx:258](../site/src/content/lessons/math/linear-algebra.mdx#L258) | `Q^{-1}=Q^\top` |
| [math/matrix-calculus.mdx:107](../site/src/content/lessons/math/matrix-calculus.mdx#L107) | `(m\times d)` |
| [math/matrix-calculus.mdx:230](../site/src/content/lessons/math/matrix-calculus.mdx#L230) | `g=(-0.5,1)^\top` |
| [math/probability.mdx:45](../site/src/content/lessons/math/probability.mdx#L45) | `P(A\cap D)=P(A\mid D)P(D)` |
| [math/probability.mdx:54](../site/src/content/lessons/math/probability.mdx#L54) | `P(A\cap D)=P(A)P(D)` |
| [math/probability.mdx:83](../site/src/content/lessons/math/probability.mdx#L83) | `p_{kr}=P(U=u_k,V=v_r)` |
| [math/probability.mdx:97](../site/src/content/lessons/math/probability.mdx#L97) | `\mathcal R(\theta)=\mathbb E[\ell(\theta;x,y)]` |
| [math/probability.mdx:97](../site/src/content/lessons/math/probability.mdx#L97) | `L(\theta)=B^{-1}\sum_i\ell(\theta;x_i,y_i)` |
| [math/probability.mdx:97](../site/src/content/lessons/math/probability.mdx#L97) | `\mathbb E[L(\theta)]=\mathcal R(\theta)` |
| [math/probability.mdx:140](../site/src/content/lessons/math/probability.mdx#L140) | `V-\mathbb E[V]=a(U-\mu)` |
| [math/probability.mdx:140](../site/src/content/lessons/math/probability.mdx#L140) | `\operatorname{Var}(V)=a^2\operatorname{Var}(U)` |
| [math/probability.mdx:146](../site/src/content/lessons/math/probability.mdx#L146) | `\operatorname{Var}(\bar U)=\sigma^2/B` |
| [math/probability.mdx:182](../site/src/content/lessons/math/probability.mdx#L182) | `\rho=\operatorname{Cov}(U,V)/(\sigma_U\sigma_V)` |
| [math/probability.mdx:284](../site/src/content/lessons/math/probability.mdx#L284) | `\sqrt{16/64}=1/2` |
| [math/training.mdx:57](../site/src/content/lessons/math/training.mdx#L57) | `s(z)=1/(1+e^{-z})` |
| [math/training.mdx:57](../site/src/content/lessons/math/training.mdx#L57) | `s(z)(1-s(z))` |
| [math/training.mdx:104](../site/src/content/lessons/math/training.mdx#L104) | `(\tanh 1,0)^\top\approx(0.7616,0)^\top` |
| [math/training.mdx:156](../site/src/content/lessons/math/training.mdx#L156) | `G_2\approx(-0.3183,0.3183)` |
| [math/training.mdx:156](../site/src/content/lessons/math/training.mdx#L156) | `G_1\approx(-0.1337,0.3183)` |
| [math/training.mdx:237](../site/src/content/lessons/math/training.mdx#L237) | `\widehat g=B^{-1}\sum_{i\in\mathcal B}\nabla_\theta\ell_i` |
| [principles/attention-order.mdx:31](../site/src/content/lessons/principles/attention-order.mdx#L31) | `q_i^\top k_j=\sum_{a=1}^{d_h}q_{i,a}k_{j,a}` |
| [principles/attention-order.mdx:33](../site/src/content/lessons/principles/attention-order.mdx#L33) | `k_1=(0,1)^\top` |
| [principles/attention-order.mdx:33](../site/src/content/lessons/principles/attention-order.mdx#L33) | `v_0=(1,0)^\top,v_1=(0,3)^\top` |
| [principles/kv-cache.mdx:50](../site/src/content/lessons/principles/kv-cache.mdx#L50) | `[B,H,L+1,D]` |
| [principles/kv-cache.mdx:50](../site/src/content/lessons/principles/kv-cache.mdx#L50) | `[B,H,1,L+1]` |
| [principles/kv-cache.mdx:58](../site/src/content/lessons/principles/kv-cache.mdx#L58) | `\widetilde k_{L-1}=R((L-1)\theta)k_{L-1}` |
| [principles/kv-cache.mdx:84](../site/src/content/lessons/principles/kv-cache.mdx#L84) | `j\in[0,L+C-1]` |
| [principles/rope.mdx:70](../site/src/content/lessons/principles/rope.mdx#L70) | `(x+\mathrm{i}y)(u+\mathrm{i}v)=(xu-yv)+\mathrm{i}(xv+yu)` |
| [principles/rope.mdx:82](../site/src/content/lessons/principles/rope.mdx#L82) | `\operatorname{Re}(\bar z w)=&#124;z&#124;&#124;w&#124;\cos(\beta-\alpha)` |
| [principles/rope.mdx:92](../site/src/content/lessons/principles/rope.mdx#L92) | `rs\cos(\beta-\alpha)` |
| [principles/rope.mdx:104](../site/src/content/lessons/principles/rope.mdx#L104) | `R(i\theta)^\top R(j\theta)=R((j-i)\theta)` |
| [principles/rope.mdx:142](../site/src/content/lessons/principles/rope.mdx#L142) | `q=(1,0)^\top,k=(\sqrt3/2,1/2)^\top` |
| [principles/rope.mdx:142](../site/src/content/lessons/principles/rope.mdx#L142) | `\cos(\pi/6+(j-i)\omega)` |
| [principles/rope.mdx:170](../site/src/content/lessons/principles/rope.mdx#L170) | `\pi/6+(3-1)\pi/6=\pi/2` |
| [principles/sinusoidal.mdx:56](../site/src/content/lessons/principles/sinusoidal.mdx#L56) | `e_1=(1,0)^\top` |
| [principles/sinusoidal.mdx:71](../site/src/content/lessons/principles/sinusoidal.mdx#L71) | `M(3,4)^\top=3(2,0)^\top+4(1,1)^\top=(10,4)^\top` |
| [principles/sinusoidal.mdx:222](../site/src/content/lessons/principles/sinusoidal.mdx#L222) | `\sin((p+\Delta)\omega)=c_\Delta\sin(p\omega)` |

## 修复顺序

1. 修正共用 CSS 层级，使既有字号和间距规则真正生效。
2. 修正 KV 表中的短形状折行；保留现有表格滚动。
3. 将桌面超宽推导及上方 12 个结构性超宽公式改为有明确对齐点的多行公式。
4. 重新测量手机宽度，对剩余长式按内容分行；横向滚动保留为矩阵和无法合理拆分结构的兜底。

上述统计记录本次排查时的原始布局。排查使用的临时脚本、测量 JSON、截图及预览服务已清理；站点未新增浏览器测试依赖。
