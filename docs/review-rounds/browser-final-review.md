# 全站浏览器验收

本轮由 root 对最终生产构建检查，范围为首页、86 篇 MDX（含八个主题索引），共 87 个入口。逐页分别使用 390、768、1440px 屏宽，共 261 轮；每轮检查明暗两种主题。逐篇结果见 [原始记录](../review-assets/browser-full.json)，没有失败项。

验证使用真实 Chromium。打开包含实验的折叠区，滚动 `astro-island` 的实际子元素，等待所有实验完成水合；等待字体与 Mermaid 完成绘制，再测量页面。各入口使用新的浏览器上下文，避免浏览器测试自身积累页面资源。

检查结果：所有入口返回 200；没有浏览器页面异常、未水合实验、Mermaid 绘制失败或重复 DOM ID；没有整页横向溢出。需要局部滚动的图框均能键盘聚焦。SVG text 使用屏幕变换矩阵测量真实像素字号，Matplotlib 路径文字按照字形组变换测量，所有可见文字达到 12px（测量容差 0.05px）。主题通过实际 `themechange` 事件切换，同时触发 Mermaid 重绘。

另对全部 18 个使用 Mermaid 的入口补测 `foreignObject` HTML 标签，三种屏宽、两种主题共 108 项。102 项包含 HTML 标签，实际最小字号约 15px，全部通过；其余六项为同一 sequence 图的不同屏宽/主题，使用 SVG text，已由全页测量覆盖。见 [标签测量记录](../review-assets/mermaid-labels.json)。

实际控件的数值断言另见 [25 项交互复核](browser-controls-review.md)。全页检查证明所有实验能加载与呈现，数值正确性由 Python/TypeScript 对照、实验模型检查及上述控件断言共同验证。

统一模型组装实验的最终截图：[手机深色](../review-assets/transformer-mobile-dark.png)、[桌面深色](../review-assets/transformer-desktop-dark.png)。截图展示共享 SVG、矩阵读数和源码样式；逐页检查结果以完整 JSON 为准。

CUDA/Triton 设备执行与性能不在本轮浏览器验证范围内。
