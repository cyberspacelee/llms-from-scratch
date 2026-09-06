# LLMs From Scratch

以问题、数学推导、小维度算例和 PyTorch 数值验证组织的中文讲义。当前以桌面端阅读为主。

## 讲义

- [训练：从第一性原理推导位置编码](training/position-encoding.html)
- [推理：位置编码与 KV cache](inference/position-encoding-and-kv-cache.html)
- [推理系统：nano-vLLM 从入门到精通](inference/nano-vllm-from-zero-to-mastery.html)

HTML 可直接用浏览器打开，无需启动服务器。Markdown 讲义的正文已嵌入 HTML，不使用 `fetch` 读取本地文件；KaTeX、markdown-it、数学插件和 highlight.js 通过固定版本 CDN 加载，nano-vLLM 讲义的 Mermaid 图也使用固定版本 CDN，因此完整渲染需要联网。原理讲义的脚本和样式使用 SRI 校验，CDN 失败时显示原始 Markdown，不会伪装成渲染成功。

## 结构与更新

```text
training/       训练原理、Markdown 源文、HTML、PyTorch 实现
inference/      推理原理、Markdown 源文、HTML、缓存验证
assets/         共享样式、交互与 HTML 生成脚本
```

前两篇原理讲义修改对应 `.md` 与 `.py` 后，在项目根目录运行：

```sh
node assets/build.mjs
```

生成脚本只使用 Node.js 标准库。Python 源码会自动嵌入讲义，避免手工维护两份代码。生成的 HTML 纳入版本控制，读者不必先构建。

nano-vLLM 讲义直接维护 `inference/nano-vllm-from-zero-to-mastery.html`，正文、样式和交互均在该文件中，不经过上述生成脚本。推荐阅读顺序为：位置编码 → KV cache 正确性 → nano-vLLM 推理系统。

渲染器只处理本仓库作者维护的内容，允许讲义里的受控 HTML 交互占位。它不是通用的不可信 Markdown 预览器；若后续接收外部输入，应关闭原始 HTML 或引入经过测试的清洗流程。

## 数值验证

环境需要 Python 和 PyTorch。当前代码已在 Python 3.13 / PyTorch 2.14.0 CPU 上验证，不需要 NumPy。

```sh
python training/position_encoding.py
python inference/position_encoding_cache.py
```

第一项验证置换等变、sin/cos 平移、相对点积核与 RoPE 梯度；第二项比较整段、逐 token、prefill 后解码及分块缓存计算，并包含错误位置的负面对照。它们是教学用单层验证，不是完整模型的训练或生成程序。

## Git

仓库已初始化，未提交或推送。`.agents/`、系统文件、Python 缓存、虚拟环境和 Node 依赖被忽略；`skills-lock.json` 保留，不会随技能目录一起删除。
