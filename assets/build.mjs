import {readFile, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dependencies = [
  ['css', 'https://cdn.jsdelivr.net/npm/katex@0.18.6/dist/katex.min.css', 'M59ezskvvpvT+a+C1x088YJ3DVmK+wZdX0UkVKalOI4Qi5Nwv0WrvpqHcfa2HQqB'],
  ['css', 'https://cdn.jsdelivr.net/npm/markdown-it-texmath@1.0.0/css/texmath.min.css', 'UttgWGqUmSzrxetIBqxigLodBLOt0G9HCTlG9hKEmS9H5A5T3PGyDhPv0G8/C1EE'],
  ['css', 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/styles/github-dark.min.css', 'wH75j6z1lH97ZOpMOInqhgKzFkAInZPPSPlZpYKYTOqsaizPvhQZmAtLcPKXpLyH'],
  ['js', 'https://cdn.jsdelivr.net/npm/markdown-it@14.1.0/dist/markdown-it.min.js', 'wLhprpjsmjc/XYIcF+LpMxd8yS1gss6jhevOp6F6zhiIoFK6AmHtm4bGKtehTani'],
  ['js', 'https://cdn.jsdelivr.net/npm/katex@0.18.6/dist/katex.min.js', '7jGyG5zFwmEamqNWdCbpsPn+GTWEis3lnV7X/jXHyhFpJG7ExABLyMapabg8F4+p'],
  ['js', 'https://cdn.jsdelivr.net/npm/markdown-it-texmath@1.0.0/texmath.min.js', 'IJTt8+6hpcfxunTzDwSuS8o8lsOFjDwoUddKsTvs+x8oa9muA1PbBBcXaz8D6yR0'],
  ['js', 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/highlight.min.js', 'RH2xi4eIQ/gjtbs9fUXM68sLSi99C7ZWBRX1vDrVv6GQXRibxXLbwO2NGZB74MbU'],
];
const tags = dependencies.map(([type, url, hash]) => type === 'css'
  ? `<link rel="stylesheet" href="${url}" integrity="sha384-${hash}" crossorigin="anonymous">`
  : `<script defer src="${url}" integrity="sha384-${hash}" crossorigin="anonymous"></script>`).join('\n');

const lessons = [
  {directory: 'foundations', name: 'mathematics',
    title: '数学基础，理解旋转', edition: '00 · 数学基础',
    subtitle: '九章图解：向量、范数、点积、矩阵、极坐标、复数与高维旋转。'},
  {directory: 'training', name: 'position-encoding', code: 'position_encoding.py',
    title: '位置编码，从第一性原理推导', edition: '01 · 位置编码',
    subtitle: '从注意力的对称性，到三角函数、矩阵指数与旋转。'},
  {directory: 'inference', name: 'position-encoding-and-kv-cache', code: 'position_encoding_cache.py',
    title: '位置编码，走进 KV cache', edition: '02 · KV cache',
    subtitle: '从整段因果计算，推到 prefill、逐 token 解码与长度外推。'},
];

for (const lesson of lessons) {
  const folder = path.join(root, lesson.directory);
  const markdown = await readFile(path.join(folder, `${lesson.name}.md`), 'utf8');
  let source = markdown;
  if (lesson.code) {
    const code = await readFile(path.join(folder, lesson.code), 'utf8');
    if (markdown.split('<!-- python-source -->').length !== 2) throw new Error('Expected one source marker');
    source = markdown.replace('<!-- python-source -->', () => `\`\`\`python\n${code}\`\`\``);
  }
  const shelf = [
    ['foundations/mathematics.html', '数学', '00 · 数学基础'],
    ['training/position-encoding.html', '位置编码', '01 · 位置编码'],
    ['inference/position-encoding-and-kv-cache.html', 'KV cache', '02 · KV cache'],
    ['inference/nano-vllm-from-zero-to-mastery.html', '推理系统', '03 · 推理系统'],
  ];
  const here = `${lesson.directory}/${lesson.name}.html`;
  const link = ([file, label]) => `<a href="../${file}"${file === here ? ' aria-current="page"' : ''}>${label}</a>`;
  const navigation = shelf.map(([file, label]) => link([file, label])).join('')
    + '<a href="https://cyberspacelee.github.io/llms-from-scratch/lessons/ai-infra/">AI Infra</a>';
  const route = shelf.map(([file, , label]) => link([file, label])).join('')
    + '<a href="https://cyberspacelee.github.io/llms-from-scratch/lessons/ai-infra/">AI Infra</a>';
  const place = shelf.findIndex(([file]) => file === here);
  const prev = place > 0 ? shelf[place - 1] : undefined;
  const next = place >= 0 && place < shelf.length - 1 ? shelf[place + 1] : undefined;
  const footer = (prev || next) ? `<nav class="series-footer" aria-label="前后篇">${
    prev ? `<a class="prev" href="../${prev[0]}" rel="prev">上一篇 · ${prev[1]}</a>` : ''
  }${next ? `<a class="next" href="../${next[0]}" rel="next">下一篇 · ${next[1]}</a>` : ''}</nav>` : '';
  if (/<\/script/i.test(source)) throw new Error('Embedded source contains an HTML script terminator');
  const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="${lesson.subtitle}">
<title>${lesson.title}</title>
<script>
(function () {
  try {
    var stored = localStorage.getItem('theme');
    var dark = stored === 'dark' || (stored !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches);
    var root = document.documentElement;
    root.classList.toggle('dark', dark);
    root.dataset.theme = stored || 'system';
    root.style.colorScheme = dark ? 'dark' : 'light';
  } catch (error) {}
})();
</script>
${tags}
<link rel="stylesheet" href="../assets/site.css">
<script defer src="../assets/lesson.js"></script>
<script defer src="../assets/reader.js"></script>
</head>
<body>
<a class="skip-link" href="#content">跳到正文</a>
<header class="site-header">
<div class="header-bar">
<div class="bar-inner">
<a class="brand" href="../foundations/mathematics.html">LLMs From Scratch</a>
<nav class="switch" aria-label="系列讲义">${navigation}</nav>
<p class="read-label" id="read-label"></p>
<button class="theme-toggle" type="button">系统</button>
</div>
<div class="read-progress" aria-hidden="true"><div id="read-bar"></div></div>
</div>
<div class="hero-inner">
<p class="edition">${lesson.edition}</p>
<h1>${lesson.title}</h1>
<p class="hero-description">${lesson.subtitle}</p>
<p class="source-row"><a href="${lesson.code || '../training/position_encoding.py'}">PyTorch 源码</a></p>
</div>
</header>
<div class="layout">
<aside class="sidebar"><nav class="route-links" aria-label="学习路线"><div class="sidebar-title">学习路线</div>${route}<div class="sidebar-title">本章目录</div><div id="contents"></div></nav></aside>
<details class="mobile-toc"><summary>目录</summary><div class="mobile-toc-list"><nav class="route-links" aria-label="学习路线"><div class="sidebar-title">学习路线</div>${route}<div class="sidebar-title">本章目录</div><div id="mobile-chapters"></div></nav></div></details>
<main id="content"><div id="load-status" role="status"></div><article id="article"></article>${footer}<noscript>此讲义需要 JavaScript 渲染。可阅读同目录的 <a href="${lesson.name}.md">Markdown 原文</a>。</noscript></main>
</div>
<script type="text/plain" id="lesson-source">${source}</script>
</body></html>
`;
  await writeFile(path.join(folder, `${lesson.name}.html`), html);
  console.log(`Built ${lesson.directory}/${lesson.name}.html (${Buffer.byteLength(html)} bytes)`);
}
