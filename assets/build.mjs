import {readFile, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dependencies = [
  ['css', 'https://cdn.jsdelivr.net/npm/katex@0.18.6/dist/katex.min.css', 'M59ezskvvpvT+a+C1x088YJ3DVmK+wZdX0UkVKalOI4Qi5Nwv0WrvpqHcfa2HQqB'],
  ['css', 'https://cdn.jsdelivr.net/npm/markdown-it-texmath@1.0.0/css/texmath.min.css', 'UttgWGqUmSzrxetIBqxigLodBLOt0G9HCTlG9hKEmS9H5A5T3PGyDhPv0G8/C1EE'],
  ['css', 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/styles/github.min.css', 'eFTL69TLRZTkNfYZOLM+G04821K1qZao/4QLJbet1pP4tcF+fdXq/9CdqAbWRl/L'],
  ['js', 'https://cdn.jsdelivr.net/npm/markdown-it@14.1.0/dist/markdown-it.min.js', 'wLhprpjsmjc/XYIcF+LpMxd8yS1gss6jhevOp6F6zhiIoFK6AmHtm4bGKtehTani'],
  ['js', 'https://cdn.jsdelivr.net/npm/katex@0.18.6/dist/katex.min.js', '7jGyG5zFwmEamqNWdCbpsPn+GTWEis3lnV7X/jXHyhFpJG7ExABLyMapabg8F4+p'],
  ['js', 'https://cdn.jsdelivr.net/npm/markdown-it-texmath@1.0.0/texmath.min.js', 'IJTt8+6hpcfxunTzDwSuS8o8lsOFjDwoUddKsTvs+x8oa9muA1PbBBcXaz8D6yR0'],
  ['js', 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.11.1/build/highlight.min.js', 'RH2xi4eIQ/gjtbs9fUXM68sLSi99C7ZWBRX1vDrVv6GQXRibxXLbwO2NGZB74MbU'],
];
const tags = dependencies.map(([type, url, hash]) => type === 'css'
  ? `<link rel="stylesheet" href="${url}" integrity="sha384-${hash}" crossorigin="anonymous">`
  : `<script defer src="${url}" integrity="sha384-${hash}" crossorigin="anonymous"></script>`).join('\n');

const lessons = [
  {directory: 'training', name: 'position-encoding', code: 'position_encoding.py',
    title: '位置编码，从第一性原理推导', edition: '01 / TRAINING',
    subtitle: '从注意力的对称性，到三角函数、矩阵指数与旋转。'},
  {directory: 'inference', name: 'position-encoding-and-kv-cache', code: 'position_encoding_cache.py',
    title: '位置编码，走进 KV cache', edition: '02 / INFERENCE',
    subtitle: '从整段因果计算，推到 prefill、逐 token 解码与长度外推。'},
];

for (const lesson of lessons) {
  const folder = path.join(root, lesson.directory);
  const markdown = await readFile(path.join(folder, `${lesson.name}.md`), 'utf8');
  const code = await readFile(path.join(folder, lesson.code), 'utf8');
  if (markdown.split('<!-- python-source -->').length !== 2) throw new Error('Expected one source marker');
  const source = markdown.replace('<!-- python-source -->', () => `\`\`\`python\n${code}\`\`\``);
  if (/<\/script/i.test(source)) throw new Error('Embedded source contains an HTML script terminator');
  const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="${lesson.subtitle}">
<title>${lesson.title}</title>
${tags}
<link rel="stylesheet" href="../assets/lesson.css">
<script defer src="../assets/lesson.js"></script>
</head>
<body>
<header><div class="switch"><a href="../training/position-encoding.html">01 位置编码</a><a href="../inference/position-encoding-and-kv-cache.html">02 KV cache</a><a href="../inference/nano-vllm-from-zero-to-mastery.html">03 推理系统</a><a href="${lesson.code}">PyTorch 源码</a></div>
<p class="edition">LLMS FROM SCRATCH · ${lesson.edition}</p><h1>${lesson.title}</h1><p>${lesson.subtitle}</p></header>
<div class="layout"><nav aria-label="章节目录"><p class="nav-label">CONTENTS / 推导路径</p><div id="contents"></div></nav>
<main><div id="load-status" role="status"></div><article id="article"></article><noscript>此讲义需要 JavaScript 渲染。可阅读同目录的 <a href="${lesson.name}.md">Markdown 原文</a>。</noscript></main></div>
<script type="text/plain" id="lesson-source">${source}</script>
</body></html>
`;
  await writeFile(path.join(folder, `${lesson.name}.html`), html);
  console.log(`Built ${lesson.directory}/${lesson.name}.html (${Buffer.byteLength(html)} bytes)`);
}
