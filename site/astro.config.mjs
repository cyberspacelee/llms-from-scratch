import { defineConfig } from 'astro/config'
import mdx from '@astrojs/mdx'
import react from '@astrojs/react'
import { unified } from '@astrojs/markdown-remark'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { rehypeMermaidBlocks } from './src/plugins/rehype-mermaid.mjs'
import { rehypeTableScroll } from './src/plugins/rehype-table-scroll.mjs'
import { rehypeBaseLinks } from './src/plugins/rehype-base-links.mjs'
import tailwindcss from '@tailwindcss/vite'

const base = '/llms-from-scratch'
// A running dev server must keep its dependency files when check/build run concurrently.
const cacheCommand = process.argv.find(arg => ['dev', 'check', 'build'].includes(arg)) ?? 'build'

export default defineConfig({
  site: 'https://cyberspacelee.github.io',
  base,
  trailingSlash: 'always',
  integrations: [react(), mdx()],
  vite: {
    cacheDir: `node_modules/.vite/${cacheCommand}`,
    plugins: [tailwindcss()],
  },
  markdown: {
    syntaxHighlight: {
      type: 'shiki',
      excludeLangs: ['mermaid'],
    },
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      defaultColor: false,
    },
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeMermaidBlocks,
        rehypeTableScroll,
        [rehypeBaseLinks, { base }],
        [rehypeKatex, { throwOnError: true, strict: 'error' }],
      ],
    }),
  },
})
