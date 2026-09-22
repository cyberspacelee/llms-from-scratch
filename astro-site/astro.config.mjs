import { defineConfig } from 'astro/config'
import mdx from '@astrojs/mdx'
import react from '@astrojs/react'
import { unified } from '@astrojs/markdown-remark'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { rehypeMermaidBlocks } from './src/plugins/rehype-mermaid.mjs'
import { rehypeTableScroll } from './src/plugins/rehype-table-scroll.mjs'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  site: 'https://cyberspacelee.github.io',
  base: '/llms-from-scratch',
  integrations: [
    react(),
    mdx(),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
  markdown: {
    syntaxHighlight: {
      type: 'shiki',
      excludeLangs: ['mermaid'],
    },
    shikiConfig: {
      theme: 'github-dark',
    },
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeMermaidBlocks,
        rehypeTableScroll,
        [rehypeKatex, { throwOnError: true, strict: 'error' }],
      ],
    }),
  },
})
