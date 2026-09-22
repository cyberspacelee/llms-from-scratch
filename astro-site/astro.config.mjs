import { defineConfig } from 'astro/config'
import mdx from '@astrojs/mdx'
import react from '@astrojs/react'
import { unified } from '@astrojs/markdown-remark'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { rehypeMermaidBlocks } from './src/plugins/rehype-mermaid.mjs'

export default defineConfig({
  site: 'https://cyberspacelee.github.io',
  base: '/llms-from-scratch',
  integrations: [
    react(),
    mdx(),
  ],
  markdown: {
    syntaxHighlight: {
      type: 'shiki',
      excludeLangs: ['mermaid'],
    },
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeMermaidBlocks, [rehypeKatex, { throwOnError: true, strict: 'error' }]],
    }),
  },
})
