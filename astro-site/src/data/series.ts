import { aiInfraChapters } from './ai-infra'

export type SeriesGroup = 'principle' | 'infra'

export type SeriesItem = {
  id: string
  nav: string
  title: string
  question: string
  href: string
  kicker: string
  sidebar: string
  group: SeriesGroup
}

const principle: SeriesItem[] = [
  {
    id: 'mathematics',
    nav: '数学',
    title: '数学基础，理解旋转',
    question: '从向量、范数、点积到旋转',
    href: 'lessons/mathematics',
    kicker: '00 · 数学基础',
    sidebar: '00 · 数学基础',
    group: 'principle',
  },
  {
    id: 'position-encoding',
    nav: '位置编码',
    title: '位置编码，从第一性原理推导',
    question: 'sin/cos、相对位置与 RoPE',
    href: 'lessons/position-encoding',
    kicker: '01 · 位置编码',
    sidebar: '01 · 位置编码',
    group: 'principle',
  },
  {
    id: 'kv-cache',
    nav: 'KV cache',
    title: '位置编码，走进 KV cache',
    question: '从整段注意力到逐 token 解码',
    href: 'lessons/kv-cache',
    kicker: '02 · KV cache',
    sidebar: '02 · KV cache',
    group: 'principle',
  },
  {
    id: 'nano-vllm',
    nav: '推理系统',
    title: 'nano-vLLM：从一条请求到一个推理引擎',
    question: '调度、分页缓存与 CUDA Graph 如何协作',
    href: 'inference/nano-vllm-from-zero-to-mastery.html',
    kicker: '03 · 推理系统',
    sidebar: '03 · 推理系统',
    group: 'principle',
  },
]

const infra: SeriesItem[] = aiInfraChapters.map((chapter) => ({
  id: chapter.slug,
  nav: chapter.nav,
  title: chapter.title,
  question: chapter.question,
  href: `lessons/${chapter.slug}`,
  kicker: chapter.slug === 'ai-infra' ? 'AI Infra · 导读' : `AI Infra · ${chapter.nav}`,
  sidebar: chapter.nav,
  group: 'infra' as const,
}))

export const series: SeriesItem[] = [...principle, ...infra]
export const principleItems = principle
export const infraItems = infra

export const topNav = [
  { id: 'mathematics', label: '数学', href: 'lessons/mathematics' },
  { id: 'position-encoding', label: '位置编码', href: 'lessons/position-encoding' },
  { id: 'kv-cache', label: 'KV cache', href: 'lessons/kv-cache' },
  { id: 'nano-vllm', label: '推理系统', href: 'inference/nano-vllm-from-zero-to-mastery.html' },
  { id: 'ai-infra', label: 'AI Infra', href: 'lessons/ai-infra' },
] as const

export function findSeries(id?: string) {
  if (!id) return undefined
  return series.find((item) => item.id === id)
}
