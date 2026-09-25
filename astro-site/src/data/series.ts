import { aiInfraChapters } from './ai-infra'

export type SeriesGroup = 'principle' | 'math' | 'infra'

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
    title: '深度学习数学，从统计到反向传播',
    question: '八章：概率、矩阵、微积分、神经网络与旋转',
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

export const mathItems: SeriesItem[] = [
  ['概率与统计', '均值、期望、方差与批量统计'],
  ['分布与信息量', '正态分布、似然、交叉熵与 KL'],
  ['向量与矩阵', '形状、点积、矩阵乘法与转置'],
  ['微积分与梯度', '导数、偏导、积分与优化步长'],
  ['矩阵求导', '从分量到批量线性层的梯度'],
  ['链式法则与反向传播', '计算图怎样累加梯度'],
  ['神经网络训练', '两层网络的前向、反向与参数更新'],
  ['旋转与位置编码', '正交变换、复数与 RoPE'],
].map(([title, question], index) => {
  const number = String(index + 1).padStart(2, '0')
  return {
    id: `math-${number}`, nav: title, title, question,
    href: `lessons/math-${number}`,
    kicker: `数学 · ${number}`, sidebar: `${number} · ${title}`,
    group: 'math',
  }
})

export const series: SeriesItem[] = [principle[0], ...mathItems, ...principle.slice(1), ...infra]
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
