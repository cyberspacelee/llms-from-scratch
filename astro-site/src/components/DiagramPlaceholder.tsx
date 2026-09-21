import React from 'react'

export default function DiagramPlaceholder({ title }: { title: string }) {
  return <section className="astro-lab" aria-label={title}><strong>{title}</strong><p>此图解组件以 SVG 静态图作为首屏内容，交互版本可按需 hydration。</p></section>
}
