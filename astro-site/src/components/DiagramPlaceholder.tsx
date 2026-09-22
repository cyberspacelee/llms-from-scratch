import React from 'react'

export default function DiagramPlaceholder({ title }: { title: string }) {
  return <section className="astro-lab" aria-label={title}><strong>{title}</strong><p>这一节可以在本地打开的原理讲义里拖动参数，看图随数值变化。</p></section>
}
