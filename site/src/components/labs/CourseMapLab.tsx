import { useState } from 'react'
import { Controls, LabFrame, pen, useWidth } from './Lab'

export type CourseNode = { track: string; code: string; title: string; question: string; href: string; order: number }
type TrackChoice = { id: string; label: string }

export default function CourseMapLab({ nodes, tracks, initialTrack, locked = false }: {
  nodes: CourseNode[]; tracks: TrackChoice[]; initialTrack: string; locked?: boolean
}) {
  const [track, setTrack] = useState(initialTrack)
  const [selected, setSelected] = useState('')
  const [practical, setPractical] = useState(false)
  const [ref, width] = useWidth<HTMLDivElement>(320)
  const chapters = nodes.filter(node => node.track === track)
  const sequence = track === 'training' && practical ? [8, 1, 2, 3, 4, 5, 6, 7, 9] : chapters.map(node => node.order)
  const ordered = sequence.map(order => chapters.find(node => node.order === order)!).filter(Boolean)
  const current = ordered.find(node => node.code === selected) ?? ordered[0]
  const columns = width >= 560 ? 2 : 1
  const canvas = columns === 2 ? 640 : 320
  const rowHeight = 94
  const position = (index: number) => {
    const row = Math.floor(index / columns)
    const column = columns === 2 && row % 2 === 1 ? 1 - index % 2 : index % columns
    return { x: 10 + column * 320, y: 12 + row * rowHeight }
  }
  const adjacent = ordered.flatMap((node, index) => {
    if (index === 0 || track === 'advanced' || (track === 'math' && node.order === 11)) return []
    const previous = position(index - 1), next = position(index)
    const sameRow = previous.y === next.y
    const x1 = sameRow ? previous.x + (next.x > previous.x ? 298 : 0) : previous.x + 149
    const x2 = sameRow ? next.x + (next.x > previous.x ? 0 : 298) : next.x + 149
    return [{ x1, y1: sameRow ? previous.y + 32 : previous.y + 64, x2, y2: sameRow ? next.y + 32 : next.y }]
  })
  if (!current) return null
  return <LabFrame title={`${tracks.find(item => item.id === track)?.label} · 阅读路线`} hint={track === 'advanced' ? '按问题选择分支' : '连线表示建议阅读次序'}>
    <div ref={ref}>
      <Controls>
        {!locked && <label className="text-sm">路线<select value={track} onChange={event => { setTrack(event.target.value); setSelected(''); setPractical(false) }} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2">{tracks.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>}
        <label className="text-sm">章节<select value={current.code} onChange={event => setSelected(event.target.value)} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2">{ordered.map(node => <option key={node.code} value={node.code}>{node.code} · {node.title}</option>)}</select></label>
        {track === 'training' && <label className="flex h-9 items-center gap-2 text-sm"><input type="checkbox" checked={practical} onChange={event => setPractical(event.target.checked)} className="accent-accent" />先处理原始数据</label>}
      </Controls>
      <svg viewBox={`0 0 ${canvas} ${Math.ceil(ordered.length / columns) * rowHeight}`} className={pen.canvas} role="group" aria-label={`${track} 章节路线，当前 ${current.code}`}>
        {adjacent.map((edge, index) => <g key={index}><line {...edge} className={pen.axis} /><circle cx={edge.x2} cy={edge.y2} r="3" className="fill-accent" /></g>)}
        {ordered.map((node, index) => {
          const { x, y } = position(index)
          const chosen = current.code === node.code
          const title = Array.from(node.title)
          return <g key={node.code} role="button" tabIndex={0} aria-pressed={chosen} aria-label={`${node.code} ${node.title}`} onClick={() => setSelected(node.code)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setSelected(node.code) } }} className="cursor-pointer focus:outline-2 focus:outline-info">
            <rect x={x} y={y} width="298" height="64" rx="3" className={chosen ? 'fill-accent-soft stroke-accent stroke-2' : 'fill-paper stroke-rule'} />
            <text x={x + 12} y={y + 36} className={pen.mono}>{node.code}</text>
            <text x={x + 54} y={y + (title.length > 17 ? 27 : 36)} className={chosen ? pen.textA : ''}>{title.slice(0, 17).join('')}</text>
            {title.length > 17 && <text x={x + 54} y={y + 47}>{title.slice(17).join('')}</text>}
            {track === 'math' && node.order === 11 && <text x={x + 12} y={y + 58} className="text-[10px] fill-muted!">分支</text>}
          </g>
        })}
      </svg>
      <div aria-live="polite" className="mt-3 border-t border-rule pt-3 text-sm">
        <p className="m-0 text-muted">{current.question}</p>
        <a href={current.href} className="mt-2 inline-block font-semibold text-accent">{current.code} · {current.title}</a>
        {track === 'math' && current.order === 11 && <p className="mt-2 mb-0 text-xs text-muted">谱分解分支从 M6、M7 接入，不是进入模型路线的门槛。</p>}
        {track === 'training' && practical && <p className="mt-2 mb-0 text-xs text-muted">实际项目先按 T8 清理和分组，再进行 T1 样本构造；T5–T9 按适配、预算与分布式需求进入。</p>}
      </div>
    </div>
  </LabFrame>
}
