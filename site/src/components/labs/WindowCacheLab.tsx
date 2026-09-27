import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function WindowCacheLab() {
  const [position, setPosition] = useState(5)
  const [window, setWindow] = useState(3)
  const first = Math.max(0, position - window + 1)
  const visible = Array.from({ length: position - first + 1 }, (_, i) => first + i)
  const tags = Array.from({ length: window }, (_, slot) => {
    const latest = position - ((position - slot) % window + window) % window
    return latest >= 0 ? latest : null
  })
  return <LabFrame title="逻辑位置与窗口物理槽">
    <Controls>
      <Range label="当前查询位置" value={position} min={0} max={11} onChange={setPosition} />
      <Range label="窗口（包含当前）" value={window} min={1} max={6} onChange={setWindow} />
    </Controls>
    <svg viewBox="0 0 460 200" className={pen.canvas} role="img" aria-label={`查询 ${position} 可见 ${visible.join('、')}，窗口 ${window}`}>
      {Array.from({ length: 12 }, (_, i) => <g key={i}>
        <rect x={12 + i * 37} y="30" width="30" height="34" rx="3" className={i >= first && i <= position ? 'fill-accent/20 stroke-accent' : 'fill-sunken stroke-rule'} />
        <text x={27 + i * 37} y="53" textAnchor="middle">{i}</text>
      </g>)}
      <text x="12" y="21">逻辑位置</text>
      <text x="12" y="106">物理槽与绝对位置 tag</text>
      {tags.map((tag, slot) => <g key={slot}>
        <rect x={20 + slot * 72} y="120" width="60" height="34" rx="3" className="fill-accent2/15 stroke-accent2" />
        <text x={50 + slot * 72} y="142" textAnchor="middle">{tag === null ? '空' : tag}</text>
        <text x={50 + slot * 72} y="176" textAnchor="middle">slot {slot}</text>
      </g>)}
    </svg>
    <Readout>可见集合 {'{' + visible.join(', ') + '}'} · 当前 slot {position % window} · RoPE 逻辑位置 {position} · 有效缓存 {visible.length}/{window}</Readout>
  </LabFrame>
}
