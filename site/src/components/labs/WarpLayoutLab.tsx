import { useState } from 'react'
import { threadCoordinates } from './GpuIndexModel'
import { LabFrame, Range, Readout, pen } from './Lab'

export default function WarpLayoutLab() {
  const [thread, setThread] = useState(32)
  const layouts = [[8, 8, 35], [16, 4, 277], [32, 2, 423]] as const
  return <LabFrame title="64 线程的三种布局与 warp 边界">
    <Range label="块内线性线程号" value={thread} min={0} max={63} onChange={setThread} />
    <svg viewBox="0 0 320 544" className={`${pen.canvas} max-w-96`} role="img" aria-label={`比较 8×8、16×4、32×2 布局，选中线程 ${thread}，warp ${Math.floor(thread / 32)}，lane ${thread % 32}`}>
      {layouts.map(([width, height, top]) => {
        const selected = threadCoordinates(thread, width, height)
        return <g key={width}>
          <text x="12" y={top - 12}>{width}×{height} · 选中 ({selected.x},{selected.y})</text>
          {Array.from({ length: 64 }, (_, t) => {
            const coordinates = threadCoordinates(t, width, height)
            return <rect key={t} x={12 + coordinates.x * (296 / width)} y={top + coordinates.y * 24} width={296 / width - 2} height="20" rx="1" className={t === thread ? 'fill-accent2 stroke-ink stroke-2' : t < 32 ? 'fill-accent-soft stroke-accent' : 'fill-info-soft stroke-info'} />
          })}
          <line x1="12" x2="308" y1={top + (32 / width) * 24 - 2} y2={top + (32 / width) * 24 - 2} className="stroke-ink stroke-2 [stroke-dasharray:4_3]" />
        </g>
      })}
      <text x="12" y="504" className={pen.textA}>绿色：warp 0</text><text x="167" y="504" className="fill-info!">蓝色：warp 1</text>
      <text x="12" y="526" className={pen.muted}>橙色：选中线程 · 虚线：warp 边界</text>
    </svg>
    <Readout>线程 {thread} · warp {Math.floor(thread / 32)} · lane {thread % 32}<br />
      {layouts.map(([width, height]) => {
        const c = threadCoordinates(thread, width, height)
        return <span key={width} className="mr-3 inline-block">{width}×{height}：({c.x},{c.y})</span>
      })}
    </Readout>
  </LabFrame>
}
