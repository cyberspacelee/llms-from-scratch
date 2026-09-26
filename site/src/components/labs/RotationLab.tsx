import { useId, useState } from 'react'
import { Arrow, Controls, fmt, LabFrame, pen, Range, Readout, useWidth } from './Lab'

const theta = Math.PI / 8

/** Same offset, different starting point: the relative rotation depends only on Δ. */
export default function RotationLab() {
  const [p, setP] = useState(2)
  const [delta, setDelta] = useState(3)
  const [ref, width] = useWidth<HTMLDivElement>()
  const id = useId().replace(/:/g, '')
  const compact = width < 560

  const phi = delta * theta
  const [c, s] = [Math.cos(phi), Math.sin(phi)]
  const view = compact ? { w: 360, h: 540, cx: 180, cy: 150, r: 104 } : { w: 720, h: 320, cx: 170, cy: 160, r: 112 }
  const point = (angle: number, radius = view.r) => [view.cx + radius * Math.cos(angle), view.cy - radius * Math.sin(angle)]
  const [qx, qy] = point(p * theta)
  const [kx, ky] = point((p + delta) * theta)
  const [sx, sy] = point(p * theta, 48)
  const [ex, ey] = point((p + delta) * theta, 48)

  const cell = { w: compact ? 120 : 100, h: 48, gap: 8 }
  const mw = cell.w * 2 + cell.gap
  const mx = compact ? (view.w - mw) / 2 : 500
  const my = compact ? 360 : 104
  const values = [c, -s, s, c]

  return (
    <LabFrame title="同样的位移，不同的起点" hint="θ = π/8，查询与键的内容都取 (1, 0)">
      <Controls>
        <Range label="起点 p" value={p} min={0} max={20} onChange={setP} />
        <Range label="位移 Δ" value={delta} min={-8} max={8} onChange={setDelta} />
      </Controls>
      <div ref={ref}>
        <svg viewBox={`0 0 ${view.w} ${view.h}`} className={pen.canvas} role="img"
          aria-label={`q 位于 ${p}，k 位于 ${p + delta}，相对旋转 ${fmt(phi)} 弧度，点积 ${fmt(c, 6)}`}>
          <defs>
            <Arrow id={`${id}-q`} className="fill-accent" />
            <Arrow id={`${id}-k`} className="fill-accent2" />
          </defs>
          <circle cx={view.cx} cy={view.cy} r={view.r} className={pen.axis} />
          <line x1={view.cx - view.r - 20} y1={view.cy} x2={view.cx + view.r + 20} y2={view.cy} className={pen.axis} />
          <line x1={view.cx} y1={view.cy - view.r - 20} x2={view.cx} y2={view.cy + view.r + 20} className={pen.axis} />
          <text x={view.cx + view.r + 18} y={view.cy - 8} textAnchor="end" className={pen.muted}>cos</text>
          <text x={view.cx + 8} y={view.cy - view.r - 12} className={pen.muted}>sin</text>
          {delta !== 0 && (
            <path d={`M ${sx} ${sy} A 48 48 0 0 ${delta > 0 ? 0 : 1} ${ex} ${ey}`} className={pen.guide} />
          )}
          <line x1={view.cx} y1={view.cy} x2={qx} y2={qy} className={pen.a} markerEnd={`url(#${id}-q)`} />
          <line x1={view.cx} y1={view.cy} x2={kx} y2={ky} className={pen.b} markerEnd={`url(#${id}-k)`} />
          <text x={Math.min(view.w - 40, qx + 10)} y={Math.max(16, qy - 10)} className={pen.textA}>q(p)</text>
          <text x={Math.min(view.w - 70, kx + 10)} y={Math.min(view.cy + view.r + 34, ky + 20)} className={pen.textB}>k(p + Δ)</text>

          <text x={mx} y={my - 20} className={pen.mono}>R(Δθ) =</text>
          <path
            d={`M ${mx - 8} ${my - 6} h -8 v ${cell.h * 2 + cell.gap + 12} h 8 M ${mx + mw + 8} ${my - 6} h 8 v ${cell.h * 2 + cell.gap + 12} h -8`}
            className={pen.axis}
          />
          {values.map((value, index) => {
            const x = mx + (index % 2) * (cell.w + cell.gap)
            const y = my + Math.floor(index / 2) * (cell.h + cell.gap)
            return (
              <g key={index}>
                <rect x={x} y={y} width={cell.w} height={cell.h} rx="4" className="fill-sunken" />
                <text x={x + cell.w / 2} y={y + 29} textAnchor="middle" className={pen.mono}>{fmt(value)}</text>
              </g>
            )
          })}
          <text x={mx + mw / 2} y={my + cell.h * 2 + cell.gap + 40} textAnchor="middle" className={`${pen.mono} ${pen.muted}`}>Δθ = {fmt(phi)} rad</text>
          <text x={mx + mw / 2} y={my + cell.h * 2 + cell.gap + 66} textAnchor="middle" className={pen.mono}>qᵀR(Δθ)k = {fmt(c, 6)}</text>
        </svg>
      </div>
      <Readout>
        p = {p} · j = {p + delta} · Δ = {delta} · 两个向量一起转了 p·θ，夹角只由 Δ 决定 · cos(Δθ) = {fmt(c, 6)}
      </Readout>
    </LabFrame>
  )
}
