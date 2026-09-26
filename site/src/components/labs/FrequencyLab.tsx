import { useId, useState } from 'react'
import { Arrow, Controls, fmt, LabFrame, pen, Range, Readout, useWidth } from './Lab'

export const frequencies = Array.from({ length: 4 }, (_, m) => 10000 ** (-2 * m / 8))

/** One frequency pair seen two ways: as sin/cos curves over position, and as a point on the unit circle. */
export default function FrequencyLab() {
  const [m, setM] = useState(0)
  const [p, setP] = useState(12)
  const [ref, width] = useWidth<HTMLDivElement>()
  const id = useId().replace(/:/g, '')
  const compact = width < 560

  const omega = frequencies[m]
  const [sin, cos] = [Math.sin(p * omega), Math.cos(p * omega)]
  const view = compact ? { w: 360, h: 580 } : { w: 720, h: 340 }
  const plot = compact ? { l: 40, r: 348, t: 40, b: 290 } : { l: 56, r: 470, t: 36, b: 280 }
  const circle = compact ? { x: 180, y: 440, r: 80 } : { x: 600, y: 160, r: 84 }
  const x = (value: number) => plot.l + (value / 64) * (plot.r - plot.l)
  const y = (value: number) => plot.t + ((1.15 - value) / 2.3) * (plot.b - plot.t)
  const curve = (fn: (value: number) => number) =>
    Array.from({ length: 257 }, (_, index) => `${index ? 'L' : 'M'} ${x(index / 4).toFixed(2)} ${y(fn(index / 4)).toFixed(2)}`).join(' ')
  const vx = circle.x + circle.r * sin
  const vy = circle.y - circle.r * cos

  return (
    <LabFrame title="一个频率对，两种看法" hint="d = 8，四个维度对 m = 0…3">
      <Controls>
        <label className="block text-sm">
          <span>维度对 m</span>
          <select
            value={m}
            onChange={(event) => setM(Number(event.target.value))}
            className="mt-1 block w-full rounded-sm border border-rule-strong bg-paper px-2 py-1"
          >
            {frequencies.map((value, index) => (
              <option key={index} value={index}>m = {index} · d{2 * index}/d{2 * index + 1} · ω = {value}</option>
            ))}
          </select>
        </label>
        <Range label="位置 p" value={p} min={0} max={64} onChange={setP} />
      </Controls>
      <div ref={ref}>
        <svg viewBox={`0 0 ${view.w} ${view.h}`} className={pen.canvas} role="img"
          aria-label={`第 ${m} 对在位置 ${p} 的正弦 ${fmt(sin)}、余弦 ${fmt(cos)}`}>
          <defs>
            <clipPath id={`${id}-clip`}><rect x={plot.l} y={plot.t} width={plot.r - plot.l} height={plot.b - plot.t} /></clipPath>
            <Arrow id={`${id}-v`} className="fill-info" />
          </defs>
          {[0, 16, 32, 48, 64].map((tick) => (
            <g key={tick}>
              <line x1={x(tick)} y1={plot.t} x2={x(tick)} y2={plot.b} className={pen.grid} />
              <text x={x(tick)} y={plot.b + 20} textAnchor="middle" className={`${pen.mono} ${pen.muted}`}>{tick}</text>
            </g>
          ))}
          {[-1, 0, 1].map((tick) => (
            <g key={tick}>
              <line x1={plot.l} y1={y(tick)} x2={plot.r} y2={y(tick)} className={tick === 0 ? pen.axis : pen.grid} />
              <text x={plot.l - 10} y={y(tick) + 4} textAnchor="end" className={`${pen.mono} ${pen.muted}`}>{tick}</text>
            </g>
          ))}
          <g clipPath={`url(#${id}-clip)`}>
            <path d={curve((v) => Math.sin(v * omega))} className={pen.a} />
            <path d={curve((v) => Math.cos(v * omega))} className={pen.b} />
            <line x1={x(p)} y1={plot.t} x2={x(p)} y2={plot.b} className={pen.guide} />
          </g>
          <circle cx={x(p)} cy={y(sin)} r="5" className="fill-accent" />
          <circle cx={x(p)} cy={y(cos)} r="5" className="fill-accent2" />
          <line x1={plot.l + 8} y1={plot.t - 16} x2={plot.l + 34} y2={plot.t - 16} className={pen.a} />
          <text x={plot.l + 40} y={plot.t - 12}>sin(pω)</text>
          <line x1={plot.l + 118} y1={plot.t - 16} x2={plot.l + 144} y2={plot.t - 16} className={pen.b} />
          <text x={plot.l + 150} y={plot.t - 12}>cos(pω)</text>
          <text x={(plot.l + plot.r) / 2} y={plot.b + 42} textAnchor="middle" className={pen.muted}>位置 p（token）</text>

          <circle cx={circle.x} cy={circle.y} r={circle.r} className={pen.axis} />
          <line x1={circle.x - circle.r - 18} y1={circle.y} x2={circle.x + circle.r + 18} y2={circle.y} className={pen.axis} />
          <line x1={circle.x} y1={circle.y - circle.r - 18} x2={circle.x} y2={circle.y + circle.r + 18} className={pen.axis} />
          <line x1={circle.x} y1={circle.y} x2={vx} y2={circle.y} className={`${pen.a} stroke-2`} />
          <line x1={vx} y1={circle.y} x2={vx} y2={vy} className={`${pen.b} stroke-2`} />
          <line x1={circle.x} y1={circle.y} x2={vx} y2={vy} className={pen.c} markerEnd={`url(#${id}-v)`} />
          <text x={circle.x + circle.r + 16} y={circle.y - 8} textAnchor="end" className={pen.muted}>sin</text>
          <text x={circle.x + 8} y={circle.y - circle.r - 12} className={pen.muted}>cos</text>
          <text x={circle.x} y={circle.y + circle.r + 44} textAnchor="middle" className={pen.mono}>u(p) = (sin, cos)</text>
        </svg>
      </div>
      <Readout>
        PE[{p}, {2 * m}] = sin({fmt(p * omega)}) = {fmt(sin, 6)} · PE[{p}, {2 * m + 1}] = cos({fmt(p * omega)}) = {fmt(cos, 6)}
      </Readout>
    </LabFrame>
  )
}
