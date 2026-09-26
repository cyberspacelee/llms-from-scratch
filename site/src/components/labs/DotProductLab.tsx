import { useId, useState } from 'react'
import { Arrow, Button, Controls, fmt, LabFrame, pen, Range, Readout } from './Lab'

const initial = { a: 2, b: 3, phase: 60, common: 0, relative: 0 }
const deg = Math.PI / 180

/** Length, angle and the dot product; a common rotation cancels, a relative one does not. */
export default function DotProductLab() {
  const [state, setState] = useState(initial)
  const set = (key: keyof typeof initial) => (value: number) => setState((current) => ({ ...current, [key]: value }))
  const id = useId().replace(/:/g, '')

  const delta = (state.phase + state.relative) * deg
  const cosine = Math.cos(delta)
  const score = state.a * state.b * cosine
  const theta = Math.acos(Math.max(-1, Math.min(1, cosine)))

  const [cx, cy, scale] = [220, 160, 32]
  const point = (r: number, angle: number) => [cx + scale * r * Math.cos(angle), cy - scale * r * Math.sin(angle)]
  const angleA = state.common * deg
  const [ax, ay] = point(state.a, angleA)
  const [bx, by] = point(state.b, angleA + delta)
  const [px, py] = point(state.b * cosine, angleA)

  return (
    <LabFrame title="长度、夹角与点积" hint="a 的相位是 γ，b 的相位是 γ + δ + φ">
      <Controls>
        <Range label="a 的长度" value={state.a} min={0.5} max={4} step={0.1} onChange={set('a')} format={(v) => v.toFixed(1)} />
        <Range label="b 的长度" value={state.b} min={0.5} max={4} step={0.1} onChange={set('b')} format={(v) => v.toFixed(1)} />
        <Range label="内容相位差 δ" value={state.phase} min={-180} max={180} onChange={set('phase')} format={(v) => `${v}°`} />
        <Range label="共同旋转 γ" value={state.common} min={-180} max={180} onChange={set('common')} format={(v) => `${v}°`} />
        <Range label="b 的额外旋转 φ" value={state.relative} min={-180} max={180} onChange={set('relative')} format={(v) => `${v}°`} />
        <div><Button onClick={() => setState(initial)}>恢复初始值</Button></div>
      </Controls>
      <svg viewBox="0 0 440 330" className={pen.canvas} role="img"
        aria-label={`a 长 ${state.a.toFixed(1)}，b 长 ${state.b.toFixed(1)}，夹角 ${fmt(theta / deg, 1)} 度，点积 ${fmt(score)}`}>
        <defs>
          <Arrow id={`${id}-a`} className="fill-accent" />
          <Arrow id={`${id}-b`} className="fill-accent2" />
        </defs>
        <circle cx={cx} cy={cy} r={state.a * scale} className={pen.grid} />
        <circle cx={cx} cy={cy} r={state.b * scale} className={pen.grid} />
        <line x1={cx - 140} y1={cy} x2={cx + 140} y2={cy} className={pen.axis} />
        <line x1={cx} y1={cy - 140} x2={cx} y2={cy + 140} className={pen.axis} />
        <line x1={bx} y1={by} x2={px} y2={py} className={pen.guide} />
        <line x1={cx} y1={cy} x2={px} y2={py} className={pen.guide} />
        <line x1={cx} y1={cy} x2={ax} y2={ay} className={pen.a} markerEnd={`url(#${id}-a)`} />
        <line x1={cx} y1={cy} x2={bx} y2={by} className={pen.b} markerEnd={`url(#${id}-b)`} />
        <circle cx={px} cy={py} r="4" className="fill-info" />
        <text x={ax} y={ay - 12} textAnchor="middle" className={pen.textA}>a</text>
        <text x={bx} y={by + 22} textAnchor="middle" className={pen.textB}>b</text>
        <text x={cx} y={322} textAnchor="middle" className={pen.muted}>圆的半径是长度；蓝色虚线是 b 在 a 方向上的投影</text>
      </svg>
      <Readout>
        相位差 δ + φ = {state.phase + state.relative}° · 夹角 θ = {fmt(theta / deg, 1)}° · 有符号投影 = {fmt(state.b * cosine)}
        <br />
        aᵀb = {state.a.toFixed(1)} × {state.b.toFixed(1)} × {fmt(cosine)} = <strong>{fmt(score)}</strong> · 只由内容决定的原点积 = {fmt(state.a * state.b * Math.cos(state.phase * deg))}
      </Readout>
    </LabFrame>
  )
}
