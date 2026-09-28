import { useId, useState } from 'react'
import { descend } from '../../lib/math-labs-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

const scale = 80, cx = 220, cy = 130
const px = (v: number) => Math.round((cx + v * scale) * 100) / 100
const py = (v: number) => Math.round((cy - v * scale) * 100) / 100
const verdicts = { converge: '两个坐标单调收敛', oscillate: '第二坐标振荡收敛', boundary: '第二坐标等幅振荡，不收敛', diverge: '第二坐标振幅扩大，发散' }

export default function DescentLab() {
  const [eta, setEta] = useState(0.2)
  const [steps, setSteps] = useState(12)
  const clip = useId()
  const run = descend(eta, [2, 1], steps)
  const last = run.path[run.path.length - 1]
  return <LabFrame title="学习率怎样决定收敛、振荡或发散" hint="L = (x₁² + 4x₂²) / 2，从 (2, 1) 出发">
    <Controls>
      <Range label="学习率 η" value={eta} min={0.05} max={0.7} step={0.05} onChange={setEta} format={v => v.toFixed(2)} />
      <Range label="迭代步数" value={steps} min={1} max={20} onChange={setSteps} />
    </Controls>
    <svg viewBox="0 0 440 260" className={`${pen.canvas} max-w-120`} role="img" aria-label={`学习率 ${eta}，${steps} 步后位于 (${fmt(last[0])}, ${fmt(last[1])})`}>
      <defs><clipPath id={clip}><rect x="0" y="0" width="440" height="260" /></clipPath></defs>
      <line x1="0" x2="440" y1={cy} y2={cy} className={pen.grid} />
      <line x1={cx} x2={cx} y1="0" y2="260" className={pen.grid} />
      {[0.125, 0.5, 1.125, 2, 3].map(c => <ellipse key={c} cx={cx} cy={cy} rx={Math.sqrt(2 * c) * scale} ry={Math.sqrt(2 * c) / 2 * scale} className={pen.grid} />)}
      <g clipPath={`url(#${clip})`}>
        <polyline points={run.path.map(([a, b]) => `${px(a)},${py(b)}`).join(' ')} className={run.verdict === 'diverge' ? pen.b : pen.a} />
        {run.path.map(([a, b], i) => <circle key={i} cx={px(a)} cy={py(b)} r={i === 0 ? 5 : 3} className={i === 0 ? 'fill-ink' : run.verdict === 'diverge' ? 'fill-accent2' : 'fill-accent'} />)}
      </g>
      <text x="430" y={cy - 6} textAnchor="end" className={pen.muted}>x₁</text>
      <text x={cx + 6} y="14" className={pen.muted}>x₂</text>
    </svg>
    <Readout>每步 x₁ ← {fmt(run.factors[0], 2)}·x₁，x₂ ← {fmt(run.factors[1], 2)}·x₂ · {verdicts[run.verdict]} · 第 {steps} 步损失 {run.losses[steps].toExponential(3)} · 收敛条件 0 &lt; η &lt; 2/λ_max = 0.5</Readout>
  </LabFrame>
}
