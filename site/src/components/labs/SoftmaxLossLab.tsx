import { useState } from 'react'
import { softmaxLoss } from '../../lib/math-labs-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

/** Server and browser may differ in the last float digit; round drawing coordinates. */
const r = (value: number) => Math.round(value * 100) / 100

export default function SoftmaxLossLab() {
  const [logits, setLogits] = useState([0, 0.7, 1.1])
  const [temperature, setTemperature] = useState(1)
  const [target, setTarget] = useState(2)
  const result = softmaxLoss(logits, temperature, target)
  const x = (c: number) => 40 + c * 90
  const gx = (c: number) => 330 + c * 70
  return <LabFrame title="softmax、交叉熵与分数梯度" hint="三类别 · 梯度是损失对原始分数 z 的导数">
    <Controls>
      {logits.map((z, c) => <Range key={c} label={`分数 z${c}`} value={z} min={-4} max={4} step={0.1} format={v => v.toFixed(1)}
        onChange={value => setLogits(logits.map((old, i) => (i === c ? value : old)))} />)}
      <Range label="温度 τ" value={temperature} min={0.25} max={3} step={0.25} onChange={setTemperature} format={v => v.toFixed(2)} />
      <label className="text-sm">正确类别
        <select value={target} onChange={event => setTarget(Number(event.target.value))} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2">
          {[0, 1, 2].map(c => <option key={c} value={c}>类别 {c}</option>)}
        </select>
      </label>
    </Controls>
    <svg viewBox="0 0 560 250" className={pen.canvas} role="img" aria-label={`概率 ${result.probs.map(p => p.toFixed(3)).join('、')}，损失 ${result.loss.toFixed(3)}`}>
      <text x="20" y="20" className={pen.muted}>概率 p = softmax(z / τ)</text>
      <line x1="20" x2="290" y1="200" y2="200" className={pen.axis} />
      {result.probs.map((p, c) => <g key={c}>
        <rect x={x(c)} y={r(200 - p * 160)} width="60" height={r(Math.max(1, p * 160))} rx="2" className={c === target ? 'fill-accent2' : 'fill-accent'} />
        <text x={x(c) + 30} y={r(192 - p * 160)} textAnchor="middle" className={pen.mono}>{p.toFixed(3)}</text>
        <text x={x(c) + 30} y="220" textAnchor="middle" className={c === target ? pen.textB : ''}>类别 {c}</text>
      </g>)}
      <text x="320" y="20" className={pen.muted}>梯度 ∂ℓ/∂z = (p − y) / τ</text>
      <line x1="310" x2="540" y1="115" y2="115" className={pen.axis} />
      {result.gradient.map((g, c) => <g key={c}>
        <rect x={gx(c)} y={r(g > 0 ? 115 - g * 80 : 115)} width="50" height={r(Math.max(1, Math.abs(g) * 80))} rx="2" className={c === target ? 'fill-accent2' : 'fill-info'} />
        <text x={gx(c) + 25} y={r(g > 0 ? 107 - g * 80 : 130 - g * 80)} textAnchor="middle" className={pen.mono}>{fmt(g)}</text>
        <text x={gx(c) + 25} y="220" textAnchor="middle" className={c === target ? pen.textB : ''}>z{c}</text>
      </g>)}
      <text x="20" y="244" className={pen.muted}>橙色是正确类别：它的梯度为负，下降会抬高它的分数；其余类别梯度为正。</text>
    </svg>
    <Readout>ℓ = −log p{target} = {result.loss.toFixed(4)} · 熵 {result.entropy.toFixed(4)} nat · 梯度之和 {fmt(result.gradient.reduce((a, b) => a + b, 0), 6)}（给所有分数加同一常数不改变结果）</Readout>
  </LabFrame>
}
