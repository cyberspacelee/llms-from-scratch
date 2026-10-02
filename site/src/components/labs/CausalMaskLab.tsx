import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Controls, LabFrame, pen, Range, Readout } from './Lab'

export default function CausalMaskLab() {
  const [query, setQuery] = useState(2)
  const [temperature, setTemperature] = useState(1)
  const scores = [0.2, 1.1, -0.4, 0.7]
  const maximum = Math.max(...scores.slice(0, query + 1))
  const numerator = scores.map((s, j) => j <= query ? Math.exp((s - maximum) / temperature) : 0)
  const denominator = numerator.reduce((a, b) => a + b, 0)
  const weights = numerator.map(v => v / denominator)
  return <LabFrame title="可见历史与注意力权重">
    <Controls>
      <Range label="查询位置 i" value={query} min={0} max={3} onChange={setQuery} />
      <Range label="分数温度 τ" value={temperature} min={0.2} max={2} step={0.1} onChange={setTemperature} format={v => v.toFixed(1)} />
    </Controls>
    <SvgCanvas viewBox="0 0 360 230" className={`${pen.canvas} max-w-110!`} role="img" aria-label={`查询位置 ${query}，未来权重为零`}>
      {scores.map((s, j) => <g key={j}>
        <rect x={26 + j * 80} y={170 - weights[j] * 125} width="54" height={Math.max(1, weights[j] * 125)} className={j <= query ? 'fill-accent' : 'fill-rule-strong'} />
        <text x={53 + j * 80} y="196" textAnchor="middle">j={j}</text>
        <text x={53 + j * 80} y="217" textAnchor="middle">{j <= query ? weights[j].toFixed(3) : '屏蔽'}</text>
        <text x={53 + j * 80} y="25" textAnchor="middle">s={s}</text>
      </g>)}
      <line x1="12" y1="170" x2="348" y2="170" className={pen.axis} />
    </SvgCanvas>
    <Readout>允许 j ≤ {query} · 权重和 {weights.reduce((a, b) => a + b, 0).toFixed(6)} · 未来权重 0</Readout>
  </LabFrame>
}
