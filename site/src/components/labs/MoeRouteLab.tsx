import { useState } from 'react'
import Formula from './Formula'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function MoeRouteLab() {
  const [firstScore, setFirstScore] = useState(4)
  const [secondScore, setSecondScore] = useState(3)
  const [topK, setTopK] = useState(2)
  const scores = [firstScore, secondScore, 1, 0]
  const selected = scores.map((score, id) => ({ score, id }))
    .sort((a, b) => b.score - a.score || a.id - b.id).slice(0, topK)
  const maximum = selected[0].score
  const denominator = selected.reduce((sum, entry) => sum + Math.exp(entry.score - maximum), 0)
  const weights = scores.map((score, id) => selected.some(entry => entry.id === id)
    ? Math.exp(score - maximum) / denominator : 0)
  const scale = weights.reduce((sum, weight, id) => sum + weight * (id + 1), 0)
  return <LabFrame title="四专家路由与加权输出">
    <Controls>
      <Range label="专家 0 分数" value={firstScore} min={-2} max={5} step={0.1} onChange={setFirstScore} format={v => v.toFixed(1)} />
      <Range label="专家 1 分数" value={secondScore} min={-2} max={5} step={0.1} onChange={setSecondScore} format={v => v.toFixed(1)} />
      <Range label="活跃专家数" value={topK} min={1} max={4} onChange={setTopK} />
    </Controls>
    <svg viewBox="0 0 440 210" className={pen.canvas} role="img" aria-label={`选中专家 ${selected.map(e => e.id).join('、')}，输出缩放 ${scale.toFixed(3)}`}>
      {weights.map((weight, id) => <g key={id}>
        <rect x={34 + id * 102} y={150 - weight * 110} width="54" height={Math.max(1, weight * 110)} className={weight ? 'fill-accent' : 'fill-rule-strong'} />
        <text x={61 + id * 102} y="175" textAnchor="middle">专家 {id}</text>
        <text x={61 + id * 102} y="195" textAnchor="middle">{weight.toFixed(3)}</text>
        <text x={61 + id * 102} y="24" textAnchor="middle">分数 {scores[id].toFixed(1)}</text>
      </g>)}
      <line x1="20" y1="150" x2="426" y2="150" className={pen.axis} />
    </svg>
    <Readout>输入 (1,2) · 专家 <Formula>{'F_e(x)=(e+1)x'}</Formula> · 输出 ({scale.toFixed(6)}, {(2 * scale).toFixed(6)}) · 权重和 {weights.reduce((a, b) => a + b, 0).toFixed(6)}</Readout>
  </LabFrame>
}
