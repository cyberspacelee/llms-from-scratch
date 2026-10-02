import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function TokenWeightLab() {
  const [firstCount, setFirstCount] = useState(2)
  const [secondCount, setSecondCount] = useState(6)
  const [firstLoss, setFirstLoss] = useState(1)
  const [secondLoss, setSecondLoss] = useState(3)
  const total = firstCount + secondCount
  const tokenMean = (firstCount * firstLoss + secondCount * secondLoss) / total
  const sequenceMean = (firstLoss + secondLoss) / 2
  return <LabFrame title="有效 token 权重与平均损失">
    <Controls>
      <Range label="微批 A 有效目标数" value={firstCount} min={1} max={16} onChange={setFirstCount} />
      <Range label="微批 B 有效目标数" value={secondCount} min={1} max={16} onChange={setSecondCount} />
      <Range label="微批 A 平均 NLL" value={firstLoss} min={0} max={4} step={0.1} onChange={setFirstLoss} format={v => v.toFixed(1)} />
      <Range label="微批 B 平均 NLL" value={secondLoss} min={0} max={4} step={0.1} onChange={setSecondLoss} format={v => v.toFixed(1)} />
    </Controls>
    <SvgCanvas viewBox="0 0 440 210" className={pen.canvas} role="img" aria-label={`token 加权损失 ${tokenMean.toFixed(3)}，微批均值平均 ${sequenceMean.toFixed(3)}`}>
      {[{ label: 'token 加权', value: tokenMean, color: 'fill-accent' }, { label: '微批均值平均', value: sequenceMean, color: 'fill-accent2' }].map((bar, i) => <g key={bar.label}>
        <rect x={90 + 190 * i} y={160 - bar.value * 30} width="70" height={Math.max(1, bar.value * 30)} className={bar.color} />
        <text x={125 + 190 * i} y="185" textAnchor="middle">{bar.label}</text>
        <text x={125 + 190 * i} y={150 - bar.value * 30} textAnchor="middle">{bar.value.toFixed(3)}</text>
      </g>)}
      <line x1="50" y1="160" x2="415" y2="160" className={pen.axis} />
    </SvgCanvas>
    <Readout>A 权重 {(firstCount / total).toFixed(3)} · B 权重 {(secondCount / total).toFixed(3)} · token PPL {Math.exp(tokenMean).toFixed(3)}</Readout>
  </LabFrame>
}
