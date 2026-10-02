import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { stateRecurrence } from '../../lib/advanced-interactive-model'

export default function StateScanLab() {
  const [step, setStep] = useState(0), [selective, setSelective] = useState(false), [retention, setRetention] = useState(0.5), [scan, setScan] = useState(false)
  const rows = stateRecurrence(selective, retention), r = rows[step]
  return <LabFrame title="同一四步递推与仿射组合">
    <Controls><Range label="t" value={step} min={0} max={3} onChange={setStep} /><Range label="保持系数 a" value={retention} min={0} max={1.2} step={0.1} onChange={setRetention} /><Toggle label="负输入保持旧状态且不写入" checked={selective} onChange={value => setSelective(value)} /><Toggle label="仿射组合路径" checked={scan} onChange={value => setScan(value)} /></Controls>
    <SvgCanvas viewBox="0 0 280 230" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`t=${step} 状态 ${r.state}，扫描参照 ${r.cumulativeB}`}>
      {rows.map((row, i) => <g key={i}><rect x="8" y={8 + i * 53} width="264" height="45" className={i === step ? 'fill-accent/15 stroke-accent' : 'fill-sunken stroke-rule'} /><text x="16" y={26 + i * 53}>t{i} · u={row.input} · (a,b)=({fmt(row.a, 1)},{row.b})</text><text x="16" y={44 + i * 53} className={pen.mono}>{scan ? `累计 (A,B)=(${fmt(row.cumulativeA)},${fmt(row.cumulativeB)})` : `${fmt(row.decayed)} + ${row.b} = ${fmt(row.state)}`}</text></g>)}
    </SvgCanvas><Readout>旧状态={fmt(r.previous)}；衰减={fmt(r.decayed)}；新写入={r.b}；状态={fmt(r.state)}<br />(a₂,b₂)∘(a₁,b₁)=(a₂a₁,a₂b₁+b₂)；累计映射 s=A·0+B={fmt(r.cumulativeB)}<br />串行与仿射组合误差={fmt(r.state - r.cumulativeB, 9)}；本图不测并行扫描速度</Readout>
  </LabFrame>
}
