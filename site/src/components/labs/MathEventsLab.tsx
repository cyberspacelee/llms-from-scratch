import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { eventProbability, type EventName } from '../../lib/math-probability-model'
import { Select, Controls, LabFrame, Readout, fmt, pen } from './Lab'

const names: Record<EventName, string> = { positive: '正类', negative: '非正类', D: '来源 D', firstTwo: '前两条记录', empty: '空事件' }
export default function MathEventsLab() {
  const [a, setA] = useState<EventName>('positive'), [b, setB] = useState<EventName>('D')
  const result = eventProbability(a, b)
  return <LabFrame title="条件集合成为新的分母">
    <Controls>{([{ label: '事件 A', value: a, change: setA }, { label: '已知事件 B', value: b, change: setB }]).map(control => <Select key={control.label} label={control.label} value={control.value} onChange={e => control.change(e.target.value as EventName)}>{Object.entries(names).map(([key, title]) => <option key={key} value={key}>{title}</option>)}</Select>)}</Controls>
    <SvgCanvas viewBox="0 0 320 286" className={`${pen.canvas} max-w-96`} role="img" aria-label={`A 有 ${result.selectedA.length} 条，条件 B 有 ${result.selectedB.length} 条，交集 ${result.intersection.length} 条`}>
      {['D · 正类', 'D · 非正类', 'D · 非正类', 'E · 正类'].map((label, i) => <g key={i}>
        <rect x="10" y={12 + i * 59} width="300" height="49" rx="4" className={result.selectedB.includes(i) ? 'fill-info-soft stroke-info stroke-2' : 'fill-sunken stroke-rule'} />
        <rect x="20" y={23 + i * 59} width="28" height="27" rx="3" className={result.selectedA.includes(i) ? 'fill-accent' : 'fill-rule-strong'} />
        <text x="60" y={42 + i * 59}>ω{i + 1} · {label}</text>
        <text x="295" y={42 + i * 59} textAnchor="end" className={pen.textB}>{result.intersection.includes(i) ? 'A∩B' : result.selectedB.includes(i) ? 'B' : ''}</text>
      </g>)}
      <text x="10" y="271">实心方块：A · 蓝色框：条件 B</text>
    </SvgCanvas>
    <Readout>P(A)={result.selectedA.length}/4={fmt(result.pA)}<br />P(A∩B)={result.intersection.length}/4={fmt(result.joint)}<br />P(A|B)={result.conditional === null ? '未定义：P(B)=0' : `${result.intersection.length}/${result.selectedB.length}=${fmt(result.conditional)}`}<br />P(A)P(B)={fmt(result.pA * result.pB)} · {result.independent ? '满足独立性' : '不独立'} · {result.intersection.length === 0 ? '互斥' : '有交集'}</Readout>
  </LabFrame>
}
