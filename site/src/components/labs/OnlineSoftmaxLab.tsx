import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { onlineAttention } from '../../lib/systems-interactive-model'

export default function OnlineSoftmaxLab() {
  const [position, setPosition] = useState(5), [block, setBlock] = useState(3), [step, setStep] = useState(0)
  const r = onlineAttention(position, block, step), s = r.state
  return <LabFrame title="在线 softmax：共同基准与重缩放">
    <Controls><Range label="查询绝对位置" value={position} min={3} max={5} onChange={setPosition} /><Range label="键块大小" value={block} min={1} max={6} onChange={v => { setBlock(v); setStep(0) }} /><Range label="已读块数" value={step} min={0} max={Math.ceil(6 / block)} onChange={setStep} /></Controls>
    <SvgCanvas viewBox="0 0 280 170" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`已读 ${step} 块，m=${s.m}, l=${s.l}, u=${s.u}`}>
      {Array.from({ length: 6 }, (_, j) => <g key={j}><rect x={8 + j * 45} y="25" width="39" height="42" className={j > position ? 'fill-sunken stroke-rule [stroke-dasharray:3_3]' : j < step * block ? 'fill-accent/20 stroke-accent' : 'fill-paper stroke-rule'} /><text x={27 + j * 45} y="43" textAnchor="middle">k{j}</text><text x={27 + j * 45} y="59" textAnchor="middle">v={j + 1}</text></g>)}
      <text x="8" y="100">旧基准 {r.previous.m === -Infinity ? '−∞' : r.previous.m} → 新基准 {s.m === -Infinity ? '−∞' : s.m}</text><text x="8" y="125">旧 l、u × {fmt(s.factor, 6)}</text><text x="8" y="150">新块贡献 → 共同分母 → 输出</text>
    </SvgCanvas>
    <Readout>m（正文 a）={s.m === -Infinity ? '−∞' : s.m}；l={fmt(s.l, 6)}；u={fmt(s.u, 6)}<br />o=u/l：{r.output === null ? '未定义（尚无可见键）' : fmt(r.output, 6)}；整行参照 {fmt(r.dense, 6)}<br />当前块可见键：{s.end > s.start ? `${s.start}–${s.end - 1}` : '空块，不进入分母'}</Readout>
  </LabFrame>
}
