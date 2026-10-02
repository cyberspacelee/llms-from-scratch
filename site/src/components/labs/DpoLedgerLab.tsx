import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { dpoLedger } from '../../lib/advanced-interactive-model'

export default function DpoLedgerLab() {
  const [lambda, setLambda] = useState(0.2), [eos, setEos] = useState(true), [selected, setSelected] = useState(0)
  const r = dpoLedger(lambda, eos)
  return <LabFrame title="逐 token 回复账目与固定参考差">
    <Controls><Range label="λ DPO" value={lambda} min={0.05} max={2} step={0.05} onChange={setLambda} /><Range label="所选预测行" value={selected} min={0} max={3} onChange={setSelected} format={v => ['提示内 ?', '首回复', 'EOS', 'padding'][v]} /><Toggle label="EOS 属于回复" checked={eos} onChange={value => setEos(value)} /></Controls>
    <SvgCanvas viewBox="0 0 280 190" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`DPO margin ${fmt(r.margin)}，损失 ${fmt(r.loss)}`}>
      {['策略 chosen', '策略 rejected', '参考 chosen', '参考 rejected'].map((name, i) => <g key={name}><text x="8" y={22 + i * 43}>{name}</text><line x1="8" x2={8 + -r.logs[i] * 120} y1={32 + i * 43} y2={32 + i * 43} className={i < 2 ? pen.a : pen.b} /><text x="205" y={32 + i * 43} className={pen.mono}>{fmt(r.logs[i])}</text></g>)}
    </SvgCanvas><Readout>预测行 {selected}：{selected === 1 ? '目标 4 或 5，计入回复' : selected === 2 ? (eos ? '目标 EOS，计入回复' : 'EOS 排除，已改变目标 span') : '不进入回复 log-prob 和；仍可能通过后续位置影响梯度'}<br />d=(logπ+−logπ−)−(logref+−logref−)={fmt(r.difference, 6)}<br />z={fmt(r.margin, 6)}；loss={fmt(r.loss, 6)}；chosen log-prob 偏导={fmt(r.gradient, 6)}；rejected 偏导={fmt(-r.gradient, 6)}</Readout>
  </LabFrame>
}
