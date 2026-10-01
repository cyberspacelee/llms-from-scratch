import { useState } from 'react'
import { softmaxLoss } from '../../lib/math-labs-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

export default function MathPredictionLab() {
  const [logits, setLogits] = useState([0, Math.log(2), Math.log(3)]), [shift, setShift] = useState(0), [target, setTarget] = useState(2)
  const result = softmaxLoss(logits.map(z => z + shift), 1, target), observations = [0, 2, 2]
  const rows = observations.map(y => softmaxLoss(logits, 1, y)), nll = rows.reduce((s, row) => s + row.loss, 0), likelihood = rows.reduce((s, row, i) => s * row.probs[observations[i]], 1)
  return <LabFrame title="从分数到目标概率与 NLL">
    <Controls>{logits.map((value, c) => <Range key={c} label={`类别 ${c + 1} 原始分数`} value={value} min={-3} max={3} step={0.05} onChange={v => setLogits(logits.map((z, i) => c === i ? v : z))} format={v => fmt(v, 3)} />)}<Range label="共同加数" value={shift} min={-1000} max={1000} step={100} onChange={setShift} /><label className="text-sm">单条目标<select value={target} onChange={e => setTarget(Number(e.target.value))} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2">{[0, 1, 2].map(c => <option key={c} value={c}>{c + 1}</option>)}</select></label></Controls>
    <svg viewBox="0 0 320 228" className={`${pen.canvas} max-w-96`} role="img" aria-label={`正确类别概率 ${result.probs[target]}，NLL ${result.loss}`}>
      <text x="10" y="20">softmax 的类别概率</text><line x1="10" x2="310" y1="182" y2="182" className={pen.axis} />
      {result.probs.map((p, c) => <g key={c}><rect x={27 + c * 99} y={182 - p * 132} width="65" height={p * 132} rx="3" className={c === target ? 'fill-accent2' : 'fill-accent'} /><text x={59 + c * 99} y={172 - p * 132} textAnchor="middle" className={pen.mono}>{fmt(p)}</text><text x={59 + c * 99} y="207" textAnchor="middle">类别 {c + 1}</text></g>)}
    </svg>
    <Readout>单条 NLL=−log {fmt(result.probs[target], 6)}={fmt(result.loss, 6)}<br />独立观测批 [1,3,3] · 似然={fmt(likelihood, 6)} · NLL 和={fmt(nll, 6)} · 平均={fmt(nll / 3, 6)}<br />−log(连乘)={fmt(-Math.log(likelihood), 6)} · 每行同时加 {shift}，概率与损失不变。</Readout>
  </LabFrame>
}
