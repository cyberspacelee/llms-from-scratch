import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { softmaxLoss } from '../../lib/math-labs-model'
import { Select, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

export default function SoftmaxLossLab() {
  const [logits, setLogits] = useState([0, -1]), [temperature, setTemperature] = useState(1), [target, setTarget] = useState(1)
  const result = softmaxLoss(logits, temperature, target), bound = Math.max(1, ...result.gradient.map(Math.abs))
  return <LabFrame title="本章两类 logits 的交叉熵与梯度" hint="默认 τ=1 对应 XOR 样本 (0,1)；τ≠1 是额外温度变式">
    <Controls>{logits.map((z, c) => <Range key={c} label={`原始分数 z${c}`} value={z} min={-4} max={4} step={0.1} format={v => fmt(v, 1)} onChange={v => setLogits(logits.map((old, i) => i === c ? v : old))} />)}<Range label="温度 τ" value={temperature} min={0.25} max={3} step={0.25} onChange={setTemperature} format={v => fmt(v, 2)} /><Select label="正确类别" value={target} onChange={e => setTarget(Number(e.target.value))} >{[0, 1].map(c => <option key={c} value={c}>{c}</option>)}</Select></Controls>
    <div className="mt-4 grid gap-4 sm:grid-cols-2">
      <SvgCanvas viewBox="0 0 320 245" className={`${pen.canvas} mt-0! max-w-96`} role="img" aria-label={`两类概率 ${result.probs.join(',')}`}>
        <text x="10" y="20">p=softmax(z/τ)</text><line x1="20" x2="300" y1="196" y2="196" className={pen.axis} />
        {result.probs.map((p, c) => <g key={c}><rect x={55 + c * 140} y={196 - p * 145} width="70" height={p * 145} className={c === target ? 'fill-accent2' : 'fill-accent'} /><text x={90 + c * 140} y={185 - p * 145} textAnchor="middle" className={pen.mono}>{fmt(p)}</text><text x={90 + c * 140} y="224" textAnchor="middle">类别 {c}</text></g>)}
      </SvgCanvas>
      <SvgCanvas viewBox="0 0 320 245" className={`${pen.canvas} mt-0! max-w-96`} role="img" aria-label={`梯度 ${result.gradient.join(',')}，当前坐标界限 ${bound}`}>
        <text x="10" y="20">∂ℓ/∂z=(p−y)/τ</text><text x="10" y="44" className={pen.muted}>纵轴 ±{fmt(bound)}</text><line x1="20" x2="300" y1="132" y2="132" className={pen.axis} />
        {result.gradient.map((g, c) => <g key={c}><rect x={55 + c * 140} y={g > 0 ? 132 - g / bound * 60 : 132} width="70" height={Math.abs(g) / bound * 60} className={c === target ? 'fill-accent2' : 'fill-info'} /><text x={90 + c * 140} y={g > 0 ? 123 - g / bound * 60 : 150 + Math.abs(g) / bound * 60} textAnchor="middle" className={pen.mono}>{fmt(g)}</text><text x={90 + c * 140} y="238" textAnchor="middle">z{c}</text></g>)}
      </SvgCanvas>
    </div>
    <Readout>ℓ=−log p{target}={fmt(result.loss, 6)} · 梯度之和={fmt(result.gradient.reduce((a, b) => a + b, 0), 6)}<br />图中是单样本梯度；四行平均的输出梯度还须除以 4。τ 改变时导数含 1/τ，坐标范围随实际梯度变化。</Readout>
  </LabFrame>
}
