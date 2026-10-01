import { useState } from 'react'
import { xorTraining } from '../../lib/math-linear-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

export default function MathTrainingLab() {
  const [eta, setEta] = useState(0.3), [steps, setSteps] = useState(0), [parameter, setParameter] = useState<'w1' | 'b1' | 'w2' | 'b2'>('w2')
  const run = xorTraining(eta, steps)
  const flatten = (value: number[] | number[][]) => value.flat()
  const old = flatten(run.parameters[parameter]), gradient = flatten(run.gradients[parameter]), next = flatten(run.next[parameter])
  return <LabFrame title="同一 XOR 网络：先求梯度，再一起更新">
    <Controls><Range label="已完成更新次数" value={steps} min={0} max={20} onChange={setSteps} /><Range label="学习率 η" value={eta} min={0.05} max={0.8} step={0.05} onChange={setEta} format={v => fmt(v, 2)} /><label className="text-sm">参数组<select value={parameter} onChange={e => setParameter(e.target.value as typeof parameter)} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2">{['w1', 'b1', 'w2', 'b2'].map(name => <option key={name} value={name}>{name}</option>)}</select></label></Controls>
    <svg viewBox="0 0 320 290" className={`${pen.canvas} max-w-96`} role="img" aria-label={`${steps} 次更新后平均损失 ${run.loss}`}>
      <text x="10" y="20">类别 1 概率 · 灰线为分类阈值 0.5</text>
      <line x1="191" x2="191" y1="32" y2="256" className={pen.guide} />
      {run.rows.map((row, i) => <g key={i}>
        <text x="10" y={60 + i * 58}>({row.x.join(',')}) → y={i === 1 || i === 2 ? 1 : 0}</text>
        <rect x="126" y={38 + i * 58} width={130 * row.probs[1]} height="28" rx="3" className="fill-accent" />
        <text x="310" y={59 + i * 58} textAnchor="end" className={pen.mono}>{fmt(row.probs[1])}</text>
        <text x="126" y={83 + i * 58} className={pen.muted}>ℓ={fmt(row.loss)} · h=({row.hidden.map(v => fmt(v, 2)).join(',')})</text>
      </g>)}
    </svg>
    <Readout>L 初始={fmt(run.initialLoss, 6)} → 当前={fmt(run.loss, 6)}<br />{parameter} · 下面各项均为当前同一次前向计算：<br />{old.map((v, i) => <span key={i}>{i}: {fmt(v, 5)} − {fmt(eta, 2)}×{fmt(gradient[i], 5)} = {fmt(next[i], 5)}<br /></span>)}下一次更新同时应用四组梯度；ReLU 在零点采用局部梯度 0。</Readout>
  </LabFrame>
}
