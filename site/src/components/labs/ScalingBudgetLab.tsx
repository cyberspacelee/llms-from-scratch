import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { budgetLoss } from '../../lib/chapter-labs-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

export default function ScalingBudgetLab() {
  const [budget, setBudget] = useState(100)
  const [size, setSize] = useState(10)
  const [alpha, setAlpha] = useState(0.6)
  const [beta, setBeta] = useState(0.4)
  const result = budgetLoss(size, budget, alpha, beta)
  const optimum = Math.min(100, Math.max(1, result.optimalSize))
  const optimumLoss = budgetLoss(optimum, budget, alpha, beta)
  const x = (n: number) => Math.round((40 + Math.log10(n) / 2 * 300) * 100) / 100
  const y = (loss: number) => Math.round((250 - loss / 10 * 220) * 100) / 100
  const samples = Array.from({ length: 81 }, (_, i) => {
    const n = 10 ** (i / 40)
    return { n, ...budgetLoss(n, budget, alpha, beta) }
  })
  return <LabFrame title="同一预算中，容量与数据怎样此消彼长" hint="教学真值 E=0.5，A=4，B=2；横轴为 log₁₀N">
    <Controls>
      <Range label="乘积预算 K = C/6 = ND" min={50} max={400} step={10} value={budget} onChange={setBudget} />
      <Range label="模型量 N（教学单位）" min={1} max={100} value={size} onChange={setSize} />
      <Range label="模型指数 α" min={0.2} max={1} step={0.1} value={alpha} onChange={setAlpha} />
      <Range label="数据指数 β" min={0.2} max={1} step={0.1} value={beta} onChange={setBeta} />
    </Controls>
    <SvgCanvas viewBox="0 0 380 295" className={`${pen.canvas} max-w-110!`} role="img" aria-label={`预算 ${budget}，当前 N ${size}，损失 ${fmt(result.loss)}`}>
      <path d="M40 20V250H350" className={pen.axis} />
      {[0, 2, 4, 6, 8, 10].map(n => <g key={n}><line x1="40" x2="350" y1={y(n)} y2={y(n)} className={pen.grid} /><text x="32" y={y(n) + 4} textAnchor="end">{n}</text></g>)}
      {[1, 10, 100].map(n => <text key={n} x={x(n)} y="274" textAnchor="middle">{n}</text>)}
      <polyline points={samples.map(s => `${x(s.n)},${y(s.loss)}`).join(' ')} className={pen.a} />
      <polyline points={samples.map(s => `${x(s.n)},${y(s.modelTerm)}`).join(' ')} className={pen.c} />
      <polyline points={samples.map(s => `${x(s.n)},${y(s.dataTerm)}`).join(' ')} className={pen.b} />
      <line x1={x(size)} x2={x(size)} y1="25" y2="250" className={pen.guide} /><circle cx={x(size)} cy={y(result.loss)} r="5" className="fill-accent" />
      <circle cx={x(optimum)} cy={y(optimumLoss.loss)} r="5" className="fill-accent2" />
      <text x="44" y="16" className={pen.muted}>损失 / 分项</text><text x="350" y="292" textAnchor="end">模型量 N</text>
    </SvgCanvas>
    <p className="mt-2 text-sm"><span className="text-accent">绿色：总损失</span> · <span className="text-info">蓝色：模型项</span> · <span className="text-accent2">橙色虚线：数据项；圆点：区间最优</span></p>
    <Readout>N = {size} · D = K/N = {fmt(result.dataSize)} · L = 0.5 + {fmt(result.modelTerm)} + {fmt(result.dataTerm)} = {fmt(result.loss)}<br />无约束 N* = {fmt(result.optimalSize)} · 区间 [1,100] 最优 N = {fmt(optimum)} · 这是假设曲线，不是实测模型结果。</Readout>
  </LabFrame>
}
