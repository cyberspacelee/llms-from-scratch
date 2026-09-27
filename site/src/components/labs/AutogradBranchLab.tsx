import { useState } from 'react'
import { branchGradient } from '../../lib/framework-tensor-model'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'
import Formula from './Formula'

export default function AutogradBranchLab() {
  const [x, setX] = useState(2)
  const [seed, setSeed] = useState(1)
  const [grad, setGrad] = useState(0)
  const state = branchGradient(x, seed, grad)
  return <LabFrame title="两条路径的梯度怎样相加" hint="每次反向都重新构建前向图">
    <Controls><Range label="叶子 x" value={x} min={-3} max={3} onChange={setX} /><Range label="反向种子 g" value={seed} min={-2} max={2} onChange={setSeed} /></Controls>
    <svg viewBox="0 0 320 320" className={`${pen.canvas} max-w-96`} role="img" aria-label={`x平方加三x，两路梯度 ${state.squareGradient} 加 ${state.linearGradient} 等于 ${state.gradient}`}>
      <path d="M160 55L77 115M160 55L242 115M77 163L160 222M242 163L160 222" className={pen.axis} />
      <path d="M145 219L62 159M175 219L257 159M62 110L145 50M257 110L175 50" className={pen.guide} />
      <rect x="105" y="18" width="110" height="38" rx="4" className="fill-accent-soft stroke-accent" /><text x="160" y="43" textAnchor="middle">叶子 x = {x}</text>
      <rect x="12" y="115" width="130" height="48" rx="4" className="fill-sunken stroke-rule" /><text x="77" y="145" textAnchor="middle">平方 = {state.square}</text>
      <rect x="177" y="115" width="130" height="48" rx="4" className="fill-sunken stroke-rule" /><text x="242" y="145" textAnchor="middle">3x = {state.linear}</text>
      <rect x="100" y="221" width="120" height="38" rx="4" className="fill-accent2-soft stroke-accent2" /><text x="160" y="247" textAnchor="middle">y = {state.output}</text>
      <text x="12" y="89" className={pen.textB}>平方路：{state.squareGradient}</text><text x="202" y="89" className={pen.textB}>线性路：{state.linearGradient}</text>
      <text x="12" y="293">实线：前向依赖 · 蓝色虚线：反向传播</text>
      <text x="12" y="315">图内相加；跨次累积到 .grad。</text>
    </svg>
    <div className="mt-3 flex flex-wrap gap-2">
      <button type="button" onClick={() => setGrad(state.accumulated)} className="h-9 rounded border border-accent bg-accent px-3 text-sm text-paper">重新前向并 backward</button>
      <button type="button" onClick={() => setGrad(0)} className="h-9 rounded border border-rule bg-paper px-3 text-sm">清空叶子梯度</button>
    </div>
    <Readout><Formula>{String.raw`J^\top g=(2x+3)g=${state.gradient}`}</Formula> · 本次种子 g={seed} · 当前 x.grad={grad} · 下一次 backward 后={state.accumulated}</Readout>
  </LabFrame>
}
