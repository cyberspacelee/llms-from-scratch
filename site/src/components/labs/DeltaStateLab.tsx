import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function DeltaStateLab() {
  const [decay, setDecay] = useState(0.5)
  const [update, setUpdate] = useState(1)
  const state = [[decay, 0], [decay * 0.5, 2 * update]]
  return <LabFrame title="两步 gated delta 状态">
    <Controls>
      <Range label="第二步保持系数 a" value={decay} min={0} max={1} step={0.05} onChange={setDecay} format={v => v.toFixed(2)} />
      <Range label="第二步更新强度 η" value={update} min={0} max={1} step={0.05} onChange={setUpdate} format={v => v.toFixed(2)} />
    </Controls>
    <svg viewBox="0 0 440 200" className={pen.canvas} role="img" aria-label={`状态矩阵第一列 ${decay}、${decay * .5}，第二列 0、${2 * update}`}>
      <text x="220" y="24" textAnchor="middle">状态 S（行：值维度，列：键方向）</text>
      {state.map((row, i) => row.map((value, j) => <g key={`${i}-${j}`}>
        <rect x={116 + j * 104} y={40 + i * 56} width="94" height="46" rx="3" className={j ? 'fill-accent2/15 stroke-accent2' : 'fill-accent/15 stroke-accent'} />
        <text x={163 + j * 104} y={69 + i * 56} textAnchor="middle">{value.toFixed(3)}</text>
      </g>))}
      <text x="163" y="170" textAnchor="middle">旧键 (1,0)</text>
      <text x="267" y="170" textAnchor="middle">新键 (0,1)</text>
    </svg>
    <Readout>第一步 S=[[1,0],[0.5,0]] · 第二步 v=(0,2), k=q=(0,1) · 当前输出 (0,{(2 * update).toFixed(3)}) · 旧方向按 a 衰减</Readout>
  </LabFrame>
}
