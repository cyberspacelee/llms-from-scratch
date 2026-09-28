import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function DeltaStateLab() {
  const [decay, setDecay] = useState(0.5)
  const [update, setUpdate] = useState(1)
  const old = [2 * decay, decay]
  const current = [old[0] * (1 - update), old[1] * (1 - update) + 2 * update]
  return <LabFrame title="位置 1：同一键的遗忘与改写">
    <Controls>
      <Range label="保持系数 a" value={decay} min={0} max={1} step={0.05} onChange={setDecay} format={v => v.toFixed(2)} />
      <Range label="更新强度 η" value={update} min={0} max={1} step={0.05} onChange={setUpdate} format={v => v.toFixed(2)} />
    </Controls>
    <svg viewBox="0 0 440 195" className={pen.canvas} role="img" aria-label={'旧预测 ' + old.map(v => v.toFixed(2)).join('、') + '，新关联 ' + current.map(v => v.toFixed(2)).join('、')}>
      <text x="220" y="25" textAnchor="middle">当前键 k=(1,0) 的关联值</text>
      {old.map((value, i) => <g key={'old-' + i}>
        <rect x="82" y={42 + i * 52} width="116" height="43" rx="3" className="fill-accent/15 stroke-accent" />
        <text x="140" y={69 + i * 52} textAnchor="middle">{value.toFixed(2)}</text>
      </g>)}
      {current.map((value, i) => <g key={'new-' + i}>
        <rect x="242" y={42 + i * 52} width="116" height="43" rx="3" className="fill-accent2/15 stroke-accent2" />
        <text x="300" y={69 + i * 52} textAnchor="middle">{value.toFixed(2)}</text>
      </g>)}
      <text x="140" y="169" textAnchor="middle">衰减后旧预测</text>
      <text x="300" y="169" textAnchor="middle">更新后新关联</text>
    </svg>
    <Readout>位置 0 的旧关联是 (2,1)，位置 1 目标是 (0,2)；更新后为 ({current.map(v => v.toFixed(2)).join(', ')})。</Readout>
  </LabFrame>
}
