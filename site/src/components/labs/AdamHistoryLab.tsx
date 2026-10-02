import { TraceTable as Trace } from './DataViews'

import { useState } from 'react'
import { adamTrajectory } from '../../lib/training-trace-model'
import { Toggle, Controls, LabFrame, Range, Readout, fmt } from './Lab'

const vector = (values: number[]) => `(${values.map(x => fmt(x, 4)).join(', ')})`

export default function AdamHistoryLab() {
  const [step, setStep] = useState(1), [reset, setReset] = useState(false), [coordinate, setCoordinate] = useState(0)
  const states = adamTrajectory(6, reset ? 1 : -1), state = states[step], previous = states[Math.max(0, step - 1)], i = coordinate
  const reference = adamTrajectory(6)[step]
  return <LabFrame title="同一参数的 AdamW 历史" hint="w₀=(1,−2)，ρ₁=.9，ρ₂=.99，η=.1，λ=.2">
    <Controls><Range label="更新步" min={0} max={6} value={step} onChange={setStep} /><Range label="参数坐标" min={0} max={1} value={coordinate} onChange={setCoordinate} /><Toggle label="第 1 步后丢失优化器状态" checked={reset} onChange={value => setReset(value)} /></Controls>
    <Trace label="数据梯度、矩状态、偏差校正与衰减更新" rows={[
      ['当前数据梯度 g（不含衰减）', fmt(state.g[i], 6)], ['一阶 m / 二阶原始矩 v', `${fmt(state.m[i], 6)} / ${fmt(state.v[i], 6)}`], ['偏差校正 m̂ / v̂', `${fmt(state.mh[i], 6)} / ${fmt(state.vh[i], 6)}`], ['参数：旧值 → 更新后值', `${fmt(previous.w[i], 6)} → ${fmt(state.w[i], 6)}`],
    ]} active={step ? 3 : 0} />
    <Readout>优化器步计数 k={state.k} · w={vector(state.w)} · 与完整历史最大差 {fmt(Math.max(...state.w.map((v, j) => Math.abs(v - reference.w[j]))), 6)}。第 0 步矩状态未形成；后续梯度由新 w 与同一四个目标重新计算。</Readout>
  </LabFrame>
}
