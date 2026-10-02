import { TraceTable as Trace } from './DataViews'

import { useState } from 'react'
import { ddpLedger } from '../../lib/training-trace-model'
import { StepControls, Toggle, Controls, LabFrame, Readout, fmt } from './Lab'


export default function DdpReplayLab() {
  const [step, setStep] = useState(0), [wrong, setWrong] = useState(false)
  const result = ddpLedger(wrong, Math.min(step, 2))
  const names = ['统计 14 个有效目标', '微批 1：暂不同步', '微批 2：累积完毕', 'DDP 平均所有 rank', 'SGD：两侧同一次更新']
  return <LabFrame title="四个微批，只有一次全局更新" hint="rank 0：3+2 个 y=1；rank 1：5+4 个 y=3">
    <Controls><StepControls labels={names} value={step} onChange={setStep} /><Toggle label="错误：各 rank 先取局部均值" checked={wrong} onChange={value => setWrong(value)} /></Controls>
    <Trace active={Math.min(step, 3)} label={names[step]} rows={[
      ['有效数计数归约', 'rank0:5 + rank1:9 = N:14'], ['局部梯度和（当前已处理微批）', `${fmt(result.localSums[0])} / ${fmt(result.localSums[1])}`], [wrong ? '错误局部分母 5 / 9' : '本地 sum × R/N', `${fmt(result.local[0], 6)} / ${fmt(result.local[1], 6)}`], ['同步平均 → SGD 参数', step >= 3 ? `${fmt(result.gradient, 6)} → ${step === 4 ? fmt(result.theta, 6) : '尚未 step()'}` : '等待所有微批，不提前更新'],
    ]} />
    <Readout>阶段：{names[step]}。单进程参考 g={fmt(result.reference, 6)}，θ₁={fmt(-0.1 * result.reference, 6)}。{step >= 3 ? `当前偏差 ${fmt(result.gradient - result.reference, 6)}` : '未完成累积的梯度不能与完整更新比较。'} 参数同步不能证明分母正确。</Readout>
  </LabFrame>
}
