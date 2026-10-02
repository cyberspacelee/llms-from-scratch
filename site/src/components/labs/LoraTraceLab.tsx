import { TraceTable as Trace } from './DataViews'

import { useState } from 'react'
import { loraTrajectory } from '../../lib/training-trace-model'
import { Toggle, Controls, LabFrame, Range, Readout, fmt } from './Lab'

const vector = (values: number[]) => `(${values.map(x => fmt(x, 4)).join(', ')})`

export default function LoraTraceLab() {
  const [step, setStep] = useState(0), [bothZero, setBothZero] = useState(false), [merged, setMerged] = useState(false), [target, setTarget] = useState(2)
  const result = loraTrajectory(2, bothZero)[step]
  const error = Math.max(...result.logits.map((v, i) => Math.abs(v - result.merged[i])))
  return <LabFrame title="零增量、首步梯度与合并" hint="正文 x=(1,2,3,4)，r=2，s=1，SGD η=.01">
    <Controls><Range label="已完成更新数" min={0} max={2} value={step} onChange={setStep} /><Range label="观察输出类别" min={0} max={5} value={target} onChange={setTarget} /><Toggle label="A、B 都初始化为零" checked={bothZero} onChange={value => setBothZero(value)} /><Toggle label="使用合并矩阵" checked={merged} onChange={value => setMerged(value)} /></Controls>
    <Trace label="冻结基座路径和低秩增量的数值" rows={merged ? [
      ['固定基座 + 固定 adapter → W′', 'W′ = W + BA'], ['合并输出 W′x（所选类）', fmt(result.merged[target], 6)], ['两路径输出差', fmt(error, 9)],
    ] : [
      ['冻结路径 Wx（所选类）', fmt(result.base[target], 6)], ['压缩表示 Ax', vector(result.compressed)], ['adapter 增量 B(Ax)（所选类）', fmt(result.delta[target], 6)], ['输出 Wx+B(Ax)', fmt(result.logits[target], 6)],
    ]} />
    <Readout>目标固定为下标 2 · CE={fmt(result.loss, 6)} · 当前 ∥∇A∥={fmt(result.gradA, 6)}，∥∇B∥={fmt(result.gradB, 6)}。{bothZero ? '两因子为零，所有步骤的数据梯度都为零。' : '初始 B=0，A 的数据梯度为零，B 先更新；第二次反向才让 A 获得梯度。'} 合并后禁止再加同一 adapter。</Readout>
  </LabFrame>
}
