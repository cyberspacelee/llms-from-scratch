import { TraceTable as Trace } from './DataViews'

import { useState } from 'react'
import { modernGate } from '../../lib/principles-trace-model'
import { StepControls, Controls, LabFrame, Range, Readout, fmt } from './Lab'

const vector = (values: number[]) => `(${values.map(x => fmt(x, 4)).join(', ')})`

export default function SwiGluLab() {
  const [multiplier, setMultiplier] = useState(1), [stage, setStage] = useState(0)
  const result = modernGate(multiplier)
  return <LabFrame title="同一个位置的 gate 与 up 怎样相乘" hint="沿用 A 位置 H=(2p,2)；up/down 仍为 I₂">
    <Controls><Range label="gate 权重倍数" min={-2} max={2} step={0.25} value={multiplier} onChange={setMultiplier} /><StepControls labels={["残差输入","归一化","gate激活","逐元素乘","残差相加"]} value={stage} onChange={setStage} /></Controls>
    <Trace active={stage} label="SwiGLU 的两路投影、激活与逐元素乘法" rows={[
      ['第二次 RMSNorm 的输入 H', vector(result.h)], ['两路输入 r（up=r）', vector(result.r)], ['SiLU(gate) = gate·sigmoid(gate)', vector(result.silu)], ['逐元素乘 up，再经 down', vector(result.branch)], ['相加写回残差流', vector(result.output)],
    ]} />
    <Readout>gate={vector(result.gate)}。倍数 1 恢复正文 Y≈(1.644970,2.752987)；倍数 0 令 FFN 写入为零；负 gate 可以产生负写入，不是仅用 sigmoid(gate)×up。</Readout>
  </LabFrame>
}
