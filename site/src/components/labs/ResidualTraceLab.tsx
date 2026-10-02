import { TraceTable as Trace } from './DataViews'

import { useState } from 'react'
import { decoderResidual } from '../../lib/principles-trace-model'
import { StepControls, Controls, LabFrame, Range, Readout, fmt } from './Lab'

const vector = (values: number[]) => `(${values.map(x => fmt(x, 4)).join(', ')})`

export default function ResidualTraceLab() {
  const [position, setPosition] = useState(0), [stage, setStage] = useState(0)
  const result = decoderResidual(position)
  return <LabFrame title="一个位置的两次残差与词表头" hint="P4 的二维手算参数；不是 P11 的现代块">
    <Controls><Range label="输入位置" min={0} max={2} value={position} onChange={setPosition} /><StepControls label="前向阶段" labels={["输入","注意力写入","残差相加","FFN写入","残差相加","词表头"]} value={stage} onChange={setStage} /></Controls>
    <Trace active={stage} label="输入、注意力写入、残差、FFN 写入、残差与词表 logits 的数值路径" rows={[
      ['原始残差流 X', vector(result.input)], ['注意力写入 branch', vector(result.branch)], ['第一次相加 H=X+branch', vector(result.first)], ['FFN 写入 GELU(1),0', vector(result.ffn)], ['第二次相加 Y=H+FFN', vector(result.second)], ['末层 Norm → 四类 logits', vector(result.logits)],
    ]} />
    <Readout>目标 {['A', 'B', 'EOS'][position]} · 正确类别概率 {fmt(result.probabilities[position + 1], 6)} · NLL {fmt(result.nll, 6)}。注意力写入为零的 A 位置仍保留原状态 (0,2)。</Readout>
  </LabFrame>
}
