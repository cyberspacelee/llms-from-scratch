import { MatrixGrid, TokenSequence } from './DataViews'
import { TraceTable as Trace } from './DataViews'

import { useState } from 'react'
import { packing } from '../../lib/training-trace-model'
import { StepControls, Toggle, Controls, LabFrame, Range, Readout } from './Lab'


export default function PackingLab() {
  const [isolated, setIsolated] = useState(true), [row, setRow] = useState(3), [stage, setStage] = useState(0)
  const result = packing(isolated)
  return <LabFrame title="两篇文档怎样成为一批预测" hint="A=[1,2,5]，B=[3,4,5]，EOS=5">
    <Controls><StepControls label="构造阶段" labels={["文档拼接","下一目标移位","位置与loss mask"]} value={stage} onChange={setStage} /><Range label="查询 / 损失行" min={0} max={4} value={row} onChange={setRow} /><Toggle label="隔离文档" checked={isolated} onChange={value => setIsolated(value)} /></Controls>
    <Trace active={stage} label="文档拼接、标签移位和三类元数据" rows={[
      ['文档拼接（ID 5 不自动隔离）', 'A:1,2,5 │ B:3,4,5'], ['input → target（本行）', `${result.stream[row]} → ${result.stream[row + 1]} · 行 ${row}`], ['位置编号 / 直接损失', `p=${result.positions[row]} · loss mask=${Number(result.valid[row])}`],
    ]} />
    <TokenSequence label="文档拼接后的 token" tokens={result.stream.map((id, i) => ({ label: String(id), detail: i < 3 ? '文档 A' : '文档 B', state: i === row ? 'current' : 'past' }))} />
    <MatrixGrid label="因果读取集合（1允许，0屏蔽）" values={result.allowed.map(keys => keys.map(allowed => Number(allowed)))} rowLabels={[0,1,2,3,4].map(i => `${i}·${i < 3 ? 'A' : 'B'}`)} columnLabels={[0,1,2,3,4]} activeRow={row} allowed={result.allowed} />
    <Readout>{result.count} 个有效目标 · 行 {row} 可读键 [{result.allowed[row].flatMap((v, j) => v ? [j] : []).join(',')}]。{isolated ? '行 2 的 EOS→B 首项没有直接监督；B 的位置从 0 重新开始。' : '连续目标计入跨文档预测，B 读取 A 的历史。'} 两种选择定义不同目标。</Readout>
  </LabFrame>
}
