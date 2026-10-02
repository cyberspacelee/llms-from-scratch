import { TokenSequence } from './DataViews'
import { TraceTable as Trace } from './DataViews'

import { useState } from 'react'

import { Toggle, Controls, LabFrame, Range, Readout, fmt } from './Lab'

const vector = (values: number[]) => `(${values.map(x => fmt(x, 4)).join(', ')})`

export default function TokenLookupLab() {
  const ids = [231, 140, 171, 256, 257], pieces = ['E7', '8C', 'AB', '20 73', '61 74']
  const embedding = [[0.1, 0.3], [-0.2, 0.4], [0.5, 0.1], [0.2, -0.1], [-0.3, 0.2]]
  const [position, setPosition] = useState(3), [special, setSpecial] = useState(false)
  return <LabFrame title="从字节片段到 embedding 行" hint="猫 sat · 两次合并后的固定词表">
    <Controls><Range label="正文 token 位置" min={0} max={4} value={position} onChange={setPosition} /><Toggle label="附加 BOS / EOS" checked={special} onChange={value => setSpecial(value)} /></Controls>
    <TokenSequence label="正文 token 与词表行地址" tokens={ids.map((id, i) => ({ label: String(id), detail: pieces[i], state: i === position ? 'current' : 'past' }))} />
    <Trace label={`第 ${position} 个 token 选 embedding 第 ${ids[position]} 行`} rows={[
      ['UTF-8 片段（十六进制）', pieces[position]], ['词表 ID → 参数行', `${ids[position]} → E[${ids[position]}, :]`], ['查表输出', vector(embedding[position])], ['序列形状', special ? '(7, 2) · BOS=258, EOS=259' : '(5, 2) · 5 个普通 token'],
    ]} active={2} />
    <Readout>ID 是行地址，不乘入向量。两份输入的 ID 256 共用参数行；上游 (1,2) 与 (3,−1) 汇合为 (4,1)。其余四行数值仅为本实验补充的查表例。</Readout>
  </LabFrame>
}
