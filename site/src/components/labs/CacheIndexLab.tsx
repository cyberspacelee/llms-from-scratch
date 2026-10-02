import { TokenSequence } from './DataViews'
import { useState } from 'react'
import Formula from './Formula'
import { Controls, LabFrame, Range, Readout } from './Lab'

/** Which keys the query at decode position t reads, and at what relative offset. */
export default function CacheIndexLab() {
  const [t, setT] = useState(3)
  return (
    <LabFrame title="当前查询读到哪些键" hint="单条序列，一个注意力头">
      <Controls>
        <Range label="解码位置 t" value={t} min={0} max={8} onChange={setT} />
      </Controls>
      <TokenSequence label="键的位置与相对位移" tokens={Array.from({ length: 9 }, (_, j) => ({ label: `j=${j}`, detail: j <= t ? `Δ=${j-t}` : '屏蔽', state: j < t ? 'past' : j === t ? 'current' : 'future' }))} />
      <p className="mt-3 mb-0 text-xs text-muted">绿色是缓存里的历史键，橙色是这一步新写入的键，灰色是还不存在的未来位置。</p>
      <Readout>
        Q: <Formula>{'(B, H_q, 1, D_h)'}</Formula> · K/V: <Formula>{`(B, H_{kv}, ${t + 1}, D_h)`}</Formula> · 分数: <Formula>{`(B, H_q, 1, ${t + 1})`}</Formula>
      </Readout>
    </LabFrame>
  )
}
