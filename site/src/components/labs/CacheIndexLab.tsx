import { useState } from 'react'
import { Controls, LabFrame, Range, Readout } from './Lab'

/** Which keys the query at decode position t reads, and at what relative offset. */
export default function CacheIndexLab() {
  const [t, setT] = useState(4)
  return (
    <LabFrame title="当前查询读到哪些键" hint="单条序列，一个注意力头">
      <Controls>
        <Range label="解码位置 t" value={t} min={0} max={8} onChange={setT} />
      </Controls>
      <ol className="mt-4 grid list-none grid-cols-9 gap-1 p-0" aria-label="键的位置">
        {Array.from({ length: 9 }, (_, j) => {
          const state = j < t ? 'past' : j === t ? 'current' : 'future'
          return (
            <li
              key={j}
              className={[
                'rounded-sm py-2 text-center font-mono text-xs leading-normal',
                state === 'past' && 'bg-accent-soft text-accent-strong',
                state === 'current' && 'bg-accent2-soft font-semibold text-accent2',
                state === 'future' && 'bg-sunken text-muted',
              ].filter(Boolean).join(' ')}
            >
              j={j}
              <br />
              {j <= t ? `Δ=${j - t}` : '屏蔽'}
            </li>
          )
        })}
      </ol>
      <p className="mt-3 mb-0 text-xs text-muted">绿色是缓存里的历史键，橙色是这一步新写入的键，灰色是还不存在的未来位置。</p>
      <Readout>
        Q: [B, H, 1, D] · K/V: [B, H, {t + 1}, D] · 分数: [B, H, 1, {t + 1}]
      </Readout>
    </LabFrame>
  )
}
