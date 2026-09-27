import { useState } from 'react'
import { Controls, LabFrame, pen, Range, Readout } from './Lab'

const stages = [
  [['a', 'b', 'a'], ['a', 'b', 'b'], ['b', 'a', 'b']],
  [['ab', 'a'], ['ab', 'b'], ['b', 'ab']],
  [['aba'], ['ab', 'b'], ['b', 'ab']],
]

export default function TokenMergeLab() {
  const [step, setStep] = useState(0)
  return <LabFrame title="BPE 合并与覆盖字节">
    <Controls><Range label="合并次数" value={step} min={0} max={2} onChange={setStep} /></Controls>
    <svg viewBox="0 0 480 180" className={pen.canvas} role="img" aria-label={`合并 ${step} 次后的三条文档切分`}>
      {stages[step].map((tokens, row) => {
        let offset = 100
        return <g key={row}>
          <text x="12" y={38 + row * 55}>{['aba × 4', 'abb × 2', 'bab × 1'][row]}</text>
          {tokens.map((token, col) => {
            const x = offset
            const width = token.length * 72
            offset += width
            return <g key={col}>
              <rect x={x} y={15 + row * 55} width={width - 6} height="36" rx="4" className="fill-accent-soft stroke-accent" />
              <text x={x + (width - 6) / 2} y={38 + row * 55} textAnchor="middle">{token}</text>
            </g>
          })}
        </g>
      })}
    </svg>
    <Readout>{['基础字节：a=97，b=98', '合并 (97,98) → 256：ab', '合并 (256,97) → 257：aba'][step]} · 词表大小 {256 + step}</Readout>
  </LabFrame>
}
