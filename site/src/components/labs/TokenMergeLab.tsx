import { useState } from 'react'
import { Controls, LabFrame, pen, Range, Readout } from './Lab'

const stages = [
  [['E7', 1], ['8C', 1], ['AB', 1], ['20', 1], ['73', 1], ['61', 1], ['74', 1]],
  [['E7', 1], ['8C', 1], ['AB', 1], ['20 73', 2], ['61', 1], ['74', 1]],
  [['E7', 1], ['8C', 1], ['AB', 1], ['20 73', 2], ['61 74', 2]],
] as const

export default function TokenMergeLab() {
  const [step, setStep] = useState(0)
  return <LabFrame title="猫 sat 的两次 BPE 合并">
    <Controls><Range label="合并次数" value={step} min={0} max={2} onChange={setStep} /></Controls>
    <svg viewBox="0 0 540 105" className={pen.canvas} role="img" aria-label={`猫 sat 合并 ${step} 次后的字节切分`}>
      {(() => {
        let offset = 25
        return stages[step].map(([label, byteCount], index) => {
          const x = offset
          const width = byteCount * 70
          offset += width
          return <g key={index}>
            <rect x={x} y="20" width={width - 6} height="40" rx="4" className="fill-accent-soft stroke-accent" />
            <text x={x + (width - 6) / 2} y="45" textAnchor="middle">{label}</text>
          </g>
        })
      })()}
      <text x="25" y="88">十六进制字节；方框表示一个 token</text>
    </svg>
    <Readout>{['七个基础字节', '合并 (32,115) → 256：空格+s', '合并 (97,116) → 257：at'][step]} · 词表大小 {256 + step}</Readout>
  </LabFrame>
}
