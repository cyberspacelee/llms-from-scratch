import { useState } from 'react'
import { lowRank } from '../../lib/chapter-labs-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

export default function LowRankLab() {
  const [rank, setRank] = useState(1)
  const result = lowRank(rank)
  return <LabFrame title="逐项保留奇异方向" hint="固定 W；奇异值 5、2、1">
    <Controls><Range label="保留秩 r" min={0} max={3} value={rank} onChange={setRank} /></Controls>
    <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
      {[{ title: '原矩阵 W', rows: result.full }, { title: '截断矩阵 Wᵣ', rows: result.approximation }, { title: '残差 W − Wᵣ', rows: result.residual }].map(({ title, rows }) => <svg key={title} viewBox="0 0 190 184" className={`${pen.canvas} mt-0! max-w-52!`} role="img" aria-label={title}>
        <text x="95" y="18" textAnchor="middle">{title}</text>
        {rows.map((row, i) => row.map((x, j) => <g key={`${i}-${j}`}>
          <rect x={20 + j * 50} y={30 + i * 50} width="48" height="48" rx="2" className={x < 0 ? 'fill-accent2' : 'fill-accent'} opacity={0.08 + Math.abs(x) / 5 * 0.45} />
          <text x={44 + j * 50} y={59 + i * 50} textAnchor="middle" className={pen.mono}>{fmt(x, 1)}</text>
        </g>))}
      </svg>)}
    </div>
    <Readout>‖W − Wᵣ‖F = {fmt(result.error)} · 保留平方能量 {(100 * result.energy).toFixed(1)}% · r = 3 时才完整恢复 W</Readout>
  </LabFrame>
}
