import { useState } from 'react'
import Formula from './Formula'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function TileReuseLab() {
  const [tile, setTile] = useState(4)
  const [step, setStep] = useState(0)
  const steps = Math.ceil(7 / tile)
  const phase = Math.min(step, steps - 1)
  const start = phase * tile
  const matrices = [
    { name: 'A · 5 × 7', rows: 5, cols: 7, x: 26, y: 36, active: (r: number, c: number) => r < tile && c >= start && c < start + tile },
    { name: 'B · 7 × 6', rows: 7, cols: 6, x: 26, y: 236, active: (r: number, c: number) => r >= start && r < start + tile && c < tile },
    { name: 'C · 5 × 6', rows: 5, cols: 6, x: 26, y: 486, active: (r: number, c: number) => r < tile && c < tile },
  ]
  const naiveReads = 2 * 5 * 7 * 6
  const tiledReads = Math.ceil(6 / tile) * 5 * 7 + Math.ceil(5 / tile) * 7 * 6
  return <LabFrame title="一个输出 tile 怎样累计 K 维贡献">
    <Controls>
      <Range label="tile 边长" value={tile} min={1} max={4} onChange={value => { setTile(value); setStep(0) }} />
      <Range label="K 维分块编号" value={phase} min={0} max={steps - 1} onChange={setStep} />
    </Controls>
    <svg viewBox="0 0 320 686" className={`${pen.canvas} max-w-96`} role="img" aria-label={`边长 ${tile}，第一输出 tile 正累计 K 索引 ${start} 至 ${Math.min(6, start + tile - 1)}`}>
      {matrices.map((matrix, id) => <g key={matrix.name}>
        <text x={matrix.x} y={matrix.y - 14}>{matrix.name}</text>
        {Array.from({ length: matrix.rows }, (_, row) => Array.from({ length: matrix.cols }, (_, col) =>
          <g key={`${row}-${col}`}>
            <rect x={matrix.x + col * 36} y={matrix.y + row * 28} width="34" height="26"
              className={matrix.active(row, col) ? id === 2 ? 'fill-info-soft stroke-info' : 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} />
            <text x={matrix.x + col * 36 + 17} y={matrix.y + row * 28 + 18} textAnchor="middle" className={pen.mono}>{row},{col}</text>
          </g>))}
      </g>)}
      <text x="26" y="206">A 的列 × B 的行 → 当前贡献</text>
      <text x="26" y="458">寄存器累计，再处理下一段 K</text>
      <text x="26" y="661">蓝色：同一组输出，直到全部 K 完成</text>
    </svg>
    <Readout><Formula>{String.raw`C_{0:${tile},0:${tile}}\mathrel{+}=A_{0:${tile},${start}:${Math.min(7, start + tile)}}B_{${start}:${Math.min(7, start + tile)},0:${tile}}`}</Formula> · 完整矩阵有效输入读取：朴素 {naiveReads}，分块 {tiledReads} · 只计逻辑元素访问，不代表 DRAM 流量或实测速度。</Readout>
  </LabFrame>
}
