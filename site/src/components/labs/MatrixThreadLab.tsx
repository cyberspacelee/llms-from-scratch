import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import Formula from './Formula'
import { matrixIndex } from './GpuIndexModel'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function MatrixThreadLab() {
  const [bx, setBx] = useState(1)
  const [by, setBy] = useState(0)
  const [tx, setTx] = useState(3)
  const [ty, setTy] = useState(0)
  const result = matrixIndex(bx, by, tx, ty)
  return <LabFrame title="二维 grid → block → 矩阵元素">
    <Controls>
      <Range label="block x" value={bx} min={0} max={1} onChange={setBx} />
      <Range label="block y" value={by} min={0} max={2} onChange={setBy} />
      <Range label="thread x" value={tx} min={0} max={3} onChange={setTx} />
      <Range label="thread y" value={ty} min={0} max={1} onChange={setTy} />
    </Controls>
    <SvgCanvas viewBox="0 0 320 505" className={`${pen.canvas} max-w-96`} role="group" aria-label="二维网格、块内线程与带尾部位置的矩阵">
      <text x="12" y="20">grid (2, 3) · 选择 block</text>
      {Array.from({ length: 6 }, (_, b) => {
        const x = b % 2, y = Math.floor(b / 2)
        const choose = () => { setBx(x); setBy(y) }
        return <g key={b} role="button" tabIndex={0} aria-label={`选择 block (${x},${y})`} aria-pressed={x === bx && y === by} onClick={choose} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose() } }} className="cursor-pointer focus:outline-2 focus:outline-info">
          <rect x={12 + x * 148} y={34 + y * 35} width="138" height="29" rx="3" className={x === bx && y === by ? 'fill-accent-soft stroke-accent stroke-2' : 'fill-sunken stroke-rule'} />
          <text x={81 + x * 148} y={54 + y * 35} textAnchor="middle">({x}, {y})</text>
        </g>
      })}
      <text x="12" y="166">block (4, 2) · x 最快变化</text>
      {Array.from({ length: 8 }, (_, t) => {
        const x = t % 4, y = Math.floor(t / 4)
        const choose = () => { setTx(x); setTy(y) }
        return <g key={t} role="button" tabIndex={0} aria-label={`选择 thread (${x},${y})`} aria-pressed={x === tx && y === ty} onClick={choose} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose() } }} className="cursor-pointer focus:outline-2 focus:outline-info">
          <rect x={12 + x * 74} y={181 + y * 35} width="66" height="29" rx="3" className={x === tx && y === ty ? 'fill-info-soft stroke-info stroke-2' : 'fill-sunken stroke-rule'} />
          <text x={45 + x * 74} y={201 + y * 35} textAnchor="middle">({x},{y})</text>
        </g>
      })}
      <text x="12" y="277">矩阵 (5, 7) · 灰格为尾部</text>
      {Array.from({ length: 48 }, (_, t) => {
        const row = Math.floor(t / 8), col = t % 8
        const valid = row < 5 && col < 7
        const selected = row === result.row && col === result.col
        return <g key={t}>
          <rect x={12 + col * 37} y={292 + row * 30} width="33" height="26" rx="2" className={selected ? result.valid ? 'fill-accent-soft stroke-accent stroke-2' : 'fill-accent2-soft stroke-accent2 stroke-2' : !valid ? 'fill-sunken stroke-rule' : Math.floor(row / 2) === by && Math.floor(col / 4) === bx ? 'fill-info-soft stroke-info' : 'fill-none stroke-rule'} />
          <text x={28.5 + col * 37} y={309 + row * 30} textAnchor="middle" className={`text-[11px] ${valid ? '' : pen.muted}`}>{valid ? row * 7 + col : '×'}</text>
        </g>
      })}
      <text x="12" y="496" className={pen.muted}>格内数字：行主序元素偏移</text>
    </SvgCanvas>
    <Readout><Formula>{`r=${by}\\times2+${ty}=${result.row},\\quad c=${bx}\\times4+${tx}=${result.col}`}</Formula><br />
      <Formula>{`p=${result.row}\\times7+${result.col}=${result.offset}`}</Formula> · 行列检查 {result.valid ? '通过' : '拒绝'} · 仅检查 p &lt; 35：{result.offsetOnly ? '通过' : '拒绝'}
      {!result.valid && result.offsetOnly && ' · 偏移合法但坐标越界，会误写下一行'}
    </Readout>
  </LabFrame>
}
