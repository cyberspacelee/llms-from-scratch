import { useState } from 'react'
import Formula from './Formula'
import { threadNumber } from './GpuIndexModel'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function BlockVolumeLab() {
  const [x, setX] = useState(0)
  const [y, setY] = useState(0)
  const [z, setZ] = useState(1)
  const thread = threadNumber(x, y, z, 8, 4)
  return <LabFrame title="三维 block 的轴测与 z 切片">
    <Controls>
      <Range label="thread x" value={x} min={0} max={7} onChange={setX} />
      <Range label="thread y" value={y} min={0} max={3} onChange={setY} />
      <Range label="z 切片" value={z} min={0} max={1} onChange={setZ} />
    </Controls>
    <svg viewBox="0 0 320 380" className={`${pen.canvas} max-w-96`} role="img" aria-label={`8×4×2 逻辑 block，选中坐标 (${x},${y},${z})，线性线程 ${thread}`}>
      <text x="12" y="19">block (8, 4, 2) · 逻辑坐标</text>
      {[0, 1].map(layer => <g key={layer}>
        <text x="286" y={155 - layer * 76} className={layer === z ? pen.textA : pen.muted}>z={layer}</text>
        {Array.from({ length: 32 }, (_, i) => {
          const tx = i % 8, ty = Math.floor(i / 8)
          const px = 15 + tx * 28 + ty * 10, py = 132 + ty * 14 - layer * 76
          const selected = tx === x && ty === y && layer === z
          return <path key={i} d={`M${px} ${py}h25l10 12h-25Z`} className={selected ? 'fill-accent2-soft stroke-accent2 stroke-2' : layer === z ? 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} />
        })}
      </g>)}
      <path d="M15 190h220m-5 -4 5 4 -5 4M15 190l30 28m-7 -1 7 1 -1 -7M8 180V70m-4 5 4 -5 4 5" className={pen.axis} />
      <text x="242" y="194">x</text><text x="50" y="223">y</text><text x="3" y="58">z</text>
      <text x="12" y="247">z={z} 切片 · x 从左向右，y 从上向下</text>
      {Array.from({ length: 32 }, (_, i) => {
        const tx = i % 8, ty = Math.floor(i / 8)
        return <g key={i}>
          <rect x={12 + tx * 37} y={261 + ty * 25} width="33" height="21" rx="2" className={tx === x && ty === y ? 'fill-accent2-soft stroke-accent2 stroke-2' : 'fill-accent-soft stroke-accent'} />
          <text x={28.5 + tx * 37} y={276 + ty * 25} textAnchor="middle" className="text-[11px]">{threadNumber(tx, ty, z, 8, 4)}</text>
        </g>
      })}
      <text x="12" y="377" className={pen.muted}>轴测示意；不是物理核心排列</text>
    </svg>
    <Readout><Formula>{`t=${x}+8(${y}+4\\times${z})=${thread}`}</Formula> · warp {Math.floor(thread / 32)} · lane {thread % 32}<br />x 每加 1，t 加 1 · y 每加 1，t 加 8 · z 每加 1，t 加 32</Readout>
  </LabFrame>
}
