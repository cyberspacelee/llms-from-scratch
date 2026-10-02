import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import Formula from './Formula'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function CoalescingLab() {
  const [stride, setStride] = useState(1)
  const [offset, setOffset] = useState(0)
  const words = Array.from({ length: 32 }, (_, lane) => offset + lane * stride)
  const sectors = new Set(words.map(word => Math.floor(word / 8)))
  return <LabFrame title="一条 warp 覆盖哪些地址段" hint="float32 · 对齐基址 · 32-byte 段">
    <Controls>
      <Range label="元素步长" value={stride} min={1} max={8} onChange={setStride} />
      <Range label="元素偏移" value={offset} min={0} max={7} onChange={setOffset} />
    </Controls>
    <SvgCanvas viewBox="0 0 360 300" className={pen.canvas} role="img" aria-label={`32 个 lane 覆盖 ${sectors.size} 个地址段`}>
      <text x="12" y="20">lane → 元素编号 / 地址段</text>
      {words.map((word, lane) => {
        const x = 12 + lane % 4 * 88, y = 36 + Math.floor(lane / 4) * 31
        return <g key={lane}>
          <rect x={x} y={y} width="82" height="26" rx="3" className="fill-accent-soft stroke-rule" />
          <text x={x + 41} y={y + 17} textAnchor="middle" className={pen.mono}>{lane}: {word} / {Math.floor(word / 8)}</text>
        </g>
      })}
      <text x="12" y="294">每段含 8 个 float32；段号相同即共享覆盖段。</text>
    </SvgCanvas>
    <Readout><Formula>{String.raw`a_\ell=4(${offset}+${stride}\ell)`}</Formula> 字节 · 覆盖 {sectors.size} 段 · 请求 128 字节 · 段覆盖 {32 * sectors.size} 字节</Readout>
  </LabFrame>
}
