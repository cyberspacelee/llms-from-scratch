import { useState } from 'react'
import { Controls, fmt, LabFrame, pen, Range, Readout } from './Lab'
import { frequencies } from './FrequencyLab'

const row = (position: number) => frequencies.flatMap((omega) => [Math.sin(position * omega), Math.cos(position * omega)])

/** 32 positions × 8 dimensions: every row is one position's encoding. */
export default function EncodingMatrixLab() {
  const [selected, setSelected] = useState(12)
  const [x0, y0, cw, ch] = [64, 56, 76, 9]
  const values = row(selected)
  const norm = values.reduce((sum, value) => sum + value ** 2, 0)

  return (
    <LabFrame title="位置 × 维度：把波形堆成矩阵" hint="点一行，或拖动滑块选择位置">
      <Controls>
        <Range label="选中位置 p" value={selected} min={0} max={31} onChange={setSelected} />
      </Controls>
      <svg viewBox="0 0 680 470" className={pen.canvas} role="img"
        aria-label={`位置编码矩阵，选中位置 ${selected}`}>
        {Array.from({ length: 8 }, (_, d) => (
          <g key={d}>
            <text x={x0 + d * cw + cw / 2} y={22} textAnchor="middle" className={pen.mono}>d{d}</text>
            <text x={x0 + d * cw + cw / 2} y={40} textAnchor="middle" className={`${pen.mono} ${pen.muted}`}>
              {d % 2 ? 'cos' : 'sin'} ω{Math.floor(d / 2)}
            </text>
          </g>
        ))}
        {Array.from({ length: 8 }, (_, index) => index * 4).map((position) => (
          <text key={position} x={x0 - 12} y={y0 + position * ch + 8} textAnchor="end" className={`${pen.mono} ${pen.muted}`}>{position}</text>
        ))}
        <text x={14} y={y0 + 16 * ch} textAnchor="middle" transform={`rotate(-90 14 ${y0 + 16 * ch})`} className={pen.muted}>位置 p</text>
        {Array.from({ length: 32 }, (_, position) =>
          row(position).map((value, d) => (
            <rect
              key={`${position}-${d}`}
              x={x0 + d * cw}
              y={y0 + position * ch}
              width={cw - 1}
              height={ch - 1}
              className={`cursor-pointer ${value >= 0 ? 'fill-accent' : 'fill-accent2'}`}
              fillOpacity={0.08 + Math.abs(value) * 0.82}
              onClick={() => setSelected(position)}
            >
              <title>{`p=${position}, d${d} = ${fmt(value, 4)}`}</title>
            </rect>
          )),
        )}
        <rect x={x0 - 3} y={y0 + selected * ch - 2} width={cw * 8 + 5} height={ch + 3} className="fill-none stroke-info stroke-2" />
        <text x={x0 - 12} y={y0 + 32 * ch + 50} textAnchor="end" className={pen.mono}>PE({selected})</text>
        {values.map((value, d) => (
          <g key={d}>
            <rect x={x0 + d * cw} y={y0 + 32 * ch + 22} width={cw - 4} height={40} rx="4" className="fill-sunken" />
            <text x={x0 + d * cw + cw / 2 - 2} y={y0 + 32 * ch + 47} textAnchor="middle" className={`${pen.mono} ${value >= 0 ? pen.textA : pen.textB}`}>
              {fmt(value)}
            </text>
          </g>
        ))}
        <text x={x0} y={y0 + 32 * ch + 92} className={pen.muted}>绿色为正、橙色为负，颜色越深绝对值越大；每个 sin/cos 对的长度平方都是 1</text>
      </svg>
      <Readout>
        PE({selected}) = [{values.map((value) => fmt(value)).join(', ')}] · 平方范数 = {fmt(norm, 4)}
      </Readout>
    </LabFrame>
  )
}
