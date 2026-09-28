import { useState } from 'react'
import { Controls, fmt, LabFrame, pen, Range, Readout, useWidth } from './Lab'

export default function SpeculativeLab() {
  const [first, setFirst] = useState(0.6)
  const [length, setLength] = useState(4)
  const [ref, width] = useWidth<HTMLDivElement>()
  const target = [0.5, 0.3, 0.2]
  const draft = [first, (1 - first) / 4, 3 * (1 - first) / 4]
  const accepted = target.map((value, i) => Math.min(value, draft[i]))
  const acceptance = accepted.reduce((sum, value) => sum + value, 0)
  const missing = target.map((value, i) => Math.max(value - draft[i], 0))
  const rejected = missing.reduce((sum, value) => sum + value, 0)
  const expected = Array.from({ length: length + 1 }, (_, i) => acceptance ** i)
    .reduce((sum, value) => sum + value, 0)
  const canvasWidth = Math.max(260, width)
  const plotWidth = canvasWidth - 76

  return (
    <LabFrame title="接受质量与残差补偿">
      <div ref={ref}>
        <Controls>
          <Range label="草稿第一类概率" value={first} min={0} max={1} step={0.02}
            onChange={setFirst} format={value => fmt(value, 2)} />
          <Range label="草稿长度 k" value={length} min={1} max={8} onChange={setLength} />
        </Controls>
        <p className="mt-3 mb-0 text-sm text-muted">目标 p = (0.5, 0.3, 0.2)</p>
        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs">
          <span className="text-accent">实线：接受质量</span>
          <span className="text-accent2">右侧：残差质量</span>
          <span className="text-info">虚线框：草稿概率</span>
        </div>
        <svg viewBox={`0 0 ${canvasWidth} 244`} className={pen.canvas} role="img"
          aria-label={`草稿分布 ${draft.map(value => fmt(value)).join(', ')}；接受率 ${fmt(acceptance)}；接受与补偿后等于目标分布`}>
          {[0, 0.5, 1].map(value => (
            <g key={value}>
              <line x1={60 + value * plotWidth} y1={25} x2={60 + value * plotWidth} y2={222}
                className={pen.grid} />
              <text x={60 + value * plotWidth} y={16} textAnchor="middle" className={pen.mono}>{value}</text>
            </g>
          ))}
          {target.map((probability, i) => {
            const y = 37 + 65 * i
            return (
              <g key={i}>
                <text x={8} y={y + 17}>ID {i}</text>
                <rect x={60} y={y} width={accepted[i] * plotWidth} height={22} className="fill-accent" />
                <rect x={60 + accepted[i] * plotWidth} y={y}
                  width={missing[i] * plotWidth} height={22} className="fill-accent2" />
                <rect x={60} y={y - 4} width={draft[i] * plotWidth} height={30}
                  className="fill-none stroke-info stroke-2 [stroke-dasharray:5_3]" />
                <text x={60} y={y + 46} className={pen.mono}>
                  {fmt(accepted[i], 2)} + {fmt(missing[i], 2)} = {fmt(probability, 2)}
                </text>
              </g>
            )
          })}
        </svg>
        <Readout>
          q = ({draft.map(value => fmt(value, 2)).join(', ')})<br />
          总接受率 = {fmt(acceptance)}；总拒绝率 = {fmt(rejected)}<br />
          拒绝后的条件分布 r = ({missing.map(value => fmt(rejected > 0 ? value / rejected : 0)).join(', ')})<br />
          固定独立接受率近似：每轮期望提交 {fmt(expected)} 个 token
        </Readout>
      </div>
    </LabFrame>
  )
}
