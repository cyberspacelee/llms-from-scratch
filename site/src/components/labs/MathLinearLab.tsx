import { useState } from 'react'
import { linearX, linearW, linearB, linearZ, linearElement, gradientContributions } from '../../lib/math-linear-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

export default function MathLinearLab({ gradient = false }: { gradient?: boolean }) {
  const [row, setRow] = useState(1), [output, setOutput] = useState(0), [weight, setWeight] = useState(1), [step, setStep] = useState(3), [mean, setMean] = useState(true)
  const element = linearElement(row, output), contributions = gradientContributions(weight, step, mean)
  const grids = gradient ? [{ title: 'X · 每行一条记录', values: linearX, selected: (i: number, j: number) => i < step && j === contributions.j }, { title: 'E = Z − 0', values: linearZ, selected: (i: number, j: number) => i < step && j === contributions.k }, { title: '累计 ∇W', values: contributions.gradient, selected: (i: number, j: number) => i === contributions.k && j === contributions.j }] : [{ title: 'X · 输入', values: linearX, selected: (i: number) => i === row }, { title: 'W · 共享权重', values: linearW, selected: (i: number) => i === output }, { title: 'Z = XWᵀ + b', values: linearZ, selected: (i: number, j: number) => i === row && j === output }]
  return <LabFrame title={gradient ? '同一参数收集三条记录的贡献' : '选择输出元素，追踪它读取的两条行'}>
    <Controls>{gradient ? <><Range label="参数位置 k×2+j" value={weight} min={0} max={3} onChange={setWeight} /><Range label="已累积记录数" value={step} min={0} max={3} onChange={setStep} /><label className="flex h-9 items-center gap-2 text-sm"><input type="checkbox" checked={mean} onChange={e => setMean(e.target.checked)} />样本 mean（分母固定为 3）</label></> : <><Range label="输出记录 i（从 0 开始）" value={row} min={0} max={2} onChange={setRow} /><Range label="输出特征 k" value={output} min={0} max={1} onChange={setOutput} /></>}</Controls>
    <div className="mt-4 grid gap-4 sm:grid-cols-3">{grids.map(grid => <svg key={grid.title} viewBox={`0 0 230 ${50 + grid.values.length * 48}`} className={`${pen.canvas} mt-0! max-w-64!`} role="img" aria-label={grid.title}>
      <text x="10" y="22">{grid.title}</text>{grid.values.map((values, i) => values.map((v, j) => <g key={`${i}-${j}`}><rect x={10 + j * 107} y={37 + i * 48} width="97" height="38" rx="4" className={grid.selected(i, j) ? 'fill-accent2-soft stroke-accent2 stroke-2' : 'fill-sunken stroke-rule'} /><text x={58 + j * 107} y={61 + i * 48} textAnchor="middle" className={pen.mono}>{fmt(v)}</text></g>))}</svg>)}</div>
    <Readout>{gradient ? <>∂L/∂W[{contributions.k},{contributions.j}] · {contributions.terms.map((v, i) => `记录 ${i}: ${fmt(v)}`).join(' + ')}<br />已取前 {step} 项={fmt(contributions.partial)} · {mean ? '每项已除以全批次 3；累积中途不重新平均' : 'sum 不除以样本数'}</> : <>Z[{row},{output}]={linearX[row].map((x, j) => `${x}×${linearW[output][j]}`).join(' + ')} + {linearB[output]}={fmt(element.value)}<br />逐项乘积 [{element.terms.join(', ')}] · 输入只读取记录 {row}；W[{output},:] 被全部记录共享。</>}</Readout>
  </LabFrame>
}
