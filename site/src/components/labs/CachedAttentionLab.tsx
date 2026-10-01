import { useState } from 'react'
import { cachedChapterAttention } from '../../lib/principles-trace-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

export default function CachedAttentionLab() {
  const [past, setPast] = useState(3)
  const [length, setLength] = useState(2)
  const [row, setRow] = useState(0)
  const [wrong, setWrong] = useState(false)
  const selected = Math.min(row, length - 1)
  const result = cachedChapterAttention(past, length, selected, wrong)
  const total = past + length
  return <LabFrame title="从因果 mask 算到真实注意力输出" hint="正文单头 dh=2；q=k=(1,0)，频率 1；Vⱼ=(j,0)">
    <Controls>
      <Range label="缓存长度 P" min={0} max={5} value={past} onChange={setPast} />
      <Range label="新增长度 U" min={1} max={4} value={length} onChange={setLength} />
      <Range label="当前查询行 i" min={0} max={length - 1} value={selected} onChange={setRow} />
      <label className="flex min-h-9 items-center gap-2 text-sm"><input type="checkbox" checked={wrong} onChange={e => setWrong(e.target.checked)} className="accent-accent" />使用错误的左上角 tril</label>
    </Controls>
    <svg viewBox={`0 0 380 ${110 + length * 30}`} className={`${pen.canvas} max-w-110!`} role="img" aria-label={`缓存因果矩阵，${wrong ? '错误' : '正确'} mask，选中行 ${selected}`}>
      {Array.from({ length: total }, (_, j) => <text key={j} x={72 + j * 32} y="20" textAnchor="middle">{j}</text>)}
      {Array.from({ length }, (_, i) => <g key={i}>
        <text x="4" y={49 + i * 30}>p={past + i}</text>
        {Array.from({ length: total }, (_, j) => <rect key={j} x={57 + j * 32} y={30 + i * 30} width="28" height="26" rx="2" className={j <= (wrong ? i : past + i) ? 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} strokeWidth={i === selected ? 2 : 1} />)}
      </g>)}
      <text x="4" y={78 + length * 30} className={wrong ? pen.textB : pen.textA}>{wrong ? 'j ≤ i：丢掉本来可见的历史' : 'j ≤ P+i：历史与当前前缀都可见'}</text>
    </svg>
    <div className="mt-3 overflow-x-auto"><table className="w-full text-right font-mono text-xs"><caption className="mb-2 text-left font-sans text-sm">查询绝对位置 p={result.position}，旋转后 Q=({fmt(result.q[0])}, {fmt(result.q[1])})</caption><thead><tr><th className="p-2">键 j</th><th className="p-2">QK/√2</th><th className="p-2">mask 后</th><th className="p-2">概率</th></tr></thead><tbody>{result.scores.map((s, j) => <tr key={j} className="border-t border-rule"><td className="p-2">{j}</td><td className="p-2">{fmt(s)}</td><td className="p-2">{result.allowed[j] ? fmt(s) : '−∞'}</td><td className="p-2">{fmt(result.probabilities[j])}</td></tr>)}</tbody></table></div>
    <Readout>Σ概率 = {fmt(result.probabilities.reduce((a, b) => a + b, 0))} · Yᵢ = Σⱼ aᵢⱼVⱼ = ({fmt(result.output[0])}, {fmt(result.output[1])}) · {wrong && past > 0 ? '这是错误 mask 的结果；归一化依然正常，不能只检查概率和。' : '正确位置偏移决定读取范围。'}</Readout>
  </LabFrame>
}
