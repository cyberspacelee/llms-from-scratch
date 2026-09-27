import { useState } from 'react'
import { storageSelection } from './NumpyContractModel'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function NumpyStorageLab() {
  const [mode, setMode] = useState('view')
  const [cell, setCell] = useState(0)
  const [value, setValue] = useState(9)
  const result = storageSelection(mode, cell, value)
  const snippets: Record<string, string> = { view: 'v = A[:, 1:]; v[r,c] = value', copy: 'v = A[:, 1:].copy(); v[r,c] = value', advanced: 'v = A[:, [1,2]]; v[r,c] = value', assign: 'A[:, [1,2]] = updated_values' }
  return <LabFrame title="切片、索引与修改传播">
    <Controls>
      <label className="block text-sm">取值或赋值方式<select value={mode} onChange={e => setMode(e.target.value)} className="mt-1 block w-full rounded border border-rule bg-paper p-2"><option value="view">基础切片视图</option><option value="copy">切片后显式 copy</option><option value="advanced">高级索引取值副本</option><option value="assign">高级索引直接赋值</option></select></label>
      <Range label="选区线性位置" value={cell} min={0} max={3} onChange={setCell} />
      <Range label="写入数值" value={value} min={-5} max={15} onChange={setValue} />
    </Controls>
    <svg viewBox="0 0 320 259" className={`${pen.canvas} max-w-96`} role="img" aria-label={`写入位置 ${cell}，原数组${result.writesBack ? '随之改变' : '保持不变'}`}>
      <text x="12" y="20">原数组 A · shape (2,3)</text>
      {result.source.map((v, i) => <g key={i}>
        <rect x={12 + i % 3 * 99} y={34 + Math.floor(i / 3) * 37} width="91" height="31" rx="3" className={i === [1, 2, 4, 5][cell] && result.writesBack ? 'fill-accent2-soft stroke-accent2' : 'fill-accent-soft stroke-accent'} />
        <text x={57.5 + i % 3 * 99} y={55 + Math.floor(i / 3) * 37} textAnchor="middle">{v}</text>
      </g>)}
      <text x="12" y="134">{result.shared ? '共享存储的选区' : mode === 'assign' ? '直接写回后的选区取值' : '独立存储的选区'} · (2,2)</text>
      {result.selected.map((v, i) => <g key={i}>
        <rect x={12 + i % 2 * 148} y={148 + Math.floor(i / 2) * 37} width="139" height="31" rx="3" className={i === cell ? 'fill-accent2-soft stroke-accent2' : 'fill-info-soft stroke-info'} />
        <text x={81.5 + i % 2 * 148} y={169 + Math.floor(i / 2) * 37} textAnchor="middle">{v}</text>
      </g>)}
      <text x="12" y="245" className={pen.muted}>每次选择从 A = [[0,1,2],[3,4,5]] 开始</text>
    </svg>
    <Readout>{snippets[mode]}<br />选区位置 ({Math.floor(cell / 2)},{cell % 2}) → A ({Math.floor(cell / 2)},{cell % 2 + 1})<br />shares_memory：{result.shared ? 'True' : 'False（取值）'} · 原数组写回：{result.writesBack ? '是' : '否'}</Readout>
  </LabFrame>
}
