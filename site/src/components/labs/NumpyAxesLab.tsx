import { useState } from 'react'
import { axisCalculation } from './NumpyContractModel'
import { Controls, LabFrame, Readout, pen } from './Lab'

export default function NumpyAxesLab() {
  const [mode, setMode] = useState('feature')
  const [axis, setAxis] = useState(1)
  const [keep, setKeep] = useState(true)
  const result = axisCalculation(mode, axis, keep)
  return <LabFrame title="广播与归约的轴契约">
    <Controls>
      <label className="block text-sm">计算方式<select value={mode} onChange={e => setMode(e.target.value)} className="mt-1 block w-full rounded border border-rule bg-paper p-2"><option value="feature">加特征偏置 (3,)</option><option value="sample">加样本偏置 (2,1)</option><option value="invalid">加错误形状 (2,)</option><option value="mean">mean 归约</option></select></label>
      <label className="block text-sm">归约轴<select value={axis} disabled={mode !== 'mean'} onChange={e => setAxis(Number(e.target.value))} className="mt-1 block w-full rounded border border-rule bg-paper p-2"><option value={0}>axis=0 跨样本</option><option value={1}>axis=1 跨特征</option></select></label>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={keep} disabled={mode !== 'mean'} onChange={e => setKeep(e.target.checked)} className="accent-accent" />keepdims</label>
    </Controls>
    <svg viewBox="0 0 320 270" className={`${pen.canvas} max-w-96`} role="img" aria-label={`输入2×3，输出${result.shape}`}>
      <text x="12" y="20">X · 样本轴 0，特征轴 1</text>
      {[1, 2, 3, 4, 5, 6].map((v, i) => <g key={i}>
        <rect x={12 + i % 3 * 99} y={35 + Math.floor(i / 3) * 36} width="91" height="30" rx="3" className="fill-accent-soft stroke-accent" /><text x={57 + i % 3 * 99} y={55 + Math.floor(i / 3) * 36} textAnchor="middle">{v}</text>
      </g>)}
      <text x="12" y="135">输出 · shape {result.shape}</text>
      {result.valid ? result.values.map((v, i) => {
        const columns = mode === 'mean' && axis === 1 ? keep ? 1 : 2 : 3
        return <g key={i}><rect x={12 + i % columns * 99} y={151 + Math.floor(i / columns) * 36} width="91" height="30" rx="3" className="fill-info-soft stroke-info" /><text x={57 + i % columns * 99} y={171 + Math.floor(i / columns) * 36} textAnchor="middle">{v}</text></g>
      }) : <text x="12" y="176" className={pen.textB}>尾轴 3 与 2 不相等且都不为 1</text>}
      <text x="12" y="257" className={pen.muted}>{mode === 'mean' ? keep ? '归约轴保留为长度 1' : '归约轴被删除' : '广播从尾轴向前对齐'}</text>
    </svg>
    <Readout>{mode === 'mean' ? `X.mean(axis=${axis}, keepdims=${keep ? 'True' : 'False'})` : mode === 'feature' ? 'X + [10,20,30]' : mode === 'sample' ? 'X + [[100],[200]]' : 'X + [100,200] → ValueError'}<br />{mode === 'mean' && axis === 1 ? keep ? '结果 (2,1) 可直接从 X 每行减去' : '结果 (2,) 不能直接从 (2,3) 的 X 减去' : `输出 ${result.shape}`}</Readout>
  </LabFrame>
}
