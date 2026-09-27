import { useState } from 'react'
import Formula from './Formula'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function ThreadIndexLab() {
  const [length, setLength] = useState(100)
  const [threads, setThreads] = useState(64)
  const blocks = Math.ceil(length / threads)
  const [selected, setSelected] = useState(1)
  const block = Math.min(selected, blocks - 1)
  const valid = Math.max(0, Math.min(threads, length - block * threads))
  return <LabFrame title="数组元素与线程编号">
    <Controls>
      <Range label="元素数" value={length} min={1} max={256} onChange={setLength} />
      <Range label="每块线程数" value={threads} min={16} max={128} step={16} onChange={setThreads} />
      <Range label="查看 block" value={block} min={0} max={blocks - 1} onChange={setSelected} />
    </Controls>
    <svg viewBox="0 0 320 350" className={`${pen.canvas} max-w-96`} role="img" aria-label={`block ${block}，有效线程 ${valid}，尾部线程 ${threads - valid}`}>
      <text x="12" y="20">block {block} · 从元素 {block * threads} 开始</text>
      {Array.from({ length: 128 }, (_, t) => <g key={t}>
        <rect x={12 + (t % 8) * 38} y={38 + Math.floor(t / 8) * 17} width="34" height="14" rx="2" className={t >= threads ? 'fill-none stroke-rule' : t < valid ? 'fill-accent-soft stroke-accent' : 'fill-accent2-soft stroke-accent2'} />
        {t < threads && <text x={29 + (t % 8) * 38} y={49 + Math.floor(t / 8) * 17} textAnchor="middle" className="text-[9px]">{block * threads + t}</text>}
      </g>)}
      <text x="12" y="329" className={pen.textA}>绿色：有效元素</text>
      <text x="166" y="329" className={pen.textB}>橙色：边界外</text>
    </svg>
    <Readout><Formula>{`G=\\lceil ${length}/${threads}\\rceil=${blocks}`}</Formula> · 启动 {blocks * threads} 个线程 · 总尾部 {blocks * threads - length} 个 · 当前有效 {valid} 个</Readout>
  </LabFrame>
}
