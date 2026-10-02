import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import Formula from './Formula'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function ThreadIndexLab() {
  const [length, setLength] = useState(100)
  const [threads, setThreads] = useState(64)
  const blocks = Math.ceil(length / threads)
  const [selected, setSelected] = useState(1)
  const [lane, setLane] = useState(0)
  const local = Math.min(lane, threads - 1)
  const block = Math.min(selected, blocks - 1)
  const valid = Math.max(0, Math.min(threads, length - block * threads))
  return <LabFrame title="数组元素与线程编号">
    <Controls>
      <Range label="元素数" value={length} min={1} max={256} onChange={setLength} />
      <Range label="每块线程数" value={threads} min={16} max={128} step={16} onChange={setThreads} />
      <Range label="查看 block" value={block} min={0} max={blocks - 1} onChange={setSelected} />
      <Range label="选中块内线程" value={local} min={0} max={threads-1} onChange={setLane} />
    </Controls>
    <SvgCanvas viewBox={`0 0 400 ${Math.ceil(threads/8)*26+90}`} className={`${pen.canvas} max-w-120`} role="img" aria-label={`block ${block}，有效线程 ${valid}，尾部线程 ${threads - valid}`}>
      <text x="12" y="20">block {block} · 从元素 {block * threads} 开始</text>
      {Array.from({ length: threads }, (_, t) => <g key={t}>
        <rect x={12 + (t % 8) * 48} y={38 + Math.floor(t / 8) * 26} width="44" height="22" rx="2" className={t < valid ? 'fill-accent-soft stroke-accent' : 'fill-accent2-soft stroke-accent2'} strokeWidth={t === local ? 3 : 1} />
        {t < threads && <text x={34 + (t % 8) * 48} y={53 + Math.floor(t / 8) * 26} textAnchor="middle" className={pen.mono}>{block * threads + t}</text>}
      </g>)}
      <text x="12" y={Math.ceil(threads/8)*26+68} className={pen.textA}>绿色：有效元素</text>
      <text x="166" y={Math.ceil(threads/8)*26+68} className={pen.textB}>橙色：边界外</text>
    </SvgCanvas>
    <Readout><Formula>{`G=\\lceil ${length}/${threads}\\rceil=${blocks}`}</Formula> · 启动 {blocks * threads} 个线程 · 总尾部 {blocks * threads - length} 个 · 当前有效 {valid} 个<br />选中 threadIdx.x={local}；global i={block}×{threads}+{local}={block*threads+local}；{local < valid ? "负责一个有效元素" : "越界，不写输出"}。</Readout>
  </LabFrame>
}
