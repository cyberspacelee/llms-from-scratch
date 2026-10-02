import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Range, Readout, pen } from './Lab'
import { hybridBoundary } from '../../lib/advanced-interactive-model'

export default function HybridBoundaryLab() {
  const [boundary, setBoundary] = useState(4), [full, setFull] = useState(true), [window, setWindow] = useState(true), [recursive, setRecursive] = useState(true)
  const r = hybridBoundary(boundary, full, window, recursive)
  const componentNames = ['FULL', 'SWA', '递归快照'], enabled = [full, window, recursive]
  return <LabFrame title="混合缓存的共同安全边界" hint="独立扩展示例：8 项前缀；SWA 窗口 3">
    <Controls><Range label="候选已处理边界 b" value={boundary} min={0} max={8} onChange={setBoundary} />{componentNames.map((name, i) => <Toggle key={name} label={name} checked={enabled[i]} onChange={value => [setFull, setWindow, setRecursive][i](value)} />)}</Controls>
    <SvgCanvas viewBox="0 0 280 235" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`候选边界 ${boundary}，各组件投票 ${r.votes.join(', ')}，可复用 ${r.valid}`}>
      {componentNames.map((name, row) => <g key={name}><text x="8" y={22 + row * 70}>{name} · {enabled[row] ? (r.votes[row] ? '通过' : '拒绝') : '未启用'}</text>{Array.from({ length: 9 }, (_, i) => { const available = row === 0 ? i < 8 : row === 1 ? r.windowAvailable.includes(i) : r.checkpoints.includes(i); return <g key={i}><rect x={8 + i * 29} y={30 + row * 70} width="24" height="25" className={available ? 'fill-accent/20 stroke-accent' : 'fill-sunken stroke-rule [stroke-dasharray:3_3]'} /><text x={20 + i * 29} y={48 + row * 70} textAnchor="middle">{i}</text></g> })}</g>)}<text x="8" y="230">FULL/SWA：token 下标；快照：处理边界</text>
    </SvgCanvas><Readout>FULL 需要 [0,{boundary})；SWA 需要最近 {'{' + r.requiredWindow.join(', ') + '}'}；递归状态需要恰好 b={boundary} 的快照<br />共同结果：{r.valid ? '可从此边界继续' : '不能复用此边界，应查更早的独立候选或重算'}</Readout>
  </LabFrame>
}
