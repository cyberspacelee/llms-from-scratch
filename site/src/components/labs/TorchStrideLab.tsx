import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { headCoordinate } from '../../lib/framework-trace-model'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function TorchStrideLab() {
  const [offset, setOffset] = useState(0), [stage, setStage] = useState(2)
  const selected = headCoordinate(offset), permuted = stage === 2
  const cells = Array.from({ length: 8 }, (_, i) => headCoordinate(i)).sort((a, b) => permuted ? (a.head * 4 + a.t * 2 + a.channel) - (b.head * 4 + b.t * 2 + b.channel) : a.offset - b.offset)
  const cols = permuted ? 2 : 4, stride = ['(8,4,1)', '(8,4,2,1)', '(8,2,4,1)', '(8,4,1)'][stage]
  return <LabFrame title="一个元素从时间/通道走到头，再合回原位置">
    <Controls><Range label="原输入元素偏移" value={offset} min={0} max={7} onChange={setOffset} /><Range label="阶段：输入/拆头/换轴/合头" value={stage} min={0} max={3} onChange={setStage} /></Controls>
    <SvgCanvas viewBox={`0 0 320 ${permuted ? 350 : 240}`} className={`${pen.canvas} max-w-96`} role="img" aria-label={`阶段 ${stage}，元素 ${selected.value} 位于头 ${selected.head}、时间 ${selected.t}、头内通道 ${selected.channel}`}>
      <text x="10" y="20">{['X: (1,2,4)', 'reshape: (1,2,2,2)', 'permute: (1,2,2,2)', '先 permute 回去，再 reshape'][stage]}</text>
      {cells.map((cell, i) => <g key={cell.offset}><rect x={10 + i % cols * (300 / cols)} y={35 + Math.floor(i / cols) * 55} width={300 / cols - 8} height="47" rx="3" className={cell.offset === offset ? 'fill-accent2-soft stroke-accent2 stroke-2' : cell.head ? 'fill-info-soft stroke-info' : 'fill-accent-soft stroke-accent'} /><text x={10 + i % cols * (300 / cols) + (300 / cols - 8) / 2} y={56 + Math.floor(i / cols) * 55} textAnchor="middle">{cell.value}</text><text x={10 + i % cols * (300 / cols) + (300 / cols - 8) / 2} y={73 + Math.floor(i / cols) * 55} textAnchor="middle" className={pen.mono}>S[{cell.offset}]</text></g>)}
      <text x="10" y={permuted ? 277 : 166}>颜色标记两组头；橙色标记同一元素</text>
      <text x="10" y={permuted ? 301 : 190}>同一存储身份 S · stride {stride}</text>
      <text x="10" y={permuted ? 325 : 214}>这条正确的合头路径能恢复原输入</text>
    </SvgCanvas>
    <Readout>X[{selected.original.join(',')}]={selected.value} → 拆头 [{selected.split.join(',')}] → 换轴 [{selected.permuted.join(',')}] → 合头 [{selected.original.join(',')}]<br />换轴后的元素偏移=头×2+时间×4+头内通道={selected.offset}，存储次序不变。<br />这里先换回原轴才合头；直接把换轴结果 reshape(1,2,4) 会按头先后重排元素，并可能复制。存储关系与梯度关系需要分别检查。</Readout>
  </LabFrame>
}
