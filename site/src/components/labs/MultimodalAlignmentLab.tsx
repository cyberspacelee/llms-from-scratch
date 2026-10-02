import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { multimodalPatch } from '../../lib/advanced-interactive-model'

export default function MultimodalAlignmentLab() {
  const [patch, setPatch] = useState(0), [position, setPosition] = useState(6), [swapped, setSwapped] = useState(false), [gradient, setGradient] = useState(false)
  const r = multimodalPatch(patch, swapped), names = ['USER', 'z0', 'z1', 'z2', 'z3', 'QUESTION', 'ASSISTANT', 'ANSWER', 'EOS']
  return <LabFrame title="像素块、视觉行与回答监督">
    <Controls><Range label="patch" value={patch} min={0} max={3} onChange={setPatch} /><Range label="预测行" value={position} min={0} max={8} onChange={setPosition} /><Toggle label="交换图像上下半部" checked={swapped} onChange={value => setSwapped(value)} /><Toggle label="显示梯度路径" checked={gradient} onChange={value => setGradient(value)} /></Controls>
    <SvgCanvas viewBox="0 0 280 370" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`patch ${patch} 位于语言模型输入 ${r.position}，预测行 ${position}`}>
      {Array.from({ length: 16 }, (_, i) => { const selected = r.indices.includes(i), v = (swapped ? (i + 8) % 16 : i) / 15; return <g key={i}><rect x={8 + i % 4 * 33} y={8 + Math.floor(i / 4) * 33} width="30" height="30" fill={`rgb(${Math.round(v * 180 + 45)},${Math.round(v * 180 + 45)},${Math.round(v * 180 + 45)})`} className={selected ? 'stroke-accent stroke-3' : 'stroke-rule'} /></g> })}<text x="150" y="45">patch {patch}</text><text x="150" y="69">h ∈ R⁶</text><text x="150" y="93">projector</text><text x="150" y="117">z ∈ R⁸</text>
      {names.map((name, i) => <g key={i} transform={`translate(${8 + i % 3 * 90},${160 + Math.floor(i / 3) * 65})`}><rect width="83" height="50" className={i === r.position ? 'fill-accent/20 stroke-accent stroke-2' : i <= position ? 'fill-info/10 stroke-info' : 'fill-sunken stroke-rule [stroke-dasharray:3_3]'} /><text x="41" y="20" textAnchor="middle" className="text-[11px]">{name}</text><text x="41" y="39" textAnchor="middle">p={i}{i === position ? ' ←' : ''}</text></g>)}<text x="8" y="365">{gradient ? '回复 loss → Decoder → z → projector' : '蓝框：当前行因果可见；虚线：未来'}</text>
    </SvgCanvas><Readout>patch 索引=[{r.indices.join(', ')}]；像素=[{r.pixels.map(v => fmt(v)).join(', ')}]；h 第一维均值={fmt(r.mean, 6)}<br />六维特征 → 八维视觉行 z{patch}，进入位置 {r.position}；本图只核对形状，未伪造其余特征值<br />当前预测目标：{position === 6 ? (swapped ? 'ANSWER（应为上半部）' : 'ANSWER（应为下半部）') : position === 7 ? 'EOS' : '忽略/无下一目标'}；有效 logits 行=[6,7]<br />{gradient ? '视觉行无直接文字标签；回复 loss 仍可回传连接器。冻结视觉编码器不要求 detach 连接器输出' : '换图必须重算视觉特征与前缀 KV；相同问题文字不代表缓存相同'}</Readout>
  </LabFrame>
}
