import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Select, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { quantizedWeights } from '../../lib/systems-interactive-model'

export default function QuantizationLab() {
  const [bits, setBits] = useState(3), [group, setGroup] = useState(8), [outlier, setOutlier] = useState(20), [asymmetric, setAsymmetric] = useState(false)
  const r = quantizedWeights(bits, group, outlier, asymmetric)
  return <LabFrame title="同一矩阵：编码、恢复、输出和存储">
    <Controls><Range label="位宽" value={bits} min={2} max={8} onChange={setBits} /><Select label="共享 scale 的元素数"  value={group} onChange={e => setGroup(Number(e.target.value))}><option value={8}>全矩阵 8</option><option value={4}>每行 4</option><option value={2}>每组 2</option></Select><Range label="第二行异常值幅度" value={outlier} min={1} max={20} onChange={setOutlier} /><Toggle label="非对称无符号码" checked={asymmetric} onChange={value => setAsymmetric(value)} /></Controls>
    <SvgCanvas viewBox="0 0 280 252" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`原值与恢复值，输出误差 ${r.output.map((v, i) => fmt(v - r.reference[i])).join(', ')}`}>
      {r.original.map((v, i) => <g key={i}><text x="6" y={24 + i * 26}>w{i}</text><line x1="42" y1={20 + i * 26} x2="264" y2={20 + i * 26} className={pen.grid} /><circle cx={153 + v / outlier * 105} cy={20 + i * 26} r="4" className="fill-accent" /><path d={`M${153 + r.reconstructed[i] / outlier * 105} ${15 + i * 26}v10`} className={pen.b} /></g>)}<text x="6" y="233">● 原值；虚线 恢复值；同一横轴</text>
    </SvgCanvas>
    <Readout>码=[{r.codes.join(', ')}]；scale=[{r.scales.map(v => fmt(v, 4)).join(', ')}]；zero=[{r.zeros.join(', ')}]<br />Wx=[{r.reference.map(v => fmt(v)).join(', ')}]；恢复输出=[{r.output.map(v => fmt(v)).join(', ')}]<br />误差=[{r.output.map((v, i) => fmt(v - r.reference[i])).join(', ')}]；紧密打包码 + fp16 scale{asymmetric ? ' + uint8 zero' : ''}={r.bytes} B；原 bf16=16 B</Readout>
  </LabFrame>
}
