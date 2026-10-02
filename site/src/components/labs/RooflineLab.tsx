import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { roofline } from '../../lib/systems-interactive-model'

export default function RooflineLab() {
  const [batch, setBatch] = useState(1), [length, setLength] = useState(8192)
  const r = roofline(batch, length), x = (i: number) => 30 + Math.log10(Math.max(0.1, i) / 0.1) / 4 * 230
  const y = (t: number) => 205 - t / 1000 * 165
  return <LabFrame title="同一 decode 负载在屋顶线上的位置" hint="规格理论上限；不含所有算子与额外流量">
    <Controls><Range label="B" value={batch} min={1} max={512} onChange={setBatch} /><Range label="T" value={length} min={1} max={16384} onChange={setLength} /></Controls>
    <SvgCanvas viewBox="0 0 280 245" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`强度 ${fmt(r.intensity)} FLOP/byte，理论上限 ${fmt(r.ceiling)} TFLOP/s`}>
      <path d="M30 25V205H264" className={pen.axis} /><path d={`M${x(0.1)} ${y(0.335)}L${x(r.ridge)} ${y(989.5)}H260`} className={pen.a} />
      <circle cx={x(r.intensity)} cy={y(r.ceiling)} r="5" className="fill-accent2" /><line x1={x(r.ridge)} y1="35" x2={x(r.ridge)} y2="205" className={pen.guide} />
      {[0.1, 1, 10, 100, 1000].map(i => <text key={i} x={x(i)} y="225" textAnchor="middle" className={pen.mono}>{i}</text>)}<text x="35" y="20">TFLOP/s</text><text x="140" y="243" textAnchor="middle">FLOP/byte · 对数横轴</text>
    </SvgCanvas>
    <Readout>I={fmt(r.intensity)}；脊点={fmt(r.ridge)}；上限={fmt(r.ceiling)} TFLOP/s<br />计算下界 {fmt(r.tCompute * 1000)} ms；搬运下界 {fmt(r.tMemory * 1000)} ms；取较大值 {fmt(Math.max(r.tCompute, r.tMemory) * 1000)} ms</Readout>
  </LabFrame>
}
