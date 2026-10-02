import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { stateAllocation } from '../../lib/training-trace-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'


export default function StateAllocationLab() {
  const [ranks, setRanks] = useState(4), [layout, setLayout] = useState(0)
  const bytes = stateAllocation(ranks), names = ['DDP', 'ZeRO-1', 'ZeRO-2', 'ZeRO-3 / FSDP']
  return <LabFrame title="相同参数坐标，由谁保存哪一份状态" hint="十亿参数；参数 2B、梯度 2B、主参数+m+v 共 12B">
    <Controls><Range label="rank 数" min={1} max={8} value={ranks} onChange={setRanks} /><Range label="状态布局" min={0} max={3} value={layout} onChange={setLayout} /></Controls>
    <SvgCanvas viewBox="0 0 360 242" className={`${pen.canvas} max-w-110!`} role="img" aria-label="各分片阶段每 rank 的常驻模型状态字节数">{bytes.map((value, i) => <g key={i}><text x="8" y={24 + i * 56}>{names[i]} · {fmt(value, 2)} GB/rank</text><rect x="8" y={34 + i * 56} width={value / 16 * 336} height="22" className={layout === i ? 'fill-accent stroke-accent' : 'fill-accent-soft stroke-rule'} /></g>)}</SvgCanvas>
    <Readout>{names[layout]}：{fmt(bytes[layout], 3)} GB/rank；所有 rank 常驻账合计 {fmt(bytes[layout] * ranks, 3)} GB。这是指定精度的模型状态，不含激活、临时参数 gather、通信缓冲和 workspace。选中布局仍对每个坐标执行相同的全局梯度更新。</Readout>
  </LabFrame>
}
