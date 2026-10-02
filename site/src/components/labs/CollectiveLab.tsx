import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Select, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { parallelMatrix } from '../../lib/systems-interactive-model'

export default function CollectiveLab() {
  const [mode, setMode] = useState<'column' | 'row'>('column'), [step, setStep] = useState(0)
  const r = parallelMatrix(mode)
  return <LabFrame title="两张卡算同一矩阵">
    <Controls><Select label="切分轴" value={mode} onChange={e => setMode(e.target.value as 'column' | 'row')} ><option value="column">输出列：拼接</option><option value="row">输入行：部分和</option></Select><Range label="步骤" value={step} min={0} max={2} onChange={setStep} format={v => ['分发', '局部计算', '合并'][v]} /></Controls>
    <SvgCanvas viewBox="0 0 280 200" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`${mode} 切分，两 rank 输出 ${r.output.join(', ')}`}>
      {[0, 1].map(rank => <g key={rank}><rect x="8" y={8 + rank * 72} width="264" height="60" className="fill-sunken stroke-rule" /><text x="18" y={30 + rank * 72}>rank {rank}：{mode === 'column' ? `输出坐标 ${rank * 2},${rank * 2 + 1}` : `输入坐标 ${rank * 2},${rank * 2 + 1}`}</text><text x="18" y={53 + rank * 72} className={pen.mono}>{step ? r.local[rank].join(', ') : '取得权重切片与所需输入'}</text></g>)}<text x="8" y="182" className={pen.textA}>{step === 2 ? `${mode === 'column' ? '拼接' : '相加'} → ${r.output.join(', ')}` : '局部结果保持 rank 身份'}</text>
    </SvgCanvas><Readout>完整 xW=[{r.reference.join(', ')}]；合并误差={fmt(Math.max(...r.output.map((v, i) => Math.abs(v - r.reference[i]))))}<br />两 rank 都要完整输出时：{mode === 'column' ? 'all-gather' : 'all-reduce'}；每 rank 发/收各 {r.sentBytes} B（半精度，教学两卡）</Readout>
  </LabFrame>
}
