import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import Formula from './Formula'
import { Select, Controls, LabFrame, Readout, pen } from './Lab'

export default function BankConflictLab() {
  const [mode, setMode] = useState('32')
  const words = Array.from({ length: 32 }, (_, lane) => mode === 'broadcast' ? 7 : lane * Number(mode))
  const banks = Array.from({ length: 32 }, (_, bank) => new Set(words.filter(word => word % 32 === bank)))
  const requests = Math.max(...banks.map(bank => bank.size))
  return <LabFrame title="同一个 bank，是否同一个地址" hint="32 banks · 32-bit 读取">
    <Controls><Select label="访问方式" value={mode} onChange={event => setMode(event.target.value)}>
        <option value="1">连续读取：步长 1</option><option value="32">读取 32 列 tile 的一列</option><option value="33">读取 33 列 tile 的一列</option><option value="broadcast">所有 lane 读取元素 7</option>
      </Select></Controls>
    <SvgCanvas viewBox="0 0 360 276" className={pen.canvas} role="img" aria-label={`每个 bank 最多 ${requests} 个不同地址`}>
      <text x="12" y="20">bank 编号：不同读取地址数</text>
      {banks.map((bank, index) => {
        const x = 12 + index % 4 * 88, y = 36 + Math.floor(index / 4) * 29
        return <g key={index}><rect x={x} y={y} width="82" height="24" rx="3" className={bank.size > 1 ? 'fill-accent2-soft stroke-rule' : bank.size ? 'fill-accent-soft stroke-rule' : 'fill-sunken stroke-rule'} /><text x={x + 41} y={y + 16} textAnchor="middle" className={pen.mono}>{index}: {bank.size}</text></g>
      })}
    </SvgCanvas>
    <Readout><Formula>{String.raw`\operatorname{bank}(w)=w\bmod32`}</Formula> · 最大不同地址数 {requests} · {mode === 'broadcast' ? '32 个 lane 同址读取，广播，无 bank 冲突' : `${requests} 路教学冲突模型`}</Readout>
  </LabFrame>
}
