import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { epilogue } from '../../lib/gpu-replay-model'
import { Toggle, Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function FusedEpilogueLab() {
  const [row, setRow] = useState(0)
  const [col, setCol] = useState(0)
  const [bias, setBias] = useState(1)
  const [stage, setStage] = useState(0)
  const [fused, setFused] = useState(true)
  const state = epilogue(row, col, bias, fused, stage)
  const values = [state.c, state.shifted, state.result]
  return <LabFrame title="完整累计后再加偏置与 ReLU" hint="沿用 5×7 乘 7×6；只计 C、偏置后结果、D 的逻辑读写">
    <Controls>
      <Range label="输出行 i" value={row} min={0} max={4} onChange={setRow} />
      <Range label="输出列 j" value={col} min={0} max={5} onChange={setCol} />
      <Range label="偏置 bⱼ" value={bias} min={-5} max={5} onChange={setBias} />
      <Range label="阶段" value={stage} min={0} max={2} onChange={setStage} format={value => ['累计完成', '加偏置', 'ReLU 与写回'][value]} />
      <Toggle label="融合写回" checked={fused} onChange={value => setFused(value)} />
    </Controls>
    <SvgCanvas viewBox="0 0 320 292" className={`${pen.canvas} max-w-96`} role="img" aria-label={`输出 ${state.result}，当前逻辑读写 ${state.bytes} 字节`}>
      {['完整 K 累计', '加偏置', 'ReLU'].map((label, index) => <g key={label}>
        <rect x="18" y={14 + index * 83} width="284" height="61" rx="3" className={index === stage ? 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} />
        <text x="30" y={37 + index * 83}>{label} = {values[index]}</text>
        <text x="30" y={61 + index * 83} className={pen.muted}>{fused ? index === 2 ? '写回 D：120 B' : '值保留在寄存器' : index === 0 ? '写 C：120 B' : '读前结果、写新结果：240 B'}</text>
        {index < 2 && <path d={`M160 ${75 + index * 83}v20m-4-5 4 5 4-5`} className={pen.axis} />}
      </g>)}
      <text x="18" y="284">此时累计读写：{state.bytes} B</text>
    </SvgCanvas>
    <Readout>C[{row},{col}]={state.c} → +{bias} → max(0, {state.shifted})={state.result}<br />30 个输出的最终逻辑读写：{state.finalBytes} B；{fused ? '融合为 4MN' : '三个独立 kernel 为 20MN'}。不计 A/B、偏置读取、缓存、launch 或真实耗时。</Readout>
  </LabFrame>
}
