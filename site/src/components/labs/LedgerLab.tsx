import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Range, Readout, pen } from './Lab'
import { resourceLedger } from '../../lib/systems-interactive-model'

export default function LedgerLab() {
  const [batch, setBatch] = useState(1), [length, setLength] = useState(5), [large, setLarge] = useState(false)
  const r = resourceLedger(batch, length, large), scale = Math.max(r.weights, r.kv)
  return <LabFrame title="容量、读取与运算分别记账">
    <Controls><Range label="请求数 B" value={batch} min={1} max={16} onChange={setBatch} /><Range label="已处理长度 T" value={length} min={1} max={large ? 16384 : 16} onChange={setLength} />
      <Toggle label="Llama 3 8B 形状" checked={large} onChange={value => { setLarge(value); setLength(value ? 8192 : 5) }} /></Controls>
    <SvgCanvas viewBox="0 0 280 150" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`权重 ${r.weights} 字节，KV ${r.kv} 字节`}>
      {[['权重容量', r.weights], ['KV 容量', r.kv]].map(([name, value], i) => <g key={name}><text x="8" y={24 + i * 66}>{name}</text><rect x="8" y={34 + i * 66} width={260 * Number(value) / scale} height="20" className={i ? 'fill-accent2' : 'fill-accent'} /></g>)}
    </SvgCanvas>
    <Readout>容量：权重 {r.weights.toLocaleString()} B；KV {r.kv.toLocaleString()} B<br />当步逻辑读取：权重 + KV = {r.read.toLocaleString()} B；新增 KV 写入 {r.write.toLocaleString()} B<br />decode：线性 {r.linearFlops.toLocaleString()} + 注意力 {r.attentionFlops.toLocaleString()} FLOP<br />同长度 causal prefill 主项：{r.prefillFlops.toLocaleString()} FLOP</Readout>
  </LabFrame>
}
