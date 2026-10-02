import SvgCanvas from './SvgCanvas'
import { useState } from 'react'

import { Select, Controls, LabFrame, Range, Readout, pen } from './Lab'


export default function GqaMapLab() {
  const [kv, setKv] = useState(1), [query, setQuery] = useState(0), [length, setLength] = useState(2)
  const key = Math.floor(query / (2 / kv))
  return <LabFrame title="两个 Q 头映射到哪些 KV 参数" hint="B=L=1，H_q=2，D_h=2；缓存不保存 Q">
    <Controls><Range label="选择 Q 头" min={0} max={1} value={query} onChange={setQuery} /><Select label="KV 头数" value={kv} onChange={e => setKv(Number(e.target.value))} ><option value="1">1 · 共用 KV</option><option value="2">2 · 独立 KV</option></Select><Range label="缓存长度 T" min={1} max={8} value={length} onChange={setLength} /></Controls>
    <SvgCanvas viewBox="0 0 360 170" className={`${pen.canvas} max-w-110!`} role="img" aria-label={`Q${query} 读取 KV${key}`}>
      {[0, 1].map(i => <g key={i}><rect x="10" y={24 + i * 74} width="88" height="42" className={query === i ? 'fill-accent-soft stroke-accent' : 'fill-paper stroke-rule'} /><text x="24" y={50 + i * 74}>Q 头 {i}</text><path d={`M98 ${45 + i * 74}L245 ${kv === 1 ? 80 : 45 + i * 74}`} className={query === i ? pen.a : pen.axis} /></g>)}
      {Array.from({ length: kv }, (_, i) => <g key={i}><rect x="245" y={kv === 1 ? 59 : 24 + i * 74} width="105" height="42" className={key === i ? 'fill-accent-soft stroke-accent' : 'fill-paper stroke-rule'} /><text x="257" y={kv === 1 ? 85 : 50 + i * 74}>KV 组 {i}</text></g>)}
    </SvgCanvas>
    <Readout>2×B×L×T×H_kv×D_h = 2×1×1×{length}×{kv}×2 = {4 * length * kv} 个缓存元素。Q/O 参数仍为 16；K/V 为 {8 * kv}。共享 KV 不强制两个 Q 使用相同 softmax 权重。</Readout>
  </LabFrame>
}
