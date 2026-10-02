import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { moeDispatch } from '../../lib/advanced-interactive-model'

export default function MoeDispatchLab() {
  const [step, setStep] = useState(0), [token, setToken] = useState(0), [crowded, setCrowded] = useState(false), [capacity, setCapacity] = useState(2), [drop, setDrop] = useState(false)
  const r = moeDispatch(crowded, capacity, drop), routes = r.routes.filter(v => v.token === token)
  return <LabFrame title="三行输入分发后仍回到原行">
    <Controls><Range label="阶段" value={step} min={0} max={3} onChange={setStep} format={v => ['路由', 'dispatch', '专家计算', '加权 gather'][v]} /><Range label="原 token 行" value={token} min={0} max={2} onChange={setToken} /><Range label="专家容量" value={capacity} min={1} max={3} onChange={setCapacity} /><Toggle label="全部选择专家 0、1" checked={crowded} onChange={value => setCrowded(value)} /><Toggle label="丢弃超容量贡献" checked={drop} onChange={value => setDrop(value)} /></Controls>
    <SvgCanvas viewBox="0 0 280 250" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`专家命中 ${r.counts.join(', ')}，原行 ${token} 的输出 ${r.output[token].join(', ')}`}>
      {[0, 1, 2, 3].map(expert => <g key={expert}><rect x="8" y={8 + expert * 58} width="264" height="48" className={routes.some(v => v.expert === expert) ? 'fill-accent/15 stroke-accent' : 'fill-sunken stroke-rule'} /><text x="16" y={27 + expert * 58}>专家 {expert} · {r.counts[expert]} 次调用</text><text x="16" y={46 + expert * 58} className={pen.mono}>{r.routes.filter(v => v.expert === expert).map(v => `行${v.token}${v.dropped ? '×' : v.overflow ? '待' : ''}`).join(' / ')}</text></g>)}
    </SvgCanvas><Readout>token {token} 输入=[{r.inputs[token].join(', ')}]<br />{routes.map(v => `专家${v.expert}：${step < 2 ? '持有原行号' : `[${v.expertOutput.join(',')}]`} × ${fmt(v.weight)}${v.dropped ? '（已丢弃）' : v.overflow ? '（额外批次等待后执行）' : ''}`).join('；')}<br />{step === 3 ? `gather 回原行=[${r.output[token].map(v => fmt(v)).join(', ')}]；未丢弃参照=[${r.reference[token].map(v => fmt(v)).join(', ')}]` : '返回时保留原行号，两个贡献相加'}</Readout>
  </LabFrame>
}
