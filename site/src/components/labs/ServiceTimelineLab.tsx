import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { serviceMetrics } from '../../lib/systems-interactive-model'

export default function ServiceTimelineLab() {
  const [request, setRequest] = useState(0), [event, setEvent] = useState(0), [ttft, setTtft] = useState(0.7), [tpot, setTpot] = useState(0.2), [total, setTotal] = useState(1)
  const all = serviceMetrics(ttft, tpot, total), r = all.requests[request]
  const selected = Math.min(event, r.events.length + 1)
  const time = selected === 0 ? r.arrival : selected === r.events.length + 1 ? r.end : r.events[selected - 1][0]
  return <LabFrame title="从客户端事件读出延迟和合格产出" hint="正文固定轨迹；不是实机测量">
    <Controls><Range label="请求" value={request} min={0} max={3} onChange={v => { setRequest(v); setEvent(0) }} format={v => 'ABCD'[v]} /><Range label="事件" value={selected} min={0} max={r.events.length + 1} onChange={setEvent} format={v => v === 0 ? '发送' : v > r.events.length ? (r.complete ? '完成' : '取消') : `输出 ${v}`} /><Range label="TTFT 上限 / s" value={ttft} min={0.1} max={1.5} step={0.1} onChange={setTtft} /><Range label="TPOT 上限 / s" value={tpot} min={0.05} max={0.5} step={0.05} onChange={setTpot} /><Range label="总延迟上限 / s" value={total} min={0.5} max={2} step={0.1} onChange={setTotal} /></Controls>
    <SvgCanvas viewBox="0 0 280 195" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`选择请求 ${r.name} 时刻 ${time} s，是否满足 SLO ${r.qualified}`}>
      <line x1={32 + time * 150} x2={32 + time * 150} y1="4" y2="175" className={pen.guide} />
      {all.requests.map((row, i) => <g key={row.name}><text x="4" y={28 + i * 42}>{row.name}</text><line x1={32 + row.arrival * 150} x2={32 + row.end * 150} y1={23 + i * 42} y2={23 + i * 42} className={row.qualified ? pen.a : pen.axis} />{row.events.map((e, ei) => <g key={ei}><circle cx={32 + e[0] * 150} cy={23 + i * 42} r={request === i && selected === ei + 1 ? 6 : 4} className="fill-accent2" /><text x={32 + e[0] * 150} y={40 + i * 42} textAnchor="middle" className={pen.mono}>+{e[1]}</text></g>)}{!row.complete && <text x={32 + row.end * 150} y={27 + i * 42}>×</text>}</g>)}<text x="32" y="192">0</text><text x="257" y="192" textAnchor="end">1.5 s</text>
    </SvgCanvas><Readout>{r.name}：所选事件 t={fmt(time, 2)} s；TTFT={r.ttft === null ? '未定义' : fmt(r.ttft)}；ITL=[{r.itl.map(v => fmt(v)).join(', ')}]；TPOT={r.tpot === null ? '未定义' : fmt(r.tpot)}<br />总延迟={fmt(r.total)} s；{r.complete ? (r.qualified ? 'SLO 合格' : 'SLO 不合格') : '取消，仍进入 4 条到达分母'}<br />合格 {all.requests.filter(v => v.qualified).length}/4 到达；goodput={fmt(all.goodput)} 请求/s；{fmt(all.tokenGoodput)} token/s（窗口 1.5 s）</Readout>
  </LabFrame>
}
