import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { timingBoundary, timingScopes } from '../../lib/gpu-replay-model'
import { Select, Controls, LabFrame, Readout, pen, useWidth } from './Lab'

export default function TimingBoundaryLab() {
  const [scope, setScope] = useState(1)
  const [ref, width] = useWidth<HTMLDivElement>(320)
  const state = timingBoundary(scope)
  const canvas = Math.max(300, Math.min(640, width))
  const x = (time: number) => 72 + time / state.total * (canvas - 92)
  return <LabFrame title="相同工作，四种计时范围" hint="教学时钟：提交 0.4、准备 1、A=5、B=3、C=2；非实测">
    <div ref={ref}>
      <Controls><Select label="计时对象" value={scope} onChange={event => setScope(Number(event.target.value))} >{timingScopes.map((label, index) => <option key={label} value={index}>{label}</option>)}</Select></Controls>
      <SvgCanvas viewBox={`0 0 ${canvas} 258`} className={pen.canvas} role="img" aria-label={`${timingScopes[scope]}，区间 ${state.start} 到 ${state.end}`}>
        <rect x={x(state.start)} y="14" width={x(state.end) - x(state.start)} height="188" className="fill-accent2/10 stroke-accent2" />
        {['CPU', 'stream 0', 'stream 1'].map((label, row) => <g key={label}><text x="2" y={51 + row * 62}>{label}</text><line x1="72" x2={canvas - 16} y1={65 + row * 62} y2={65 + row * 62} className={pen.axis} /></g>)}
        {state.segments.map(segment => <g key={segment.label}><rect x={x(segment.start)} y={30 + segment.row * 62} width={x(segment.end) - x(segment.start)} height="30" rx="2" className="fill-accent-soft stroke-accent" />{segment.end - segment.start >= 1 && <text x={(x(segment.start) + x(segment.end)) / 2} y={50 + segment.row * 62} textAnchor="middle">{segment.label}</text>}</g>)}
        {[0, 1.4, 6.4, 8.4].map(time => <text key={time} x={x(time)} y="223" textAnchor="middle" className={pen.mono}>{time}</text>)}
        <text x={canvas / 2} y="248" textAnchor="middle" className={pen.muted}>同一教学时钟上的逻辑时间</text>
      </SvgCanvas>
      <Readout>{timingScopes[scope]}：{state.start.toFixed(1)} → {state.end.toFixed(1)}，差值 {state.duration.toFixed(1)}<br />A/B 在准备后开始，C 等到两者完成；CPU 提交结束时结果尚未就绪。橙色区间是当前分母，各种范围不可混用。</Readout>
    </div>
  </LabFrame>
}
