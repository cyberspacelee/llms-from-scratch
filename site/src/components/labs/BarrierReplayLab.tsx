import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { barrierTrace, type BarrierScenario } from '../../lib/gpu-replay-model'
import { Select, Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function BarrierReplayLab() {
  const [scenario, setScenario] = useState<BarrierScenario>('protected')
  const [step, setStep] = useState(0)
  const trace = barrierTrace(scenario)
  const state = trace[step]
  return <LabFrame title="shared 的写入、发布、消费与覆盖" hint="一个可能的线程交错，非设备时间线">
    <Controls>
      <Select label="同步场景" value={scenario} onChange={event => { setScenario(event.target.value as BarrierScenario); setStep(0) }} >
        <option value="protected">发布与回收都正确</option><option value="publish">缺少发布屏障</option><option value="reclaim">缺少回收屏障</option>
      </Select>
      <Range label="执行步骤" min={0} max={trace.length - 1} value={step} onChange={setStep} />
    </Controls>
    <SvgCanvas viewBox={`0 0 320 ${155 + trace.length * 42}`} className={`${pen.canvas} max-w-96`} role="img" aria-label={`步骤 ${step}，shared 值 ${state.slot}，${state.label}`}>
      <rect x="20" y="10" width="280" height="48" rx="3" className="fill-accent-soft stroke-accent" />
      <text x="160" y="39" textAnchor="middle">shared 槽 = {state.slot} · 本轮应读 7</text>
      {trace.map((frame, index) => <g key={index}>
        <rect x="20" y={78 + index * 42} width="280" height="34" rx="3" className={index === step ? 'fill-accent2-soft stroke-accent2' : 'fill-sunken stroke-rule'} />
        <text x="30" y={100 + index * 42}>{index}. {frame.label}</text>
      </g>)}
      <text x="20" y={135 + trace.length * 42} className={pen.muted}>读取者：{state.reads.map(read => `${read.actor.slice(-1)}=${read.value}`).join('，') || '尚未读取'}</text>
    </SvgCanvas>
    <Readout>{state.actor} · {state.label}<br />{state.reads.map(read => `${read.actor} 读到 ${read.value}：${read.valid ? '本轮值正确' : '不是本轮值'}`).join('；') || '尚未消费 shared'}<br />{scenario === 'protected' ? '全部消费者读完后，生产者才覆盖。' : '这里只展示一个允许出现的错误交错；缺少依赖时不能保证其他交错正确。'}</Readout>
  </LabFrame>
}
