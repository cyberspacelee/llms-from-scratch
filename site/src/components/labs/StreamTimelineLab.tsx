import SvgCanvas from './SvgCanvas'
import { useId, useState } from 'react'
import { streamSchedule } from '../../lib/gpu-models'
import { Select, Toggle, Arrow, Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function StreamTimelineLab() {
  const id = useId()
  const [twoStreams, setTwoStreams] = useState(true)
  const [waitEvent, setWaitEvent] = useState(true)
  const [producer, setProducer] = useState(5)
  const [independent, setIndependent] = useState(3)
  const [consumer, setConsumer] = useState(2)
  const schedule = streamSchedule(producer, independent, consumer, twoStreams, waitEvent)
  const x = (time: number) => 85 + time / schedule.total * 340
  return <LabFrame title="跨 stream 的依赖与时间线" hint="逻辑时间单位 · 假设资源允许重叠">
    <Controls>
      <Select label="执行队列" value={twoStreams ? 'two' : 'one'} onChange={event => setTwoStreams(event.target.value === 'two')}>
          <option value="one">同一 stream</option><option value="two">两个 stream</option>
        </Select>
      <Toggle label="C 等待 A 的完成 event" checked={waitEvent} onChange={value => setWaitEvent(value)} />
      <Range label="A 生产数据" value={producer} min={1} max={10} onChange={setProducer} />
      <Range label="B 独立任务" value={independent} min={1} max={10} onChange={setIndependent} />
      <Range label="C 消费 A 的结果" value={consumer} min={1} max={10} onChange={setConsumer} />
    </Controls>
    <SvgCanvas viewBox="0 0 440 230" className={pen.canvas} role="img" aria-label={`完成时刻 ${schedule.total}，${schedule.ordered ? '依赖已建立' : '缺少跨 stream 依赖'}`}>
      <defs><Arrow id={`${id}-event`} className="fill-info" /></defs>
      {[0, 1].map(stream => <g key={stream}>
        <text x="4" y={91 + stream * 70}>stream {stream}</text>
        <line x1="85" y1={100 + stream * 70} x2="428" y2={100 + stream * 70} className={pen.axis} />
      </g>)}
      {schedule.tasks.map(task => <g key={task.label}>
        <rect x={x(task.start)} y={65 + task.stream * 70} width={x(task.end) - x(task.start)} height="30" rx="2"
          className={task.label === 'A' ? 'fill-accent-soft stroke-accent' : task.label === 'B' ? 'fill-info-soft stroke-info' : 'fill-accent2-soft stroke-accent2'} />
        <text x={(x(task.start) + x(task.end)) / 2} y={85 + task.stream * 70} textAnchor="middle">{task.label}</text>
      </g>)}
      {twoStreams && waitEvent && <path d={`M ${x(producer)} 95 V 117 H ${x(schedule.tasks[2].start)} V 130`} className={pen.guide} markerEnd={`url(#${id}-event)`} />}
      {[0, schedule.total / 2, schedule.total].map(time => <text key={time} x={x(time)} y="203" textAnchor="middle" className={pen.mono}>{time}</text>)}
      <text x="255" y="226" textAnchor="middle" className={pen.muted}>逻辑时间（示意）</text>
    </SvgCanvas>
    <Readout>A 完成于 {producer} · C 开始于 {schedule.tasks[2].start} · 总完成时刻 {schedule.total}<br />
      {schedule.ordered ? 'A → C 依赖已建立' : schedule.safe ? '缺少 A → C 依赖；当前时长碰巧没有提前读取' : '缺少 A → C 依赖；C 在 A 完成前读取结果'}
    </Readout>
  </LabFrame>
}
