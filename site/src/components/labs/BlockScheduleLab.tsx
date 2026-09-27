import { useState } from 'react'
import { schedulerFrames } from '../../lib/gpu-models'
import { LabFrame, Range, Readout, pen } from './Lab'

export default function BlockScheduleLab() {
  const [step, setStep] = useState(1)
  const frame = schedulerFrames[step]
  const resident = frame.sms.flat()
  const pending = Array.from({ length: 6 }, (_, i) => i).filter(i => !resident.includes(i) && !frame.done.includes(i))
  return <LabFrame title="从待执行 block 到可运行 warp" hint="一种合法调度 · 步骤不代表时间">
    <Range label="调度步骤" value={step} min={0} max={schedulerFrames.length - 1} onChange={setStep} />
    <svg viewBox="0 0 360 436" className={pen.canvas} role="img" aria-label={`步骤 ${step}，待执行 ${pending.length}，驻留 ${resident.length}，完成 ${frame.done.length}`}>
      <text x="12" y="25">kernel → grid → 6 blocks → 每块 2 warps</text>
      <text x="12" y="57">待执行</text>
      {Array.from({ length: 6 }, (_, block) => <g key={block}>
        <rect x={12 + block * 56} y="69" width="48" height="30" rx="2" className={pending.includes(block) ? 'fill-sunken stroke-rule-strong' : 'fill-none stroke-rule'} />
        <text x={36 + block * 56} y="89" textAnchor="middle" className={pending.includes(block) ? '' : pen.muted}>{pending.includes(block) ? `B${block}` : '—'}</text>
      </g>)}
      {frame.sms.map((blocks, sm) => <g key={sm}>
        <rect x={12 + sm * 176} y="121" width="160" height="240" rx="4" className="fill-none stroke-rule-strong" />
        <text x={24 + sm * 176} y="146">SM {sm} · 2 个教学槽位</text>
        {[0, 1].map(slot => {
          const block = blocks[slot]
          return <g key={slot}>
            <rect x={24 + sm * 176} y={160 + slot * 96} width="136" height="84" rx="2" className={block === undefined ? 'fill-none stroke-rule' : 'fill-accent-soft stroke-accent'} />
            <text x={32 + sm * 176} y={179 + slot * 96}>{block === undefined ? '空槽位' : `block ${block} · 64 threads`}</text>
            {block !== undefined && [0, 1].map(warp => {
              const waiting = step === 2 && block === 0 && warp === 0
              const issued = step === 2 && block === 2 && warp === 0
              return <g key={warp}>
                <rect x={32 + sm * 176} y={187 + slot * 96 + warp * 24} width="120" height="20" rx="2" className={waiting ? 'fill-accent2-soft stroke-accent2' : issued ? 'fill-info-soft stroke-info' : 'fill-raised stroke-rule'} />
                <text x={38 + sm * 176} y={201 + slot * 96 + warp * 24} className="text-[11px]">w{warp} · {waiting ? '等待数据' : issued ? '本步发射' : '可运行'}</text>
              </g>
            })}
          </g>
        })}
      </g>)}
      <text x="12" y="388">已完成：{frame.done.length ? frame.done.map(i => `B${i}`).join('、') : '无'}</text>
      <text x="12" y="417" className={pen.muted}>等待数据的 warp 仍占用驻留资源。</text>
    </svg>
    <Readout>待执行 {pending.length} · 驻留 {resident.length} · 完成 {frame.done.length}<br />{frame.note}</Readout>
  </LabFrame>
}
