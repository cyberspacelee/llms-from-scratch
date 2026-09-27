import { useState } from 'react'
import { residency, type ResidencyLimits } from '../../lib/gpu-models'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function OccupancyLab() {
  const [threads, setThreads] = useState(256)
  const [registers, setRegisters] = useState(32)
  const [shared, setShared] = useState(16)
  const [limits, setLimits] = useState<ResidencyLimits>({ warps: 64, registers: 65536, sharedKiB: 96, blocks: 32 })
  const result = residency(threads, registers, shared, limits)
  const capacities = [
    ['warps', 'SM warp 上限', 1, 64, 1],
    ['registers', 'SM 32-bit 寄存器数', 1024, 262144, 1024],
    ['sharedKiB', 'SM shared memory（KiB）', 1, 256, 1],
    ['blocks', 'SM block 上限', 1, 32, 1],
  ] as const
  return <LabFrame title="一个 SM 能驻留多少 block" hint="教学资源模型 · 未计分配粒度">
    <Controls>
      <Range label="每块线程数" value={threads} min={32} max={1024} step={32} onChange={setThreads} />
      <Range label="每线程寄存器数" value={registers} min={8} max={128} step={8} onChange={setRegisters} />
      <Range label="每块 shared memory（KiB）" value={shared} min={0} max={128} step={4} onChange={setShared} />
    </Controls>
    <fieldset className="mt-4 border-t border-rule pt-3">
      <legend className="text-xs font-semibold text-muted">教学 SM 容量</legend>
      <div className="grid grid-cols-2 gap-3">
        {capacities.map(([key, label, min, max, step]) => <label key={key} className="text-xs">
          <span className="block">{label}</span>
          <input type="number" min={min} max={max} step={step} value={limits[key]}
            onChange={event => {
              const value = Number(event.target.value)
              if (Number.isInteger(value) && value >= min && value <= max) setLimits({ ...limits, [key]: value })
            }} className="mt-1 w-full rounded-sm border border-rule-strong bg-paper px-2 py-1 font-mono" />
        </label>)}
      </div>
    </fieldset>
    <svg viewBox="0 0 440 240" className={pen.canvas} role="img" aria-label={`驻留 ${result.blocks} 个 block，${result.activeWarps} 个 warp，总容量 ${limits.warps} 个 warp`}>
      <text x="12" y="18">驻留 warp</text>
      {Array.from({ length: limits.warps }, (_, index) => <rect key={index}
        x={12 + index % 8 * 53} y={32 + Math.floor(index / 8) * 24} width="45" height="16" rx="2"
        className={index >= result.activeWarps ? 'fill-sunken stroke-rule' : Math.floor(index / result.warpsPerBlock) % 2 ? 'fill-info' : 'fill-accent'} />)}
      <text x="12" y="236" className={pen.muted}>绿色与蓝色区分 block；空格是未驻留的 warp 容量。</text>
    </svg>
    <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
      {result.bounds.map(([name, count]) => <div key={name} className="flex justify-between border-b border-rule py-1">
        <dt>{name} 允许的 block</dt><dd className="m-0 font-mono">{count}</dd>
      </div>)}
    </dl>
    <Readout>驻留 {result.blocks} block · {result.activeWarps}/{limits.warps} warp · 理论 occupancy {(100 * result.occupancy).toFixed(1)}%{result.blocks === 0 && ' · 当前资源无法容纳一个 block'}</Readout>
  </LabFrame>
}
