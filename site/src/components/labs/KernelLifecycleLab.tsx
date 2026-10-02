import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { kernelLifecycle, kernelPhases } from '../../lib/kernel-lifecycle'
import Formula from './Formula'
import { Toggle, Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function KernelLifecycleLab() {
  const [tile, setTile] = useState(4)
  const [step, setStep] = useState(0)
  const [edge, setEdge] = useState(false)
  const state = kernelLifecycle(tile, step, edge)
  const maximum = state.rounds * 4
  const labels = [
    ['global → shared', `K 索引 ${state.start}–${state.start + state.width - 1}`, `有效装载 ${state.validLoads} · 补零 ${state.paddingSlots}`],
    ['发布屏障 · __syncthreads()', '全块当前轮装载完成', '随后读取 shared 才有先写后读保证'],
    ['shared → 寄存器累计', `累计 K 位置数 ${state.completedK} / 7`, `C(${state.row},${state.col}) 暂存值 ${state.sample}`],
    ['回收屏障 · __syncthreads()', '全块当前轮读取完成', '随后才能覆盖同一组 shared 数组'],
    [state.done ? '寄存器 → global C' : '下一段 K / 最终写回', state.done ? `写回 ${state.validOutputs} 个有效输出` : '还有 K 则回到装载，全部完成才写回', state.done ? '其余逻辑输出不执行 store' : '两个输入数组，未做拷贝与计算重叠'],
  ]
  return <LabFrame title="分块计算与资源生命周期">
    <Controls>
      <Range label="tile 边长" value={tile} min={1} max={4} onChange={value => { setTile(value); setStep(0) }} />
      <Range label="执行步骤" value={step} min={0} max={maximum} onChange={setStep} />
      <Toggle label="右下边缘输出块" checked={edge} onChange={value => { setEdge(value); setStep(0) }} />
    </Controls>
    <div className="mt-3 flex items-center gap-3">
      <button type="button" title="上一步" aria-label="上一步" disabled={step === 0} onClick={() => setStep(step - 1)} className="h-9 w-9 rounded border border-rule bg-paper text-lg disabled:opacity-40">←</button>
      <output className="min-w-0 flex-1 text-center text-sm">第 {state.round + 1}/{state.rounds} 轮 · {kernelPhases[state.phase]}</output>
      <button type="button" title="下一步" aria-label="下一步" disabled={step === maximum} onClick={() => setStep(step + 1)} className="h-9 w-9 rounded border border-rule bg-paper text-lg disabled:opacity-40">→</button>
    </div>
    <SvgCanvas viewBox="0 0 340 642" className={`${pen.canvas} max-w-96`} role="img" aria-label={`当前阶段 ${kernelPhases[state.phase]}，每块 shared ${state.sharedBytes} 字节，${state.accumulatorSlots} 个累计槽，样本累计值 ${state.sample}`}>
      <path d="M310 379H330V8H160V22m-4-5 4 5 4-5" className={pen.guide} />
      {labels.map(([label, detail, note], index) => <g key={label}>
        {index > 0 && <path d={`M160 ${22 + (index - 1) * 106 + 78}v20m-4-5 4 5 4-5`} className={pen.axis} />}
        <rect x="10" y={22 + index * 106} width="300" height="78" rx="4" className={index === state.phase ? 'fill-accent-soft stroke-accent stroke-2' : 'fill-sunken stroke-rule'} />
        <text x="22" y={44 + index * 106} className={index === state.phase ? pen.textA : ''}>{label}</text>
        <text x="22" y={66 + index * 106}>{detail}</text>
        <text x="22" y={86 + index * 106} className="text-[11px]">{note}</text>
      </g>)}
      <text x="12" y="570">每块 shared：2 × {tile}² × 4 = {state.sharedBytes} B</text>
      <text x="12" y="595">累计槽：{tile}² = {state.accumulatorSlots} 个 float32 值</text>
      <text x="12" y="620" className="text-[11px]">槽数不等于编译器报告的总寄存器数</text>
    </SvgCanvas>
    <Readout><Formula>{state.completedK === 0 ? String.raw`C_{${state.row},${state.col}}=0` : String.raw`C_{${state.row},${state.col}}=\sum_{q=0}^{${state.completedK - 1}} A_{${state.row},q} B_{q,${state.col}}=${state.sample}`}</Formula> · {state.outputWritten ? '已写回' : '尚未写回'} · {state.phase === 0 ? '本轮装载阶段' : state.phase === 3 || state.done ? '读取已回收，可进入下一轮' : '未完成回收，禁止覆盖本轮 shared'} · 教学顺序，非 GPU 时间线</Readout>
  </LabFrame>
}
