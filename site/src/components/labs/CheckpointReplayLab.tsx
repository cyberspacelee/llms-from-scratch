import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { checkpointComparison } from '../../lib/training-trace-model'
import { Select, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'


export default function CheckpointReplayLab() {
  const [omit, setOmit] = useState<'none' | 'optimizer' | 'rng' | 'cursor'>('none'), [step, setStep] = useState(0)
  const result = checkpointComparison(omit), reference = result.reference[step], restored = result.resumed[step]
  const firstMismatch = result.reference.findIndex((row, i) => row.document !== result.resumed[i].document || row.w !== result.resumed[i].w)
  return <LabFrame title="漏存状态后，哪一步首先偏离" hint="独立缩小例：单参数、三篇目标、5 步后保存，再运行 7 步">
    <Controls><Select label="恢复状态" value={omit} onChange={e => setOmit(e.target.value as typeof omit)} ><option value="none">全部恢复</option><option value="optimizer">漏掉 m、v、优化器步数</option><option value="rng">漏掉 RNG</option><option value="cursor">漏掉当前顺序与游标</option></Select><Range label="恢复后的更新序号" min={0} max={6} value={step} onChange={setStep} /></Controls>
    <SvgCanvas viewBox="0 0 360 230" className={`${pen.canvas} max-w-110!`} role="img" aria-label="连续和恢复路径的七步参数轨迹">
      <line x1="30" y1="180" x2="335" y2="180" className={pen.axis} />
      <polyline points={result.reference.map((r, i) => `${40 + i * 46},${180 - r.w * 180}`).join(' ')} className={pen.a} /><polyline points={result.resumed.map((r, i) => `${40 + i * 46},${180 - r.w * 180}`).join(' ')} className={pen.b} />
      {result.reference.map((r, i) => <g key={i}><circle cx={40 + i * 46} cy={180 - r.w * 180} r={i === step ? 6 : 3} className="fill-accent" /><text x={40 + i * 46} y="203" textAnchor="middle">+{i + 1}</text></g>)}
      <text x="30" y="222" className={pen.muted}>实线：连续训练；虚线：恢复训练</text>
    </SvgCanvas>
    <Readout>连续：文档 {reference.document}，loss={fmt(reference.loss, 6)}，w={fmt(reference.w, 6)}<br />恢复：文档 {restored.document}，loss={fmt(restored.loss, 6)}，w={fmt(restored.w, 6)}<br />{firstMismatch < 0 ? '七步参数与输入事件完全相同。' : `第一处偏离：恢复后第 ${firstMismatch + 1} 步。`} 重新初始化 RNG 不保证下一轮 shuffle 相同。</Readout>
  </LabFrame>
}
