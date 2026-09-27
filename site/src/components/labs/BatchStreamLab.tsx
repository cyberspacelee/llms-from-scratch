import { useState } from 'react'
import { batchPlan } from '../../lib/batch-plan'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

export default function BatchStreamLab() {
  const [samples, setSamples] = useState(5)
  const [batchSize, setBatchSize] = useState(2)
  const [dropLast, setDropLast] = useState(false)
  const plan = batchPlan(samples, batchSize, dropLast)
  return <LabFrame title="末尾小批次与平均损失的分母" hint="样本损失 (2i−3)² · 固定顺序">
    <Controls>
      <Range label="样本数" value={samples} min={1} max={10} onChange={setSamples} />
      <Range label="batch size" value={batchSize} min={1} max={8} onChange={setBatchSize} />
      <label className="flex h-8 items-center gap-2 text-sm"><input type="checkbox" checked={dropLast} onChange={e => setDropLast(e.target.checked)} className="accent-accent" />drop_last</label>
    </Controls>
    <svg viewBox="0 0 360 466" className={pen.canvas} role="img" aria-label={`${plan.batches.length} 个批次，保留 ${plan.retained} 个样本，丢弃 ${plan.dropped} 个样本`}>
      <text x="12" y="24">数据集：样本编号与各自损失</text>
      {Array.from({ length: 10 }, (_, i) => <g key={i}>
        <rect x={12 + i * 34} y="38" width="30" height="52" rx="2" className={i >= samples ? 'fill-none stroke-rule' : i >= plan.retained ? 'fill-accent2-soft stroke-accent2' : 'fill-accent-soft stroke-accent'} />
        <text x={27 + i * 34} y="58" textAnchor="middle" className="text-[11px]">{i < samples ? i : '—'}</text>
        <text x={27 + i * 34} y="79" textAnchor="middle" className="text-[10px]">{i < samples ? (2 * i - 3) ** 2 : ''}</text>
      </g>)}
      <text x="12" y="119">DataLoader → {plan.batches.length} 个实际批次</text>
      {Array.from({ length: 10 }, (_, i) => {
        const b = plan.batches[i], x = 12 + i % 2 * 174, y = 135 + Math.floor(i / 2) * 58
        return <g key={i}>
          <rect x={x} y={y} width="162" height="48" rx="2" className={b ? 'fill-info-soft stroke-info' : 'fill-none stroke-rule'} />
          <text x={x + 8} y={y + 18} className="text-[11px]">{b ? `batch ${i} · n=${b.indices.length} · mean=${b.mean.toFixed(2)}` : '—'}</text>
          <text x={x + 8} y={y + 37} className="text-[10px]">{b ? `[${b.indices.join(', ')}]` : ''}</text>
        </g>
      })}
      <text x="12" y="450" className={pen.muted}>橙色：丢弃样本；空框：没有这样的批次</text>
    </svg>
    <Readout>批次 {plan.batches.length} · 保留 {plan.retained} · 丢弃 {plan.dropped}<br />
      {plan.weighted === null ? '没有有效批次，不能计算均值或执行参数更新' : `按样本平均 ${plan.weighted.toFixed(6)} · 批均值直接平均 ${plan.unweighted!.toFixed(6)}`}
    </Readout>
  </LabFrame>
}
