import { useState } from 'react'
import { maskedLoss, type LossReduction } from '../../lib/masked-loss-model'
import Formula from './Formula'
import { Controls, LabFrame, Readout, pen } from './Lab'

export default function MaskedLossLab() {
  const [targets, setTargets] = useState([0, 2, 1])
  const [ignored, setIgnored] = useState([false, false, true])
  const [reduction, setReduction] = useState<LossReduction>('mean')
  const result = maskedLoss(targets, ignored, reduction)
  const reduced = Array.isArray(result.result) ? `[${result.result.map(value => value.toFixed(6)).join(', ')}]` : result.result.toFixed(6)
  return <LabFrame title="从 logits 到有效位置交叉熵">
    <Controls>
      {targets.map((target, row) => <label key={row} className="text-sm">位置 {row} 目标
        <select value={target} onChange={event => setTargets(targets.map((value, index) => index === row ? Number(event.target.value) : value))} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2">
          {[0, 1, 2].map(value => <option key={value} value={value}>类别 {value}</option>)}
        </select>
      </label>)}
      <label className="text-sm">归约方式<select value={reduction} onChange={event => setReduction(event.target.value as LossReduction)} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2"><option value="none">none · 逐位置</option><option value="sum">sum · 有效和</option><option value="mean">mean · 有效平均</option></select></label>
    </Controls>
    <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2">
      {ignored.map((value, row) => <label key={row} className="flex items-center gap-2 text-sm"><input type="checkbox" checked={value} onChange={event => setIgnored(ignored.map((entry, index) => index === row ? event.target.checked : entry))} className="accent-accent" />忽略位置 {row}</label>)}
    </div>
    <svg viewBox="0 0 320 510" className={`${pen.canvas} max-w-96`} role="img" aria-label={`有效位置 ${result.valid} 个，损失 ${reduced}`}>
      {result.rows.map((row, index) => <g key={index}>
        <text x="10" y={22 + index * 144}>位置 {index} · logits ({row.logits.join(', ')})</text>
        <text x="10" y={43 + index * 144} className={ignored[index] ? pen.muted : pen.textA}>{ignored[index] ? '忽略 · 本行损失置零' : `目标 ${targets[index]} · 损失 ${row.loss.toFixed(6)}`}</text>
        {row.probability.map((probability, cls) => <g key={cls}>
          <rect x={20 + cls * 101} y={111 + index * 144 - probability * 60} width="60" height={probability * 60} className={ignored[index] ? 'fill-rule-strong' : targets[index] === cls ? 'fill-accent2' : 'fill-accent'} />
          <text x={50 + cls * 101} y={126 + index * 144} textAnchor="middle" className={pen.mono}>{cls}: {probability.toFixed(3)}</text>
        </g>)}
        <line x1="10" x2="307" y1={111 + index * 144} y2={111 + index * 144} className={pen.axis} />
      </g>)}
      <text x="10" y="454">有效和 {result.total.toFixed(6)}</text>
      <text x="10" y="477">mean 分母：有效位置数 {result.valid}</text>
      <text x="10" y="500" className={pen.muted}>橙色：目标概率 · 灰色：忽略位置</text>
    </svg>
    <Readout><Formula>{String.raw`\ell_i=\log\sum_c e^{z_{ic}}-z_{i,y_i}`}</Formula> · {reduction} = {reduced} · {result.empty ? '全部忽略：本实验返回 0；原始 mean 分母为 0' : `有效 ${result.valid}/3`}</Readout>
  </LabFrame>
}
