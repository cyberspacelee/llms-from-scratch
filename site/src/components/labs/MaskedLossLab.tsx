import { useState } from 'react'
import { maskedLoss, type LossReduction } from '../../lib/masked-loss-model'
import { Controls, LabFrame, Readout, fmt, pen } from './Lab'

export default function MaskedLossLab() {
  const [targets, setTargets] = useState([0, 1, 0, 1, 0, 0]), [ignored, setIgnored] = useState([false, false, true, false, false, true]), [selected, setSelected] = useState(0), [reduction, setReduction] = useState<LossReduction>('mean')
  const result = maskedLoss(targets, ignored, reduction), row = result.rows[selected]
  const reduced = Array.isArray(result.result) ? `[${result.result.map(value => fmt(value, 6)).join(', ')}]` : fmt(result.result, 6)
  return <LabFrame title="六个位置、四个类别、四个有效目标">
    <Controls><label className="text-sm">选中位置<select value={selected} onChange={e => setSelected(Number(e.target.value))} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2">{targets.map((_, i) => <option key={i} value={i}>({Math.floor(i / 3)},{i % 3})</option>)}</select></label><label className="text-sm">此位置目标<select value={targets[selected]} onChange={e => setTargets(targets.map((v, i) => i === selected ? Number(e.target.value) : v))} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2">{[0, 1, 2, 3].map(c => <option key={c} value={c}>{c}</option>)}</select></label><label className="text-sm">归约方式<select value={reduction} onChange={e => setReduction(e.target.value as LossReduction)} className="mt-1 block h-9 w-full rounded border border-rule bg-paper px-2"><option value="none">none · 逐位置</option><option value="sum">sum · 有效和</option><option value="mean">mean · 有效平均</option></select></label></Controls>
    <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">{ignored.map((value, i) => <label key={i} className="flex items-center gap-2 text-sm"><input type="checkbox" checked={value} onChange={e => setIgnored(ignored.map((v, j) => i === j ? e.target.checked : v))} />忽略 ({Math.floor(i / 3)},{i % 3})</label>)}</div>
    <div className="mt-4 grid gap-4 sm:grid-cols-2">
      <svg viewBox="0 0 320 250" className={`${pen.canvas} mt-0! max-w-96`} role="img" aria-label={`选中位置的四类概率 ${row.probability.join(',')}`}>
        <text x="10" y="20">位置 ({Math.floor(selected / 3)},{selected % 3}) 的全部类别</text>
        <line x1="10" x2="310" y1="195" y2="195" className={pen.axis} />
        {row.probability.map((p, c) => <g key={c}><rect x={16 + c * 77} y={195 - p * 220} width="54" height={p * 220} className={c === targets[selected] ? 'fill-accent2' : 'fill-accent'} /><text x={43 + c * 77} y={183 - p * 220} textAnchor="middle" className={pen.mono}>{fmt(p)}</text><text x={43 + c * 77} y="220" textAnchor="middle">类 {c}</text></g>)}
        <text x="10" y="245">{ignored[selected] ? '忽略本行目标，不删除任何类别' : `目标 ${targets[selected]} · NLL ${fmt(row.rawLoss)}`}</text>
      </svg>
      <svg viewBox="0 0 320 250" className={`${pen.canvas} mt-0! max-w-96`} role="img" aria-label={`有效位置 ${result.valid} 个，有效和 ${result.total}`}>
        <text x="10" y="20">六个位置的损失项</text>
        {result.rows.map((r, i) => <g key={i}><rect x="10" y={33 + i * 34} width="300" height="29" rx="3" className={i === selected ? 'fill-info-soft stroke-info' : 'fill-sunken stroke-rule'} /><text x="22" y={53 + i * 34}>({Math.floor(i / 3)},{i % 3}) · {ignored[i] ? '忽略 → 0' : `y=${targets[i]} → ${fmt(r.loss, 6)}`}</text></g>)}
      </svg>
    </div>
    <Readout>logits=({row.logits.map(v => fmt(v, 6)).join(',')})<br />有效和={fmt(result.total, 6)} · 有效数量={result.valid}/6 · {reduction}={reduced}<br />{result.empty ? '原始 mean 为 0/0，没有定义；本章保护策略用 sum / max(1,有效数) 返回 0。这不决定是否执行 optimizer.step。' : `正确有效平均=${fmt(result.total / result.valid, 6)} · 错误地除以全部六个=${fmt(result.total / 6, 6)}`}</Readout>
  </LabFrame>
}
