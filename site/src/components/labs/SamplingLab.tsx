import { useState } from 'react'
import { rawProbabilities, sampleDistribution } from '../../lib/sampling-model'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'

const words = ['BOS', 'A', 'B', 'EOS']
const logits = [0.1, 0.4, 0.3, 0.2].map(Math.log)
/** Server and browser may differ in the last float digit; round drawing coordinates. */
const r = (value: number) => Math.round(value * 100) / 100

export default function SamplingLab() {
  const [temperature, setTemperature] = useState(1)
  const [topK, setTopK] = useState(4)
  const [topP, setTopP] = useState(1)
  const [minP, setMinP] = useState(0)
  const [penalty, setPenalty] = useState(1)
  const [repeated, setRepeated] = useState(true)
  const history = repeated ? [1] : []
  const result = sampleDistribution(logits, { temperature, topK, topP, minP, penalty, history })
  const raw = rawProbabilities(logits)
  const removedBy = (i: number) => result.stages.find(stage => !stage.kept[i])?.name
  const bar = 62, gap = 24, base = 196
  // The axis tops out at the next quarter above the tallest bar, so small probabilities stay visible.
  const ceiling = Math.ceil(Math.max(...raw, ...result.final) * 4) / 4
  const scale = 150 / ceiling
  return <LabFrame title="温度、截断与重复惩罚怎样改写下一 token 分布" hint="历史 [BOS,A] · 空心框是正文原始概率">
    <Controls>
      <Range label="温度 τ" value={temperature} min={0.2} max={2} step={0.1} onChange={setTemperature} format={v => v.toFixed(1)} />
      <Range label="top-k" value={topK} min={1} max={4} onChange={setTopK} />
      <Range label="top-p" value={topP} min={0.05} max={1} step={0.05} onChange={setTopP} format={v => v.toFixed(2)} />
    </Controls>
    <details className="mt-3 text-sm"><summary>min-p 与重复惩罚</summary><Controls>
      <Range label="min-p" value={minP} min={0} max={0.5} step={0.05} onChange={setMinP} format={v => v.toFixed(2)} />
      <Range label="重复惩罚 θ" value={penalty} min={1} max={3} step={0.1} onChange={setPenalty} format={v => v.toFixed(1)} />
      <label className="flex h-8 items-center gap-2 text-sm"><input type="checkbox" checked={repeated} onChange={event => setRepeated(event.target.checked)} className="accent-accent" />历史里已出现 A</label>
    </Controls></details>
    <svg viewBox="0 0 360 250" className={`${pen.canvas} max-w-110!`} role="img" aria-label={`保留 ${result.support} 个候选；${words.map((w, i) => `${w} ${result.final[i].toFixed(2)}`).join('，')}`}>
      <line x1="10" x2="350" y1={base} y2={base} className={pen.axis} />
      <line x1="10" x2="350" y1={base - 150} y2={base - 150} className={pen.grid} />
      <text x="350" y={base - 156} textAnchor="end" className={`${pen.mono} text-[11px] ${pen.muted}`}>{ceiling.toFixed(2)}</text>
      {words.map((word, i) => {
        const x = 12 + i * (bar + gap)
        const removed = removedBy(i)
        const penalized = repeated && i === 1 && penalty !== 1
        return <g key={word}>
          {!removed && <rect x={x} y={r(base - result.final[i] * scale)} width={bar} height={r(Math.max(1, result.final[i] * scale))} rx="2" className={penalized ? 'fill-accent2' : 'fill-accent'} />}
          <rect x={x} y={r(base - raw[i] * scale)} width={bar} height={r(raw[i] * scale)} rx="2" className="fill-none stroke-ink [stroke-dasharray:3_3]" />
          <text x={x + bar / 2} y={r(base - Math.max(raw[i], result.final[i]) * scale - 8)} textAnchor="middle" className={`${pen.mono} text-[11px] ${removed ? pen.muted : ''}`}>{removed ? '0' : result.final[i].toFixed(2)}</text>
          <text x={x + bar / 2} y={base + 20} textAnchor="middle" className={penalized ? pen.textB : ''}>{word}</text>
          <text x={x + bar / 2} y={base + 40} textAnchor="middle" className={`${pen.mono} text-[11px] ${pen.muted}`}>{removed ?? `z ${result.penalized[i].toFixed(2)}`}</text>
        </g>
      })}
    </svg>
    <Readout>
      顺序：惩罚 → ÷τ → top-k → top-p → min-p → 归一化 · 保留 {result.support}/4 项 · 最大概率 {Math.max(...result.final).toFixed(3)} · 熵 {result.entropy.toFixed(3)} nat（原始 {(-raw.reduce((s, p) => s + p * Math.log(p), 0)).toFixed(3)}）
      {repeated && penalty !== 1 ? ` · A 的负 logit ${logits[1].toFixed(2)} → ${result.penalized[1].toFixed(2)}` : ''}
    </Readout>
  </LabFrame>
}
