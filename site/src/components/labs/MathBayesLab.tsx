import { useState } from 'react'
import { bayesPaths, sequencePath } from '../../lib/math-probability-model'
import { Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

export default function MathBayesLab({ sequence = false }: { sequence?: boolean }) {
  const [prior, setPrior] = useState(0.75), [step, setStep] = useState(0)
  const tree = bayesPaths(prior), chain = sequencePath(Math.min(step, 3))
  return <LabFrame title={sequence ? '完整历史上的概率连乘' : '先沿路径相乘，再在正类中归一化'}>
    <Controls>{!sequence && <Range label="来源 D 的先验" value={prior} min={0} max={1} step={0.05} onChange={setPrior} format={v => fmt(v, 2)} />}<Range label="计算阶段" value={step} min={0} max={3} onChange={setStep} /></Controls>
    {sequence ? <svg viewBox="0 0 320 275" className={`${pen.canvas} max-w-96`} role="img" aria-label={`已展开 ${step} 步，前缀联合概率 ${chain.joint}`}>
      {['BOS → A', 'BOS,A → B', 'BOS,A,B → EOS'].map((label, i) => <g key={i}>
        {i > 0 && <path d={`M160 ${i * 78 - 4}V${i * 78 + 12}`} className={pen.axis} />}
        <rect x="10" y={12 + i * 78} width="300" height="62" rx="4" className={i < step ? 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} />
        <text x="24" y={36 + i * 78}>{label}</text><text x="24" y={58 + i * 78} className={pen.mono}>条件概率 {chain.probabilities[i]} · 前缀 {fmt(sequencePath(i + 1).joint)}</text>
      </g>)}<text x="10" y="269">每一步保留此前全部历史</text>
    </svg> : <svg viewBox="0 0 320 300" className={`${pen.canvas} max-w-96`} role="img" aria-label={`D 正类路径 ${tree.jointD}，E 正类路径 ${tree.jointE}，后验 D ${tree.posteriorD}`}>
      <path d="M160 44L84 98M160 44L236 98M84 139V188M236 139V188" className={pen.axis} />
      <rect x="100" y="10" width="120" height="35" rx="4" className="fill-sunken stroke-rule" /><text x="160" y="34" textAnchor="middle">抽一条记录</text>
      {[{ x: 10, title: 'D', prior, likelihood: 1 / 3, joint: tree.jointD, posterior: tree.posteriorD }, { x: 165, title: 'E', prior: 1 - prior, likelihood: 1, joint: tree.jointE, posterior: tree.posteriorE }].map(row => <g key={row.title}>
        <text x={row.x + 72} y="84" textAnchor="middle">P({row.title})={fmt(row.prior)}</text>
        <rect x={row.x} y="98" width="145" height="42" rx="4" className="fill-info-soft stroke-info" /><text x={row.x + 72} y="125" textAnchor="middle">正类比例 {fmt(row.likelihood)}</text>
        <rect x={row.x} y="188" width="145" height="42" rx="4" className={step >= 1 ? 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} /><text x={row.x + 72} y="214" textAnchor="middle">联合 {fmt(row.joint)}</text>
        <text x={row.x + 72} y="254" textAnchor="middle" className={step >= 3 ? pen.textB : pen.muted}>后验 {fmt(row.posterior)}</text>
      </g>)}<text x="160" y="290" textAnchor="middle" className={step >= 2 ? pen.textA : pen.muted}>P(正类)={fmt(tree.positive)}</text>
    </svg>}
    <Readout>{sequence ? <>已取 {step} 项 · 连乘 {chain.probabilities.slice(0, step).join(' × ') || '1'} = {fmt(chain.joint, 6)}{step === 0 && '（空乘积为 1）'}</> : <>{['0 · 先列来源与条件概率', `1 · 两条正类路径：${fmt(tree.jointD)} 与 ${fmt(tree.jointE)}`, `2 · 全概率相加：${fmt(tree.positive)}`, `3 · 后验 D：${fmt(tree.jointD)} / ${fmt(tree.positive)} = ${fmt(tree.posteriorD)}`][step]}<br />P(正类|D)=1/3 保持不变；改变先验会同时改变整体和后验。</>}</Readout>
  </LabFrame>
}
