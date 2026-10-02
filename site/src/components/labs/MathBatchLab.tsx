import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { batchDistribution } from '../../lib/math-probability-model'
import { Select, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

export default function MathBatchLab() {
  const [size, setSize] = useState(4), [mode, setMode] = useState<'independent' | 'copied' | 'complete'>('independent')
  const result = batchDistribution(size, mode), maximum = Math.max(...result.rows.map(row => row.probability))
  return <LabFrame title="同样 B 个位置，抽样关系决定均值波动" hint="精确枚举损失 5 出现次数的概率；不是有限次模拟">
    <Controls><Range label="batch size B" value={size} min={1} max={64} onChange={setSize} /><Select label="抽样关系" value={mode} onChange={e => setMode(e.target.value as typeof mode)} ><option value="independent">独立、有放回</option><option value="copied">抽一次并复制 B 份</option><option value="complete">无放回取完整四条</option></Select></Controls>
    <SvgCanvas viewBox="0 0 320 264" className={`${pen.canvas} max-w-96`} role="img" aria-label={`均值分布标准差 ${result.standardError}`}>
      <line x1="25" x2="295" y1="210" y2="210" className={pen.axis} />
      <line x1="92.5" x2="92.5" y1="25" y2="210" className={pen.guide} />
      <text x="10" y="18">概率质量（最大 {fmt(maximum)}）</text>
      {result.rows.map((row, i) => <rect key={i} x={25 + (row.mean - 1) * 67.5 - Math.min(10, 250 / result.rows.length) / 2} y={210 - row.probability / maximum * 165} width={Math.min(10, 250 / result.rows.length)} height={row.probability / maximum * 165} className={Math.abs(row.mean - 2) < 1 - 1e-12 ? 'fill-accent' : 'fill-accent2'} />)}
      {[1, 2, 3, 4, 5].map(mean => <text key={mean} x={25 + (mean - 1) * 67.5} y="232" textAnchor="middle">{mean}</text>)}
      <text x="160" y="249" textAnchor="middle">batch 均值 · 虚线为目标 2</text>
    </SvgCanvas>
    <Readout>有效 B={mode === 'complete' ? 4 : size} · E[均值]={fmt(result.mean)} · Var={fmt(result.variance, 6)} · 标准误={fmt(result.standardError, 6)}<br />P(|均值−2|&lt;1)={fmt(result.closeProbability * 100, 2)}% · {mode === 'independent' ? `理论 √(3/B)=${fmt(Math.sqrt(3 / size), 6)}` : mode === 'copied' ? '共同复制不产生独立样本，方差仍为 3' : '完整四条均值恒为 2；此场景的 B 固定为 4'}</Readout>
  </LabFrame>
}
