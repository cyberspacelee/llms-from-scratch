import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { informationCost } from '../../lib/math-probability-model'
import { Toggle, Select, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

const number = (v: number) => Number.isFinite(v) ? fmt(v, 6) : '∞'
export default function MathInformationLab() {
  const [q, setQ] = useState(0.75), [source, setSource] = useState<'all' | 'D' | 'E'>('all'), [bits, setBits] = useState(false), [selected, setSelected] = useState(0)
  const result = informationCost(q, source, bits)
  const finite = [...result.entropyTerms, ...result.crossTerms].filter(Number.isFinite), maximum = Math.max(1, ...finite)
  return <LabFrame title="真实不确定性与额外预测代价">
    <Controls><Range label="模型 Q(正类)" value={q} min={0} max={1} step={0.05} onChange={setQ} format={v => fmt(v, 2)} /><Select label="真实分布" value={source} onChange={e => setSource(e.target.value as typeof source)} ><option value="all">整体 P=(1/2,1/2)</option><option value="D">已知 D · P=(1/3,2/3)</option><option value="E">已知 E · P=(1,0)</option></Select><Select label="观察结果" value={selected} onChange={e => setSelected(Number(e.target.value))} ><option value="0">正类</option><option value="1">非正类</option></Select><Toggle label="以 2 为底（bit）" checked={bits} onChange={value => setBits(value)} /></Controls>
    <SvgCanvas viewBox="0 0 320 258" className={`${pen.canvas} max-w-96`} role="img" aria-label={`熵 ${result.entropy}，交叉熵 ${result.crossEntropy}，KL ${result.kl}`}>
      <text x="10" y="19">绿色：−p log p · 橙色：−p log q</text>
      {['正类', '非正类'].map((label, c) => <g key={c}>
        <rect x={20 + c * 155} y="32" width="125" height="195" rx="4" className={c === selected ? 'fill-sunken stroke-info stroke-2' : 'fill-none stroke-rule'} />
        {[result.entropyTerms[c], result.crossTerms[c]].map((value, index) => <g key={index}><rect x={36 + c * 155 + index * 45} y={193 - (Number.isFinite(value) ? value / maximum : 1) * 118} width="30" height={(Number.isFinite(value) ? value / maximum : 1) * 118} className={index ? 'fill-accent2' : 'fill-accent'} /><text x={51 + c * 155 + index * 45} y="212" textAnchor="middle" className={pen.mono}>{Number.isFinite(value) ? fmt(value) : '∞'}</text></g>)}
        <text x={82 + c * 155} y="54" textAnchor="middle">p={fmt(result.p[c], 2)} q={fmt(result.model[c], 2)}</text><text x={82 + c * 155} y="249" textAnchor="middle">{label}</text>
      </g>)}
    </SvgCanvas>
    <Readout>选中结果 I={number(result.information[selected])} {bits ? 'bit' : 'nat'}{result.p[selected] === 0 && '（零概率结果不可能出现，其熵贡献按 0 处理）'}<br />H={number(result.entropy)} · CE={number(result.crossEntropy)} · KL={number(result.kl)}<br />CE=H+KL：{number(result.entropy)} + {number(result.kl)}<br />H(U|来源)=3/4×H(U|D)+1/4×0={number(result.conditionalEntropy)}<br />{result.crossEntropy === Infinity ? 'P 有正质量而 Q=0：交叉熵与 KL 发散' : '逐类贡献相加得到平均代价'}</Readout>
  </LabFrame>
}
