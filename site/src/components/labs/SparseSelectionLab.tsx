import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Select, Controls, LabFrame, Readout, fmt, pen } from './Lab'
import { sparseSelection } from '../../lib/advanced-interactive-model'

export default function SparseSelectionLab() {
  const [selected, setSelected] = useState([0, 3, 7])
  const r = sparseSelection(selected)
  return <LabFrame title="候选集合改变分母与输出">
    <Controls><Select label="固定 indexer 场景" value={selected.join(',')} onChange={e => setSelected(e.target.value ? e.target.value.split(',').map(Number) : [])} ><option value="0,3,7">top-3：0、3、7</option><option value="0,4,7">漏读 3：0、4、7</option><option value="5,6,7">最近窗口</option><option value="0,1,2,3,4,5,6,7">全部读取</option><option value="">清空集合</option></Select></Controls>
    <div className="mt-3 grid grid-cols-4 gap-2">{r.weights.map((_, j) => <Toggle key={j} label={`键 ${j}`} checked={selected.includes(j)} onChange={checked => setSelected(checked ? [...selected, j].sort((a, b) => a - b) : selected.filter(v => v !== j))} />)}</div>
    <SvgCanvas viewBox="0 0 280 252" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`保留 ${selected.join(', ')}，丢弃质量 ${r.discarded}，输出 ${r.output}`}>
      {r.probabilities.map((p, j) => <g key={j}><text x="8" y={25 + j * 26}>k{j}</text><rect x="38" y={9 + j * 26} width={p * 225} height="16" className="fill-accent" /><path d={`M38 ${29 + j * 26}h${r.weights[j] / 12 * 225}`} className={pen.b} /></g>)}<text x="8" y="234">实柱：集合内概率；虚线：完整概率</text>
    </SvgCanvas><Readout>indexer 仅决定集合，主指数=[{r.weights.join(', ')}]；分母={r.denominator}<br />被丢弃质量 δ={fmt(r.discarded)}；稀疏输出={r.output === null ? '未定义：禁止空集合' : fmt(r.output, 6)}；完整输出={fmt(r.full, 6)}<br />输出差={r.error === null ? '未定义' : fmt(r.error, 6)}；固定值界 2Mδ={fmt(r.bound)}（M=70）</Readout>
  </LabFrame>
}
