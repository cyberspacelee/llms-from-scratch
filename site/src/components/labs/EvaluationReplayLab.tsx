import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { evaluationLedger, evaluationWindows } from '../../lib/training-trace-model'
import { Toggle, Select, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'


export default function EvaluationReplayLab() {
  const [stride, setStride] = useState(2), [step, setStep] = useState(0), [model, setModel] = useState<'A' | 'B'>('A'), [included, setIncluded] = useState([true, true, true, true])
  const windows = evaluationWindows(stride), selected = Math.min(step, windows.length - 1), current = windows[selected]
  const visited = windows.slice(0, selected + 1).flatMap(w => w.targets), result = evaluationLedger(model, included)
  return <LabFrame title="重读上下文，只计新增目标" hint="D2 长度 7，输入上限 K=4；下方 NLL 是正文固定观测">
    <Controls><Range label="新增目标步幅 s" min={1} max={4} value={stride} onChange={setStride} /><Range label="窗口序号" min={0} max={windows.length - 1} value={selected} onChange={setStep} /><Select label="checkpoint" value={model} onChange={e => setModel(e.target.value as 'A' | 'B')} ><option>A</option><option>B</option></Select></Controls>
    <SvgCanvas viewBox="0 0 360 244" className={`${pen.canvas} max-w-110!`} role="img" aria-label="输入 token、计分目标和累计覆盖的错位一格关系">
      {['输入 token', '新增目标', '累计计分'].map((name, i) => <text key={name} x="5" y={35 + i * 71}>{name}</text>)}
      {[0, 1, 2, 3, 4, 5, 6].map(j => <g key={j}>{[current.inputs.includes(j), current.targets.includes(j), visited.includes(j)].map((v, i) => <g key={i}><rect x={7 + j * 49} y={45 + i * 71} width="43" height="34" className={v ? 'fill-accent-soft stroke-accent' : 'fill-sunken stroke-rule'} /><text x={28 + j * 49} y={68 + i * 71} textAnchor="middle">{j}</text></g>)}</g>)}
    </SvgCanvas>
    <div className="mt-3 flex flex-wrap gap-4">{included.map((value, i) => <Toggle key={i} label={`D${i + 1}`} checked={value} onChange={checked => setIncluded(values => values.map((v, j) => j === i ? checked : v))} />)}</div>
    <Readout>本窗输入 [{current.inputs.join(',')}] → 新目标 [{current.targets.join(',')}]；累计 {visited.length}/6 项。每个目标在完整回放中恰好一次。<br />{result.count ? `固定 s=2 观测：ΣNLL=${fmt(result.nll)}，N=${result.count}，token 均值=${fmt(result.mean!)}，PPL=${fmt(Math.exp(result.mean!), 4)}；文档等权均值=${fmt(result.documentMean!)}` : '无有效目标：NLL 与 PPL 未定义。'}<br />改变步幅仅重算覆盖；不同上下文的真实 NLL 必须重新运行模型，不能沿用固定观测。</Readout>
  </LabFrame>
}
