import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Select, Controls, LabFrame, Range, Readout, pen } from './Lab'
import { speculativeTrace } from '../../lib/systems-interactive-model'

export default function SpeculativeReplayLab() {
  const [step, setStep] = useState(0), [scenario, setScenario] = useState(1)
  const draws = [[0.9, 0.5], [0.5, 0.9], [0.5, 0.5]][scenario], r = speculativeTrace(draws[0], draws[1], step)
  return <LabFrame title="两步验证与 KV 提交边界">
    <Controls><Select label="抽数轨迹" value={scenario} onChange={e => { setScenario(Number(e.target.value)); setStep(0) }} ><option value={0}>第一项拒绝</option><option value={1}>第二项拒绝</option><option value={2}>全部接受</option></Select><Range label="验证步骤" value={step} min={0} max={3} onChange={setStep} format={v => ['初始历史', '验证第 1 项', '验证第 2 项', '提交/补偿'][v]} /></Controls>
    <SvgCanvas viewBox="0 0 280 145" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`提交 ID ${r.submitted.join(', ')}；有效 KV ${r.cached.join(', ')}`}>
      {[['ID', r.submitted], ['KV', r.cached]].map(([label, values], row) => <g key={String(label)}><text x="4" y={35 + row * 65}>{String(label)}</text>{(values as number[]).map((v, i) => <g key={i}><rect x={38 + i * 44} y={12 + row * 65} width="36" height="35" className={row ? 'fill-accent/20 stroke-accent' : 'fill-accent2/15 stroke-accent2'} /><text x={56 + i * 44} y={35 + row * 65} textAnchor="middle">{v}</text></g>)}</g>)}
    </SvgCanvas><Readout>草稿=[0,2]；接受阈值=[5/6,2/3]；固定抽数=[{draws.join(', ')}]<br />已接受 {r.accepted} 项；{r.rejected ? '首次拒绝后停止，残差只含 ID 1' : '按真实已接受历史继续'}<br />有效 KV=[{r.cached.join(', ')}]；下轮待输入={r.pending === null ? '尚未提交新末项' : r.pending}；截断的是验证路径，已提交输出不回滚</Readout>
  </LabFrame>
}
