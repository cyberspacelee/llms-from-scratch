import { TraceTable as Trace } from './DataViews'

import { useState } from 'react'
import { sequenceLedger } from '../../lib/principles-trace-model'
import { Toggle, Controls, LabFrame, Range, Readout, fmt } from './Lab'


export default function SequencePplLab() {
  const [row, setRow] = useState(0), [valid, setValid] = useState([true, true, true]), [pads, setPads] = useState(2)
  const p = [0.6, 0.7, 0.6], histories = ['BOS', 'BOS,A', 'BOS,A,B'], targets = ['A', 'B', 'EOS']
  const result = sequenceLedger(valid)
  return <LabFrame title="三个下一 token 目标的概率账" hint="自然对数 · 未加权 NLL">
    <Controls><Range label="预测行 i" min={0} max={2} value={row} onChange={setRow} /><Range label="PAD 占位数" min={0} max={5} value={pads} onChange={setPads} /></Controls>
    <div className="mt-3 flex flex-wrap gap-4">{targets.map((name, i) => <Toggle key={name} label={`计分目标 ${name}`} checked={valid[i]} onChange={value => setValid(current => current.map((v, j) => j === i ? value : v))} />)}</div>
    <Trace label={`历史 ${histories[row]} 预测 ${targets[row]}，汇总 ${result.count} 个有效目标`} rows={[
      ['已知历史 → 下一目标', `${histories[row]} → ${targets[row]}`], ['条件概率 → 本行 NLL', `${p[row]} → ${fmt(-Math.log(p[row]), 6)} nat`], ['损失和 / 有效目标数', `${fmt(result.sum, 6)} / ${result.count}`], ['平均 NLL → PPL', result.mean === null ? '未定义：零有效目标' : `${fmt(result.mean, 6)} → ${fmt(result.ppl!, 6)}`],
    ]} active={1} />
    <Readout>数组长度 {3 + pads}，PAD 不改变有效分母。{result.count === 3 ? `整段 P(A,B,EOS|BOS)=${fmt(result.joint, 3)}` : '取消任一真实目标后，当前乘积不再代表完整文档概率。'} · 有效概率的连乘 {fmt(result.joint, 6)}</Readout>
  </LabFrame>
}
