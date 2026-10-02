import { MatrixGrid } from './DataViews'


import { useState } from 'react'
import { permutationAttention } from '../../lib/principles-trace-model'
import { Toggle, Controls, LabFrame, Range, Readout, fmt } from './Lab'

const vector = (values: number[]) => `(${values.map(x => fmt(x, 4)).join(', ')})`

export default function PermutationLab() {
  const [swapped, setSwapped] = useState(true), [causal, setCausal] = useState(false), [position, setPosition] = useState(false), [move, setMove] = useState(false), [row, setRow] = useState(1)
  const result = permutationAttention(swapped, causal, position, move)
  const baseline = permutationAttention(false, causal, position, false)
  const expected = result.permutation.map(i => baseline.output[i])
  const error = Math.max(...result.output.map((x, i) => Math.abs(x - expected[i])))
  return <LabFrame title="交换内容，还是同时交换槽位结构" hint="X=(1,2,3)，位置表 E=(0,1,0)，单位 Q/K/V">
    <Controls>{[['交换 BOS 与 A', swapped, setSwapped], ['causal mask', causal, setCausal], ['加位置表', position, setPosition], ['mask / 位置表随身份移动', move, setMove]].map(([name, value, setter]) => <Toggle key={String(name)} label={String(name)} checked={value as boolean} onChange={value => (setter as (v: boolean) => void)(value)} />)}<Range label="查询槽位" min={0} max={2} value={row} onChange={setRow} /></Controls>
    <MatrixGrid label="固定槽位或随身份移动的分数与允许集合" values={result.scores.map((scores, i) => scores.map((score, j) => result.allowed[i][j] ? score : '−∞'))} rowLabels={[0,1,2].map(i => `槽 ${i}`)} columnLabels={result.permutation.map(id => ['BOS','A','B'][id])} activeRow={row} allowed={result.allowed} />
    <Readout>当前输入 {vector(result.values)}<br />查询行概率 {vector(result.probabilities[row])} · 输出 {vector(result.output)}<br />单纯重排旧输出 {vector(expected)} · 最大差 {fmt(error, 6)}</Readout>
  </LabFrame>
}
