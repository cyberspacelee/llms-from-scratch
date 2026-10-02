import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { duplicateGroups } from '../../lib/training-trace-model'
import { Toggle, Select, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'


export default function DuplicateGraphLab() {
  const [threshold, setThreshold] = useState(0.4), [chain, setChain] = useState(false), [selected, setSelected] = useState(0)
  const [assignments, setAssignments] = useState<Record<number, boolean>>({})
  const result = duplicateGroups(threshold, chain), points = [[70, 55], [180, 115], [290, 55], [90, 210], [270, 210]]
  const component = result.group[selected]
  const training = (group: number) => assignments[group] ?? group % 2 === 0
  return <LabFrame title="边的阈值与整组划分" hint="正文 A–E 的完整三词 shingle 复核">
    <Controls><Range label="Jaccard 阈值" min={0.1} max={1} step={0.1} value={threshold} onChange={value => { setThreshold(value); setAssignments({}) }} /><Range label="选择文档" min={0} max={4} value={selected} onChange={setSelected} /><Select label="选中整组的用途" value={training(component) ? 'training' : 'validation'} onChange={e => setAssignments(current => ({ ...current, [component]: e.target.value === 'training' }))} ><option value="training">训练</option><option value="validation">验证</option></Select><Toggle label="独立链反例 A–B–C" checked={chain} onChange={value => { setChain(value); setAssignments({}) }} /></Controls>
    <SvgCanvas viewBox="0 0 360 260" className={`${pen.canvas} max-w-110!`} role="img" aria-label="Jaccard 达阈值的边及选中文档的连通分量">
      {result.edges.map(([i, j]) => <g key={`${i}-${j}`}><line x1={points[i][0]} y1={points[i][1]} x2={points[j][0]} y2={points[j][1]} className={result.group[i] === component ? pen.a : pen.axis} /><text x={(points[i][0] + points[j][0]) / 2} y={(points[i][1] + points[j][1]) / 2 - 9} textAnchor="middle" className={pen.mono}>{fmt(result.similarities[i][j], 1)}</text></g>)}
      {points.map(([x, y], i) => <g key={i}><circle cx={x} cy={y} r="23" className={result.group[i] === component ? 'fill-accent-soft stroke-accent' : 'fill-paper stroke-rule'} strokeWidth={i === selected ? 3 : 1} /><text x={x} y={y + 5} textAnchor="middle">{'ABCDE'[i]}</text><text x={x} y={y + 40} textAnchor="middle" className={pen.muted}>{training(result.group[i]) ? '训练' : '验证'}</text></g>)}
    </SvgCanvas>
    <Readout>选中组 [{result.group.flatMap((g, i) => g === component ? ['ABCDE'[i]] : []).join(',')}] · {new Set(result.group).size} 个不可拆分的组。{chain ? '链场景使用明确另设的相似度：AB=BC=.6、AC=0；同组不是两两相似。' : 'A=C 的 Jaccard=1，AB=BC=.4；阈值 .4 恢复正文组 {A,B,C}、{D}、{E}。'}<br />训练 [{result.group.flatMap((g, i) => training(g) ? ['ABCDE'[i]] : []).join(',')}]；验证 [{result.group.flatMap((g, i) => !training(g) ? ['ABCDE'[i]] : []).join(',')}]。移动选中组会移动组内全部记录。改阈值重置划分；默认按组最小 ID 的奇偶分配，初始结果与正文一致，交互变式不沿用正文 seed。</Readout>
  </LabFrame>
}
