import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Range, Readout, pen, useWidth } from './Lab'
import { packedRows } from '../../lib/systems-interactive-model'

export default function PackedExecutionLab() {
  const [decode, setDecode] = useState(false), [selected, setSelected] = useState(0), [stage, setStage] = useState(0)
  const rows = packedRows(decode), row = rows[Math.min(selected, rows.length - 1)]
  const [ref, width] = useWidth<HTMLDivElement>(), columns = width < 500 ? 3 : 5
  return <LabFrame title="token、逻辑位置、物理槽与 Graph 缓冲区">
    <div ref={ref}><Controls><Toggle label="下一轮 decode" checked={decode} onChange={value => { setDecode(value); setSelected(0) }} /><Range label="packed 下标" value={Math.min(selected, rows.length - 1)} min={0} max={rows.length - 1} onChange={setSelected} /><Range label="执行阶段" value={stage} min={0} max={3} onChange={setStage} format={v => ['准备', 'eager', 'capture', 'replay'][v]} /></Controls>
      <SvgCanvas viewBox={`0 0 ${columns * 88} ${Math.ceil(rows.length / columns) * 78 + 20}`} className={pen.canvas} role="img" aria-label={`选择请求 ${row.request} 的 ID ${row.id}，逻辑位置 ${row.position}，写槽 ${row.slot}`}>
        {rows.map((r, i) => <g key={i} transform={`translate(${(i % columns) * 88 + 4},${Math.floor(i / columns) * 78 + 8})`}><rect width="80" height="66" className={i === row.index ? 'fill-accent/20 stroke-accent stroke-2' : 'fill-sunken stroke-rule'} /><text x="6" y="18">{r.request} · ID {r.id}</text><text x="6" y="37">p={r.position}</text><text x="6" y="55">slot={r.slot}</text></g>)}
      </SvgCanvas></div>
    <Readout>packed[{row.index}] → {row.request}；RoPE 位置 {row.position}；可读本请求位置 0–{row.position}<br />block table=[{row.blocks.join(', ')}]；{row.blocks[Math.floor(row.position / 4)]}×4+{row.position % 4}={row.slot}<br />{stage < 2 ? '本轮按有效长度准备并执行' : !decode ? '该固定快照的 prefill 使用 eager；本阶段不能捕获/重放 prefill' : `固定地址 buf_ids / buf_positions / buf_slots；Graph 4 行，有效 3 行；padding 行 slot=-1、context_len=0；${stage === 2 ? '捕获固定运算依赖' : '复制新值后 replay，仅前三行采样'}`}</Readout>
  </LabFrame>
}
