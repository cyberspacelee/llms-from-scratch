import { useState } from 'react'
import { Controls, LabFrame, Range, Readout, pen } from './Lab'
import { pageSnapshot } from '../../lib/systems-interactive-model'

const stages = [
  { title: 'A 已计算六项', next: '位置 6 尚未写入' },
  { title: 'B 共享六项', next: '共享的尾块 1 还不能直接写' },
  { title: 'A 写入 7', next: '复制块 1 → 2；A 写槽 10' },
  { title: 'B 写入 70', next: 'B 独占块 1；写槽 6' },
  { title: 'A 退出', next: '块 2 可回收，B 仍持有块 0、1' },
  { title: 'B 退出', next: '所有块均可回收' },
]

function Blocks({ name, table }: { name: string; table: number[] }) {
  return (
    <div className="flex min-h-18 flex-wrap items-center gap-2 border-b border-rule py-3 last:border-0">
      <strong className="w-5 text-sm">{name}</strong>
      {table.length ? table.map((physical, logical) => (
        <div key={logical} className="min-w-30 border border-rule-strong bg-paper px-3 py-2 text-sm">
          <span className="block text-xs text-muted">逻辑块 {logical}</span>
          <span className="font-mono">物理块 {physical}</span>
        </div>
      )) : <span className="text-sm text-muted">未持有缓存</span>}
    </div>
  )
}

export default function PagedKvLab() {
  const [step, setStep] = useState(0)
  const snapshot = pageSnapshot(step)
  const stage = { ...stages[step], a: snapshot.a, b: snapshot.b, refs: snapshot.refs }

  return (
    <LabFrame title="同一前缀，两个写入位置" hint="教学用标量槽位；块大小为 4">
      <Controls>
        <Range label="操作步骤" value={step} min={0} max={stages.length - 1}
          onChange={setStep} format={() => stage.title} />
      </Controls>
      <div className="mt-4 bg-sunken px-4">
        <Blocks name="A" table={stage.a} />
        <Blocks name="B" table={stage.b} />
      </div>
      <svg viewBox="0 0 280 218" className={pen.canvas} role="img" aria-label={`物理块、槽内容与空闲块；引用数 ${stage.refs.join(', ')}`}>
        {[0, 1, 2].map(block => {
          const values = snapshot.memory[block]
          return <g key={block}><text x="8" y={20 + block * 68}>块 {block} · ref={stage.refs[block]} · {stage.refs[block] ? '持有' : '空闲'}</text>{values.map((value, slot) => <g key={slot}><rect x={8 + slot * 65} y={28 + block * 68} width="58" height="33" className={stage.refs[block] ? 'fill-accent/15 stroke-accent' : 'fill-sunken stroke-rule [stroke-dasharray:3_3]'} /><text x={37 + slot * 65} y={50 + block * 68} textAnchor="middle">{stage.refs[block] && value !== null ? value : '—'}</text></g>)}</g>
        })}
      </svg>
      <Readout>物理块 0/1/2 引用数：{stage.refs.join(' / ')}。{stage.next}。</Readout>
      <Readout>A 逻辑读取=[{snapshot.aValues.join(', ')}]；B 逻辑读取=[{snapshot.bValues.join(', ')}]<br />空闲块=[{snapshot.free.join(', ')}]；只有有效且持有的槽可读。释放不保证清零旧内存，重新分配必须重建有效长度与身份。</Readout>
    </LabFrame>
  )
}
