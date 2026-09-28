import { useState } from 'react'
import { Controls, LabFrame, Range, Readout } from './Lab'

const stages = [
  { title: 'A 已计算六项', a: [0, 1], b: [] as number[], refs: [1, 1, 0], next: '位置 6 尚未写入' },
  { title: 'B 共享六项', a: [0, 1], b: [0, 1], refs: [2, 2, 0], next: '共享的尾块 1 还不能直接写' },
  { title: 'A 写入 7', a: [0, 2], b: [0, 1], refs: [2, 1, 1], next: '复制块 1 → 2；A 写槽 10' },
  { title: 'B 写入 70', a: [0, 2], b: [0, 1], refs: [2, 1, 1], next: 'B 独占块 1；写槽 6' },
  { title: 'A 退出', a: [], b: [0, 1], refs: [1, 1, 0], next: '块 2 可回收，B 仍持有块 0、1' },
  { title: 'B 退出', a: [], b: [], refs: [0, 0, 0], next: '所有块均可回收' },
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
  const stage = stages[step]

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
      <Readout>物理块 0/1/2 引用数：{stage.refs.join(' / ')}。{stage.next}。</Readout>
    </LabFrame>
  )
}
