import { useState } from 'react'
import { Button, Controls, LabFrame, Range, Readout, pen } from './Lab'

type Seq = { id: string; prompt: number; cached: number; completion: number; maxCompletion: number; state: 'waiting' | 'running' | 'finished' | 'cancelled' }

const start = (): Seq[] => [
  { id: 'A', prompt: 3, cached: 0, completion: 0, maxCompletion: 2, state: 'waiting' },
  { id: 'B', prompt: 5, cached: 0, completion: 0, maxCompletion: 2, state: 'waiting' },
]
const ready = 'READY    A 的提示长 3，B 长 5；C 将在第 3 轮加入。'

/** A small synchronous reference for the chapter's prefill-first policy. */
export default function SchedulerLab() {
  const [budget, setBudget] = useState(4)
  const [maxSeqs, setMaxSeqs] = useState(2)
  const [seqs, setSeqs] = useState(start)
  const [step, setStep] = useState(0)
  const [trace, setTrace] = useState(ready)
  const [selected, setSelected] = useState('A')

  function changeRequest(cancel: boolean) {
    setSeqs(current => current.map(seq => seq.id === selected && seq.state !== 'finished' && seq.state !== 'cancelled'
      ? { ...seq, cached: 0, state: cancel ? 'cancelled' : 'waiting' } : seq))
    setTrace(`${cancel ? 'CANCEL' : 'PREEMPT'} ${selected}：保留已有 ID 和已交付输出；${cancel ? '终止请求，安全收尾后释放 KV' : '释放 KV 后回到等待，重算已有历史，不重复发送旧输出'}`)
  }

  function advance() {
    const next = seqs.map((seq) => ({ ...seq }))
    const number = step + 1
    if (number === 3) next.push({ id: 'C', prompt: 2, cached: 0, completion: 0, maxCompletion: 1, state: 'waiting' })
    const lines = [`STEP ${String(number).padStart(2, '0')}  budget=${budget}, max_num_seqs=${maxSeqs}`]
    const waiting = next.filter((seq) => seq.state === 'waiting')
    if (waiting.length) {
      let used = 0
      let scheduled = 0
      for (const seq of waiting) {
        if (scheduled >= maxSeqs) break
        const remaining = budget - used
        if (remaining === 0) break
        const total = seq.prompt + seq.completion
        const need = total - seq.cached
        if (remaining < need && scheduled > 0) {
          lines.push(`PREFILL  ${seq.id} 延后：剩余预算 ${remaining} 装不下 ${need} 个 token`)
          break
        }
        const take = Math.min(need, remaining)
        seq.cached += take
        used += take
        scheduled += 1
        if (seq.cached === total) {
          seq.state = 'running'
          seq.completion += 1
          lines.push(`PREFILL  ${seq.id} +${take} token，写完当前历史后采样下一输出 token`)
          if (seq.completion >= seq.maxCompletion) seq.state = 'finished'
        } else {
          lines.push(`CHUNK    ${seq.id} +${take} token，缓存 ${seq.cached}/${total}`)
        }
      }
    } else {
      const running = next.filter((seq) => seq.state === 'running').slice(0, Math.min(maxSeqs, budget))
      if (!running.length) lines.push('DONE     所有序列都已结束')
      for (const seq of running) {
        seq.cached += 1
        seq.completion += 1
        if (seq.completion >= seq.maxCompletion) seq.state = 'finished'
        lines.push(`DECODE   ${seq.id} 写入上一步的 token，再采样 1 个${seq.state === 'finished' ? ' → FINISHED' : ''}`)
      }
    }
    const finished = next.filter((seq) => seq.state === 'finished').map((seq) => seq.id)
    if (finished.length) lines.push(`FINISHED ${finished.join(', ')}`)
    setSeqs(next)
    setStep(number)
    setTrace(lines.join('\n'))
  }

  function reset() {
    setSeqs(start())
    setStep(0)
    setTrace(ready)
  }

  const queue = (state: Seq['state'], title: string) => {
    const items = seqs.filter((seq) => seq.state === state)
    return (
      <div>
        <h4 className="mt-0 mb-2 text-xs font-semibold text-muted">{title}</h4>
        {items.length === 0 && <p className="m-0 border-t border-rule py-2 text-sm text-muted">队列为空</p>}
        {items.map((seq) => {
          const total = seq.prompt + seq.completion
          const promptCached = (Math.min(seq.cached, seq.prompt) / total) * 100
          const generated = (seq.completion / total) * 100
          return (
            <div key={seq.id} className="grid grid-cols-[1.5rem_minmax(0,1fr)_auto] items-center gap-3 border-t border-rule py-2">
              <span className="font-mono font-semibold text-accent">{seq.id}</span>
              <div className="flex h-2.5 overflow-hidden rounded-full bg-sunken" role="img"
                aria-label={`${seq.id}：缓存 ${seq.cached}/${total}，已生成 ${seq.completion}`}>
                <span className="bg-accent" style={{ width: `${promptCached}%` }} />
                <span className="ml-auto bg-accent2" style={{ width: `${generated}%` }} />
              </div>
              <span className="text-right font-mono text-xs leading-snug text-muted">
                cache {seq.cached}/{total}
                <br />
                out {seq.completion}/{seq.maxCompletion}
              </span>
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <LabFrame title="调度器单步推演" hint="默认预算 4、最多 2 条请求；C 在第 3 轮到达">
      <Controls>
        <Range label="本步 token 预算" value={budget} min={2} max={8} onChange={setBudget} />
        <Range label="最多序列数" value={maxSeqs} min={1} max={3} onChange={setMaxSeqs} />
        <div className="flex gap-2">
          <Button primary onClick={advance}>执行一步</Button>
          <Button onClick={reset}>重置</Button>
        </div>
        <label className="text-sm">请求身份<select className="block h-9 w-full border border-rule bg-paper" value={selected} onChange={e => setSelected(e.target.value)}>{seqs.map(seq => <option key={seq.id}>{seq.id}</option>)}</select></label>
        <div className="flex flex-wrap gap-2"><Button onClick={() => changeRequest(false)}>抢占</Button><Button onClick={() => changeRequest(true)}>取消</Button></div>
      </Controls>
      <div className="mt-4 grid grid-cols-[repeat(auto-fit,minmax(15rem,1fr))] gap-4">
        {queue('waiting', 'WAITING · 等待 prefill')}
        {queue('running', 'RUNNING · 逐步 decode')}
        {queue('finished', 'FINISHED · 已完成')}
        {queue('cancelled', 'CANCELLED · 已取消')}
      </div>
      <svg viewBox="0 0 280 145" className={pen.canvas} role="img" aria-label={`${selected} 已有 token 与已写入 KV 的逐位置状态`}>
        {(() => { const seq = seqs.find(item => item.id === selected)!, total = seq.prompt + seq.completion; return Array.from({ length: total }, (_, i) => <g key={i}><rect x={8 + i % 6 * 44} y={10 + Math.floor(i / 6) * 63} width="38" height="48" className={i < seq.cached ? 'fill-accent/20 stroke-accent' : 'fill-accent2/15 stroke-accent2 [stroke-dasharray:3_3]'} /><text x={27 + i % 6 * 44} y={29 + Math.floor(i / 6) * 63} textAnchor="middle">p{i}</text><text x={27 + i % 6 * 44} y={48 + Math.floor(i / 6) * 63} textAnchor="middle">{i < seq.cached ? 'KV' : 'ID'}</text></g>) })()}
      </svg>
      <Readout>{selected}：{(() => { const seq = seqs.find(item => item.id === selected)!; return `已有 N=${seq.prompt + seq.completion} 个 ID；已写 K=${seq.cached} 个 KV；未处理位置=[${Array.from({ length: seq.prompt + seq.completion - seq.cached }, (_, i) => seq.cached + i).join(', ')}]；已交付 ${seq.completion} 个输出` })()}。图中位置保留请求身份，不跨请求读取。</Readout>
      <p className="mt-3 mb-0 text-xs text-muted">条形代表整条序列：左侧绿色是已写入 KV 的 prompt，右侧橙色是已生成的输出 token。</p>
      <pre aria-live="polite" className="mt-4 mb-0 min-h-22 rounded-lg bg-sunken px-4 py-3 font-mono text-xs leading-relaxed whitespace-pre-wrap">{trace}</pre>
    </LabFrame>
  )
}
