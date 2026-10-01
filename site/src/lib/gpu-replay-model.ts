export type BarrierScenario = 'protected' | 'publish' | 'reclaim'
type Event = { actor: string; action: 'write' | 'read' | 'barrier'; value?: number; label: string }

export function barrierTrace(scenario: BarrierScenario) {
  const write: Event = { actor: '生产者', action: 'write', value: 7, label: '写入本轮值 7' }
  const publish: Event = { actor: '全块', action: 'barrier', label: '发布屏障完成' }
  const fast: Event = { actor: '消费者 A', action: 'read', label: '读取本轮值' }
  const slow: Event = { actor: '消费者 B', action: 'read', label: '读取本轮值' }
  const overwrite: Event = { actor: '生产者', action: 'write', value: 9, label: '下一轮覆盖为 9' }
  const scenarios: Record<BarrierScenario, Event[]> = {
    protected: [write, publish, fast, slow, { actor: '全块', action: 'barrier', label: '回收屏障完成' }, overwrite],
    publish: [slow, write, fast, overwrite],
    reclaim: [write, publish, fast, overwrite, slow],
  }
  if (!(scenario in scenarios)) throw new RangeError('Unknown barrier scenario')
  let slot = 0
  const reads: { actor: string; value: number; valid: boolean }[] = []
  const initial = { actor: '初始', label: 'shared 保留旧值 0', slot, reads: [...reads] }
  return [initial, ...scenarios[scenario].map(event => {
    if (event.action === 'write') slot = event.value!
    if (event.action === 'read') reads.push({ actor: event.actor, value: slot, valid: slot === 7 })
    return { actor: event.actor, label: event.label, slot, reads: [...reads] }
  })]
}

export function epilogue(row: number, col: number, bias: number, fused: boolean, stage: number) {
  if (!Number.isInteger(row) || row < 0 || row > 4 || !Number.isInteger(col) || col < 0 || col > 5 || !Number.isFinite(bias) || !Number.isInteger(stage) || stage < 0 || stage > 2) throw new RangeError('Invalid epilogue input')
  const c = Array.from({ length: 7 }, (_, q) => (row + q + 1) * (q - col)).reduce((a, b) => a + b, 0)
  const shifted = c + bias
  const values = [c, shifted, Math.max(0, shifted)]
  const transfers = fused ? [0, 0, 30 * 4] : [30 * 4, 30 * 12, 30 * 20]
  return { c, shifted, result: values[2], current: values[stage], bytes: transfers[stage], finalBytes: transfers[2] }
}

export const timingScopes = ['CPU 提交', 'C 的设备区间', 'A/B/C 的设备区间', '端到端'] as const
export function timingBoundary(scope: number) {
  if (!Number.isInteger(scope) || scope < 0 || scope > 3) throw new RangeError('Invalid timing scope')
  // One declared teaching clock: submit 0.4, setup 1, then A=5/B=3/C=2.
  const segments = [
    { label: 'CPU 提交', row: 0, start: 0, end: 0.4 },
    { label: '准备', row: 0, start: 0.4, end: 1.4 },
    { label: 'A', row: 1, start: 1.4, end: 6.4 },
    { label: 'B', row: 2, start: 1.4, end: 4.4 },
    { label: 'C', row: 2, start: 6.4, end: 8.4 },
  ]
  const [start, end] = [[0, 0.4], [6.4, 8.4], [1.4, 8.4], [0, 8.4]][scope]
  return { segments, start, end, duration: end - start, total: 8.4 }
}
