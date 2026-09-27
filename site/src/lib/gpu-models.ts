export type ResidencyLimits = {
  warps: number
  registers: number
  sharedKiB: number
  blocks: number
}

export function residency(threads: number, registers: number, sharedKiB: number, limits: ResidencyLimits) {
  const values = [threads, registers, sharedKiB, ...Object.values(limits)]
  if (values.some(value => !Number.isInteger(value) || value < 0) || threads < 1 || registers < 1 ||
    limits.warps < 1 || limits.registers < 1 || limits.sharedKiB < 1 || limits.blocks < 1) {
    throw new RangeError('Resource counts must be nonnegative integers with positive capacities.')
  }
  const warpsPerBlock = Math.ceil(threads / 32)
  // ponytail: ignores architecture allocation granularity; use CUDA occupancy APIs for real devices.
  const bounds = [
    ['warp', Math.floor(limits.warps / warpsPerBlock)],
    ['register', Math.floor(limits.registers / (threads * registers))],
    ['shared memory', sharedKiB === 0 ? limits.blocks : Math.floor(limits.sharedKiB / sharedKiB)],
    ['block', limits.blocks],
  ] as const
  const blocks = Math.min(...bounds.map(([, count]) => count))
  return { bounds, blocks, warpsPerBlock, activeWarps: blocks * warpsPerBlock, occupancy: blocks * warpsPerBlock / limits.warps }
}

export function streamSchedule(producer: number, independent: number, consumer: number, twoStreams: boolean, waitEvent: boolean) {
  if ([producer, independent, consumer].some(value => !Number.isFinite(value) || value <= 0)) {
    throw new RangeError('Durations must be finite and positive.')
  }
  // ponytail: assumes resources permit overlap; actual schedules require a GPU trace.
  const independentStart = twoStreams ? 0 : producer
  const consumerStart = Math.max(independentStart + independent, twoStreams && waitEvent ? producer : 0)
  const tasks = [
    { label: 'A', stream: 0, start: 0, end: producer },
    { label: 'B', stream: twoStreams ? 1 : 0, start: independentStart, end: independentStart + independent },
    { label: 'C', stream: twoStreams ? 1 : 0, start: consumerStart, end: consumerStart + consumer },
  ]
  return { tasks, total: Math.max(producer, consumerStart + consumer), safe: consumerStart >= producer, ordered: !twoStreams || waitEvent }
}
// ponytail: fixed legal snapshots illustrate scheduling; use device traces for actual timing or issue behavior.
export const schedulerFrames: { sms: number[][]; done: number[]; note: string }[] = [
  { sms: [[], []], done: [], note: '提交六个 block；尚未分配 SM 资源。' },
  { sms: [[0, 2], [1, 3]], done: [], note: '四个 block 驻留；编号没有规定分配顺序。' },
  { sms: [[0, 2], [1, 3]], done: [], note: 'B0 的 w0 等待数据；SM 0 可发射 B2 的 w0，等待不会释放 B0 的资源。' },
  { sms: [[0, 2], [3, 4]], done: [1], note: 'B1 先完成；SM 1 释放其资源，再接收 B4。' },
  { sms: [[5], [3, 4]], done: [0, 1, 2], note: 'B0、B2 已完成；SM 0 接收 B5。blocks 可分批推进。' },
  { sms: [[], []], done: [0, 1, 2, 3, 4, 5], note: '全部 block 完成。图示没有建立跨 block 的数据依赖。' },
]
