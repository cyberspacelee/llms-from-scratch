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
