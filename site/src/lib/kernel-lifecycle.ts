export const kernelPhases = ['装载 A / B', '发布屏障', '寄存器累计', '回收屏障', '写回 C'] as const

export function kernelLifecycle(tile: number, step: number, edge = false) {
  if (!Number.isInteger(tile) || tile < 1 || tile > 4 || !Number.isInteger(step) || step < 0 || step > Math.ceil(7 / tile) * 4) {
    throw new RangeError('Expected tile 1–4 and an in-range lifecycle step')
  }
  const rounds = Math.ceil(7 / tile)
  const done = step === rounds * 4
  const round = done ? rounds - 1 : Math.floor(step / 4)
  const phase = done ? 4 : step % 4
  const row = edge ? Math.floor(4 / tile) * tile : 0
  const col = edge ? Math.floor(5 / tile) * tile : 0
  const rows = Math.min(tile, 5 - row), cols = Math.min(tile, 6 - col)
  const start = round * tile, width = Math.min(tile, 7 - start)
  const completedK = done ? 7 : phase >= 2 ? Math.min(7, start + tile) : start
  // One accumulator per logical output; actual register allocation comes from the compiler.
  let sample = 0
  for (let q = 0; q < completedK; q++) sample += (row + q + 1) * (q - col)
  return { rounds, round, phase, done, row, col, rows, cols, start, width, completedK, sample,
    sharedBytes: 8 * tile * tile, accumulatorSlots: tile * tile,
    validLoads: width * (rows + cols), paddingSlots: 2 * tile * tile - width * (rows + cols),
    validOutputs: rows * cols, canOverwrite: phase === 0 || phase === 3 || done,
    outputWritten: done }
}
