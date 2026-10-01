export type LossReduction = 'none' | 'sum' | 'mean'
const s = 1 / Math.sqrt(2)
export const lossLogits = [[s, -s, 0, 0], [-s, s, 0, 0], [s, -s, 0, 0], [-s, s, 0, 0], [s, -s, 0, 0], [s, -s, 0, 0]] as const

export function maskedLoss(targets: readonly number[], ignored: readonly boolean[], reduction: LossReduction) {
  if (targets.length !== 6 || ignored.length !== 6 || targets.some(value => !Number.isInteger(value) || value < 0 || value > 3)
      || ignored.some(value => typeof value !== 'boolean') || !['none', 'sum', 'mean'].includes(reduction)) throw new RangeError('Expected 6 targets in 0..3, boolean masks, and a reduction')
  const rows = lossLogits.map((logits, index) => {
    const maximum = Math.max(...logits)
    const exps = logits.map(value => Math.exp(value - maximum))
    const denominator = exps.reduce((sum, value) => sum + value, 0)
    const probability = exps.map(value => value / denominator)
    const rawLoss = maximum + Math.log(denominator) - logits[targets[index]]
    return { logits, probability, rawLoss, loss: ignored[index] ? 0 : rawLoss }
  })
  const valid = ignored.filter(value => !value).length
  const total = rows.reduce((sum, row) => sum + row.loss, 0)
  // Teaching policy: all ignored returns zero; raw framework mean has zero denominator.
  const result = reduction === 'none' ? rows.map(row => row.loss) : reduction === 'sum' ? total : total / Math.max(1, valid)
  return { rows, valid, total, result, empty: valid === 0 }
}
